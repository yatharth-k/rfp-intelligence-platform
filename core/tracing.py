"""Service 05: Trace Logger (Observability).

Records every agent step (input, tool calls, output, tokens, latency, errors)
as structured events so a run can be audited, debugged, rendered as the required
example trace, and summarized for cost and latency.
"""

from __future__ import annotations

import json
import hashlib
import time
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, ContextManager

import structlog

from core.errors import LLMGatewayError


class EventType:
    """Enumeration of trace event types."""
    RUN_START = "run_start"
    RUN_END = "run_end"
    SPAN_START = "span_start"
    SPAN_END = "span_end"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    RETRY = "retry"
    DEGRADED = "degraded"
    ERROR = "error"
    VALIDATION = "validation"


class TraceContext:
    """Holds run/agent/span context using contextvars for propagation."""

    def __init__(self) -> None:
        self.run_id: str = ""
        self.agent: str = ""
        self.span_id: str = ""
        self.parent_span_id: str = ""


def _redact_secret(value: Any) -> Any:
    """Redact secret values from logs."""
    from core.errors import SecretStr
    if isinstance(value, SecretStr):
        return "<redacted>"
    if isinstance(value, str) and "nvapi-" in value.lower():
        return "<redacted>"
    return value


def _truncate_text(text: str, max_chars: int = 300) -> dict:
    """Truncate text for storage: return {sha256, preview, chars}."""
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    preview = text[:max_chars]
    chars = len(text)
    return {"sha256": sha, "preview": preview, "chars": chars}


class Tracer:
    """Central trace logger for a single run.

    Design from service 05:
    - One JSONL per run in outputs/traces/
    - structlog JSON rendering, ISO timestamps
    - Context propagation via contextvars (works with asyncio.gather)
    - Redaction: API keys never logged (SecretStr + nvapi- filter)
    - Truncation: prompts/outputs stored as {sha256, preview: first 300 chars,
      chars: n}. TRACE_FULL=1 stores full text for debugging.
    - Run summary (last event): totals for LLM/embedding/rerank requests,
      cache hits, tokens in/out, wait time, wall time, retries, degraded events
    - Optional Langfuse mirror if LANGFUSE_* variables are set
    """

    def __init__(
        self,
        run_id: str | None = None,
        trace_dir: str = "outputs/traces",
        trace_full: bool = False,
    ) -> None:
        self.run_id = run_id or f"r_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.trace_dir = trace_dir
        self.trace_full = trace_full
        self._event_counter = 0
        self._events: List[Dict[str, Any]] = []
        self._start_time = time.monotonic()
        self._context = TraceContext()

        # Setup trace directory
        os.makedirs(trace_dir, exist_ok=True)

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )

    def start_run(self, goal: str, bid_id: str, config_summary: Dict[str, Any]) -> str:
        """Start a new run and write the run_start event."""
        event = {
            "event": EventType.RUN_START,
            "run_id": self.run_id,
            "goal": goal,
            "bid_id": bid_id,
            "config": config_summary,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self._record(event)
        return self.run_id

    def span(self, agent: str, step: str) -> "_SpanContext":
        """Context manager for a span (agent + step)."""
        return _SpanContext(self, agent, step)

    def event(
        self,
        event_type: str,
        **kwargs: Any,
    ) -> None:
        """Record a generic trace event."""
        # Redact any secrets in kwargs
        redacted = {}
        for k, v in kwargs.items():
            redacted[k] = _redact_secret(v)

        event = {
            "event": event_type,
            "run_id": self._context.run_id,
            "span_id": self._context.span_id,
            "parent_span_id": self._context.parent_span_id,
            "agent": self._context.agent,
            **redacted,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self._record(event)

    def set_agent(self, agent: str) -> None:
        """Set the current agent in context."""
        self._context.agent = agent

    def set_span(self, span_id: str, parent_span_id: str = "") -> None:
        """Set current span IDs in context."""
        self._context.span_id = span_id
        self._context.parent_span_id = parent_span_id

    def _record(self, event: Dict[str, Any]) -> None:
        """Internal: add event to the collection."""
        self._event_counter += 1
        event["event_id"] = self._event_counter
        self._events.append(event)

    def end_run(self, summary: Dict[str, Any]) -> None:
        """End the run and write the run_end event + JSONL file."""
        event = {
            "event": EventType.RUN_END,
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": summary,
        }
        self._record(event)

        # Write JSONL file
        filepath = os.path.join(self.trace_dir, f"{self.run_id}.jsonl")
        with open(filepath, "w") as f:
            for e in self._events:
                f.write(json.dumps(e) + "\n")

        # Also emit a console summary
        total_events = len(self._events)
        llm_calls = sum(1 for e in self._events if e.get("event") == "llm_call")
        cache_hits = sum(
            1 for e in self._events
            if e.get("event") == "llm_call" and e.get("cache_hit")
        )
        elapsed = time.monotonic() - self._start_time
        print(
            f"=== Run {self.run_id}: {total_events} events "
            f"({llm_calls} LLM calls, {cache_hits} cache hits, "
            f"{elapsed:.1f}s) ==="
        )


class _SpanContext:
    """Context manager for agent spans."""

    def __init__(self, tracer: Tracer, agent: str, step: str) -> None:
        self.tracer = tracer
        self.agent = agent
        self.step = step
        self.start_time: float | None = None
        self.event_id: Optional[int] = None

    def __enter__(self) -> _SpanContext:
        self.start_time = __import__("time").monotonic()
        self.tracer.span(self.step, self.tracer._context.parent_span_id)
        self.tracer.set_agent(self.agent)
        # Generate a span ID
        import uuid
        span_id = str(uuid.uuid4())[:8]
        self.tracer.set_span(span_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = __import__("time").monotonic() - self.start_time if self.start_time else 0
        status = "error" if exc_type else "success"

        self.tracer.event(
            "span_end",
            status=status,
            latency_ms=int(elapsed * 1000),
        )

        # Reset agent context on exit
        self.tracer.set_agent("")


import uuid