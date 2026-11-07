"""Service 12: Orchestrator and Shared State - Agent State.

Defines the shared BidState TypedDict used across all agents, along with
reducer functions and state checkpointing utilities. Follows service 12
design sections 5-7.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from schemas import Chunk, DocType, FieldResult, SourceRef


def merge_dicts(
    d1: Dict[str, Any], d2: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two dicts; for conflicting keys, d2 values take precedence.

    Used as a LangGraph state reducer so parallel agent branches can write
    to the same shared state dict without overwriting each other's disjoint
    key sets.
    """
    result = dict(d1)
    result.update(d2)
    return result


class BidState(TypedDict, total=False):
    """Shared state for the orchestrator graph (services 12-18).

    Each agent returns a partial update containing only the keys it owns.
    Reducers (`merge_dicts`, `operator.add`) handle parallel writes safely.
    """
    run_id: str
    goal: str  # "extract", "ask", or "index"
    bid_id: str
    question: str | None
    plan: dict  # RunPlan structure
    bids_in_scope: list[dict]  # BidInfo entries
    evidence: dict  # field -> list[Evidence]
    draft_fields: dict  # field -> FieldResult
    reconciled_fields: dict  # field -> FieldResult
    addendum_changes: list[dict]
    validation: dict  # field -> ValidationVerdict
    retries: dict  # field -> int count
    final_output: dict | None
    answer: dict | None  # QAAnswer structure
    errors: list[dict]  # ErrorRecord entries
    degraded: list[str]
    llm_calls_used: int


def init_run(bid_id: str, goal: str, question: str | None = None) -> BidState:
    """Initialize a new run state.

    Args:
        bid_id: The bid identifier
        goal: "extract", "ask", or "index"
        question: Optional question (for "ask" goal)

    Returns:
        Initialized BidState
    """
    run_id = str(uuid.uuid4())[:8]
    return {
        "run_id": run_id,
        "goal": goal,
        "bid_id": bid_id,
        "question": question,
        "plan": {},
        "bids_in_scope": [],
        "evidence": {},
        "draft_fields": {},
        "reconciled_fields": {},
        "addendum_changes": [],
        "validation": {},
        "retries": {},
        "final_output": None,
        "answer": None,
        "errors": [],
        "degraded": [],
        "llm_calls_used": 0,
    }


def checkpoint_state(state: BidState, output_dir: str = "outputs") -> str:
    """Dump state to disk for checkpointing.

    Args:
        state: The BidState to checkpoint
        output_dir: Base output directory

    Returns:
        Path to the checkpoint file
    """
    run_id = state.get("run_id", "unknown")
    run_dir = os.path.join(output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "state.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> BidState:
    """Load a checkpointed state from disk.

    Args:
        checkpoint_path: Path to the state.json file

    Returns:
        Loaded BidState
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    # Ensure lists are proper Python lists
    state.setdefault("errors", [])
    state.setdefault("degraded", [])
    state.setdefault("retries", {})
    return state


def increment_llm_calls(state: BidState, count: int = 1) -> BidState:
    """Increment the LLM call counter in state."""
    state["llm_calls_used"] = state.get("llm_calls_used", 0) + count
    return state


def add_error(state: BidState, error_dict: dict) -> BidState:
    """Add an error record to state."""
    errors = state.get("errors", [])
    errors.append(error_dict)
    state["errors"] = errors
    return state


def add_degraded(state: BidState, component: str) -> BidState:
    """Add a degraded component to state."""
    degraded = state.get("degraded", [])
    if component not in degraded:
        degraded.append(component)
    state["degraded"] = degraded
    return state
