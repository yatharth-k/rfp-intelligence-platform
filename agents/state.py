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
from typing import Any, Dict, List, Literal, Optional, TypedDict

from schemas import Chunk, DocType, Evidence, FieldResult, SourceRef
from pydantic import BaseModel, Field


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


# ============================================================
# QA and Final Record schemas
# ============================================================


class Citation(BaseModel):
    """A citation referencing a source document."""
    file_name: str = Field(..., description="Source file name")
    page: int = Field(..., ge=1, description="Page number")
    text: Optional[str] = Field(None, description="Excerpt from the source")


class QAAnswer(BaseModel):
    """Answer to a free-form question with citations."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer text")
    citations: List[Citation] = Field(
        default_factory=list, description="Supporting citations"
    )
    per_bid: Optional[Dict[str, str]] = Field(
        None, description="Per-bid answers for comparison questions"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the answer"
    )
    not_found: bool = Field(
        False, description="Whether the answer was not found in documents"
    )
    notes: List[str] = Field(
        default_factory=list, description="Additional notes"
    )
    plan: Dict[str, Any] = Field(
        default_factory=dict, description="Execution plan/sub-queries used"
    )


class FinalBidRecord(BaseModel):
    """Final per-bid record with all extracted data, validation, and summary."""
    bid_id: str = Field(..., description="Bid identifier")
    fields: Dict[str, Dict[str, Any]] = Field(
        ..., description="Extracted fields with values, sources, confidence, status"
    )
    addendum_changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected addendum changes"
    )
    validation: Dict[str, int] = Field(
        ..., description="Validation summary: passed, failed, not_found counts"
    )
    summary_text: str = Field(..., description="3-6 sentence bid summary")
    run: Dict[str, Any] = Field(
        ..., description="Run metadata: models, timestamps, degraded components, warnings"
    )


# ============================================================
# Extraction schemas
# ============================================================


class FieldExtraction(BaseModel):
    """Single field extraction result from LLM."""
    name: str = Field(..., description="Field name")
    status: Literal["found", "not_found", "explicit_negative"] = Field(
        "not_found", description="Extraction status"
    )
    value: Optional[Any] = Field(None, description="Extracted value")
    evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence IDs supporting this value"
    )
    quote: Optional[str] = Field(None, description="Verbatim quote from evidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    note: Optional[str] = Field(None, description="Reasoning note")
