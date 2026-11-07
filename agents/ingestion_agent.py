"""Service 13: Ingestion Agent.

Wrapper around the ingestion and indexing pipeline (services 06-09). Given a
bid folder, ensures it is parsed, cleaned, tagged, chunked and indexed, and
reports what happened in a structured way the orchestrator (12) can act on.

Follows service 13 design: deterministic Python calling services 06-09,
LLM gateway only as fallback for document-type classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.cleaning import clean_document
from ingestion.report import ingest_folder
from search.indexer import Indexer
from search.keyword_store import KeywordStore
from search.vector_store import QdrantVectorStore
from agents.state import BidState, add_degraded, increment_llm_calls

logger = logging.getLogger(__name__)


class FileProblem(Dict[str, Any]):
    """Problem with a single file during ingestion."""
    pass


class IngestionResult:
    """Structured result from the ingestion agent."""

    def __init__(
        self,
        bid_id: str,
        files_total: int,
        files_indexed: int,
        files_skipped_unchanged: int,
        files_failed: List[FileProblem],
        pages_needing_ocr: List[Tuple[str, int]],
        chunks_total: int,
        doc_types_found: Dict[str, int],
        addendum_numbers: List[int],
        proceed: bool,
        warnings: List[str],
    ):
        self.bid_id = bid_id
        self.files_total = files_total
        self.files_indexed = files_indexed
        self.files_skipped_unchanged = files_skipped_unchanged
        self.files_failed = files_failed
        self.pages_needing_ocr = pages_needing_ocr
        self.chunks_total = chunks_total
        self.doc_types_found = doc_types_found
        self.addendum_numbers = addendum_numbers
        self.proceed = proceed
        self.warnings = warnings

    def to_state_update(self) -> Dict[str, Any]:
        """Convert to a partial BidState update."""
        return {
            "bids_in_scope": [
                {
                    "bid_id": self.bid_id,
                    "file_count": self.files_total,
                    "chunk_count": self.chunks_total,
                }
            ],
            "evidence": {},  # Would be populated with field evidence
            "degraded": self.warnings if self.warnings else [],
            "llm_calls_used": 0,  # Ingestion is deterministic; no LLM calls unless fallback
        }


def _decide_proceed(report: Any, doc_types_found: Dict[str, int],
                    addendum_numbers: List[int]) -> Tuple[bool, List[str]]:
    """Determine whether the orchestrator should proceed with extraction.

    Rules (service 13 section 6.5):
    - At least one document with usable text indexed → proceed=True
    - No RFP-type document but a bid page → proceed with warning
    - Nothing indexed → proceed=False
    """
    warnings: List[str] = []

    if report.files_indexed > 0:
        # At least something was indexed
        if "rfp" in doc_types_found or "addendum" in doc_types_found:
            return True, warnings
        # No RFP doc but bid page was indexed
        if not warnings:
            warnings.append(
                "No RFP-type document indexed; fields may be incomplete. "
                "Proceeding with bid-page-only data."
            )
        return True, warnings
    else:
        # Nothing indexed
        return False, [
            "No documents with usable text were indexed. "
            "Cannot run extraction on an empty index."
        ]


def run_ingestion(
    state: BidState,
    *,
    folder: Optional[Path] = None,
    bid_id: Optional[str] = None,
    force: bool = False,
) -> BidState:
    """Run the ingestion agent from the given state.

    Args:
        state: BidState with run context
        folder: Path to the bid folder (default: DATA_DIR/<bid_id>)
        bid_id: The bid identifier (default: from state)
        force: If True, reindex all files regardless of registry

    Returns:
        Updated BidState with ingestion results
    """
    # Resolve bid_id and folder
    if bid_id is None:
        bid_id = state.get("bid_id", "")
    if folder is None:
        # Default to a path based on bid_id
        bid_folder = Path(f"./data/{bid_id}") if bid_id else Path("./data")
    else:
        bid_folder = folder

    # Validate folder exists
    if not bid_folder.exists():
        add_error(state, {
            "type": "folder_not_found",
            "message": f"Bid folder not found: {bid_folder}",
        })
        state["degraded"].append("folder_not_found")
        return state

    # Run the ingestion pipeline
    try:
        parsed_docs, report = ingest_folder(
            bid_folder,
            bid_id=bid_id or bid_folder.name,
            only_files=None,
            recursive=False,
        )
    except Exception as e:
        add_error(state, {
            "type": "ingestion_error",
            "message": f"Ingestion pipeline failed: {str(e)}",
        })
        state["degraded"].append("ingestion_error")
        return state

    # Update LLM call count (ingestion may use LLM fallback)
    increment_llm_calls(state, report.warnings.count("LLM fallback")) if hasattr(report, 'warnings') else None

    # Determine whether to proceed
    proceed, warnings = _decide_proceed(
        report,
        report.doc_types_found if hasattr(report, 'doc_types_found') else {},
        report.addendum_numbers if hasattr(report, 'addendum_numbers') else [],
    )

    # Convert to structured result
    result = IngestionResult(
        bid_id=bid_id or bid_folder.name,
        files_total=len(report.files) if report.files else 0,
        files_indexed=report.files_indexed if hasattr(report, 'files_indexed') else 0,
        files_skipped_unchanged=getattr(report, 'files_skipped_unchanged', 0),
        files_failed=[
            FileProblem(file=f.file, error=f.error or "Parse or cleaning failed")
            for f in getattr(report, 'file_reports', [])
            if f.status == "failed"
        ],
        pages_needing_ocr=[
            (f.file, getattr(f, 'empty_pages', 0) + getattr(f, 'needs_ocr_pages', 0))
            for f in getattr(report, 'file_reports', [])
            if getattr(f, 'needs_ocr', False) or getattr(f, 'empty_pages', 0) > 0
        ],
        chunks_total=sum(
            getattr(d, 'n_chunks', 0) for d in getattr(report, 'parsed_docs', [])
        ) if hasattr(report, 'parsed_docs') else 0,
        doc_types_found={
            dt: count for dt, count in (
                getattr(report, 'doc_types_found', {}) or {}
            ).items()
        } if hasattr(report, 'doc_types_found') else {},
        addendum_numbers=[
            getattr(f, 'addendum_number', None)
            for f in getattr(report, 'file_reports', [])
            if getattr(f, 'addendum_number', None)
        ],
        proceed=proceed,
        warnings=warnings,
    )

    # Update state with results
    state.update(result.to_state_update())

    # Add any report warnings as degraded components
    for warning in warnings:
        add_degraded(state, f"ingestion_warning: {warning}")

    # Log the results
    logger.info(
        f"Ingestion for bid {state.get('bid_id', 'unknown')}: "
        f"{result.files_indexed}/{result.files_total} files indexed, "
        f"proceed={proceed}"
    )

    return state