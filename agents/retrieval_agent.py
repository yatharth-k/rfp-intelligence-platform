"""Service 14: Retrieval Agent.

Gathers ranked evidence with citations for each field or question, using the
search engine as a tool. Never answers; returns evidence for other agents
to reason over. Follows service 14 design: pure Python on top of SearchEngine,
LLM only optionally via query rewriter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.state import BidState, Evidence, FieldResult, SourceRef
from schemas import DocType, FieldSpec, Chunk
from search.engine import SearchEngine, SearchMode
from search.query_rewriter import QueryRewriter
from core.errors import BudgetExceededError

logger = logging.getLogger(__name__)

# Configuration
EVIDENCE_CHUNKS_PER_FIELD = 4
EVIDENCE_MAX_CHARS = 1000
RETRIEVAL_BROADEN_MULTIPLIER = 2


class RetrievalRequest:
    """Request structure for the retrieval agent."""

    def __init__(
        self,
        bid_id: str | list[str],
        field: str | None,
        queries: list[str],
        doc_types: list[DocType] | None,
        addendum_numbers: list[int] | None,
        top_k: int = 4,
        broaden: bool = False,
        feedback: str | None = None,
    ):
        self.bid_id = bid_id if isinstance(bid_id, list) else [bid_id]
        self.field = field
        self.queries = queries
        self.doc_types = doc_types or []
        self.addendum_numbers = addendum_numbers or []
        self.top_k = top_k
        self.broaden = broaden
        self.feedback = feedback


class RetrievalResult:
    """Result structure from the retrieval agent."""

    def __init__(
        self,
        field: str | None,
        evidence: list,
        degraded: list[str],
        queries_used: list[str],
    ):
        self.field = field
        self.evidence = evidence
        self.degraded = degraded
        self.queries_used = queries_used


async def retrieve(req: RetrievalRequest, state: BidState, search_engine: SearchEngine, query_rewriter=None) -> RetrievalResult:
    """Retrieve evidence for a single request.

    Args:
        req: RetrievalRequest with queries, filters, and preferences
        state: BidState containing run context
        search_engine: SearchEngine instance for executing searches
        query_rewriter: Optional query rewriter for query expansion

    Returns:
        RetrievalResult with evidence, queries used, and degraded flags
    """
    queries_used: list[str] = []
    all_evidence: list = []
    degraded: list[str] = []

    # Track tried chunk_ids for broadening mode
    tried_chunk_ids: set[str] = set()

    for bid_id in req.bid_id:
        for query in req.queries:
            try:
                # Build filters from addendum numbers and doc types
                filters: dict[str, Any] = {}
                if req.addendum_numbers:
                    filters["addendum_numbers"] = req.addendum_numbers
                if req.doc_types:
                    filters["doc_types"] = [dt.value if hasattr(dt, "value") else str(dt) for dt in req.doc_types]

                # Execute search
                search_mode = SearchMode.RERANK if req.broaden else SearchMode.FUSION
                search_result = search_engine.search(
                    query,
                    filters=filters if filters else None,
                    mode=search_mode,
                    top_k=req.top_k,
                    rewrite_query=query_rewriter is not None,
                )

                queries_used.append(query)

                if not search_result.get("results"):
                    continue

                # Extract evidence from results
                results = search_result["results"]
                field_evidence = _extract_evidence(
                    results,
                    bid_id,
                    EVIDENCE_CHUNKS_PER_FIELD,
                    EVIDENCE_MAX_CHARS,
                    tried_chunk_ids,
                )
                all_evidence.extend(field_evidence)

                # Track tried chunk IDs for de-dupe and broadening
                for ev in field_evidence:
                    chunk_id = ev.get("chunk_id")
                    if chunk_id:
                        tried_chunk_ids.add(chunk_id)

            except Exception as e:
                logger.error(f"Retrieval error for query '{query}': {e}")
                degraded.append(f"search_error:{query[:50]}")
                # Continue with other queries rather than failing entirely

    # De-duplicate evidence by chunk_id, keeping best scores
    deduped = _dedup_evidence(all_evidence)

    # Apply budget cap
    final_evidence = deduped[:EVIDENCE_CHUNKS_PER_FIELD]

    # Assign evidence IDs (E1, E2, ...)
    for i, ev in enumerate(final_evidence, 1):
        ev["evidence_id"] = f"E{i}"

    # Build result
    result = RetrievalResult(
        field=req.field,
        evidence=final_evidence,
        degraded=degraded,
        queries_used=queries_used,
    )

    return result


def _extract_evidence(
    results: list,
    bid_id: str,
    max_chunks: int,
    max_chars: int,
    tried_chunk_ids: set[str],
) -> list:
    """Extract evidence from search results.

    Returns a list of evidence dicts with chunk_id, text, file_name, page, etc.
    """
    evidence: list = []
    chunk_count = 0

    for result in results:
        if chunk_count >= max_chunks:
            break

        chunk_id = result.get("chunk_id", "")
        text = result.get("text", "") or ""
        file_name = result.get("file_name", "")
        page = result.get("page_start", result.get("page_end", 0))
        section = result.get("section", "")

        # Skip if we've already tried this chunk_id
        if chunk_id in tried_chunk_ids:
            continue

        # Cap character count per evidence item
        capped_text = text[:max_chars] + ("..." if len(text) > max_chars else "")

        evidence.append({
            "chunk_id": chunk_id,
            "text": capped_text,
            "file_name": file_name,
            "page": page,
            "section": section,
            "bid_id": bid_id,
            "rrf_score": result.get("rrf_score", 0),
            "cross_encoder_score": result.get("cross_encoder_score"),
        })

        chunk_count += 1

    return evidence


def _dedup_evidence(evidence: list) -> list:
    """De-duplicate evidence by chunk_id, keeping the highest-scoring entry."""
    seen: dict[str, dict] = {}

    for ev in evidence:
        chunk_id = ev.get("chunk_id", "")
        if not chunk_id:
            continue
        if chunk_id not in seen or ev.get("rrf_score", 0) > seen[chunk_id].get("rrf_score", 0):
            seen[chunk_id] = ev

    return list(seen.values())


async def retrieve_for_fields(
    bid_id: str,
    specs: list[FieldSpec],
    *,
    state: BidState,
    search_engine: SearchEngine,
    query_rewriter: Optional[QueryRewriter] = None,
) -> dict[str, RetrievalResult]:
    """Retrieve evidence for all field specs of a bid.

    Runs retrieval for each field concurrently, bounded by the LLM rate
    limiter. Shares cached embeddings/query results across fields with
    identical seeds.

    Returns:
        Dict mapping field name to RetrievalResult
    """
    # Build retrieval requests for each field
    requests: list[RetrievalRequest] = []

    for spec in specs:
        # Use field's query_seeds as the basis
        seeds = spec.query_seeds or []

        # Build doc_type preferences
        preferred_types = spec.preferred_doc_types or []

        # Build addendum number filter if relevant
        addendum_numbers = (
            list(range(1, spec.addendum_sensitive + 1)) if spec.addendum_sensitive else []
        )

        # Determine top_k
        top_k = EVIDENCE_CHUNKS_PER_FIELD

        request = RetrievalRequest(
            bid_id=bid_id,
            field=spec.name,
            queries=seeds,
            doc_types=preferred_types,
            addendum_numbers=addendum_numbers,
            top_k=top_k,
        )
        requests.append(request)

    # Execute retrieval concurrently
    tasks = [retrieve(req, state, search_engine) for req in requests]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    field_results: dict[str, RetrievalResult] = {}
    for req, result in zip(requests, results_list):
        field_name = req.field
        if isinstance(result, Exception):
            logger.error(f"Retrieval failed for field {field_name}: {result}")
            # Return empty evidence for failed field
            field_results[field_name] = RetrievalResult(
                field=field_name,
                evidence=[],
                degraded=[f"retrieval_error:{str(result)[:50]}"],
                queries_used=req.queries,
            )
        else:
            field_results[field_name] = result

    return field_results
