"""Service 11: Hybrid Search - RRF Fusion.

Implements Reciprocal Rank Fusion (RRF) for combining results from
multiple retrieval modes (BM25/FTS5 keyword search, vector similarity,
and reranking). RRF is aggregation-independent and works well with
diverse retrieval methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def rrf_fuse(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and vector search results using Reciprocal Rank Fusion.

    RRF formula: score = sum(1 / (k + rank)) for each method that returns
    the document. The resulting scores are sorted descending, with higher
    scores indicating better combined relevance.

    Args:
        bm25_results: List of dicts with 'chunk_id' and optionally 'bm25_score'
                      (negative BM25 scores converted to positive ranks internally)
        vector_results: List of dicts with 'chunk_id' and 'vector_score' (cosine similarity)
        k: RRF constant (default 60, standard value for fused retrieval)

    Returns:
        List of dicts with 'chunk_id', 'rrf_score', and merged metadata,
        sorted by rrf_score descending.
    """
    # Build rank mappings: chunk_id -> rank (1-indexed) for each method
    bm25_rank: Dict[str, int] = {}
    for i, result in enumerate(bm25_results):
        chunk_id = result.get("chunk_id")
        if chunk_id and chunk_id not in bm25_rank:
            bm25_rank[chunk_id] = i + 1  # 1-indexed rank

    vector_rank: Dict[str, int] = {}
    for i, result in enumerate(vector_results):
        chunk_id = result.get("chunk_id")
        if chunk_id and chunk_id not in vector_rank:
            vector_rank[chunk_id] = i + 1  # 1-indexed rank

    # Collect all unique chunk IDs
    all_chunk_ids = set(bm25_rank.keys()) | set(vector_rank.keys())

    # Compute RRF scores
    rrf_scores: Dict[str, float] = {}
    for chunk_id in all_chunk_ids:
        bm25_rank = bm25_rank.get(chunk_id, 0)
        vector_rank_val = vector_rank.get(chunk_id, 0)

        # RRF score: 1/(k + rank) for each method present
        bm25_contribution = 1.0 / (k + bm25_rank) if bm25_rank > 0 else 0.0
        vector_contribution = 1.0 / (k + vector_rank_val) if vector_rank_val > 0 else 0.0

        rrf_scores[chunk_id] = bm25_contribution + vector_contribution

    # Sort by RRF score descending
    sorted_chunks = sorted(
        all_chunk_ids,
        key=lambda cid: rrf_scores.get(cid, 0),
        reverse=True,
    )

    # Build result objects with metadata from both result sets
    results: List[Dict[str, Any]] = []
    # Build lookup dicts for metadata
    bm25_meta = {}
    for result in bm25_results:
        cid = result.get("chunk_id")
        if cid:
            bm25_meta[cid] = result

    vector_meta = {}
    for result in vector_results:
        cid = result.get("chunk_id")
        if cid:
            vector_meta[cid] = result

    for chunk_id in sorted_chunks:
        rrf_score = rrf_scores.get(chunk_id, 0)
        # Combine metadata from both sources
        meta = {}
        if chunk_id in bm25_meta:
            meta.update(bm25_meta[chunk_id])
        if chunk_id in vector_meta:
            meta.update(vector_meta[chunk_id])

        results.append({
            "chunk_id": chunk_id,
            "rrf_score": rrf_score,
            "rank": len(results) + 1,  # 1-indexed position in fused results
            **meta,
        })

    return results


def fuse_with_rerank(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    rerank_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse three retrieval modes: BM25, vector, and reranker.

    First fuses BM25 and vector via RRF, then incorporates reranker
    results by adjusting positions based on cross-encoder scores.

    Args:
        bm25_results: BM25/FTS5 search results
        vector_results: Vector similarity search results
        rerank_results: Cross-encoder reranker results (already sorted by score)
        k: RRF constant

    Returns:
        Fused and reordered results list
    """
    # Step 1: RRF fuse BM25 and vector
    fused = rrf_fuse(bm25_results, vector_results, k=k)

    # Step 2: Incorporate reranker results
    # Build a position map from rerank results (lower position = higher relevance)
    rerank_position: Dict[str, int] = {}
    for i, result in enumerate(rerank_results):
        chunk_id = result.get("chunk_id")
        if chunk_id:
            rerank_position[chunk_id] = i  # 0-indexed position

    # Adjust: give a boost to chunks that appear in rerank results
    # by slightly increasing their RRF score contribution
    for result in fused:
        chunk_id = result.get("chunk_id")
        if chunk_id in rerank_position:
            # Boost score: add a small fraction based on rerank position
            rerank_pos = rerank_position[chunk_id]
            boost = 1.0 / (60 + rerank_pos + 1)  # Similar formula, offset
            result["rrf_score"] += boost * 0.1  # 10% of a typical RRF contribution

    # Re-sort by final score
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)

    return fused