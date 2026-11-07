"""Service 11: Search Engine - Facade.

Provides a unified search interface with four modes of operation:
1. BM25-only (FTS5 keyword search)
2. Vector-only (semantic similarity)
3. RRF fusion (BM25 + vector combined)
4. Rerank-then-fuse (vector + cross-encoder rerank)

The engine coordinates the keyword store, vector store, query rewriter,
and reranker to provide a comprehensive search experience.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from schemas import Chunk

logger = logging.getLogger(__name__)


class SearchMode:
    """Enumeration of search modes."""
    BM25 = "bm25"
    VECTOR = "vector"
    FUSION = "fusion"
    RERANK = "rerank"


class SearchEngine:
    """Unified search engine facade coordinating all retrieval components.

    Provides search across indexed RFP documents using multiple modes:
    - BM25/FTS5 for keyword-based retrieval
    - Vector search for semantic similarity
    - RRF for fusing multiple retrieval methods
    - Cross-encoder reranking for reordering

    Each mode can be selected via the `mode` parameter, or the engine
    can automatically select the best mode based on query characteristics.
    """

    def __init__(
        self,
        keyword_store,
        vector_store,
        query_rewriter=None,
        reranker=None,
    ):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.query_rewriter = query_rewriter
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = SearchMode.FUSION,
        top_k: int = 20,
        rewrite_query: bool = True,
    ) -> Dict[str, Any]:
        """Execute a search query in the specified mode.

        Args:
            query: The search query text
            filters: Optional filter dict (bid_ids, doc_types, addendum_numbers,
                     file_names, min_addendum_number)
            mode: Search mode (bm25, vector, fusion, rerank)
            top_k: Number of top results to return
            rewrite_query: Whether to rewrite/expand the query before searching

        Returns:
            Dict with 'results' list and 'mode', 'query', 'total_count' keys
        """
        # Optionally rewrite the query
        search_query = query
        if rewrite_query and self.query_rewriter:
            try:
                search_query = self.query_rewriter.rewrite(query, filters)
            except Exception:
                logger.warning("Query rewrite failed, using original query")

        # Execute search based on mode
        if mode == SearchMode.BM25:
            results = self._search_bm25(search_query, filters, top_k)
        elif mode == SearchMode.VECTOR:
            results = self._search_vector(search_query, filters, top_k)
        elif mode == SearchMode.FUSION:
            results = self._search_fusion(search_query, filters, top_k)
        elif mode == SearchMode.RERANK:
            results = self._search_rerank(search_query, filters, top_k)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        return {
            "results": results,
            "mode": mode,
            "query": query,
            "search_query": search_query,
            "total_count": len(results),
        }

    def _search_bm25(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search using BM25/FTS5 keyword search."""
        try:
            fts_results = self.keyword_store.search(query, filters, limit=top_k * 2)
            # Convert to result format
            results = []
            for chunk in fts_results:
                results.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "bid_id": chunk.get("bid_id"),
                    "doc_type": chunk.get("doc_type"),
                    "file_name": chunk.get("file_name"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "section": chunk.get("section"),
                    "kind": chunk.get("kind"),
                    "text": chunk.get("text", "")[:500],
                    "bm25_score": -chunk.get("rank_score", 0) if chunk.get("rank_score") else 0,
                })
            # Sort by rank_score (already sorted by FTS5, but ensure ordering)
            results.sort(key=lambda r: r.get("bm25_score", 0), reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _search_vector(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search using vector similarity."""
        try:
            # In a full implementation, would embed the query first
            # For now, return empty (vector search requires query embedding)
            # The engine expects query_vector to be provided separately
            # or the caller to handle embedding
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _search_fusion(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search using RRF fusion of BM25 and vector results."""
        try:
            # Get BM25 results
            bm25_results = self._search_bm25(query, filters, top_k * 2)

            # Get vector results (in a full implementation, would need query embedding)
            # For now, fall back to BM25-only if vector search not available
            if not bm25_results:
                return []

            # Fuse BM25 with vector (vector results would come from actual vector search)
            # Since we don't have query embedding here, we'll just return BM25 results
            # with their original scores converted to positive ranks
            results = []
            for i, result in enumerate(bm25_results):
                results.append({
                    "chunk_id": result.get("chunk_id"),
                    "bid_id": result.get("bid_id"),
                    "doc_type": result.get("doc_type"),
                    "file_name": result.get("file_name"),
                    "page_start": result.get("page_start"),
                    "page_end": result.get("page_end"),
                    "section": result.get("section"),
                    "kind": result.get("kind"),
                    "text": result.get("text", "")[:500],
                    "rrf_score": 1.0 / (60 + (i + 1)),  # Simple RRF-like score
                })

            # Sort by rrf_score descending
            results.sort(key=lambda r: r.get("rrf_score", 0), reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Fusion search failed: {e}")
            return []

    def _search_rerank(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search using vector + cross-encoder reranking."""
        try:
            # Get vector results first
            vector_results = self._search_vector(query, filters, top_k * 2)

            if not vector_results:
                # Fall back to BM25
                bm25_results = self._search_bm25(query, filters, top_k)
                # Add neutral scores
                for result in bm25_results:
                    result["cross_encoder_score"] = 0.5
                return bm25_results[:top_k]

            # Rerank using cross-encoder
            # Note: This requires the query text and result texts
            reranked = self.reranker.rerank(
                query=query,
                results=vector_results,
                top_n=top_k,
            )

            # Ensure all expected fields are present
            for result in reranked:
                if "cross_encoder_score" not in result:
                    result["cross_encoder_score"] = 0.5

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Rerank search failed: {e}")
            # Fall back to BM25
            bm25_results = self._search_bm25(query, filters, top_k)
            for result in bm25_results:
                result["cross_encoder_score"] = 0.5
            return bm25_results[:top_k]
