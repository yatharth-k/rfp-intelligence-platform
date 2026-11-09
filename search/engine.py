"""Service 11: Search Engine - Facade.

Provides a unified search interface with four modes of operation:
1. BM25-only (FTS5 keyword search)
2. Vector-only (semantic similarity)
3. RRF fusion (BM25 + vector combined)
4. Hybrid + Rerank (vector + cross-encoder rerank)

The engine coordinates the keyword store, vector store, query rewriter,
and reranker to provide a comprehensive search experience.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from schemas import Chunk
from core.embeddings import EmbeddingClient

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
        embedding_client: Optional[EmbeddingClient] = None,
        query_rewriter=None,
        reranker=None,
    ):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.embedding_client = embedding_client
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
            degraded = False
        elif mode == SearchMode.VECTOR:
            results = self._search_vector(search_query, filters, top_k)
            degraded = False
        elif mode == SearchMode.FUSION:
            results = self._search_fusion(search_query, filters, top_k)
            degraded = False
        elif mode == SearchMode.RERANK:
            results, degraded = self._search_rerank(search_query, filters, top_k)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        return {
            "results": results,
            "mode": mode,
            "query": query,
            "search_query": search_query,
            "total_count": len(results),
            "degraded": degraded,
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
            if self.embedding_client is None:
                logger.warning("No embedding client available for vector search")
                return []

            # Embed query with input_type="query" (asymmetric embedding)
            query_vector = self.embedding_client.embed_query(query)

            # Build filter for vector store
            vector_filters = None
            if filters:
                # Vector store expects single values, not lists
                if filters.get("bid_ids"):
                    vector_filters = {"bid_id": filters["bid_ids"][0]}  # Single bid for now
                # Add other filters if needed

            # Search vector store
            vs_results = self.vector_store.search(
                query_vector=query_vector.tolist(),
                filters=vector_filters,
                limit=top_k * 2,
            )

            # Convert to result format
            results = []
            for chunk in vs_results:
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
                    "vector_score": chunk.get("score", 0),
                })
            # Sort by vector score descending
            results.sort(key=lambda r: r.get("vector_score", 0), reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _search_fusion(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search using RRF fusion of BM25 and vector results.

        RRF (Reciprocal Rank Fusion) combines ranked lists:
        score = sum(1 / (k + rank_i)) for each list i
        where k=60 is the standard RRF parameter.
        """
        try:
            # Get BM25 results (fetch more for better fusion - BM25 can have relevant results at lower ranks)
            bm25_limit = max(top_k * 10, 50)
            bm25_results = self._search_bm25(query, filters, bm25_limit)

            # Get vector results
            vector_results = self._search_vector(query, filters, top_k * 3)

            if not bm25_results and not vector_results:
                return []

            # Build RRF scores
            # Create a map of chunk_id -> {result_data, bm25_rank, vector_rank}
            fused = {}

            # Assign BM25 ranks
            for rank, result in enumerate(bm25_results):
                cid = result.get("chunk_id")
                if cid:
                    fused[cid] = {
                        "data": result,
                        "bm25_rank": rank + 1,
                        "vector_rank": None,
                    }

            # Assign vector ranks
            for rank, result in enumerate(vector_results):
                cid = result.get("chunk_id")
                if cid:
                    if cid in fused:
                        fused[cid]["vector_rank"] = rank + 1
                    else:
                        fused[cid] = {
                            "data": result,
                            "bm25_rank": None,
                            "vector_rank": rank + 1,
                        }

            # Calculate RRF scores
            K = 60  # RRF constant
            results = []
            for cid, info in fused.items():
                rrf_score = 0.0
                if info["bm25_rank"] is not None:
                    rrf_score += 1.0 / (K + info["bm25_rank"])
                if info["vector_rank"] is not None:
                    rrf_score += 1.0 / (K + info["vector_rank"])

                result = dict(info["data"])
                result["rrf_score"] = rrf_score
                result["bm25_rank"] = info["bm25_rank"]
                result["vector_rank"] = info["vector_rank"]
                results.append(result)

            # Sort by RRF score descending
            results.sort(key=lambda r: r.get("rrf_score", 0), reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Fusion search failed: {e}")
            # Fall back to BM25
            return self._search_bm25(query, filters, top_k)

    def _search_rerank(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Search using vector search + cross-encoder reranking.

        Returns (results, degraded) where degraded=True if reranker failed
        and we fell back to hybrid (fusion) results.
        """
        try:
            if self.reranker is None:
                logger.warning("No reranker available, falling back to fusion")
                return self._search_fusion(query, filters, top_k), True

            # Get vector results first (more candidates for reranking)
            vector_results = self._search_vector(query, filters, top_k * 4)

            if not vector_results:
                # Fall back to BM25 if vector search fails
                logger.warning("Vector search returned no results, falling back to fusion")
                return self._search_fusion(query, filters, top_k), True

            # Rerank using cross-encoder
            reranked = self.reranker.rerank(
                query=query,
                results=vector_results,
                top_n=top_k,
            )

            # Ensure all expected fields are present
            for result in reranked:
                if "cross_encoder_score" not in result:
                    result["cross_encoder_score"] = 0.5

            return reranked[:top_k], False

        except Exception as e:
            logger.error(f"Rerank search failed: {e}")
            # Fall back to fusion (not BM25, fusion is better)
            return self._search_fusion(query, filters, top_k), True
