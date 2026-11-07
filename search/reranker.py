"""Service 11: Reranker (Cross-Encoder Reranking).

Re-ranks search results from BM25/vector retrieval using a cross-encoder
model. The cross-encoder evaluates (query, passage) pairs and produces
a relevance score that is more accurate than individual encoders alone.

Results are reordered by the cross-encoder score, and a configurable
number of top results are returned.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from schemas import Chunk

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for search results.

    Takes an initial set of search results (from BM25/vector search)
    and re-ranks them using a cross-encoder model that scores (query,
    passage) pairs. This provides more accurate relevance ordering.

    Attributes:
        model_name: Name of the cross-encoder model to use
        top_n: Number of top results to return after reranking
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base",
                 top_n: int = 20):
        self.model_name = model_name
        self.top_n = top_n
        self._model = None

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not available; "
                    "reranker will use deterministic fallback"
                )
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        *,
        model_name: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank a list of search results using cross-encoder scores.

        Args:
            query: The original search query
            results: List of search result dicts, each must contain
                     'chunk_id' and 'text' (the passage text)
            model_name: Optional override for the model name
            top_n: Optional override for number of results to return

        Returns:
            Reranked results list, sorted by cross-encoder score descending,
            with 'cross_encoder_score' added to each result dict.
        """
        if not results:
            return []

        n_to_return = top_n or self.top_n
        n_to_return = min(n_to_return, len(results))

        model = self._get_model() or self._get_fallback_model(model_name)

        # Prepare (query, passage) pairs for the cross-encoder
        # Only include results that have text content
        pairs = []
        result_indices = []

        for i, result in enumerate(results):
            text = result.get("text", "")
            if text and text.strip():
                pairs.append((query, text))
                result_indices.append(i)

        if not pairs:
            # No results have text; return original order
            for i, result in enumerate(results[:n_to_return]):
                result["cross_encoder_score"] = 0.0
            return results[:n_to_return]

        try:
            # Get cross-encoder scores
            scores = self._model.predict(pairs)

            # Build result dicts with scores
            scored_results = []
            for j, (pair, score) in enumerate(zip(pairs, scores)):
                idx = result_indices[j]
                result = results[idx].copy()
                # Convert score to float; handle numpy scalars
                score_float = float(score) if not hasattr(score, 'item') else float(score.item())
                result["cross_encoder_score"] = score_float
                scored_results.append(result)

            # Sort by cross-encoder score descending
            scored_results.sort(key=lambda r: r["cross_encoder_score"], reverse=True)

            # Return only top n
            return scored_results[:n_to_return]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fall back to returning results with neutral scores
            for i, result in enumerate(results[:n_to_return]):
                result["cross_encoder_score"] = 0.5
            return results[:n_to_return]

    def _get_fallback_model(self, model_name: Optional[str] = None) -> Any:
        """Return a simple fallback scorer when the cross-encoder model is unavailable.

        Uses a deterministic heuristic based on query-term overlap with
        the passage text.
        """
        # This is a simple overlap-based fallback
        # In production, would use a lightweight model or the rerank API
        return _DeterministicReranker()


class _DeterministicReranker:
    """Deterministic fallback reranker using query-term overlap."""

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-passage pairs by inverse document frequency-like overlap.

        For each (query, passage) pair, computes the proportion of query
        terms that appear in the passage (case-insensitive, after basic
        normalization). Returns a list of scores in [0, 1].
        """
        scores = []
        for query, passage in pairs:
            # Normalize: lowercase, remove punctuation, split into terms
            query_terms = set(
                re.sub(r"[^\w\s]", "", query.lower()).split()
            )
            passage_terms = set(
                re.sub(r"[^\w\s]", "", passage.lower()).split()
            )

            if not query_terms:
                scores.append(0.0)
                continue

            # Jaccard-like overlap: |intersection| / |query terms|
            intersection = len(query_terms & passage_terms)
            score = intersection / len(query_terms)
            # Cap at 1.0
            score = min(score, 1.0)
            scores.append(score)

        return scores
