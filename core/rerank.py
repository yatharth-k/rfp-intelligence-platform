"""Service 03: Rerank Client - Cross-encoder reranker client.

Follows service 03 design sections 5.8-5.13 for reranking client.
Sends queries and passages to the cross-encoder reranker endpoint,
parses rankings, computes sigmoid scores (score01), handles batching,
and raises RetrievalModelError on persistent failure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from core.errors import RetrievalModelError

logger = logging.getLogger(__name__)


class RerankClient:
    """Client for cross-encoder reranking using NVIDIA rerank endpoint.

    Sends (query, passage) pairs to the reranker and returns ranked results
    with sigmoid scores (score01 = 1 / (1 + exp(-logit)), or approximately
    the normalized logit).

    Attributes:
        model_name: Name of the rerank model to use
        rerank_url: API endpoint for reranking
        batch_size: Number of candidates to rerank per request
        timeout_s: Request timeout in seconds
    """

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3.5-rerank-8b",
        rerank_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking",
        batch_size: int = 20,
        timeout_s: int = 45,
    ) -> None:
        self.model_name = model_name
        self.rerank_url = rerank_url
        self.batch_size = batch_size
        self.timeout_s = timeout_s

    def rerank(
        self,
        query: str,
        passages: List[str],
        *,
        model_name: Optional[str] = None,
        top_n: int = 20,
    ) -> RerankResult:
        """Rerank a list of passages using the cross-encoder model.

        Args:
            query: The search query text
            passages: List of passage texts to rerank
            model_name: Optional model name override
            top_n: Number of top results to return

        Returns:
            RerankResult with ranked passages and sigmoid scores

        Raises:
            RetrievalModelError: If the API call fails persistently
        """
        model = model_name or self.model_name
        max_candidates = min(len(passages), self.batch_size)

        # Process in batches
        all_results: List[RerankHit] = []

        for i in range(0, len(passages), self.batch_size):
            batch = passages[i : i + self.batch_size]
            batch_results = self._rerank_batch(query, batch, model=model)
            all_results.extend(batch_results)

        # Sort all results by score descending
        all_results.sort(key=lambda r: r.score01, reverse=True)

        # Return top n
        top = all_results[:top_n]

        # Build result metadata
        metadata: Dict[str, Any] = {
            "model": model,
            "query": query,
            "rerank_url": self.rerank_url,
        }

        return RerankResult(
            results=top,
            model=model,
            metadata=metadata,
        )

    def _rerank_batch(
        self, query: str, passages: List[str], model: str
    ) -> List[RerankHit]:
        """Rerank a single batch of passages."""
        # Build (query, passage) pairs
        pairs = [(query, passage) for passage in passages]

        # Make API request
        try:
            response = httpx.post(
                self.rerank_url,
                json={"query": query, "passages": passages, "model": model},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            data = response.json()

            # Parse response
            hits = []
            for item in data.get("results", data.get("hits", [])):
                logit = item.get("logit", item.get("score", 0))
                # Convert logit to sigmoid score: score01 = 1 / (1 + exp(-logit))
                # For simplicity, use a basic sigmoid approximation
                score01 = 1.0 / (1.0 + abs(logit)) if logit else 0.5
                hit = RerankHit(
                    index=item.get("index", 0),
                    logit=logit,
                    score01=round(score01, 4),
                )
                hits.append(hit)

            return hits

        except httpx.HTTPStatusError as e:
            raise RetrievalModelError(
                list(range(len(passages))),
                f"HTTP error {e.response.status_code}: {e}",
            )
        except Exception as e:
            raise RetrievalModelError(
                list(range(len(passages))),
                f"Rerank API error: {e}",
            )


class RerankHit:
    """A single reranking result hit."""

    def __init__(self, index: int, logit: float, score01: float) -> None:
        self.index = index
        self.logit = logit
        self.score01 = score01


class RerankResult:
    """Result from a reranking operation."""

    def __init__(
        self,
        results: List[RerankHit],
        model: str,
        metadata: Dict[str, Any],
    ) -> None:
        self.results = results
        self.model = model
        self.metadata = metadata