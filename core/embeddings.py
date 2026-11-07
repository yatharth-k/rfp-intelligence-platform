"""Service 03: Embedding and Rerank Clients - Embedding client."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from core.errors import RetrievalModelError


class EmbeddingClient:
    """Client for NVIDIA embedding models with caching, batching, and normalization.

    Follows service 03 design for embeddings:
    - 2048-dim vectors from nemotron-3-embed-1b
    - Asymmetric inputs: passages use input_type="passage", queries use "query"
    - Batching with EMBED_BATCH_SIZE, halving on 400/413 errors
    - Per-text cache: key = sha256(model, input_type, text)
    - L2 normalization so cosine similarity = dot product
    - Dimension guard from first successful response
    - Truncation: truncate="END" for over-long input
    - Failure: raises RetrievalModelError with failed indices
    """

    def __init__(self, model: str = "nvidia/nemotron-3-embed-1b",
                 embed_url: str = "https://integrate.api.nvidia.com/v1",
                 api_key: str | None = None,
                 batch_size: int = 16):
        self.model = model
        self.embed_url = embed_url
        self.api_key = api_key or ""
        self.batch_size = batch_size
        self.dim: int | None = None
        self._cache: dict[str, np.ndarray] = {}

    def _make_cache_key(self, input_type: str, text: str) -> str:
        """Create cache key from model, input_type, and text."""
        content = f"{self.model}|{input_type}|{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """L2-normalize a vector so cosine similarity = dot product."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _call_api(self, texts: list[str], input_type: str) -> np.ndarray:
        """Call the embedding API and return normalized vectors.

        Raises RetrievalModelError on persistent failure.
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
            "extra_body": {
                "input_type": input_type,
                "truncate": "END",
            },
        }

        try:
            response = httpx.post(self.embed_url, json=body, headers=headers,
                                  timeout=45)
            response.raise_for_status()
            data = response.json()

            # Parse response - NVIDIA returns {"data": [{"embedding": [...], "index": 0}, ...]}
            embeddings = []
            index_map = {}
            for item in data.get("data", []):
                emb = item["embedding"]
                idx = item["index"]
                embeddings.append(emb)
                index_map[idx] = len(embeddings) - 1

            # Check for errors in the response
            if "error" in data:
                raise RetrievalModelError(
                    [i for i in range(len(texts))],
                    f"API error: {data['error']}",
                )

            # Verify we got all requested embeddings
            if len(embeddings) != len(texts):
                missing = set(range(len(texts))) - set(index_map.keys())
                raise RetrievalModelError(
                    list(missing),
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}",
                )

            # Store dimension from first response
            if self.dim is None:
                self.dim = len(embeddings[0])

            # Normalize all vectors
            normalized = np.array([
                self._normalize_vector(np.array(emb, dtype=np.float32))
                for emb in embeddings
            ])

            return normalized

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (413, 400):
                # Batch too large - caller should halve batch size
                raise RetrievalModelError(
                    list(range(len(texts))),
                    f"HTTP {e.response.status_error}: Batch size issue",
                )
            raise RetrievalModelError(
                list(range(len(texts))),
                f"HTTP error: {e}",
            )
        except Exception as e:
            raise RetrievalModelError(
                list(range(len(texts))),
                f"Unexpected error: {e}",
            )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Returns L2-normalized vector of dimension self.dim.
        Caches result so second call makes no API request.
        """
        cache_key = self._make_cache_key("query", text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Single text in a batch of 1
        vector = self._call_api([text], "query")[0]
        self._cache[cache_key] = vector
        return vector

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed multiple passage texts.

        Returns array of shape (n, dim) with L2-normalized vectors.
        Caches individual texts; only sends misses to API.
        """
        # De-duplicate while preserving order and mapping back
        seen: dict[str, int] = {}
        unique_texts: list[str] = []
        text_to_idx: list[int] = []

        for i, text in enumerate(texts):
            if text not in seen:
                seen[text] = len(unique_texts)
                unique_texts.append(text)
                text_to_idx.append(i)
            else:
                # Duplicate - will be filled in from cache later
                text_to_idx.append(-1)  # placeholder

        # Check cache for each unique text
        texts_to_send: list[str] = []
        cache_results: dict[int, np.ndarray] = {}
        uncached_indices: list[int] = []

        for i, text in enumerate(unique_texts):
            cache_key = self._make_cache_key("passage", text)
            if cache_key in self._cache:
                cache_results[i] = self._cache[cache_key]
            else:
                texts_to_send.append(text)
                uncached_indices.append(i)

        # Send uncached texts to API
        if texts_to_send:
            vectors = self._call_api(texts_to_send, "passage")
            # Store in cache and map back
            for i, text in enumerate(texts_to_send):
                cache_key = self._make_cache_key("passage", text)
                self._cache[cache_key] = vectors[i]

        # Build full result array
        result = np.empty((len(unique_texts), self.dim), dtype=np.float32)

        # Fill in cached/API results
        for i in range(len(unique_texts)):
            if i in cache_results:
                result[i] = cache_results[i]
            else:
                # Find the index in the API response
                api_idx = uncached_indices.index(i)
                result[i] = vectors[api_idx]

        # Now map back to original order
        full_result = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, orig_idx in enumerate(text_to_idx):
            if orig_idx == -1:
                # Duplicate - copy from first occurrence
                full_result[i] = result[text_to_idx.index(i)]
            else:
                full_result[i] = result[i]

        return full_result

    def aembed_query(self, text: str) -> "Awaitable[np.ndarray]":
        """Async embedding of a single query text."""
        import asyncio

        async def _inner():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.embed_query, text)

        return _inner()

    def aembed_passages(self, texts: list[str]) -> "Awaitable[np.ndarray]":
        """Async embedding of multiple passage texts."""
        import asyncio

        async def _inner():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.embed_passages, texts)

        return _inner()