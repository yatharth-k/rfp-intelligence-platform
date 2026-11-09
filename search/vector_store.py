"""Service 09: Vector Store (Qdrant).

Implements a local Qdrant vector store for chunk embeddings with
2048-dimensional cosine distance, payload filtering, and idempotent
upserts. Designed for single-process CLI/API usage.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import Chunk


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


try:
    import qdrant_client
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        PayloadSchemaType,
        FieldCondition,
        Range,
        KeywordIndexParams,
        IntegerIndexParams,
        Filter,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantVectorStore:
    """Qdrant vector store for chunk embeddings.

    Uses local embedded mode (single-process) or remote server. Collection "rfp_chunks"
    holds 2048-dim vectors with cosine distance. Payload fields enable
    fast filtering by bid_id, doc_type, addendum_number, etc. The
    actual chunk text stays in the SQLite keyword store; vectors alone
    live in Qdrant.
    """

    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection_name: str = "rfp_chunks", dim: int = 2048,
                 path: Optional[str] = None):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.path = path
        self.client: Any = None
        self._collection_initialized = False

        if QDRANT_AVAILABLE:
            try:
                if path:
                    # Embedded mode - runs in-process
                    self.client = qdrant_client.QdrantClient(path=path)
                else:
                    # Remote server mode
                    self.client = qdrant_client.QdrantClient(
                        host=host, port=port,
                    )
            except Exception:
                self.client = None

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        if self.client is None:
            raise VectorStoreError(
                "Qdrant client not available. "
                "Install qdrant-client or run a Qdrant server."
            )

        if not self._collection_initialized:
            # Check if collection exists
            try:
                self.client.get_collection(
                    collection_name=self.collection_name,
                )
            except Exception:
                # Collection doesn't exist; create it
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dim,
                        distance=Distance.COSINE,
                    ),
                )
                # Create payload indexes for filtering
                self._create_payload_indexes()
            self._collection_initialized = True

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for commonly filtered fields."""
        index_fields = [
            ("bid_id", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("addendum_number", PayloadSchemaType.KEYWORD),
            ("file_name", PayloadSchemaType.KEYWORD),
            ("page_start", PayloadSchemaType.INTEGER),
            ("page_end", PayloadSchemaType.INTEGER),
            ("section", PayloadSchemaType.KEYWORD),
            ("kind", PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_schema in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as e:
                # Index might already exist or field might not be used
                pass

    def _point_id_from_chunk_id(self, chunk_id: str) -> str:
        """Derive a deterministic point ID from chunk_id for idempotent upserts."""
        # UUID v5 based on chunk_id namespace + the chunk_id as name
        # This ensures the same chunk_id always maps to the same point ID
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rfp_chunk:{chunk_id}"))

    def upsert_points(self, chunks_with_vectors: List[Tuple[Chunk, List[float]]]) -> int:
        """Upsert chunks with their embedding vectors.

        Args:
            chunks_with_vectors: List of (Chunk, embedding_vector) tuples

        Returns:
            Number of points upserted
        """
        if not chunks_with_vectors:
            return 0

        self._ensure_collection()

        points = []
        for chunk, vector in chunks_with_vectors:
            point_id = self._point_id_from_chunk_id(chunk.chunk_id)
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "bid_id": chunk.bid_id,
                    "doc_type": chunk.doc_type.value if hasattr(chunk.doc_type, "value") else str(chunk.doc_type),
                    "addendum_number": chunk.addendum_number or "",
                    "file_name": chunk.file_name,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section or "",
                    "kind": chunk.kind,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:500] if len(chunk.text) > 500 else chunk.text,
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using a query vector.

        Args:
            query_vector: 2048-dimensional embedding vector
            filters: Optional payload filters (bid_id, doc_type, addendum_number, etc.)
            limit: Maximum number of results to return

        Returns:
            List of dicts with chunk data and similarity scores
        """
        if self.client is None:
            raise VectorStoreError(
                "Qdrant client not available. "
                "Connect to a Qdrant server or start local mode."
            )

        self._ensure_collection()

        # Build filter conditions
        query_filter = None
        if filters:
            conditions = []
            if filters.get("bid_id") is not None:
                conditions.append(
                    FieldCondition(key="bid_id", match=MatchValue(value=filters["bid_id"]))
                )
            if filters.get("doc_type") is not None:
                conditions.append(
                    FieldCondition(key="doc_type", match=MatchValue(value=filters["doc_type"]))
                )
            if filters.get("addendum_number") is not None:
                conditions.append(
                    FieldCondition(key="addendum_number", match=MatchValue(value=str(filters["addendum_number"])))
                )
            if filters.get("file_name") is not None:
                conditions.append(
                    FieldCondition(key="file_name", match=MatchValue(value=filters["file_name"]))
                )
            if filters.get("min_page_start") is not None:
                conditions.append(
                    Range(key="page_start", gte=filters["min_page_start"])
                )
            if filters.get("max_page_end") is not None:
                if conditions and conditions[-1].key == "page_end":
                    # Extend existing page_end condition
                    conditions[-1] = Range(
                        key="page_end",
                        lte=filters["max_page_end"],
                    )
                else:
                    conditions.append(
                        Range(key="page_end", lte=filters["max_page_end"])
                    )
            if conditions:
                query_filter = Filter(must=conditions)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        results = []
        for hit in search_result.points:
            payload = hit.payload
            results.append({
                "chunk_id": payload.get("chunk_id"),
                "bid_id": payload.get("bid_id"),
                "doc_type": payload.get("doc_type"),
                "file_name": payload.get("file_name"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "section": payload.get("section"),
                "kind": payload.get("kind"),
                "score": hit.score,
                "text": payload.get("text", ""),
        })

        return results

    def delete_by_payload_filter(self, filter_dict: Dict[str, Any]) -> int:
        """Delete points matching a payload filter. Returns number of points deleted."""
        if self.client is None:
            raise VectorStoreError("Qdrant client not available.")

        self._ensure_collection()

        # Build filter for deletion
        conditions = []
        if filter_dict.get("bid_id") is not None:
            conditions.append(
                FieldCondition(key="bid_id", match=MatchValue(value=filter_dict["bid_id"]))
            )
        if filter_dict.get("doc_type") is not None:
            conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=filter_dict["doc_type"]))
            )
        if filter_dict.get("addendum_number") is not None:
            conditions.append(
                FieldCondition(key="addendum_number", match=MatchValue(value=str(filter_dict["addendum_number"])))
            )
        if filter_dict.get("file_name") is not None:
            conditions.append(
                FieldCondition(key="file_name", match=MatchValue(value=filter_dict["file_name"]))
            )

        if not conditions:
            return 0

        query_filter = Filter(must=conditions)

        # Delete by filter
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=query_filter,
        )

        # Qdrant delete returns the count of deleted points
        deleted_count = getattr(result, 'deleted_count', 0) or 0
        return deleted_count

    def rebuild_collection(self) -> None:
        """Drop the collection and re-create it (empty)."""
        if self.client is None:
            raise VectorStoreError("Qdrant client not available.")

        self.client.delete_collection(
            collection_name=self.collection_name,
        )
        self._collection_initialized = False
        self._ensure_collection()

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection metadata (embed_model, dim, point count)."""
        if self.client is None:
            raise VectorStoreError("Qdrant client not available.")

        col_info = self.client.get_collection(
            collection_name=self.collection_name,
        )

        return {
            "collection_name": self.collection_name,
            "dim": col_info.config.params.vectors.size,
            "distance": col_info.config.params.vectors.distance.name,
            "point_count": col_info.points_count,
        }

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self.client is not None:
            # In local mode, there's typically nothing to close explicitly
            pass
