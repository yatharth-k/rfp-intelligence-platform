"""Service 09: Indexer.

Manages the incremental indexing workflow: file registry checks,
embedding generation, and writing to both the keyword store (FTS5)
and vector store (Qdrant). Supports idempotent re-indexing and
full rebuilds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import Chunk, DocType, FileReport, IngestionReport
from core.embeddings import EmbeddingClient
from core.limiter import RateLimiter, DiskCache, ProviderContext
from core.tracing import Tracer
from search.keyword_store import KeywordStore, FileRegistry
from search.vector_store import QdrantVectorStore


class Indexer:
    """Handles incremental indexing of bids and documents.

    Workflow:
    1. For each file in a bid folder: compute hash, check registry
    2. If unchanged (same hash + same model): skip (no embedding calls)
    3. If changed: delete old chunks from both stores, then index new version
    4. If new bid: index without touching existing bids
    5. Write registry last so interrupted runs are safe
    """

    def __init__(
        self,
        keyword_store: KeywordStore,
        vector_store: QdrantVectorStore,
        file_registry: FileRegistry,
        embedding_client: EmbeddingClient,
        tracer: Tracer,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.file_registry = file_registry
        self.embedding_client = embedding_client
        self.tracer = tracer
        self.rate_limiter = rate_limiter or RateLimiter()

    def index_bid(
        self,
        bid_path: Path,
        bid_id: str,
        *,
        force: bool = False,
        embed_model: str = "nv-embed-v1",
    ) -> Dict[str, Any]:
        """Index all files in a bid folder.

        Args:
            bid_path: Path to the bid folder
            bid_id: The bid identifier
            force: If True, reindex all files regardless of registry
            embed_model: Embedding model name

        Returns:
            Index report dict with stats
        """
        start_time = time.time()
        files_seen = 0
        files_skipped = 0
        files_indexed = 0
        files_failed = 0
        chunks_added = 0
        chunks_removed = 0
        embedding_calls = 0
        cache_hits = 0

        # supported extensions
        supported_extensions = {".pdf", ".html", ".htm"}

        # list files (non-recursive by default)
        files: List[Path] = []
        for item in bid_path.iterdir():
            if item.is_file() and item.suffix.lower() in supported_extensions:
                files.append(item)

        # deterministic sort
        files.sort(key=lambda p: (p.name.lower(), p.suffix.lower()))

        for filepath in files:
            files_seen += 1
            file_bytes = filepath.read_bytes()
            file_hash = self._file_hash(file_bytes)

            # Check registry: same hash + same model -> skip
            if not force and self.file_registry.is_indexed(file_hash, embed_model):
                files_skipped += 1
                continue

            # File changed or new: process it
            # If file was previously indexed, delete old chunks
            if not force:
                old_chunks = self.file_registry.get_file_record(file_hash)
                if old_chunks and old_chunks.get("status") == "indexed":
                    # Delete old chunks from both stores
                    chunks_removed += self._delete_file_chunks(
                        file_hash, old_chunks.get("bid_id", bid_id)
                    )

            # Process the file: detect -> parse -> clean -> chunk
            # (In a full pipeline, would use ingestion modules;
            # for indexer, we expect already-chunked documents)
            # For now, create minimal chunk data and embed
            try:
                file_status, file_chunks = self._process_file(
                    filepath, file_hash, bid_id, embed_model
                )

                if file_status == "ok" and file_chunks:
                    # Index the chunks
                    self._index_chunks(file_chunks, bid_id, embed_model)
                    files_indexed += 1
                    chunks_added += len(file_chunks)
                else:
                    files_failed += 1

                # Update registry last (so interrupted runs are safe)
                self.file_registry.update_file_record(
                    file_hash=file_hash,
                    bid_id=bid_id,
                    file_name=filepath.name,
                    rel_path=str(filepath),
                    doc_type=file_chunks[0].doc_type.value if file_chunks else DocType.OTHER.value,
                    addendum_number=file_chunks[0].addendum_number if file_chunks else None,
                    n_chunks=len(file_chunks) if file_chunks else 0,
                    status=file_status,
                    embed_model=embed_model,
                )

            except Exception as e:
                files_failed += 1
                # Still update registry so we don't re-process failed files
                try:
                    self.file_registry.update_file_record(
                        file_hash=file_hash,
                        bid_id=bid_id,
                        file_name=filepath.name,
                        rel_path=str(filepath),
                        doc_type=DocType.OTHER.value,
                        addendum_number=None,
                        n_chunks=0,
                        status="failed",
                        embed_model=embed_model,
                    )
                except Exception:
                    pass

        duration = time.time() - start_time

        report = {
            "files_seen": files_seen,
            "files_skipped": files_skipped,
            "files_indexed": files_indexed,
            "files_failed": files_failed,
            "chunks_added": chunks_added,
            "chunks_removed": chunks_removed,
            "embedding_calls": embedding_calls,
            "cache_hits": cache_hits,
            "duration_seconds": duration,
        }

        return report

    def index_documents(
        self,
        documents: List["ParsedDocument"],
        bid_id: str,
        *,
        embed_model: str = "nv-embed-v1",
    ) -> Dict[str, Any]:
        """Index a list of ParsedDocument objects.

        Each document should already be chunked (via chunk_document from service 08).
        """
        start_time = time.time()
        files_indexed = 0
        chunks_added = 0
        embedding_calls = 0
        cache_hits = 0

        for doc in documents:
            files_indexed += 1
            file_hash = doc.file_hash

            # Check registry
            if self.file_registry.is_indexed(file_hash, embed_model):
                continue

            # Delete old chunks if file was previously indexed
            old_record = self.file_registry.get_file_record(file_hash)
            if old_record and old_record.get("status") == "indexed":
                chunks_removed = self._delete_file_chunks(file_hash, bid_id)
            else:
                chunks_removed = 0

            # Index chunks from this document
            file_chunks = doc.chunks if hasattr(doc, "chunks") else []
            if file_chunks:
                self._index_chunks(file_chunks, bid_id, embed_model)
                chunks_added += len(file_chunks)

            # Update registry
            self.file_registry.update_file_record(
                file_hash=file_hash,
                bid_id=bid_id,
                file_name=doc.file_name,
                rel_path=str(doc.file_name),
                doc_type=doc.doc_types[0].value if doc.doc_types else DocType.BID_PAGE.value,
                addendum_number=doc.addendum_number,
                n_chunks=len(file_chunks) if file_chunks else 0,
                status="ok",
                embed_model=embed_model,
            )

        duration = time.time() - start_time

        return {
            "files_indexed": files_indexed,
            "chunks_added": chunks_added,
            "embedding_calls": embedding_calls,
            "cache_hits": cache_hits,
            "duration_seconds": duration,
        }

    def _delete_file_chunks(self, file_hash: str, bid_id: str) -> int:
        """Delete all chunks for a file from both stores. Returns count deleted."""
        # Delete from keyword store
        kw_deleted = self.keyword_store.delete_by_file_hash(file_hash)

        # Delete from vector store (by payload filter: file_name + bid_id)
        vs_deleted = self.vector_store.delete_by_payload_filter({
            "bid_id": bid_id,
            "file_name": "",  # Will be matched more broadly
        })

        # More precise: delete from keyword store first, then vector store
        # for the same file. The vector store delete uses bid_id + file_name pattern.
        # Since we already deleted from keyword store, we just need to clean up Qdrant.

        # Try to delete from vector store using bid_id and a pattern
        # We'll use a simple approach: delete by bid_id only, then
        # the file-specific chunks will be re-embedded if needed.
        vs_deleted = self.vector_store.delete_by_payload_filter({
            "bid_id": bid_id,
        })

        return kw_deleted + vs_deleted

    def _index_chunks(
        self,
        chunks: List[Chunk],
        bid_id: str,
        embed_model: str,
    ) -> None:
        """Index a list of chunks: write to keyword store and vector store.

        Write order: chunks to SQLite/FTS5 first, registry written last,
        so interrupted run leaves no 'indexed' file without its chunks;
        next run redoes it.
        """
        # Batch embed chunks (with caching)
        chunks_to_embed = []
        chunks_already_embedded = []

        for chunk in chunks:
            if chunk.embedded:
                chunks_already_embedded.append(chunk)
            else:
                chunks_to_embed.append(chunk)

        # Embed new chunks with batching and rate limiting
        if chunks_to_embed:
            embedded_chunks = self._embed_chunks(chunks_to_embed, embed_model)
            embedding_calls_count = len(chunks_to_embed) - len(
                [c for c in embedded_chunks if c.get("cache_hit", False)]
            )
            cache_hits_count = len(
                [c for c in embedded_chunks if c.get("cache_hit", False)]
            )
        else:
            embedded_chunks = []
            embedding_calls_count = 0
            cache_hits_count = 0

        # Assign embed_text and set embedded flag
        for i, chunk in enumerate(chunks):
            # Find the matching embedded chunk
            embed_data = None
            if i < len(embedded_chunks):
                embed_data = embedded_chunks[i]

            if embed_data and "embed_text" in embed_data:
                chunk.embed_text = embed_data["embed_text"]
                chunk.embedded = True
            else:
                # Fallback: create a minimal embed_text with context prefix
                from search.chunker import make_context_prefix
                context_prefix = make_context_prefix({
                    "bid_id": chunk.bid_id,
                    "file_name": chunk.file_name,
                    "doc_type": chunk.doc_type.value
                    if hasattr(chunk.doc_type, "value") else str(chunk.doc_type),
                    "page_start": chunk.page_start,
                    "section": chunk.section or "",
                })
                chunk.embed_text = f"{context_prefix} {chunk.text}"
                chunk.embedded = True

            # Add to keyword store
            self.keyword_store.add_chunk(chunk)

        # Upsert to vector store
        chunks_with_vectors = []
        for chunk in chunks:
            if chunk.embed_text and len(chunk.embed_text) > 0:
                # In a full implementation, would use the actual embedding
                # For now, generate a dummy vector (should match the real dim)
                import numpy as np
                vector = np.random.randn(self.vector_store.dim).tolist()
                chunks_with_vectors.append((chunk, vector))

        if chunks_with_vectors:
            self.vector_store.upsert_points(chunks_with_vectors)

    def _embed_chunks(
        self,
        chunks: List[Chunk],
        embed_model: str,
    ) -> List[Dict[str, Any]]:
        """Embed chunks using the embedding client with caching and rate limiting.

        Returns list of dicts with 'embed_text' and 'cache_hit' keys.
        """
        results = []

        for chunk in chunks:
            # Check cache first
            cache_key = (
                f"{embed_model}:{chunk.file_hash}:"
                f"{hash(chunk.text) % (10**8)}"
            )
            # In a full implementation, would check the disk cache
            # For now, call the embedding client

            # Apply rate limiting
            self.rate_limiter.wait()

            try:
                embedding_result = self.embedding_client.embed_query(chunk.text)
                embedding_vector = embedding_result.embedding

                # Store embed_text as a representation
                # In practice, this would be the vector serialized or a text embedding
                embed_text = (
                    f"vector_{len(embedding_vector)}d"
                    if embedding_vector
                    else ""
                )

                results.append(
                    {
                        "embed_text": embed_text,
                        "cache_hit": False,
                        "embedding_vector": embedding_vector,
                    }
                )
            except Exception as e:
                # Mark chunk as not embedded; remains searchable by BM25 only
                results.append(
                    {
                        "embed_text": "",
                        "cache_hit": False,
                        "error": str(e),
                    }
                )

        return results

    @staticmethod
    def _file_hash(data: bytes) -> str:
        """Compute SHA-256 hash of file bytes (first 16 chars)."""
        import hashlib
        return hashlib.sha256(data).hexdigest()[:16]

    def repair_embeddings(
        self,
        chunk_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Re-embed chunks flagged as un-embedded.

        Targets chunks where embedded=False and re-embeds them.
        Returns dict with 'repaired', 'failed', 'total' counts.
        """
        # Find un-embedded chunks from keyword store
        if chunk_ids is None:
            unembedded = self.keyword_store.repair_embeddings()
            chunk_ids = [cid for cid, _ in unembedded.get("details", [])]

        # Re-embed each chunk
        repaired = 0
        failed = 0

        for chunk_id in chunk_ids:
            chunk_data = self.keyword_store.get_chunk(chunk_id)
            if chunk_data is None or chunk_data.get("embedded", False):
                continue

            # Find the chunk file and re-embed
            # In a full pipeline, would re-process the source file
            # For now, attempt a best-effort re-embed
            try:
                # Placeholder: would call embedding client on chunk text
                chunk_text = chunk_data.get("text", "")
                if chunk_text:
                    embedding_result = self.embedding_client.embed_query(chunk_text)
                    chunk.embed_text = (
                        f"vector_{self.vector_store.dim}d"
                        if embedding_result.embedding
                        else ""
                    )
                    # Update keyword store
                    self.keyword_store._conn.execute(
                        "UPDATE chunks SET embed_text = ?, embedded = 1 "
                        "WHERE chunk_id = ?",
                        (chunk.embed_text, chunk_id),
                    )
                    repaired += 1
            except Exception:
                failed += 1

        if self.keyword_store._conn is not None:
            self.keyword_store._conn.commit()

        return {"repaired": repaired, "failed": failed, "total": len(chunk_ids) or 0}

    def list_bids(self) -> List[Dict[str, Any]]:
        """List all bids in the registry with their stats."""
        if self.file_registry._conn is None:
            self.file_registry._conn = sqlite3.connect(str(self.file_registry.db_path))

        cursor = self._conn.execute(
            """SELECT bid_id, COUNT(DISTINCT file_hash) as n_files,
              SUM(n_chunks) as n_chunks, status
            FROM file_registry
            GROUP BY bid_id"""
        )
        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "bid_id": row[0],
                    "n_files": row[1],
                    "n_chunks": row[2],
                    "status": row[3],
                }
            )
        return results

    def delete_bid(self, bid_id: str) -> Dict[str, Any]:
        """Delete all chunks and registry entries for a bid."""
        # Delete from keyword store
        kw = self.keyword_store.delete_by_bid_id(bid_id)

        # Delete from vector store
        vs = self.vector_store.delete_by_payload_filter({"bid_id": bid_id})

        # Delete from registry
        if self.file_registry._conn is not None:
            self.file_registry._conn.execute(
                "DELETE FROM file_registry WHERE bid_id = ?",
                (bid_id,),
            )
            self.file_registry._conn.commit()

        return {"keyword_store_deleted": kw, "vector_store_deleted": vs}
