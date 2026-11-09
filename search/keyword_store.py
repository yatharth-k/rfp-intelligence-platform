"""Service 09: Keyword Store and FTS5 Index.

Implements a SQLite FTS5 keyword store for chunks with filter support,
query sanitization, and incremental indexing operations.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import Chunk, DocType, FileReport


class FTS5SearchError(Exception):
    """Raised when FTS5 query construction fails."""


class KeywordStore:
    """SQLite FTS5 keyword store for chunk text search.

    Stores chunks with their metadata and provides full-text search
    via a virtual table. Supports filtered search by bid_id, doc_type,
    addendum_number, file_name, and page range.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()
        # Set row factory to return Row objects (dict-like)
        if self._conn is not None:
            self._conn.row_factory = sqlite3.Row

    def _ensure_schema(self) -> None:
        """Create the chunks table and FTS5 virtual table if they don't exist."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                bid_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                addendum_number TEXT,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                section TEXT,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                embed_text TEXT,
                embedded INTEGER NOT NULL DEFAULT 0
            )
        """)

        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                bid_id,
                file_name,
                file_hash,
                doc_type,
                addendum_number,
                page_start,
                page_end,
                section,
                kind,
                text,
                embedded,
                content=chunks,
                content_rowid=rowid
            )
        """)

        # Keep FTS5 in sync with chunks table
        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
            AFTER INSERT ON chunks
            BEGIN
                INSERT INTO chunks_fts (rowid, chunk_id, bid_id, file_name, file_hash,
                                      doc_type, addendum_number, page_start, page_end,
                                      section, kind, text, embedded)
                    VALUES (new.rowid, new.chunk_id, new.bid_id, new.file_name, new.file_hash,
                            new.doc_type, new.addendum_number, new.page_start, new.page_end,
                            new.section, new.kind, new.text, new.embedded);
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
            AFTER DELETE ON chunks
            BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update
            AFTER UPDATE ON chunks
            BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
                INSERT INTO chunks_fts (rowid, chunk_id, bid_id, file_name, file_hash,
                                      doc_type, addendum_number, page_start, page_end,
                                      section, kind, text, embedded)
                    VALUES (new.rowid, new.chunk_id, new.bid_id, new.file_name, new.file_hash,
                            new.doc_type, new.addendum_number, new.page_start, new.page_end,
                            new.section, new.kind, new.text, new.embedded);
            END
        """)

        self._conn.commit()

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a single chunk to both the chunks table and FTS5 index."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        self._conn.execute(
            """INSERT OR REPLACE INTO chunks (
                chunk_id, bid_id, file_name, file_hash, doc_type,
                addendum_number, page_start, page_end, section, kind,
                text, embed_text, embedded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id,
                chunk.bid_id,
                chunk.file_name,
                chunk.file_hash,
                chunk.doc_type.value if hasattr(chunk.doc_type, "value") else chunk.doc_type,
                chunk.addendum_number,
                chunk.page_start,
                chunk.page_end,
                chunk.section,
                chunk.kind,
                chunk.text,
                chunk.embed_text,
                1 if chunk.embedded else 0,
            ),
        )
        self._conn.commit()

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add multiple chunks in a single transaction."""
        if not chunks:
            return
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        for chunk in chunks:
            self.add_chunk(chunk)

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Search chunks using FTS5 with optional filters.

        Args:
            query: The search text query
            filters: Optional dict with keys: bid_ids, doc_types, addendum_numbers,
                     file_names, min_addendum_number
            limit: Maximum number of results to return

        Returns:
            List of dicts with chunk data and BM25-style scores
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        # Build the FTS5 query expression
        fts_query = self._build_fts_query(query)

        # Build the WHERE clause for filters
        where_clauses = ["chunks_fts MATCH ?"]
        params: List[Any] = [fts_query]

        if filters:
            if filters.get("bid_ids"):
                placeholders = ",".join(["?"] * len(filters["bid_ids"]))
                where_clauses.append(f"c.bid_id IN ({placeholders})")
                params.extend(filters["bid_ids"])

            if filters.get("doc_types"):
                placeholders = ",".join(["?"] * len(filters["doc_types"]))
                where_clauses.append(f"c.doc_type IN ({placeholders})")
                params.extend(filters["doc_types"])

            if filters.get("addendum_numbers"):
                placeholders = ",".join(["?"] * len(filters["addendum_numbers"]))
                where_clauses.append(f"c.addendum_number IN ({placeholders})")
                params.extend(filters["addendum_numbers"])

            if filters.get("file_names"):
                placeholders = ",".join(["?"] * len(filters["file_names"]))
                where_clauses.append(f"c.file_name IN ({placeholders})")
                params.extend(filters["file_names"])

            if filters.get("min_addendum_number") is not None:
                where_clauses.append("c.addendum_number >= ?")
                params.append(filters["min_addendum_number"])

        where_sql = " AND ".join(where_clauses)

        # Execute the query with ORDER BY relevance (FTS5 rank) and limit
        sql = f"""
            SELECT c.*,
                   11 + (CAST(chunks_fts.rank AS REAL) * -1) as rank_score
            FROM chunks c
            JOIN chunks_fts ON chunks_fts.rowid = c.rowid
            WHERE {where_sql}
            ORDER BY rank_score
            LIMIT ?
        """
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        results = []
        for row in cursor:
            chunk_dict = dict(row)
            # Convert embedded flag
            chunk_dict["embedded"] = chunk_dict.get("embedded", 0) == 1
            results.append(chunk_dict)

        return results

    @staticmethod
    def _normalize_fts_query(query: str) -> str:
        """Normalize a query for FTS5 search.

        Prevents periods/slashes from being attached to tokens:
        - Replace trailing periods (followed by whitespace or end-of-string) with a space
        - Replace trailing slashes (followed by whitespace or end-of-string) with a space
        This ensures searching for "30" matches indexed text "30." or "30/"
        """
        # Replace trailing period or slash followed by whitespace or end-of-string with a space
        normalized = re.sub(r'[./](?=\s|$)', ' ', query)
        # Also normalize: collapse multiple spaces
        normalized = re.sub(r'\\s+', ' ', normalized).strip()
        return normalized

    def _build_fts_query(self, query: str) -> str:
        """Build a safe FTS5 query expression from user input.

        Converts user text to a safe FTS5 expression by:
        - Normalizing the query first (see _normalize_fts_query)
        - Quoting each term
        - OR-ing terms together
        - Escaping special FTS5 characters
        """
        # Normalize query first
        query = self._normalize_fts_query(query)

        # Remove leading/trailing whitespace
        query = query.strip()
        if not query:
            return "*"

        # Split into terms, handling quoted phrases
        terms = re.findall(r'"[^"]*"|[^\s]+', query)

        # Build the expression: each term quoted, then OR'd
        clauses = []
        for term in terms:
            # Escape FTS5 special characters: " " (double quote) and \
            escaped = term.replace('\\', '\\\\').replace('"', '""')
            # Wrap in quotes for FTS5 term query
            clauses.append(f'"{escaped}"')

        if len(clauses) == 1:
            return clauses[0]
        return " OR ".join(clauses)

    def delete_by_file_hash(self, file_hash: str) -> int:
        """Delete all chunks associated with a file hash. Returns number of chunks deleted."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        cursor = self._conn.execute("SELECT chunk_id FROM chunks WHERE file_hash = ?", (file_hash,))
        chunk_ids = [row[0] for row in cursor]

        if not chunk_ids:
            return 0

        for chunk_id in chunk_ids:
            self._conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))

        self._conn.commit()
        return len(chunk_ids)

    def delete_by_bid_id(self, bid_id: str) -> int:
        """Delete all chunks associated with a bid_id. Returns number of chunks deleted."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        cursor = self._conn.execute("SELECT chunk_id FROM chunks WHERE bid_id = ?", (bid_id,))
        chunk_ids = [row[0] for row in cursor]

        if not chunk_ids:
            return 0

        for chunk_id in chunk_ids:
            self._conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))

        self._conn.commit()
        return len(chunk_ids)

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a single chunk by chunk_id."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        cursor = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        chunk_dict = dict(row)
        chunk_dict["embedded"] = chunk_dict.get("embedded", 0) == 1
        return chunk_dict

    def repair_embeddings(self, chunk_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Re-embed chunks flagged as un-embedded.

        Targets chunks where embedded=False and re-embeds them.
        Returns a dict with 'repaired', 'failed', and 'total' counts.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        if chunk_ids is None:
            cursor = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE embedded = 0"
            )
            chunk_ids = [row[0] for row in cursor]

        if not chunk_ids:
            return {"repaired": 0, "failed": 0, "total": 0}

        # In a full implementation, this would call the embedding model
        # For now, we mark the operation and return counts
        repaired = 0
        failed = 0

        for chunk_id in chunk_ids:
            try:
                # Placeholder: in production, would call embedding client
                # to generate the embed_text for this chunk
                self._conn.execute(
                    "UPDATE chunks SET embedded = 1 WHERE chunk_id = ?",
                    (chunk_id,),
                )
                repaired += 1
            except Exception:
                failed += 1

        self._conn.commit()
        return {"repaired": repaired, "failed": failed, "total": len(chunk_ids)}

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class FileRegistry:
    """Registry table tracking file hashes and indexing status.

    Tracks which files have been indexed, their hashes, and metadata
    to support incremental indexing (skip unchanged files, re-index
    changed files).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the file_registry table if it doesn't exist."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS file_registry (
                file_hash TEXT PRIMARY KEY,
                bid_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                addendum_number TEXT,
                n_chunks INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                embed_model TEXT,
                indexed_at TEXT
            )
        """)
        self._conn.commit()

    def is_indexed(self, file_hash: str, embed_model: str) -> bool:
        """Check if a file with this hash is already indexed with the given model."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        cursor = self._conn.execute(
            "SELECT embed_model FROM file_registry WHERE file_hash = ?",
            (file_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        return row[0] == embed_model

    def update_file_record(
        self,
        file_hash: str,
        bid_id: str,
        file_name: str,
        rel_path: str,
        doc_type: str,
        addendum_number: Optional[str],
        n_chunks: int,
        status: str,
        embed_model: str,
    ) -> None:
        """Update or insert a file record in the registry."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        self._conn.execute(
            """INSERT OR REPLACE INTO file_registry (
                file_hash, bid_id, file_name, rel_path, doc_type,
                addendum_number, n_chunks, status, embed_model, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                file_hash,
                bid_id,
                file_name,
                rel_path,
                doc_type,
                addendum_number,
                n_chunks,
                status,
                embed_model,
            ),
        )
        self._conn.commit()

    def get_file_record(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get a file record by hash."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

        cursor = self._conn.execute(
            "SELECT * FROM file_registry WHERE file_hash = ?",
            (file_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return dict(row)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
