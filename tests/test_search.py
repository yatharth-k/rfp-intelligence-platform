"""Unit tests for search layer (services 08-11)."""

import pytest
from search.chunker import chunk_document, estimate_tokens, make_context_prefix
from search.keyword_store import KeywordStore, FTS5SearchError, FileRegistry
from search.vector_store import QdrantVectorStore
from search.indexer import Indexer
from search.query_rewriter import QueryRewriter, QueryRewriteLog
from search.hybrid import rrf_fuse, fuse_with_rerank
from search.reranker import Reranker, _DeterministicReranker
from search.engine import SearchEngine, SearchMode


class TestChunker:
    """Test chunker functionality."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        result = estimate_tokens("this is a test with several words")
        assert isinstance(result, int)
        assert result > 0

    def test_make_context_prefix(self):
        """Test context prefix generation."""
        result = make_context_prefix({
            "bid_id": "Bid1",
            "file_name": "doc.pdf",
            "doc_type": "rfp",
            "page_start": 3,
            "section": "4 Payment",
        })
        assert "Bid1" in result
        assert "doc.pdf" in result


class TestKeywordStore:
    """Test keyword store (FTS5)."""

    def test_keyword_store_creation(self):
        """Test KeywordStore can be created."""
        import tempfile
        import os
        db_path = tempfile.mktemp(suffix=".db")
        try:
            store = KeywordStore(Path(db_path))
            assert store is not None
            store.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_file_registry_creation(self):
        """Test FileRegistry can be created."""
        import tempfile
        import os
        db_path = tempfile.mktemp(suffix=".db")
        try:
            registry = FileRegistry(Path(db_path))
            assert registry is not None
            registry.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


class TestVectorStore:
    """Test vector store (Qdrant)."""

    def test_vector_store_import(self):
        """Test QdrantVectorStore can be imported."""
        from search.vector_store import QdrantVectorStore
        assert QdrantVectorStore is not None


class TestIndexer:
    """Test indexer functionality."""

    def test_indexer_import(self):
        """Test Indexer can be imported."""
        from search.indexer import Indexer
        assert Indexer is not None


class TestQueryRewriter:
    """Test query rewriter."""

    def test_query_rewriter_import(self):
        """Test QueryRewriter can be imported."""
        from search.query_rewriter import QueryRewriter
        assert QueryRewriter is not None

    def test_query_rewrite_basic(self):
        """Test basic query rewrite."""
        rewriter = QueryRewriter(use_llm=False)
        result = rewriter.rewrite("test query")
        assert result == "test query" or len(result) > 0


class TestHybrid:
    """Test hybrid/RRF fusion."""

    def test_rrf_fuse(self):
        """Test RRF fusion function."""
        bm25_results = [{"chunk_id": "c1", "bm25_score": 1.0}]
        vector_results = [{"chunk_id": "c1", "vector_score": 0.8}]
        result = rrf_fuse(bm25_results, vector_results, k=60)
        assert isinstance(result, list)


class TestReranker:
    """Test reranker functionality."""

    def test_reranker_import(self):
        """Test Reranker can be imported."""
        from search.reranker import Reranker
        assert Reranker is not None

    def test_deterministic_reranker_import(self):
        """Test deterministic reranker can be imported."""
        from search.reranker import _DeterministicReranker
        assert _DeterministicReranker is not None


class TestEngine:
    """Test search engine."""

    def test_engine_import(self):
        """Test SearchEngine can be imported."""
        from search.engine import SearchEngine
        assert SearchEngine is not None

    def test_search_mode_enum(self):
        """Test SearchMode enum values."""
        from search.engine import SearchMode
        assert SearchMode.BM25.value == "bm25"
        assert SearchMode.VECTOR.value == "vector"
        assert SearchMode.FUSION.value == "fusion"
        assert SearchMode.RERANK.value == "rerank"