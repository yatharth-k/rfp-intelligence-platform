"""Unit tests for core foundation layer (services 01-05)."""

import pytest
from core.llm import LLMGateway, LLMResult, complete_json, acomplete_json
from core.errors import (
    LLMGatewayError,
    AuthError,
    ModelUnavailableError,
    RateLimitError,
    ProviderError,
    StructuredOutputError,
    BudgetExceededError,
)
from core.embeddings import EmbeddingClient
from core.rerank import RerankClient
from core.limiter import RateLimiter, DiskCache, CircuitBreaker, ProviderContext
from core.tracing import Tracer


class TestCoreErrors:
    """Test error hierarchy."""

    def test_error_inheritance(self):
        """All error types should inherit from LLMGatewayError."""
        assert issubclass(AuthError, LLMGatewayError)
        assert issubclass(ModelUnavailableError, LLMGatewayError)
        assert issubclass(RateLimitError, LLMGatewayError)
        assert issubclass(ProviderError, LLMGatewayError)
        assert issubclass(StructuredOutputError, LLMGatewayError)
        assert issubclass(BudgetExceededError, LLMGatewayError)

    def test_error_instantiation(self):
        """Test error can be instantiated."""
        errors = [
            AuthError(),
            ModelUnavailableError(),
            RateLimitError(),
            ProviderError(),
            StructuredOutputError(),
            BudgetExceededError(),
        ]
        for error in errors:
            assert str(error)  # Should not raise


class TestCoreLlmGateway:
    """Test LLM gateway basic functionality."""

    def test_complete_json_structure(self):
        """Test complete_json with valid structured output."""
        # This tests the schema; actual LLM call would need API key
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        result = complete_json(
            prompt="Extract name and value",
            temperature=0,
            max_tokens=100,
            schema=TestSchema,
        )
        assert hasattr(result, "name")
        assert hasattr(result, "value")


class TestCoreEmbeddings:
    """Test embedding client (basic import test)."""

    def test_embedding_client_import(self):
        """Test EmbeddingClient can be imported."""
        from core.embeddings import EmbeddingClient
        assert EmbeddingClient is not None


class TestCoreRerank:
    """Test rerank client basic functionality."""

    def test_rerank_client_import(self):
        """Test RerankClient can be imported."""
        from core.rerank import RerankClient
        assert RerankClient is not None


class TestCoreLimiter:
    """Test rate limiter and cache."""

    def test_rate_limiter_import(self):
        """Test RateLimiter can be imported."""
        from core.limiter import RateLimiter
        assert RateLimiter is not None

    def test_disk_cache_import(self):
        """Test DiskCache can be imported."""
        from core.limiter import DiskCache
        assert DiskCache is not None

    def test_circuit_breaker_import(self):
        """Test CircuitBreaker can be imported."""
        from core.limiter import CircuitBreaker
        assert CircuitBreaker is not None


class TestCoreTracing:
    """Test trace logger basic functionality."""

    def test_tracer_import(self):
        """Test Tracer can be imported."""
        from core.tracing import Tracer
        assert Tracer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])