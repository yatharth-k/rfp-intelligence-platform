"""Core module for the RFP Intelligence Platform.

Contains foundational services used across the codebase:
- LLM gateway and error types
- Embedding and rerank clients
- Rate limiter and cache
- Trace logger
- Configuration and schemas
- Ingestion modules
"""

from .llm import LLMGateway, LLMResult, complete_json, acomplete_json
from .errors import (
    LLMGatewayError,
    AuthError,
    ModelUnavailableError,
    RateLimitError,
    ProviderError,
    StructuredOutputError,
    BudgetExceededError,
    RetrievalModelError,
)
from .embeddings import EmbeddingClient
from .limiter import RateLimiter, DiskCache, CircuitBreaker, ProviderContext
from .rerank import RerankClient
from .tracing import Tracer, EventType, TraceContext, _SpanContext
from .cleaning import clean_document

# Note: Ingestion functions (detect, parse_pdf, parse_html, _extract_tables_pdf, ingest_folder)
# are imported lazily to avoid triggering pdfplumber and other external dependencies
# at module load time. They can be imported via: from core import detect, etc.

# Wire ProviderContext for rate limiting, caching, circuit breaking, and trace events
# This connects the rate limiter, cache, and circuit breaker to the LLM gateway,
# embedding client, and rerank client, and enables trace event logging.
provider_context = ProviderContext()

# Export provider_context for use by other modules
# LLM gateway will use provider_context.limiter, provider_context.cache, provider_context.breaker
# Embedding client will use provider_context.cache and provider_context.limiter
# Rerank client will use provider_context.breaker and provider_context.limiter
# Tracer will use provider_context for llm_call trace events

__all__ = [
    "LLMGateway",
    "LLMResult",
    "complete_json",
    "acomplete_json",
    "LLMGatewayError",
    "AuthError",
    "ModelUnavailableError",
    "RateLimitError",
    "ProviderError",
    "StructuredOutputError",
    "BudgetExceededError",
    "RetrievalModelError",
    "EmbeddingClient",
    "RerankClient",
    "RateLimiter",
    "DiskCache",
    "CircuitBreaker",
    "ProviderContext",
    "provider_context",
    "Tracer",
    "EventType",
    "TraceContext",
    "_SpanContext",
    "clean_document",
    "detect",
    "parse_pdf",
    "parse_html",
    "_extract_tables_pdf",
    "ingest_folder",
    "DocType",
    "DetectionResult",
    "FileReport",
    "IngestionReport",
    "Block",
    "Page",
    "ParsedDocument",
]
