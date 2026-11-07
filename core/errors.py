"""Service 02: LLM Gateway - Error hierarchy."""

from __future__ import annotations


class LLMGatewayError(Exception):
    """Base exception for all LLM gateway errors."""


class AuthError(LLMGatewayError):
    """Raised when authentication fails (401/403)."""
    def __init__(self, message: str = "Authentication failed. Check NVIDIA API key.") -> None:
        super().__init__(message)


class ModelUnavailableError(LLMGatewayError):
    """Raised when the requested model is not available (404/410)."""
    def __init__(self, model_id: str, message: str | None = None) -> None:
        self.model_id = model_id
        self._message = message or f"Model {model_id} is not served"
        super().__init__(self._message)


class RateLimitError(LLMGatewayError):
    """Raised when rate limit is exceeded after retries."""
    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        if retry_after:
            super().__init__(
                f"Rate limit exceeded. Retry after {retry_after} seconds"
            )
        else:
            super().__init__("Rate limit exceeded. No retry-after header received")


class ProviderError(LLMGatewayError):
    """Raised for provider errors (5xx, timeouts, etc.)."""
    def __init__(self, message: str = "Provider error. Check logs for details.") -> None:
        super().__init__(message)


class StructuredOutputError(LLMGatewayError):
    """Raised when LLM output cannot be parsed as valid JSON after repair."""
    def __init__(
        self,
        message: str,
        raw_output: str | None = None,
    ) -> None:
        self.raw_output = raw_output
        super().__init__(message)


class BudgetExceededError(LLMGatewayError):
    """Raised when MAX_LLM_CALLS_PER_RUN is exceeded."""
    def __init__(self, limit: int = 150) -> None:
        super().__init__(
            f"Budget exceeded: {limit} LLM calls per run limit reached. "
            "Consider increasing the limit or optimizing call usage."
        )


class RetrievalModelError(Exception):
    """Raised when embedding retrieval fails.

    Attributes:
        failed_indices: List of chunk indices that failed to embed
        message: Human-readable error message
    """

    def __init__(self, failed_indices: list[int], message: str) -> None:
        self.failed_indices = failed_indices
        self.message = message
        super().__init__(message)