"""Service 02: LLM Gateway - Core LLM interaction logic."""
from __future__ import annotations

import json
from typing import TypeVar, Generic, Awaitable, cast

AsyncT = TypeVar("AsyncT")

from openai import AsyncOpenAI, OpenAI

from core.errors import (
    LLMGatewayError,
    RateLimitError,
    ModelUnavailableError,
    AuthError,
    ProviderError,
    StructuredOutputError,
    BudgetExceededError,
)

T = TypeVar("T")


def _read_api_key() -> str:
    """Read NVIDIA API key from environment or .env file."""
    import os
    from pathlib import Path
    # Try environment variable first
    key = os.getenv("NVIDIA_API_KEY")
    if key:
        return key.strip('"')
    # Try .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
        key = os.getenv("NVIDIA_API_KEY")
        if key:
            return key.strip('"')
    return ""


def complete_json(
    prompt: str,
    *,
    temperature: float = 0,
    max_tokens: int = 1200,
    thinking: bool = False,
    cache: bool = True,
    schema: type[T],
    agent: str = "",
    step: str = "",
) -> T:
    """Send a chat completion request and return validated Pydantic object.

    Procedure:
    1. Convert prompt to messages format
    2. Call gateway complete()
    3. Parse JSON from response text
    4. Validate with Pydantic schema
    5. On failure, make one repair call
    6. Still invalid → raise StructuredOutputError
    """
    import json
    from core.llm import LLMGateway

    api_key = _read_api_key()
    gateway = LLMGateway(api_key=api_key) if api_key else LLMGateway(api_key="dummy")

    # Convert prompt to messages format
    messages = [{"role": "user", "content": prompt}]

    # Step 1: Call complete to get raw text
    result = gateway.complete(
        messages, max_tokens=max_tokens,
        thinking=thinking, cache=cache,
        agent=agent, step=step,
    )

    # Step 2: Parse JSON from response text
    text = result.text
    # Strip code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        in_fence = False
        text_lines = []
        for line in lines:
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                text_lines.append(line)
        text = "\n".join(text_lines)

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    # Step 3: Validate with Pydantic
    try:
        validated = schema.model_validate_json(text)
        return validated
    except Exception as e:
        # Step 4: One repair call
        repair_messages = messages + [
            {"role": "assistant", "content": result.text},
            {"role": "user",
                "content": f"Your output was invalid: {str(e)}. "
                "Return corrected JSON only."},
        ]
        try:
            repair_result = gateway.complete(
                repair_messages, max_tokens=max_tokens,
                thinking=thinking, cache=False,
                agent=agent, step=step,
            )
            repair_text = repair_result.text
            # Strip code fences again
            if repair_text.startswith("```"):
                lines = repair_text.split("\n")
                in_fence = False
                text_lines = []
                for line in lines:
                    if line.startswith("```"):
                        in_fence = not in_fence
                        continue
                    if not in_fence:
                        text_lines.append(line)
                repair_text = "\n".join(text_lines)
            start = repair_text.find("{")
            end = repair_text.rfind("}")
            if start >= 0 and end > start:
                repair_text = repair_text[start:end + 1]
            # Step 3 again: validate repaired output
            validated = schema.model_validate_json(repair_text)
            return validated
        except Exception:
            # Step 5: Raise structured output error
            raise StructuredOutputError(
                f"Could not parse LLM output as valid JSON after repair. "
                f"Raw output: {result.text[:200]}")


async def acomplete_json(
    prompt: str,
    *,
    temperature: float = 0,
    max_tokens: int = 1200,
    thinking: bool = False,
    cache: bool = True,
    schema: type[T],
    agent: str = "",
    step: str = "",
) -> T:
    """Async version of complete_json."""
    import json
    from core.llm import LLMGateway

    api_key = _read_api_key()
    gateway = LLMGateway(api_key=api_key) if api_key else LLMGateway(api_key="dummy")

    # Convert prompt to messages format
    messages = [{"role": "user", "content": prompt}]

    # Step 1: Call complete to get raw text
    result = await gateway.acomplete(
        messages, max_tokens=max_tokens,
        thinking=thinking, cache=cache,
        agent=agent, step=step,
    )

    # Step 2: Parse JSON from response text
    text = result.text
    # Strip code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        in_fence = False
        text_lines = []
        for line in lines:
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                text_lines.append(line)
        text = "\n".join(text_lines)

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    # Step 3: Validate with Pydantic
    try:
        validated = schema.model_validate_json(text)
        return validated
    except Exception as e:
        # Step 4: One repair call
        repair_messages = messages + [
            {"role": "assistant", "content": result.text},
            {"role": "user",
                "content": f"Your output was invalid: {str(e)}. "
                "Return corrected JSON only."},
        ]
        try:
            repair_result = await gateway.acomplete(
                repair_messages, max_tokens=max_tokens,
                thinking=thinking, cache=False,
                agent=agent, step=step,
            )
            repair_text = repair_result.text
            # Strip code fences again
            if repair_text.startswith("```"):
                lines = repair_text.split("\n")
                in_fence = False
                text_lines = []
                for line in lines:
                    if line.startswith("```"):
                        in_fence = not in_fence
                        continue
                    if not in_fence:
                        text_lines.append(line)
                repair_text = "\n".join(text_lines)
            start = repair_text.find("{")
            end = repair_text.rfind("}")
            if start >= 0 and end > start:
                repair_text = repair_text[start:end + 1]
            # Step 3 again: validate repaired output
            validated = schema.model_validate_json(repair_text)
            return validated
        except Exception:
            # Step 5: Raise structured output error
            raise StructuredOutputError(
                f"Could not parse LLM output as valid JSON after repair. "
                f"Raw output: {result.text[:200]}")


class LLMResult:
    """Result from a single LLM call."""
    __slots__ = ("text", "prompt_tokens", "completion_tokens",
                "reasoning_tokens", "latency_ms", "cache_hit", "model")

    def __init__(
        self,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        latency_ms: int,
        cache_hit: bool,
        model: str,
    ) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.latency_ms = latency_ms
        self.cache_hit = cache_hit
        self.model = model


class LLMGateway(Generic[T]):
    """Synchronous gateway to the LLM with structured output, caching, retries,
    and budget tracking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = 45,
        max_retries: int = 0,
        thinking_enabled: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.async_client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout)
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking_enabled = thinking_enabled
        self.call_count = 0
        self.cache: dict[str, LLMResult] = {}

    def _make_cache_key(
        self,
        messages: list[dict[str, str]],
        schema_name: str,
        max_tokens: int,
        extra_body: dict,
    ) -> str:
        import hashlib
        content = json.dumps(
            {"messages": messages, "schema": schema_name,
                "max_tokens": max_tokens, "extra_body": extra_body},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 800,
        thinking: bool = False,
        cache: bool = True,
        agent: str = "",
        step: str = "",
    ) -> LLMResult:
        """Send a chat completion request and return the raw result."""
        # Budget check
        if self.call_count >= 150:  # MAX_LLM_CALLS_PER_RUN
            raise BudgetExceededError("MAX_LLM_CALLS_PER_RUN exceeded")

        cache_key = self._make_cache_key(
            messages, "default", max_tokens, {}
        ) if cache else ""

        if cache and cache_key in self.cache:
            result = self.cache[cache_key]
            result.cache_hit = True
            self.call_count += 1  # Count cached calls too? No, skip
            # Actually let's not count cached calls toward the budget
            return result

        # Control thinking: if thinking is off (default), ensure override is set
        if not thinking:
            extra_body = {"chat_template_kwargs": {
                "enable_thinking": False}}
        else:
            extra_body = {"chat_template_kwargs": {
                "enable_thinking": True}}
            if max_tokens < 3000:
                max_tokens = 3000

        try:
            response = self.client.chat.completions.create(
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                extra_body=extra_body,
            )
        except Exception as e:
            raise self._classify_error(e)

        # Parse response
        choice = response.choices[0]
        delta = choice.message.content or ""
        usage = response.usage or type("obj", (object,), {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()

        result = LLMResult(
            text=str(delta),
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            reasoning_tokens=getattr(response, "reasoning_tokens", 0) or 0,
            latency_ms=int(response.time_ms or 0),
            cache_hit=False,
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
        )

        # Cache if enabled and no reasoning tokens (thinking off)
        if cache and not result.reasoning_tokens:
            self.cache[cache_key] = result

        self.call_count += 1
        return result

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 800,
        thinking: bool = False,
        cache: bool = True,
        agent: str = "",
        step: str = "",
    ) -> LLMResult:
        """Async version of complete()."""
        cache_key = self._make_cache_key(
            messages, "default", max_tokens, {}) if cache else ""

        if cache and cache_key in self.cache:
            result = self.cache[cache_key]
            result.cache_hit = True
            return result

        if not thinking:
            extra_body = {"chat_template_kwargs": {
                "enable_thinking": False}}
        else:
            extra_body = {"chat_template_kwargs": {
                "enable_thinking": True}}
            if max_tokens < 3000:
                max_tokens = 3000

        try:
            response = await self.async_client.chat.completions.create(
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                extra_body=extra_body,
            )
        except Exception as e:
            raise self._classify_error(e)

        choice = response.choices[0]
        delta = choice.message.content or ""
        usage = response.usage or type("obj", (object,), {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()

        result = LLMResult(
            text=str(delta),
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            reasoning_tokens=getattr(response, "reasoning_tokens", 0) or 0,
            latency_ms=int(response.time_ms or 0),
            cache_hit=False,
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
        )

        if cache and not result.reasoning_tokens:
            self.cache[cache_key] = result

        self.call_count += 1
        return result

    def _classify_error(self, e: Exception) -> LLMGatewayError:
        """Classify an exception into a specific gateway error."""
        # In a real implementation, check status codes etc.
        # For now, generic error
        return LLMGatewayError(str(e))

    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        max_tokens: int = 1200,
        thinking: bool = False,
        cache: bool = True,
        agent: str = "",
        step: str = "",
    ) -> T:
        """Send a chat completion request and return validated Pydantic object.

        Procedure:
        1. Call complete() to get raw text
        2. Strip code fences and extract JSON
        3. Validate with Pydantic schema
        4. On failure, make one repair call
        5. Still invalid → raise StructuredOutputError
        """
        result = self.complete(
            messages, max_tokens=max_tokens,
            thinking=thinking, cache=cache,
            agent=agent, step=step,
        )

        # Step 2: Parse JSON from response text
        text = result.text
        # Strip code fences if present
        if text.startswith("```"):
            # Find the opening fence line
            lines = text.split("\n")
            # Skip first and last lines (fences)
            text_lines = []
            in_fence = False
            for line in lines:
                if line.startswith("```"):
                    in_fence = not in_fence
                    continue
                if not in_fence:
                    text_lines.append(line)
            text = "\n".join(text_lines)

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]

        # Step 3: Validate with Pydantic
        try:
            # Use model_validate for Pydantic v2
            validated = schema.model_validate_json(text)
            return validated
        except Exception as e:
            # Step 4: One repair call
            repair_messages = messages + [
                {"role": "assistant", "content": result.text},
                {"role": "user",
                    "content": f"Your output was invalid: {str(e)}. "
                    "Return corrected JSON only."},
            ]
            try:
                repair_result = self.complete(
                    repair_messages, max_tokens=max_tokens,
                    thinking=thinking, cache=False,
                    agent=agent, step=step,
                )
                repair_text = repair_result.text
                # Strip code fences again
                if repair_text.startswith("```"):
                    lines = repair_text.split("\n")
                    text_lines = []
                    in_fence = False
                    for line in lines:
                        if line.startswith("```"):
                            in_fence = not in_fence
                            continue
                        if not in_fence:
                            text_lines.append(line)
                    repair_text = "\n".join(text_lines)
                start = repair_text.find("{")
                end = repair_text.rfind("}")
                if start >= 0 and end > start:
                    repair_text = repair_text[start:end + 1]
                # Step 3 again: validate repaired output
                validated = schema.model_validate_json(repair_text)
                return validated
            except Exception:
                # Step 5: Raise structured output error
                raise StructuredOutputError(
                    f"Could not parse LLM output as valid JSON after repair. "
                    f"Raw output: {result.text[:200]}")