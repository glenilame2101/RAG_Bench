"""
Simple OpenAI client wrapper that reads configuration from environment variables.
Replaces litellm for OpenAI-compatible APIs.
"""
import os
from typing import Any, Optional

from openai import OpenAI, AsyncOpenAI


class OpenAIClient:
    """Simple OpenAI client with env var configuration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
        self.base_url = base_url or os.getenv("URL", "") or os.getenv("OPENAI_BASE_URL", "")
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL_NAME", "bge-m3-Q8_0")
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chatcompletion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> Any:
        """Synchronous chat completion."""
        return self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )

    async def achatcompletion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> Any:
        """Asynchronous chat completion."""
        return await self.async_client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )

    def embedding(
        self,
        texts: list[str],
        model: Optional[str] = None,
        **kwargs,
    ) -> list[list[float]]:
        """Synchronous embedding."""
        response = self.client.embeddings.create(
            model=model or self.embedding_model,
            input=texts,
            **kwargs,
        )
        return [item.embedding for item in response.data]

    async def aembedding(
        self,
        texts: list[str],
        model: Optional[str] = None,
        **kwargs,
    ) -> list[list[float]]:
        """Asynchronous embedding."""
        response = await self.async_client.embeddings.create(
            model=model or self.embedding_model,
            input=texts,
            **kwargs,
        )
        return [item.embedding for item in response.data]


_default_client: Optional[OpenAIClient] = None


def get_default_client() -> OpenAIClient:
    """Get or create the default client singleton."""
    global _default_client
    if _default_client is None:
        _default_client = OpenAIClient()
    return _default_client


def reset_default_client():
    """Reset the default client (useful for testing)."""
    global _default_client
    _default_client = None
