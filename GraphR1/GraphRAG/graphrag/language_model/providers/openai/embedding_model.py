# Copyright (c) 2025 RAGSearch Contributors.
# Licensed under MIT License

"""Simple OpenAI-compatible embedding model implementation."""

from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig


class SimpleEmbeddingModel:
    """Simple OpenAI-compatible Embedding Model using direct OpenAI SDK."""

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        **kwargs: Any,
    ):
        self.name = name
        self.config = config

        api_key = config.api_key or kwargs.get("api_key")
        base_url = config.api_base or kwargs.get("base_url")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        model = config.deployment_name or config.model
        if "/" not in model and config.model_provider:
            model = f"{config.model_provider}/{model}"

        self.model = model
        self.request_timeout = config.request_timeout

    async def aembed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Asynchronous embedding."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            timeout=self.request_timeout,
        )
        return [item.embedding for item in response.data]

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Synchronous embedding (wrapper around async)."""
        import asyncio
        return asyncio.run(self.aembed(texts, **kwargs))
