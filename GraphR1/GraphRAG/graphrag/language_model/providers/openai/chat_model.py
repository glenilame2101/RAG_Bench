# Copyright (c) 2025 RAGSearch Contributors.
# Licensed under MIT License

"""Simple OpenAI-compatible chat model implementation."""

import json
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any, Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.response.base import ModelResponse as MR


class SimpleModelOutput(BaseModel):
    content: str = Field(description="The generated text content")
    full_response: Optional[Any] = Field(default=None)


class SimpleModelResponse(BaseModel):
    output: SimpleModelOutput = Field(description="The output from the model")
    parsed_response: Optional[BaseModel] = Field(default=None)
    history: list = Field(default_factory=list)


class SimpleChatModel:
    """Simple OpenAI-compatible Chat Model using direct OpenAI SDK."""

    def __init__(
        self,
        name: str,
        config: "LanguageModelConfig",
        cache: "PipelineCache | None" = None,
        **kwargs: Any,
    ):
        self.name = name
        self.config = config
        self.cache = cache.child(self.name) if cache else None

        api_key = config.api_key or kwargs.get("api_key")
        base_url = config.api_base or kwargs.get("base_url")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        model = config.deployment_name or config.model
        if "/" not in model and config.model_provider:
            model = f"{config.model_provider}/{model}"

        self.model = model
        self.request_timeout = config.request_timeout
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens
        self.max_completion_tokens = config.max_completion_tokens
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty

    def _get_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        args_to_include = [
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "logprobs",
            "top_logprobs",
            "extra_headers",
        ]
        new_args = {k: v for k, v in kwargs.items() if k in args_to_include}

        if kwargs.get("json"):
            new_args["response_format"] = {"type": "json_object"}

        return new_args

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> "MR":
        new_kwargs = self._get_kwargs(**kwargs)
        messages: list[dict[str, str]] = list(history) if history else []
        messages.append({"role": "user", "content": prompt})

        common_params = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "timeout": self.request_timeout,
        }

        if self.max_tokens:
            common_params["max_tokens"] = self.max_tokens
        elif self.max_completion_tokens:
            common_params["max_tokens"] = self.max_completion_tokens

        if self.frequency_penalty:
            common_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            common_params["presence_penalty"] = self.presence_penalty

        response = await self.async_client.chat.completions.create(
            messages=messages,
            stream=False,
            **{**common_params, **new_kwargs},
        )

        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content or "",
        })

        parsed_response: Optional[BaseModel] = None
        if "response_format" in new_kwargs:
            parsed_dict = json.loads(response.choices[0].message.content or "{}")
            parsed_response = parsed_dict

        return SimpleModelResponse(
            output=SimpleModelOutput(content=response.choices[0].message.content or ""),
            parsed_response=parsed_response,
            history=messages,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        new_kwargs = self._get_kwargs(**kwargs)
        messages: list[dict[str, str]] = list(history) if history else []
        messages.append({"role": "user", "content": prompt})

        common_params = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "timeout": self.request_timeout,
        }

        if self.max_tokens:
            common_params["max_tokens"] = self.max_tokens
        elif self.max_completion_tokens:
            common_params["max_tokens"] = self.max_completion_tokens

        response = await self.async_client.chat.completions.create(
            messages=messages,
            stream=True,
            **{**common_params, **new_kwargs},
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> "MR":
        new_kwargs = self._get_kwargs(**kwargs)
        messages: list[dict[str, str]] = list(history) if history else []
        messages.append({"role": "user", "content": prompt})

        common_params = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "timeout": self.request_timeout,
        }

        if self.max_tokens:
            common_params["max_tokens"] = self.max_tokens
        elif self.max_completion_tokens:
            common_params["max_tokens"] = self.max_completion_tokens

        if self.frequency_penalty:
            common_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            common_params["presence_penalty"] = self.presence_penalty

        response = self.client.chat.completions.create(
            messages=messages,
            stream=False,
            **{**common_params, **new_kwargs},
        )

        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content or "",
        })

        parsed_response: Optional[BaseModel] = None
        if "response_format" in new_kwargs:
            parsed_dict = json.loads(response.choices[0].message.content or "{}")
            parsed_response = parsed_dict

        return SimpleModelResponse(
            output=SimpleModelOutput(content=response.choices[0].message.content or ""),
            parsed_response=parsed_response,
            history=messages,
        )

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        new_kwargs = self._get_kwargs(**kwargs)
        messages: list[dict[str, str]] = list(history) if history else []
        messages.append({"role": "user", "content": prompt})

        common_params = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "timeout": self.request_timeout,
        }

        if self.max_tokens:
            common_params["max_tokens"] = self.max_tokens
        elif self.max_completion_tokens:
            common_params["max_tokens"] = self.max_completion_tokens

        response = self.client.chat.completions.create(
            messages=messages,
            stream=True,
            **{**common_params, **new_kwargs},
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
