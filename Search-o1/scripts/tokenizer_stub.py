"""Tokenizer stub used by Search-o1 run scripts after the OpenAI-only refactor.

The original scripts used ``transformers.AutoTokenizer`` to apply a chat
template before sending prompts to a local vLLM model. With the OpenAI-
compatible backend the *server* applies the chat template, so all the client
needs is an object that:

  * has ``pad_token`` / ``eos_token`` attributes
  * exposes ``apply_chat_template`` returning the prompt unchanged

This module provides exactly that, with no external dependencies.
"""
from __future__ import annotations

from typing import Any, Iterable


class StubTokenizer:
    pad_token: str = ""
    eos_token: str = ""
    padding_side: str = "left"

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "StubTokenizer":
        return cls()

    def apply_chat_template(
        self,
        messages: Iterable[Any],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **_kwargs: Any,
    ) -> str:
        if isinstance(messages, str):
            return messages
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(msg))
        return "\n\n".join(parts)


# Drop-in alias so ``from tokenizer_stub import AutoTokenizer`` works.
AutoTokenizer = StubTokenizer
