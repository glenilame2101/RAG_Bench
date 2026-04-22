# Copyright (c) 2025 RAGSearch Contributors.
# Licensed under MIT License

"""Simple OpenAI provider."""

from .chat_model import SimpleChatModel
from .embedding_model import SimpleEmbeddingModel

__all__ = ["SimpleChatModel", "SimpleEmbeddingModel"]
