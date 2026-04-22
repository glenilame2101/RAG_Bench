"""Embedding model selector — OpenAI-compatible only."""

from .base import BaseEmbeddingModel, EmbeddingConfig
from .OpenAI import OpenAIEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = ""):
    """All embedding traffic goes through the OpenAI HTTP class."""
    return OpenAIEmbeddingModel


__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "OpenAIEmbeddingModel",
    "_get_embedding_model_class",
]
