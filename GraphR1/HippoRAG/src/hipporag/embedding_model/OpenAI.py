"""HippoRAG embedding model adapter for any OpenAI-compatible /v1/embeddings server.

This is the *only* embedding backend HippoRAG ships with after the OpenAI-only
refactor; the GritLM / NV-Embed-v2 / Contriever / VLLM / Transformers /
Cohere variants have been removed.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np
from openai import OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Calls the configured OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        embedding_model_name: Optional[str] = None,
    ) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}"
            )

        self._init_embedding_config()

        base_url = self.global_config.embedding_base_url
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = [t.replace("\n", " ") if t else " " for t in texts]
        response = self.client.embeddings.create(
            input=texts, model=self.embedding_model_name
        )
        return np.array([v.embedding for v in response.data], dtype=np.float32)

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = int(params.pop("batch_size", 16) or 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            chunks = []
            for i in range(0, len(texts), batch_size):
                chunks.append(self.encode(texts[i : i + batch_size]))
            results = np.concatenate(chunks)

        if self.embedding_config.norm:
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            results = results / norms

        return results
