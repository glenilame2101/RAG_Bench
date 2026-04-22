"""RAPTOR embedding model — OpenAI-compatible HTTP endpoint only."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import requests

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        ...


class HTTPEmbeddingModel(BaseEmbeddingModel):
    """Calls a /v1/embeddings endpoint that follows the OpenAI schema."""

    def __init__(self, base_url: str, model_name: str):
        base_url = (base_url or "").rstrip("/")
        if base_url and not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        self.base_url = base_url
        self.model_name = model_name

    def _post(self, texts: List[str]) -> dict:
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(
            f"{self.base_url}/embeddings", json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()

    def create_embedding(self, text):
        if isinstance(text, str):
            text = [text]
        result = self._post(text)
        return np.array(result["data"][0]["embedding"], dtype=np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        result = self._post(list(texts))
        return np.array(
            [item["embedding"] for item in result["data"]], dtype=np.float32
        )
