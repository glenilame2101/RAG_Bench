import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class HTTPEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, base_url: str, model_name: str = "bge-m3-Q8_0"):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model_name = model_name

    def create_embedding(self, text):
        if isinstance(text, str):
            text = [text]
        payload = {"model": self.model_name, "input": text}
        response = requests.post(f"{self.base_url}/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()
        return np.array(result["data"][0]["embedding"])

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(f"{self.base_url}/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()
        return np.array([item["embedding"] for item in result["data"]])


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
