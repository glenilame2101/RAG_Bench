"""Base classes for HippoRAG embedding models.

Pruned to remove torch / sqlite caching helpers that only the deleted
GritLM/NV-Embed-v2/Contriever paths needed.
"""
import json
import multiprocessing
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, key: str) -> Any:
        if any(key.startswith(prefix) for prefix in ("_ipython_", "_repr_")):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        raise KeyError(f"'{key}' not found in configuration.")

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            raise KeyError(f"'{key}' not found in configuration.")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def batch_upsert(self, updates: Dict[str, Any]) -> None:
        self._data.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def to_json(self) -> str:
        return json.dumps(self._data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EmbeddingConfig":
        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> "EmbeddingConfig":
        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:
        return json.dumps(self._data, indent=4)


class BaseEmbeddingModel:
    global_config: BaseConfig
    embedding_model_name: str
    embedding_config: EmbeddingConfig
    embedding_dim: int

    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")
        self.embedding_model_name = self.global_config.embedding_model_name

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        raise NotImplementedError

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        return np.dot(query_vec, doc_vecs.T)


class EmbeddingCache:
    """A multiprocessing-safe in-memory cache for embeddings."""

    _manager = None
    _cache = None
    _lock = None

    @classmethod
    def _init_manager(cls):
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
            cls._cache = cls._manager.dict()
            cls._lock = threading.Lock()

    @classmethod
    def get(cls, content):
        cls._init_manager()
        return cls._cache.get(content)

    @classmethod
    def set(cls, content, embedding):
        cls._init_manager()
        with cls._lock:
            cls._cache[content] = embedding

    @classmethod
    def contains(cls, content):
        cls._init_manager()
        return content in cls._cache

    @classmethod
    def clear(cls):
        cls._init_manager()
        with cls._lock:
            cls._cache.clear()
