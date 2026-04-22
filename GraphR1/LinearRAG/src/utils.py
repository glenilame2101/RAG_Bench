"""LinearRAG utility helpers — uses the canonical OpenAI-compatible env names."""
from __future__ import annotations

import logging
import os
import re
import string
from hashlib import md5
from typing import Iterable

import numpy as np
from openai import OpenAI


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


class LLM_Model:
    """Thin LLM wrapper used by LinearRAG indexing/retrieval pipelines.

    Reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment if no explicit
    base_url / api_key is passed.
    """

    def __init__(self, llm_model: str = None):
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_name = llm_model or os.getenv("OPENAI_MODEL")
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_config = {
            "model": model_name,
            "max_tokens": 2000,
            "temperature": 0,
        }

    def infer(self, messages: Iterable[dict]) -> str:
        response = self.openai_client.chat.completions.create(
            **self.llm_config, messages=list(messages)
        )
        return response.choices[0].message.content or ""


def normalize_answer(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def setup_logging(log_file: str) -> None:
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def min_max_normalize(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return arr
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val - min_val == 0:
        return np.ones_like(arr)
    return (arr - min_val) / (max_val - min_val)
