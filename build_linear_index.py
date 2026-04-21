"""
Build LinearRAG index using HTTP embedding server.
"""
import json
import os
import argparse
from pathlib import Path

import sys
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "GraphR1"))
sys.path.insert(0, str(REPO_ROOT / "GraphR1" / "LinearRAG"))

from LinearRAG.src.LinearRAG import LinearRAG
from LinearRAG.src.config import LinearRAGConfig
from LinearRAG.src.utils import LLM_Model


class HTTPEmbeddingModel(ABC):
    """Simple HTTP embedding model for LinearRAG index builder."""
    def __init__(self, base_url: str, model_name: str = "bge-m3-Q8_0"):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(f"{self.base_url}/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()
        embeddings = np.array([item["embedding"] for item in result["data"]])
        if normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def __call__(self, text):
        return self.encode([text])[0]


class DummyLLMModel:
    """Dummy LLM for indexing (not used during retrieval)."""
    def infer(self, messages):
        return ""


def load_corpus_from_dir(input_dir: str) -> list:
    """Load text files from directory as list of passages."""
    passages = []
    input_path = Path(input_dir)
    for i, file_path in enumerate(sorted(input_path.glob("*.txt"))):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            passages.append(f"{i}: {content}")
    return passages


def load_corpus_from_jsonl(jsonl_path: str) -> list:
    """Load corpus from JSONL file as list of passages."""
    passages = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line.strip())
            content = obj.get('contents', obj.get('text', ''))
            if content:
                passages.append(f"{i}: {content}")
    return passages


def build_linear_index(
    passages: list,
    embedding_url: str,
    embedding_model: str,
    output_dir: str,
    dataset_name: str = "corpus",
    spacy_model: str = "en_core_web_trf",
    max_workers: int = 16,
):
    """Build LinearRAG index."""
    print(f"[LinearRAG Index Builder] Building index for {len(passages)} passages...")

    emb_model = HTTPEmbeddingModel(base_url=embedding_url, model_name=embedding_model)
    llm_model = DummyLLMModel()

    config = LinearRAGConfig(
        dataset_name=dataset_name,
        embedding_model=emb_model,
        spacy_model=spacy_model,
        llm_model=llm_model,
        max_workers=max_workers,
        working_dir=output_dir,
    )

    rag = LinearRAG(global_config=config)

    print(f"[LinearRAG Index Builder] Indexing passages...")
    rag.index(passages)

    print(f"[LinearRAG Index Builder] Index built successfully!")
    print(f"[LinearRAG Index Builder] Files saved to: {output_dir}/{dataset_name}/")


def main():
    parser = argparse.ArgumentParser(description="Build LinearRAG index")
    parser.add_argument("--input_dir", type=str, help="Directory with text files")
    parser.add_argument("--input_jsonl", type=str, help="JSONL file with corpus")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="corpus", help="Dataset name")
    parser.add_argument("--embedding_url", type=str, default="http://127.0.0.1:8080/v1", help="Embedding server URL")
    parser.add_argument("--embedding_model", type=str, default="bge-m3-Q8_0", help="Embedding model name")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="Spacy model for NER")
    parser.add_argument("--max_workers", type=int, default=16, help="Max workers for parallel processing")
    args = parser.parse_args()

    if args.input_dir:
        passages = load_corpus_from_dir(args.input_dir)
    elif args.input_jsonl:
        passages = load_corpus_from_jsonl(args.input_jsonl)
    else:
        raise ValueError("Must specify either --input_dir or --input_jsonl")

    print(f"[LinearRAG Index Builder] Loaded {len(passages)} passages")

    build_linear_index(
        passages=passages,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()