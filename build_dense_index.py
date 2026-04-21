"""
Build a Dense retriever index using HTTP embedding server.
"""
import json
import os
import argparse
import requests
import numpy as np
import faiss
from pathlib import Path


class HTTPEmbeddingClient:
    def __init__(self, base_url: str, model_name: str = "bge-m3-Q8_0"):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings via HTTP API."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "input": batch,
                "model": self.model_name
            }
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings, dtype=np.float32)


def load_corpus_from_dir(input_dir: str) -> list[dict]:
    """Load text files from directory and create corpus JSONL format."""
    corpus = []
    input_path = Path(input_dir)
    for file_path in sorted(input_path.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        corpus.append({
            "id": file_path.stem,
            "contents": content
        })
    return corpus


def load_corpus_from_jsonl(jsonl_path: str) -> list[dict]:
    """Load corpus from JSONL file."""
    corpus = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    return corpus


def build_index(
    corpus: list[dict],
    embedding_url: str,
    embedding_model: str,
    output_dir: str,
    dimension: int = 1024,
    batch_size: int = 32
):
    """Build FAISS index from corpus using HTTP embeddings."""
    print(f"[Index Builder] Loading {len(corpus)} documents...")

    # Initialize embedding client
    embedding_client = HTTPEmbeddingClient(embedding_url, embedding_model)

    # Get dimension from a test embedding
    print(f"[Index Builder] Testing embedding dimension...")
    test_emb = embedding_client.encode(["test"])[0]
    dim = len(test_emb)
    print(f"[Index Builder] Embedding dimension: {dim}")

    # Build FAISS index
    print(f"[Index Builder] Building FAISS index...")
    index = faiss.IndexFlatL2(dim)

    # Encode all documents in batches
    texts = [doc["contents"] for doc in corpus]
    embeddings = embedding_client.encode(texts, batch_size=batch_size)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add to index
    index.add(embeddings)

    # Save index
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "dense_index.faiss")
    faiss.write_index(index, index_path)
    print(f"[Index Builder] Index saved to: {index_path}")

    # Save corpus
    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[Index Builder] Corpus saved to: {corpus_path}")

    return index_path, corpus_path


def main():
    parser = argparse.ArgumentParser(description="Build Dense retriever index")
    parser.add_argument("--input_dir", type=str, help="Directory with text files")
    parser.add_argument("--input_jsonl", type=str, help="JSONL file with corpus")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--embedding_url", type=str, default="http://127.0.0.1:8080/v1", help="Embedding server URL")
    parser.add_argument("--embedding_model", type=str, default="bge-m3-Q8_0", help="Embedding model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    args = parser.parse_args()

    # Load corpus
    if args.input_dir:
        corpus = load_corpus_from_dir(args.input_dir)
    elif args.input_jsonl:
        corpus = load_corpus_from_jsonl(args.input_jsonl)
    else:
        raise ValueError("Must specify either --input_dir or --input_jsonl")

    # Build index
    build_index(
        corpus=corpus,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
