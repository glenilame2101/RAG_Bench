"""
Build HypergraphRAG index.

This creates the required FAISS indexes and KV stores for HypergraphRAG:
- index_entity.bin (FAISS index for entities)
- index_hyperedge.bin (FAISS index for hyperedges)
- kv_store_entities.json (entity data)
- kv_store_hyperedges.json (hyperedge content)

Usage:
    # Default: Use embedding/reranker from .env
    python build_hypergraph_index.py --input_jsonl corpus.jsonl --output_dir hypergraph_index

    # Override models via CLI
    python build_hypergraph_index.py --input_jsonl corpus.jsonl --output_dir hypergraph_index --embed-model bge-m3 --reranker-model bge-reranker-v2-m3

Environment variables (in .env):
    EMBEDDING_URL - Base URL for embedding model (e.g., http://localhost:8080/v1)
    EMBEDDING_MODEL - Embedding model name (e.g., bge-m3)
    RERANKER_URL - Base URL for reranker model (e.g., http://localhost:8080/v1)
    RERANKER_MODEL - Reranker model name (e.g., bge-reranker-v2-m3)
"""
import json
import os
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
import requests


TEXT_FIELDS = ["contents", "text", "content", "document", "body"]


def load_dotenv(path: str = ".env") -> None:
    """Load .env file into environment variables."""
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ[key] = value


def load_corpus_from_jsonl(jsonl_path: str) -> List[str]:
    """Load corpus from JSONL file with flexible field detection."""
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                text = None
                for field in TEXT_FIELDS:
                    if field in doc:
                        text = doc[field]
                        break
                if text is None:
                    text = doc.get("question", "") + " " + doc.get("answer", "")
                if isinstance(text, list):
                    text = " ".join(str(t) for t in text)
                text = str(text).strip()
                if text:
                    texts.append(text)
            except Exception as e:
                print(f"[Warning] Skipping line {line_num}: {e}")
                continue
    return texts


def load_corpus_from_dir(input_dir: str) -> List[str]:
    """Load text files from directory."""
    texts = []
    input_path = Path(input_dir)
    for file_path in sorted(input_path.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            texts.append(content)
    return texts


class EmbeddingClient:
    """Embedding client for llama.cpp compatible servers."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("EMBEDDING_URL", os.getenv("URL", "http://localhost:8080/v1"))
        self.model = model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        print(f"[Embedding] Using {self.base_url} with model {self.model}")

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings via llama.cpp server."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            for emb in data["data"]:
                vec = np.array(emb["embedding"], dtype=np.float32)
                if normalize:
                    vec = vec / np.linalg.norm(vec)
                all_embeddings.append(vec)

        return np.array(all_embeddings)


class RerankerClient:
    """Reranker client for llama.cpp compatible servers (via OpenAI-compatible rerank endpoint)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("RERANKER_URL", os.getenv("URL", "http://localhost:8080/v1"))
        self.model = model or os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3")
        print(f"[Reranker] Using {self.base_url} with model {self.model}")

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[tuple]:
        """Rerank documents using cross-attention reranking."""
        if not documents:
            return []

        response = requests.post(
            f"{self.base_url}/rerank",
            json={
                "model": self.model,
                "query": query,
                "input": documents,
                "top_n": top_n,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data["results"]:
            results.append((item["index"], item["relevance_score"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def extract_entities_and_hyperedges(texts: list) -> tuple:
    """Extract entities and hyperedges from texts using simple NLP."""
    entities = []
    hyperedges = []
    entity_texts = set()

    for text in texts:
        words = text.replace(".", " ").replace(",", " ").split()
        words = [w for w in words if len(w) > 3]

        for i in range(len(words)):
            for j in range(i + 1, min(i + 8, len(words) + 1)):
                phrase = " ".join(words[i:j])
                if len(phrase) > 5 and phrase not in entity_texts:
                    entities.append(phrase)
                    entity_texts.add(phrase)

        sentences = text.split(".")
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:
                hyperedges.append(sent)

    return list(set(entities))[:1000], list(set(hyperedges))[:500]


def build_hypergraph_index(
    texts: list,
    output_dir: str,
    embedding_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """Build HypergraphRAG index."""
    if not texts:
        raise ValueError("No texts provided for indexing")

    print(f"[HypergraphRAG] Building index for {len(texts)} texts...")

    emb_client = EmbeddingClient(base_url=embedding_url, model=embedding_model)

    print("[HypergraphRAG] Extracting entities and hyperedges...")
    entity_list, hyperedge_list = extract_entities_and_hyperedges(texts)

    if not entity_list:
        entity_list = texts[:100]
    if not hyperedge_list:
        hyperedge_list = texts[:50]

    print(f"[HypergraphRAG] {len(entity_list)} entities, {len(hyperedge_list)} hyperedges")

    print("[HypergraphRAG] Encoding entities...")
    entity_embeddings = emb_client.encode(entity_list, batch_size=32, normalize=True)

    print("[HypergraphRAG] Encoding hyperedges...")
    hyperedge_embeddings = emb_client.encode(hyperedge_list, batch_size=32, normalize=True)

    if entity_embeddings.shape[0] == 0:
        raise ValueError("No entity embeddings generated - check input data")

    print("[HypergraphRAG] Building FAISS indexes...")
    dim = entity_embeddings.shape[1]
    entity_index = faiss.IndexFlatIP(dim)
    entity_index.add(entity_embeddings.astype(np.float32))

    hyperedge_index = faiss.IndexFlatIP(dim)
    hyperedge_index.add(hyperedge_embeddings.astype(np.float32))

    print("[HypergraphRAG] Saving indexes and KV stores...")
    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(entity_index, os.path.join(output_dir, "index_entity.bin"))
    faiss.write_index(hyperedge_index, os.path.join(output_dir, "index_hyperedge.bin"))

    kv_entities = {str(i): {"entity_name": e, "embedding": entity_embeddings[i].tolist()} for i, e in enumerate(entity_list)}
    with open(os.path.join(output_dir, "kv_store_entities.json"), "w", encoding="utf-8") as f:
        json.dump(kv_entities, f, ensure_ascii=False)

    kv_hyperedges = {str(i): {"content": h, "embedding": hyperedge_embeddings[i].tolist()} for i, h in enumerate(hyperedge_list)}
    with open(os.path.join(output_dir, "kv_store_hyperedges.json"), "w", encoding="utf-8") as f:
        json.dump(kv_hyperedges, f, ensure_ascii=False)

    print(f"[HypergraphRAG] Index built successfully at: {output_dir}")
    print(f"[HypergraphRAG] Files created:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build HypergraphRAG index")
    parser.add_argument("--input_dir", type=str, help="Directory with text files")
    parser.add_argument("--input_jsonl", type=str, help="JSONL file with corpus")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--embed-url", type=str, default=None, help="Embedding server URL (overrides EMBEDDING_URL env)")
    parser.add_argument("--embed-model", type=str, default=None, help="Embedding model name (overrides EMBEDDING_MODEL env)")
    parser.add_argument("--reranker-url", type=str, default=None, help="Reranker server URL (overrides RERANKER_URL env)")
    parser.add_argument("--reranker-model", type=str, default=None, help="Reranker model name (overrides RERANKER_MODEL env)")
    args = parser.parse_args()

    if args.input_jsonl:
        texts = load_corpus_from_jsonl(args.input_jsonl)
    elif args.input_dir:
        texts = load_corpus_from_dir(args.input_dir)
    else:
        raise ValueError("Must specify --input_jsonl or --input_dir")

    print(f"[HypergraphRAG] Loaded {len(texts)} documents")

    build_hypergraph_index(
        texts=texts,
        output_dir=args.output_dir,
        embedding_url=args.embed_url,
        embedding_model=args.embed_model,
    )

    if args.reranker_url or args.reranker_model or os.getenv("RERANKER_MODEL"):
        print(f"[HypergraphRAG] Reranker configured: {args.reranker_url or os.getenv('RERANKER_URL')} / {args.reranker_model or os.getenv('RERANKER_MODEL')}")
        print("[HypergraphRAG] Note: Reranking is applied at query time, not during indexing.")


if __name__ == "__main__":
    main()