"""
Dense Retriever API server using HTTP embeddings.
Search-R1 compatible API format.
"""
import json
import os
import argparse
import requests
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


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


app = FastAPI()

index = None
corpus = []
embedding_client = None


class SearchRequest(BaseModel):
    queries: List[str]
    topk: int = 5
    return_scores: bool = True


@app.post("/retrieve")
def retrieve(request: SearchRequest):
    """Search-R1 compatible /retrieve endpoint."""
    global index, corpus, embedding_client

    results = []
    for query in request.queries:
        # Encode query
        query_emb = embedding_client.encode([query])
        faiss.normalize_L2(query_emb)

        # Search
        D, I = index.search(query_emb, min(request.topk, len(corpus)))

        # Format results in Search-R1 format
        hits = []
        for i, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(corpus):
                hits.append({
                    "document": {
                        "id": corpus[idx]["id"],
                        "contents": corpus[idx]["contents"]
                    },
                    "score": float(score)
                })

        results.append(hits)

    return {"result": results}


@app.get("/status")
def status():
    return {
        "status": "OK",
        "index_size": len(corpus) if corpus else 0
    }


def load_index(index_path: str, corpus_path: str, embedding_url: str, embedding_model: str):
    global index, corpus, embedding_client

    print(f"[Startup] Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)

    print(f"[Startup] Loading corpus from: {corpus_path}")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line.strip()))

    print(f"[Startup] Loaded {len(corpus)} documents")

    embedding_client = HTTPEmbeddingClient(embedding_url, embedding_model)
    print(f"[Startup] Embedding client configured: {embedding_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Retriever API server")
    parser.add_argument("--index_path", type=str, required=True, help="FAISS index path")
    parser.add_argument("--corpus_path", type=str, required=True, help="Corpus JSONL path")
    parser.add_argument("--embedding_url", type=str, default="http://127.0.0.1:8080/v1", help="Embedding server URL")
    parser.add_argument("--embedding_model", type=str, default="bge-m3-Q8_0", help="Embedding model name")
    parser.add_argument("--port", type=int, default=8306, help="Server port")
    args = parser.parse_args()

    load_index(args.index_path, args.corpus_path, args.embedding_url, args.embedding_model)

    print(f"[Startup] Starting Dense Retriever API on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
