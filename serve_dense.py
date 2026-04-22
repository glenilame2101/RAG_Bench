"""Dense FAISS retriever HTTP server.

Usage:
    python serve_dense.py --index-dir <path> --port <port>

The --index-dir must contain `dense_index.faiss` and `corpus.jsonl`,
typically produced by `build_dense_index.py`.

Endpoints:
    POST /search    -> standard {"queries":[...]} interface
    POST /retrieve  -> Search-o1 compatible alias
    GET  /status    -> liveness + size
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from rag_clients import EmbeddingClient, load_env


app = FastAPI(title="RAGSearch Dense Retriever")

INDEX: Optional[faiss.Index] = None
CORPUS: List[dict] = []
EMBEDDER: Optional[EmbeddingClient] = None


class SearchRequest(BaseModel):
    queries: List[str]
    topk: int = 5
    return_scores: bool = True


def _search(queries: List[str], topk: int) -> List[List[dict]]:
    assert INDEX is not None and EMBEDDER is not None
    embeddings = EMBEDDER.encode(queries, normalize=True)
    topk = max(1, min(int(topk), len(CORPUS)))
    D, I = INDEX.search(embeddings.astype(np.float32), topk)
    out = []
    for row_d, row_i in zip(D, I):
        hits = []
        for score, idx in zip(row_d, row_i):
            if 0 <= idx < len(CORPUS):
                hits.append({
                    "document": {
                        "id": CORPUS[idx].get("id"),
                        "contents": CORPUS[idx].get("contents", ""),
                    },
                    "score": float(score),
                })
        out.append(hits)
    return out


@app.post("/retrieve")
def retrieve(request: SearchRequest):
    return {"result": _search(request.queries, request.topk)}


@app.post("/search")
def search(request: SearchRequest):
    hits_per_query = _search(request.queries, request.topk)
    return [
        {"results": "\n\n".join(h["document"].get("contents", "") for h in hits)}
        for hits in hits_per_query
    ]


@app.get("/status")
def status():
    return {"status": "ok", "retriever": "dense", "index_size": len(CORPUS)}


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--index-dir", required=True, help="Directory with dense_index.faiss + corpus.jsonl")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index_path = index_dir / "dense_index.faiss"
    corpus_path = index_dir / "corpus.jsonl"
    if not index_path.is_file() or not corpus_path.is_file():
        raise SystemExit(f"Missing dense_index.faiss or corpus.jsonl in {index_dir}")

    global INDEX, CORPUS, EMBEDDER
    INDEX = faiss.read_index(str(index_path))
    CORPUS = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    EMBEDDER = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    print(f"[serve_dense] Loaded {len(CORPUS)} docs from {index_dir}")
    print(f"[serve_dense] Embedding via {EMBEDDER.base_url} ({EMBEDDER.model})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
