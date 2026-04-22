"""HypergraphRAG retriever HTTP server.

Usage:
    python serve_hypergraph.py --index-dir <path> --port <port>

The --index-dir must contain `index_entity.bin`, `index_hyperedge.bin`,
`kv_store_entities.json`, and `kv_store_hyperedges.json`, produced by
`build_hypergraph_index.py`.
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

from rag_clients import EmbeddingClient, RerankerClient, load_env


app = FastAPI(title="RAGSearch HypergraphRAG Retriever")

ENTITY_INDEX: Optional[faiss.Index] = None
HYPEREDGE_INDEX: Optional[faiss.Index] = None
ENTITIES: List[str] = []
HYPEREDGES: List[str] = []
EMBEDDER: Optional[EmbeddingClient] = None
RERANKER: Optional[RerankerClient] = None


class SearchRequest(BaseModel):
    queries: List[str]


def _maybe_rerank(query: str, documents: List[str], top_n: int) -> List[str]:
    if not documents:
        return []
    if RERANKER is None:
        return documents[:top_n]
    ranked = RERANKER.rerank(query, documents, top_n=top_n)
    return [documents[idx] for idx, _ in ranked]


@app.post("/search")
def search(request: SearchRequest):
    assert EMBEDDER is not None and ENTITY_INDEX is not None
    embeddings = EMBEDDER.encode(request.queries, normalize=True).astype(np.float32)

    _, entity_ids = ENTITY_INDEX.search(embeddings, 5)
    _, hyper_ids = HYPEREDGE_INDEX.search(embeddings, 5)

    out = []
    for i, query in enumerate(request.queries):
        ent = [ENTITIES[idx] for idx in entity_ids[i] if 0 <= idx < len(ENTITIES)]
        hyp = [HYPEREDGES[idx] for idx in hyper_ids[i] if 0 <= idx < len(HYPEREDGES)]
        ent = _maybe_rerank(query, ent, top_n=3)
        hyp = _maybe_rerank(query, hyp, top_n=2)
        context = "\n".join(ent) + "\n---\n" + "\n".join(hyp)
        out.append({"results": context})
    return out


@app.get("/status")
def status():
    return {
        "status": "ok",
        "retriever": "hypergraph",
        "entities": len(ENTITIES),
        "hyperedges": len(HYPEREDGES),
    }


def _load_kv_list(path: Path, field: str) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [data[key][field] for key in data]


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--reranker-base-url", default=None)
    parser.add_argument("--reranker-model", default=None)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    required = [
        index_dir / "index_entity.bin",
        index_dir / "index_hyperedge.bin",
        index_dir / "kv_store_entities.json",
        index_dir / "kv_store_hyperedges.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    global ENTITY_INDEX, HYPEREDGE_INDEX, ENTITIES, HYPEREDGES, EMBEDDER, RERANKER
    ENTITY_INDEX = faiss.read_index(str(index_dir / "index_entity.bin"))
    HYPEREDGE_INDEX = faiss.read_index(str(index_dir / "index_hyperedge.bin"))
    ENTITIES = _load_kv_list(index_dir / "kv_store_entities.json", "entity_name")
    HYPEREDGES = _load_kv_list(index_dir / "kv_store_hyperedges.json", "content")

    EMBEDDER = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    print(f"[serve_hypergraph] Loaded {len(ENTITIES)} entities, {len(HYPEREDGES)} hyperedges from {index_dir}")
    print(f"[serve_hypergraph] Embedding via {EMBEDDER.base_url} ({EMBEDDER.model})")

    try:
        RERANKER = RerankerClient(base_url=args.reranker_base_url, model=args.reranker_model)
        print(f"[serve_hypergraph] Reranker via {RERANKER.base_url} ({RERANKER.model})")
    except ValueError as exc:
        RERANKER = None
        print(f"[serve_hypergraph] Reranker disabled (passthrough): {exc}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
