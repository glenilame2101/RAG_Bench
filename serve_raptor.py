"""RAPTOR retriever HTTP server.

Usage:
    python serve_raptor.py --index-dir <path> --port <port>

The --index-dir must contain `tree.pkl`, produced by `build_raptor_index.py`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "GraphR1" / "raptor"))

from rag_clients import EmbeddingClient, load_env  # noqa: E402

# build_raptor_index.py uses an inline RaptorTree class; we accept either
# that pickle shape or the legacy `Tree` shape from the vendored raptor lib.
from build_raptor_index import RaptorTree, TreeNode  # noqa: E402


app = FastAPI(title="RAGSearch RAPTOR Retriever")

TREE: Optional[RaptorTree] = None
EMBEDDER: Optional[EmbeddingClient] = None


class SearchRequest(BaseModel):
    queries: List[str]


def _retrieve(query: str, top_k: int = 5) -> str:
    assert TREE is not None and EMBEDDER is not None
    if not TREE.nodes:
        return ""
    query_emb = EMBEDDER.encode([query], normalize=True)[0]
    candidates = []
    for node in TREE.nodes.values():
        if node.embedding is None:
            continue
        emb = np.asarray(node.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm == 0:
            continue
        score = float(np.dot(query_emb, emb / norm))
        candidates.append((score, node.content))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    selected = candidates[: max(1, top_k)]
    return "\n".join(content for _, content in selected)


@app.post("/search")
def search(request: SearchRequest):
    return [{"results": _retrieve(q, top_k=5)} for q in request.queries]


@app.get("/status")
def status():
    return {
        "status": "ok",
        "retriever": "raptor",
        "index_size": len(TREE.nodes) if TREE else 0,
    }


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--index-dir", required=True, help="Directory with tree.pkl")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    tree_path = Path(args.index_dir) / "tree.pkl"
    if not tree_path.is_file():
        raise SystemExit(f"Missing tree.pkl in {args.index_dir}")

    global TREE, EMBEDDER
    with tree_path.open("rb") as fh:
        TREE = pickle.load(fh)
    EMBEDDER = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    print(f"[serve_raptor] Loaded tree with {len(TREE.nodes)} nodes from {tree_path}")
    print(f"[serve_raptor] Embedding via {EMBEDDER.base_url} ({EMBEDDER.model})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
