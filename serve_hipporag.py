"""HippoRAG retriever HTTP server.

Usage:
    python serve_hipporag.py --index-dir <path> --port <port>

The --index-dir is the directory produced by `build_hipporag_index.py`
(typically `<output-dir>/<name>/hipporag`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "GraphR1" / "HippoRAG" / "src"))

from rag_clients import EmbeddingClient, LLMClient, load_env  # noqa: E402

from hipporag import HippoRAG  # noqa: E402


app = FastAPI(title="RAGSearch HippoRAG Retriever")

RAG: Optional[HippoRAG] = None
INDEX_DIR: Optional[Path] = None


class SearchRequest(BaseModel):
    queries: List[str]


@app.post("/search")
def search(request: SearchRequest):
    assert RAG is not None
    results = RAG.retrieve(queries=list(request.queries), num_to_retrieve=5)
    out = []
    for i, _ in enumerate(request.queries):
        item = results[i]
        passages = getattr(item, "docs", None) or []
        rendered = "\n\n".join(f"Wikipedia Title: {p}" for p in passages[:5])
        out.append({"results": rendered})
    return out


@app.get("/status")
def status():
    return {
        "status": "ok",
        "retriever": "hipporag",
        "index_dir": str(INDEX_DIR) if INDEX_DIR else None,
    }


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    args = parser.parse_args()

    embedder_check = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    llm_check = LLMClient(base_url=args.llm_base_url, model=args.llm_model)
    os.environ.setdefault("OPENAI_API_KEY", llm_check.api_key)

    global RAG, INDEX_DIR
    INDEX_DIR = Path(args.index_dir)
    if not INDEX_DIR.is_dir():
        raise SystemExit(f"--index-dir does not exist: {INDEX_DIR}")
    RAG = HippoRAG(
        save_dir=str(INDEX_DIR),
        llm_model_name=llm_check.model,
        llm_base_url=llm_check.base_url,
        embedding_model_name=embedder_check.model,
        embedding_base_url=embedder_check.base_url,
    )
    print(f"[serve_hipporag] Loaded HippoRAG from {INDEX_DIR}")
    print(f"[serve_hipporag] LLM via {llm_check.base_url} ({llm_check.model})")
    print(f"[serve_hipporag] Embedding via {embedder_check.base_url} ({embedder_check.model})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
