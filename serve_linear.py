"""LinearRAG retriever HTTP server.

Usage:
    python serve_linear.py --index-dir <path> --port <port> --name <dataset>

The --index-dir is the directory passed as `--output-dir` to
`build_linear_index.py`; --name is the dataset subdirectory name.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "GraphR1"))

from rag_clients import EmbeddingClient, LLMClient, load_env  # noqa: E402

from LinearRAG.src.LinearRAG import LinearRAG  # noqa: E402
from LinearRAG.src.config import LinearRAGConfig  # noqa: E402
from LinearRAG.src.utils import LLM_Model  # noqa: E402


app = FastAPI(title="RAGSearch LinearRAG Retriever")

RAG: Optional[LinearRAG] = None


class SearchRequest(BaseModel):
    queries: List[str]


@app.post("/search")
def search(request: SearchRequest):
    assert RAG is not None
    results = RAG.retrieve(list(request.queries))
    out = []
    for i, _ in enumerate(request.queries):
        item = results[i]
        passages = item.get("sorted_passage", [])
        rendered = "\n".join(passages)
        out.append({"results": rendered})
    return out


@app.get("/status")
def status():
    return {"status": "ok", "retriever": "linear"}


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", required=True, help="Dataset subdirectory under --index-dir")
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    embedder = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    LLMClient(base_url=args.llm_base_url, model=args.llm_model)  # validate config
    llm = LLM_Model(args.llm_model)

    index_dir = Path(args.index_dir)
    dataset_dir = index_dir / args.name
    if not dataset_dir.is_dir():
        raise SystemExit(f"Index subdirectory not found: {dataset_dir}")

    config = LinearRAGConfig(
        dataset_name=args.name,
        embedding_model=embedder,
        spacy_model=args.spacy_model,
        llm_model=llm,
        max_workers=args.max_workers,
        working_dir=str(index_dir),
        batch_size=args.batch_size,
        retrieval_top_k=5,
    )
    global RAG
    RAG = LinearRAG(global_config=config)
    print(f"[serve_linear] Loaded LinearRAG from {dataset_dir}")
    print(f"[serve_linear] Embedding via {embedder.base_url} ({embedder.model})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
