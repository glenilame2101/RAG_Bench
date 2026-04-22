"""GraphRAG retriever HTTP server.

Usage:
    python serve_graphrag.py --index-dir <path> --port <port> [--mode local|global]

Pass the same directory you gave to `build_graphrag_index.py` as
`--output-dir`. Loads the parquet artifacts written under <index-dir>/output/
and exposes the canonical /search and /status endpoints.

Returns the *retrieved context text* (the text_units backing the retrieval),
not the LLM-generated answer — keeps the contract apples-to-apples with the
other retrievers (dense, hipporag, raptor, hypergraph, linear).
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from rag_clients import EmbeddingClient, LLMClient, load_env


app = FastAPI(title="RAGSearch GraphRAG Retriever")


CONFIG = None
ENTITIES: Optional[pd.DataFrame] = None
COMMUNITIES: Optional[pd.DataFrame] = None
COMMUNITY_REPORTS: Optional[pd.DataFrame] = None
TEXT_UNITS: Optional[pd.DataFrame] = None
RELATIONSHIPS: Optional[pd.DataFrame] = None
COVARIATES: Optional[pd.DataFrame] = None
INDEX_DIR: Optional[Path] = None
MODE: str = "local"


class SearchRequest(BaseModel):
    queries: List[str]


def _format_context(context_dict: dict, top_k: int = 5) -> str:
    """Extract retrieved passage text from GraphRAG's returned context dict.

    GraphRAG's local_search returns a dict whose keys vary by version; the
    most stable place to find the underlying text chunks is the 'sources'
    DataFrame (column 'text'), with a fallback to entity descriptions and
    community report excerpts so we never hand the eval an empty string.
    """
    parts: List[str] = []

    sources = context_dict.get("sources")
    if sources is not None and hasattr(sources, "head"):
        for _, row in sources.head(top_k).iterrows():
            text = row.get("text") or row.get("content") or ""
            if text:
                parts.append(str(text).strip())

    if not parts:
        entities = context_dict.get("entities")
        if entities is not None and hasattr(entities, "head"):
            for _, row in entities.head(top_k).iterrows():
                desc = row.get("description") or row.get("entity") or ""
                if desc:
                    parts.append(str(desc).strip())

    if not parts:
        reports = context_dict.get("reports") or context_dict.get("community_reports")
        if reports is not None and hasattr(reports, "head"):
            for _, row in reports.head(top_k).iterrows():
                text = row.get("content") or row.get("summary") or ""
                if text:
                    parts.append(str(text).strip())

    return "\n\n".join(parts) if parts else ""


async def _run_search(query: str) -> str:
    import graphrag.api as api

    if MODE == "global":
        _response, context = await api.global_search(
            config=CONFIG,
            entities=ENTITIES,
            communities=COMMUNITIES,
            community_reports=COMMUNITY_REPORTS,
            community_level=2,
            dynamic_community_selection=False,
            response_type="Multiple Paragraphs",
            query=query,
        )
    else:
        _response, context = await api.local_search(
            config=CONFIG,
            entities=ENTITIES,
            communities=COMMUNITIES,
            community_reports=COMMUNITY_REPORTS,
            text_units=TEXT_UNITS,
            relationships=RELATIONSHIPS,
            covariates=COVARIATES,
            community_level=2,
            response_type="Multiple Paragraphs",
            query=query,
        )

    if isinstance(context, dict):
        return _format_context(context)
    return str(context)


@app.post("/search")
def search(request: SearchRequest):
    out = []
    for query in request.queries:
        text = asyncio.run(_run_search(query))
        out.append({"results": text})
    return out


@app.get("/status")
def status():
    return {
        "status": "ok",
        "retriever": "graphrag",
        "mode": MODE,
        "index_dir": str(INDEX_DIR) if INDEX_DIR else None,
        "entities": int(len(ENTITIES)) if ENTITIES is not None else 0,
        "communities": int(len(COMMUNITIES)) if COMMUNITIES is not None else 0,
        "text_units": int(len(TEXT_UNITS)) if TEXT_UNITS is not None else 0,
    }


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=["local", "global"], default="local",
                        help="GraphRAG query mode. 'local' = entity-anchored "
                             "(retrieval-like, default); 'global' = community-summary based.")
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    args = parser.parse_args()

    embed_check = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    llm_check = LLMClient(base_url=args.llm_base_url, model=args.llm_model)

    os.environ["EMBEDDING_BASE_URL"] = embed_check.base_url
    os.environ["EMBEDDING_MODEL"] = embed_check.model
    os.environ["OPENAI_BASE_URL"] = llm_check.base_url
    os.environ["OPENAI_MODEL"] = llm_check.model
    os.environ.setdefault("OPENAI_API_KEY", llm_check.api_key or "EMPTY")

    from graphrag.config.load_config import load_config

    global CONFIG, ENTITIES, COMMUNITIES, COMMUNITY_REPORTS, TEXT_UNITS, RELATIONSHIPS, COVARIATES, INDEX_DIR, MODE
    INDEX_DIR = Path(args.index_dir).resolve()
    MODE = args.mode

    if not INDEX_DIR.is_dir():
        raise SystemExit(f"--index-dir does not exist: {INDEX_DIR}")
    output_dir = INDEX_DIR / "output"
    if not output_dir.is_dir():
        raise SystemExit(f"Expected GraphRAG output dir not found: {output_dir} "
                         f"(did build_graphrag_index.py finish?)")

    required = [
        output_dir / "entities.parquet",
        output_dir / "communities.parquet",
        output_dir / "community_reports.parquet",
        output_dir / "text_units.parquet",
        output_dir / "relationships.parquet",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing GraphRAG artifacts: {missing}")

    CONFIG = load_config(INDEX_DIR)
    ENTITIES = pd.read_parquet(output_dir / "entities.parquet")
    COMMUNITIES = pd.read_parquet(output_dir / "communities.parquet")
    COMMUNITY_REPORTS = pd.read_parquet(output_dir / "community_reports.parquet")
    TEXT_UNITS = pd.read_parquet(output_dir / "text_units.parquet")
    RELATIONSHIPS = pd.read_parquet(output_dir / "relationships.parquet")
    covariates_path = output_dir / "covariates.parquet"
    COVARIATES = pd.read_parquet(covariates_path) if covariates_path.is_file() else None

    print(f"[serve_graphrag] Loaded from {INDEX_DIR}")
    print(f"[serve_graphrag] mode={MODE}  entities={len(ENTITIES)}  "
          f"communities={len(COMMUNITIES)}  text_units={len(TEXT_UNITS)}")
    print(f"[serve_graphrag] LLM via {llm_check.base_url} ({llm_check.model})")
    print(f"[serve_graphrag] Embedding via {embed_check.base_url} ({embed_check.model})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
