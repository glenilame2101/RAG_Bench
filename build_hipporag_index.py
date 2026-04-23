"""Build a HippoRAG index using the OpenAI-compatible LLM + embeddings endpoints.

Usage:
    python build_hipporag_index.py --corpus <path> --output-dir <dir>

The index is written directly under --output-dir. Point `serve_hipporag.py`
at the same directory with `--index-dir`. HippoRAG itself creates an internal
`<llm>_<embedding>/` subdirectory for its embedding stores and graph; you do
not need to know or care about that path.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "GraphR1" / "HippoRAG" / "src"))

from rag_clients import EmbeddingClient, LLMClient, load_env  # noqa: E402

from hipporag import HippoRAG  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str, partial_pct: Optional[float] = None) -> List[str]:
    from corpus_loader import load_corpus as _load_corpus
    docs = _load_corpus(corpus_path, partial_pct=partial_pct, label="HippoRAG")
    return [d["text"] for d in docs]


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus or directory of .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory where the HippoRAG index will be written")
    parser.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--llm-base-url", default=None, help="Override OPENAI_BASE_URL")
    parser.add_argument("--llm-model", default=None, help="Override OPENAI_MODEL")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of docs per index() call")
    parser.add_argument(
        "--partial-index",
        type=float,
        default=None,
        help="Index only the first N%% of the corpus (e.g., 10 = first 10%%)",
    )
    args = parser.parse_args()

    if args.partial_index is not None and not (0 < args.partial_index <= 100):
        raise SystemExit("--partial-index must be in (0, 100]")

    docs = load_corpus(args.corpus, partial_pct=args.partial_index)
    if not docs:
        raise SystemExit(f"No documents loaded from {args.corpus}")
    logger.info(f"Loaded {len(docs)} documents from {args.corpus}")

    # Validate the env config eagerly with explicit, helpful errors.
    embedder_check = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    llm_check = LLMClient(base_url=args.llm_base_url, model=args.llm_model)

    embedding_base_url = embedder_check.base_url
    embedding_model = embedder_check.model
    llm_base_url = llm_check.base_url
    llm_model = llm_check.model

    # HippoRAG vendored code reads OPENAI_API_KEY directly when constructing
    # its CacheOpenAI client; ensure it's exported.
    os.environ.setdefault("OPENAI_API_KEY", llm_check.api_key)

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"HippoRAG: save_dir={save_dir}")
    logger.info(f"HippoRAG: llm={llm_base_url} ({llm_model})")
    logger.info(f"HippoRAG: embedding={embedding_base_url} ({embedding_model})")

    rag = HippoRAG(
        save_dir=str(save_dir),
        llm_model_name=llm_model,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model,
        embedding_base_url=embedding_base_url,
    )

    for start in tqdm(
        range(0, len(docs), args.batch_size),
        desc="[HippoRAG] Indexing batches",
        unit="batch",
    ):
        batch = docs[start : start + args.batch_size]
        rag.index(docs=batch)

    logger.info(f"Index built at: {save_dir}")


if __name__ == "__main__":
    main()
