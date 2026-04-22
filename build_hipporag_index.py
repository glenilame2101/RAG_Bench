"""Build a HippoRAG index using the OpenAI-compatible LLM + embeddings endpoints.

Usage:
    python build_hipporag_index.py --corpus <path> --output-dir <dir> --name <dataset>

The output directory will contain a subdirectory `<name>/hipporag/` with the
chunk / entity / fact embedding stores and the constructed graph.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "GraphR1" / "HippoRAG" / "src"))

from rag_clients import EmbeddingClient, LLMClient, load_env  # noqa: E402

from hipporag import HippoRAG  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TEXT_FIELDS = ("contents", "text", "content", "document", "body")


def load_corpus(corpus_path: str) -> List[str]:
    path = Path(corpus_path)
    if path.is_dir():
        docs = []
        for file_path in sorted(path.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                docs.append(content)
        return docs

    if not path.is_file():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    docs: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = None
            for field in TEXT_FIELDS:
                if field in doc:
                    text = doc[field]
                    break
            if text is None:
                text = (doc.get("question", "") + " " + doc.get("answer", "")).strip()
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            text = str(text).strip()
            if text:
                docs.append(text)
    return docs


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus or directory of .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory containing the HippoRAG index files")
    parser.add_argument("--name", required=True, help="Dataset subdirectory name under --output-dir")
    parser.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--llm-base-url", default=None, help="Override OPENAI_BASE_URL")
    parser.add_argument("--llm-model", default=None, help="Override OPENAI_MODEL")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of docs per index() call")
    args = parser.parse_args()

    docs = load_corpus(args.corpus)
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

    save_dir = Path(args.output_dir) / args.name / "hipporag"
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

    for start in range(0, len(docs), args.batch_size):
        batch = docs[start : start + args.batch_size]
        rag.index(docs=batch)
        logger.info(f"Indexed {min(start + args.batch_size, len(docs))}/{len(docs)} docs")

    logger.info(f"Index built at: {save_dir}")


if __name__ == "__main__":
    main()
