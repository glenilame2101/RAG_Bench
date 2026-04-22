"""Build a GraphRAG index using the OpenAI-compatible LLM + embeddings endpoints.

Usage:
    python build_graphrag_index.py --corpus <path> --output-dir <dir>

The output directory is treated as a GraphRAG project root. After a successful
run it contains:

    settings.yaml          (generated, references .env vars)
    input/doc_*.txt        (one file per passage from --corpus)
    output/*.parquet       (entities, relationships, communities, ...)
    output/lancedb/        (vector store)
    cache/                 (LLM response cache)

Requires `pip install graphrag lancedb` in the active venv. These are NOT in
requirements.txt because they pull in heavy deps the rest of the stack avoids;
install them manually only when you want to use the graphrag retriever.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from rag_clients import EmbeddingClient, LLMClient, load_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TEXT_FIELDS = ("contents", "text", "content", "document", "body")


SETTINGS_TEMPLATE = """\
completion_models:
  default_completion_model:
    model_provider: openai
    model: ${OPENAI_MODEL}
    api_base: ${OPENAI_BASE_URL}
    api_key: ${OPENAI_API_KEY}
    auth_method: api_key
    retry:
      type: exponential_backoff
      max_retries: 7
      base_delay: 2.0

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: ${EMBEDDING_MODEL}
    api_base: ${EMBEDDING_BASE_URL}
    api_key: ${OPENAI_API_KEY}
    auth_method: api_key

input:
  type: text
  storage:
    type: file
    base_dir: input
  file_pattern: ".*\\\\.txt$$"

output:
  type: file
  base_dir: output

cache:
  type: json
  storage:
    type: file
    base_dir: cache

vector_store:
  default_vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default

chunks:
  size: 1200
  overlap: 100

extract_graph:
  completion_model_id: default_completion_model
  max_gleanings: 1
  entity_types:
    - person
    - organization
    - location
    - event

cluster_graph:
  max_cluster_size: 10
  use_lcc: true

community_reports:
  completion_model_id: default_completion_model
  max_length: 2000
  max_input_length: 8000

embed_text:
  embedding_model_id: default_embedding_model
  vector_store_id: default_vector_store

local_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  text_unit_prop: 0.5
  community_prop: 0.1
  top_k_entities: 10
  top_k_relationships: 10
  max_context_tokens: 12000

global_search:
  completion_model_id: default_completion_model
  data_max_tokens: 12000
  map_max_length: 1000
  reduce_max_length: 2000
"""


def load_corpus(corpus_path: str, partial_pct: Optional[float] = None) -> List[str]:
    path = Path(corpus_path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if partial_pct is not None:
            n = max(1, int(len(files) * partial_pct / 100))
            print(f"[GraphRAG] --partial-index {partial_pct}%: using {n} of {len(files)} files")
            files = files[:n]
        docs = []
        for file_path in tqdm(files, desc="[GraphRAG] Loading files", unit="file"):
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                docs.append(content)
        return docs

    if not path.is_file():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    target_lines: Optional[int] = None
    if partial_pct is not None:
        print(f"[GraphRAG] --partial-index {partial_pct}%: counting lines in {path}...")
        with path.open("r", encoding="utf-8") as fh:
            total = sum(1 for _ in fh)
        target_lines = max(1, int(total * partial_pct / 100))
        print(f"[GraphRAG] Loading first {target_lines} of {total} lines")

    docs: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        bar = tqdm(fh, desc="[GraphRAG] Loading corpus", unit=" line", total=target_lines)
        for i, raw in enumerate(bar):
            if target_lines is not None and i >= target_lines:
                break
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


def write_input_files(docs: List[str], input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for i, doc in enumerate(tqdm(docs, desc="[GraphRAG] Writing input files", unit="file")):
        (input_dir / f"doc_{i:08d}.txt").write_text(doc, encoding="utf-8")


def write_settings(project_root: Path) -> Path:
    settings_path = project_root / "settings.yaml"
    settings_path.write_text(SETTINGS_TEMPLATE, encoding="utf-8")
    return settings_path


async def build_index_async(project_root: Path) -> None:
    # Local imports — graphrag is not in requirements.txt
    import graphrag.api as api
    from graphrag.config.load_config import load_config
    from graphrag.config.enums import IndexingMethod

    config = load_config(project_root)
    results = await api.build_index(
        config=config,
        method=IndexingMethod.Standard,
    )
    failed = 0
    for result in results:
        # Field name varies across graphrag versions: try `error` (singular,
        # current), then fall back to `errors` (plural, older).
        err = getattr(result, "error", None) or getattr(result, "errors", None)
        if err:
            logger.error(f"Workflow {result.workflow}: ERROR\n{err}")
            failed += 1
        else:
            logger.info(f"Workflow {result.workflow}: OK")
    if failed:
        raise SystemExit(f"{failed} workflow(s) failed; see errors above")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus or directory of .txt files")
    parser.add_argument("--output-dir", required=True, help="GraphRAG project root (will be created)")
    parser.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--llm-base-url", default=None, help="Override OPENAI_BASE_URL")
    parser.add_argument("--llm-model", default=None, help="Override OPENAI_MODEL")
    parser.add_argument(
        "--partial-index",
        type=float,
        default=None,
        help="Index only the first N%% of the corpus (e.g., 10 = first 10%%)",
    )
    args = parser.parse_args()

    if args.partial_index is not None and not (0 < args.partial_index <= 100):
        raise SystemExit("--partial-index must be in (0, 100]")

    # Validate env eagerly so we fail before writing files / spinning up graphrag.
    embed_check = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    llm_check = LLMClient(base_url=args.llm_base_url, model=args.llm_model)

    # GraphRAG's settings.yaml uses ${VAR} substitution from os.environ at load
    # time, so push any CLI overrides + the api_key into the environment.
    os.environ["EMBEDDING_BASE_URL"] = embed_check.base_url
    os.environ["EMBEDDING_MODEL"] = embed_check.model
    os.environ["OPENAI_BASE_URL"] = llm_check.base_url
    os.environ["OPENAI_MODEL"] = llm_check.model
    os.environ.setdefault("OPENAI_API_KEY", llm_check.api_key or "EMPTY")

    docs = load_corpus(args.corpus, partial_pct=args.partial_index)
    if not docs:
        raise SystemExit(f"No documents loaded from {args.corpus}")
    logger.info(f"Loaded {len(docs)} documents from {args.corpus}")

    project_root = Path(args.output_dir).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    write_input_files(docs, project_root / "input")
    settings_path = write_settings(project_root)
    logger.info(f"GraphRAG project root: {project_root}")
    logger.info(f"  settings: {settings_path}")
    logger.info(f"  llm:      {llm_check.base_url} ({llm_check.model})")
    logger.info(f"  embed:    {embed_check.base_url} ({embed_check.model})")
    logger.info(f"  docs:     {len(docs)} -> input/")

    asyncio.run(build_index_async(project_root))
    logger.info(f"Index built at: {project_root / 'output'}")


if __name__ == "__main__":
    main()
