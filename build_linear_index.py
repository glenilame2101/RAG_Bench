"""Build a LinearRAG index using the OpenAI-compatible embeddings + LLM endpoints.

Usage:
    python build_linear_index.py --corpus <path> --output-dir <dir> --name <dataset>

The output directory will contain a subdirectory `<name>/` with the parquet
embedding stores, the GraphML graph, and the JSON entity/sentence maps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
# LinearRAG.src.LinearRAG resolves through a namespace package rooted at
# GraphR1/, so put GraphR1/ on sys.path.
sys.path.insert(0, str(REPO_ROOT / "GraphR1"))

from rag_clients import EmbeddingCache, EmbeddingClient, LLMClient, load_env  # noqa: E402

from LinearRAG.src.LinearRAG import LinearRAG  # noqa: E402
from LinearRAG.src.config import LinearRAGConfig  # noqa: E402
from LinearRAG.src.utils import LLM_Model  # noqa: E402


TEXT_FIELDS = ("contents", "text", "content", "document", "body")


def load_corpus(corpus_path: str, partial_pct: Optional[float] = None) -> List[str]:
    path = Path(corpus_path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if partial_pct is not None:
            n = max(1, int(len(files) * partial_pct / 100))
            print(f"[Linear] --partial-index {partial_pct}%: using {n} of {len(files)} files")
            files = files[:n]
        passages: List[str] = []
        for i, file_path in enumerate(tqdm(files, desc="[Linear] Loading files", unit="file")):
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                passages.append(f"{i}: {content}")
        return passages

    if not path.is_file():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    target_lines: Optional[int] = None
    if partial_pct is not None:
        print(f"[Linear] --partial-index {partial_pct}%: counting lines in {path}...")
        with path.open("r", encoding="utf-8") as fh:
            total = sum(1 for _ in fh)
        target_lines = max(1, int(total * partial_pct / 100))
        print(f"[Linear] Loading first {target_lines} of {total} lines")

    passages: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        bar = tqdm(fh, desc="[Linear] Loading corpus", unit=" line", total=target_lines)
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
                passages.append(f"{i}: {text}")
    return passages


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus or directory of .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory containing the LinearRAG index files")
    parser.add_argument(
        "--name",
        default="default",
        help="Dataset subdirectory under --output-dir (default: 'default'). "
             "Override only if you want to keep multiple LinearRAG datasets in one --output-dir.",
    )
    parser.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--llm-base-url", default=None, help="Override OPENAI_BASE_URL (used for LLM_Model)")
    parser.add_argument("--llm-model", default=None, help="Override OPENAI_MODEL")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model used for NER")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--partial-index",
        type=float,
        default=None,
        help="Index only the first N%% of the corpus (e.g., 10 = first 10%%)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable embedding cache (default: cache every embedding to "
             "<output-dir>/.embedding_cache/ so re-runs skip already-embedded text)",
    )
    args = parser.parse_args()

    if args.partial_index is not None and not (0 < args.partial_index <= 100):
        raise SystemExit("--partial-index must be in (0, 100]")

    passages = load_corpus(args.corpus, partial_pct=args.partial_index)
    if not passages:
        raise SystemExit(f"No passages loaded from {args.corpus}")
    print(f"[Linear] Loaded {len(passages)} passages from {args.corpus}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wire a persistent embedding cache into the EmbeddingClient so LinearRAG's
    # internal calls to embedder.encode(...) reuse previously-computed vectors
    # across runs. The cache is content-addressed by (model, normalize, text).
    embed_cache: Optional[EmbeddingCache] = None
    if not args.no_checkpoint:
        cache_dir = output_dir / ".embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        embed_cache = EmbeddingCache(cache_dir / "embeddings.sqlite")

    embedder = EmbeddingClient(
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        cache=embed_cache,
    )
    # LLMClient validates env early; LLM_Model below is what LinearRAG actually uses.
    LLMClient(base_url=args.llm_base_url, api_key=None, model=args.llm_model)
    llm = LLM_Model(args.llm_model)

    config = LinearRAGConfig(
        dataset_name=args.name,
        embedding_model=embedder,
        spacy_model=args.spacy_model,
        llm_model=llm,
        max_workers=args.max_workers,
        working_dir=str(output_dir),
        batch_size=args.batch_size,
        retrieval_top_k=5,
        damping=0.4,
        iteration_threshold=0.01,
        max_iterations=10,
    )

    rag = LinearRAG(global_config=config)
    print(f"[Linear] Building index for dataset '{args.name}' under {output_dir}")
    try:
        rag.index(passages)
    finally:
        if embed_cache is not None:
            embed_cache.close()

    target_dir = output_dir / args.name
    print("[Linear] Files in index directory:")
    for entry in sorted(target_dir.iterdir()):
        print(f"  - {entry.name}")


if __name__ == "__main__":
    main()
