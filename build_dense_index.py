"""Build a dense FAISS index using the OpenAI-compatible embeddings endpoint.

Usage:
    python build_dense_index.py --corpus <path-to-corpus> --output-dir <dir>

`--corpus` may be a `.jsonl` file (one JSON document per line), a `.parquet`
file (one document per row), or a directory of `*.txt` files. Text is
read from the first present field among `contents`, `text`, `content`,
`document`, `body` (or `question`+`answer`).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from corpus_loader import load_corpus as _load_corpus
from rag_clients import EmbeddingClient, load_env


def load_corpus(corpus_path: str, partial_pct: Optional[float] = None) -> List[dict]:
    docs = _load_corpus(corpus_path, partial_pct=partial_pct, label="Dense")
    return [{"id": d["id"], "contents": d["text"]} for d in docs]


def build_index(
    corpus: List[dict],
    output_dir: Path,
    embedding_base_url: str | None,
    embedding_model: str | None,
    batch_size: int,
    checkpoint_dir: Optional[Path] = None,
    max_chars: Optional[int] = None,
) -> None:
    embedder = EmbeddingClient(base_url=embedding_base_url, model=embedding_model)
    print(f"[Dense] Embedding {len(corpus)} documents via {embedder.base_url} ({embedder.model})")

    texts = [doc["contents"] for doc in corpus]
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        embeddings = embedder.encode_with_checkpoint(
            texts,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            normalize=True,
            save_every_pct=1.0,
            max_chars=max_chars,
        )
    else:
        embeddings = embedder.encode(
            texts, batch_size=batch_size, normalize=True, max_chars=max_chars
        )
    if embeddings.size == 0:
        raise RuntimeError("No embeddings generated — corpus likely empty")

    dim = int(embeddings.shape[1])
    print(f"[Dense] Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    index_path = output_dir / "dense_index.faiss"
    corpus_path = output_dir / "corpus.jsonl"
    faiss.write_index(index, str(index_path))
    with corpus_path.open("w", encoding="utf-8") as fh:
        for doc in corpus:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[Dense] Wrote {index_path}")
    print(f"[Dense] Wrote {corpus_path}")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus or directory of .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory for the FAISS index + corpus.jsonl")
    parser.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
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
        help="Disable embedding checkpointing. Default: checkpoint every 1%% to "
             "<output-dir>/.checkpoint/. The checkpoint persists across runs and "
             "acts as a prefix cache — re-running with a larger --partial-index "
             "reuses embeddings from any previous run that shares the same prefix.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Truncate each text to this many characters before embedding "
             "(default: 8000, safe for any script under a typical 8192-token "
             "embedding context). Pass 0 to disable. Changing this value "
             "invalidates an existing checkpoint.",
    )
    args = parser.parse_args()

    if args.partial_index is not None and not (0 < args.partial_index <= 100):
        raise SystemExit("--partial-index must be in (0, 100]")

    corpus = load_corpus(args.corpus, partial_pct=args.partial_index)
    if not corpus:
        raise SystemExit(f"No documents loaded from {args.corpus}")
    print(f"[Dense] Loaded {len(corpus)} documents from {args.corpus}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = None if args.no_checkpoint else output_dir / ".checkpoint"

    build_index(
        corpus=corpus,
        output_dir=output_dir,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
        max_chars=args.max_chars if args.max_chars > 0 else None,
    )


if __name__ == "__main__":
    main()
