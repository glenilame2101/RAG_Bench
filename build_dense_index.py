"""Build a dense FAISS index using the OpenAI-compatible embeddings endpoint.

Usage:
    python build_dense_index.py --corpus <path-to-corpus> --output-dir <dir>

`--corpus` may be a `.jsonl` file (one JSON document per line, with any of
the fields `contents`, `text`, `content`, `document`, `body`) or a directory
of `*.txt` files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np

from rag_clients import EmbeddingClient, load_env


TEXT_FIELDS = ("contents", "text", "content", "document", "body")


def load_corpus(corpus_path: str) -> List[dict]:
    path = Path(corpus_path)
    if path.is_dir():
        corpus = []
        for file_path in sorted(path.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                corpus.append({"id": file_path.stem, "contents": content})
        return corpus

    if not path.is_file():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    corpus = []
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
            if not text:
                continue
            corpus.append({"id": str(doc.get("id", len(corpus))), "contents": text})
    return corpus


def build_index(
    corpus: List[dict],
    output_dir: Path,
    embedding_base_url: str | None,
    embedding_model: str | None,
    batch_size: int,
) -> None:
    embedder = EmbeddingClient(base_url=embedding_base_url, model=embedding_model)
    print(f"[Dense] Embedding {len(corpus)} documents via {embedder.base_url} ({embedder.model})")

    texts = [doc["contents"] for doc in corpus]
    embeddings = embedder.encode(texts, batch_size=batch_size, normalize=True)
    if embeddings.size == 0:
        raise RuntimeError("No embeddings generated — corpus likely empty")

    dim = int(embeddings.shape[1])
    print(f"[Dense] Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
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
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    if not corpus:
        raise SystemExit(f"No documents loaded from {args.corpus}")
    print(f"[Dense] Loaded {len(corpus)} documents from {args.corpus}")

    build_index(
        corpus=corpus,
        output_dir=Path(args.output_dir),
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
