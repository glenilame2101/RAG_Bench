"""Load a parquet file with precomputed embeddings into a ChromaDB collection.

Expected parquet schema (matches the Cohere/Wikipedia-style dumps):
    id, title, text, url, wiki_id, views, paragraph_id, langs, emb

The `emb` column must contain per-row vectors (list/np.ndarray of floats) —
they are used as-is, so no re-embedding happens.

Usage:
    python load_parquet_to_chroma.py \
        --parquet /path/to/file.parquet \
        --persist-dir ./chroma_db \
        --collection wiki \
        --batch-size 2000

Requires: `pip install chromadb pandas pyarrow`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

import chromadb


# Columns used as metadata (everything other than id / text / emb).
METADATA_COLUMNS = ("title", "url", "wiki_id", "views", "paragraph_id", "langs")


def _coerce_metadata_value(value):
    """Chroma metadata values must be str/int/float/bool/None."""
    if value is None:
        return None
    # pandas/numpy NaN
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, bool, int, float)):
        return value
    # numpy scalar types
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def load_parquet(parquet_path: Path, limit: int | None) -> pd.DataFrame:
    print(f"[chroma-load] Reading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"[chroma-load] Loaded {len(df):,} rows, columns: {list(df.columns)}")
    if "emb" not in df.columns:
        raise SystemExit("Parquet file has no 'emb' column with precomputed embeddings.")
    if "text" not in df.columns:
        raise SystemExit("Parquet file has no 'text' column.")
    if limit is not None:
        df = df.head(limit)
        print(f"[chroma-load] --limit applied: using first {len(df):,} rows")
    return df


def build_batch(batch: pd.DataFrame) -> tuple[List[str], List[List[float]], List[str], List[dict]]:
    ids = [str(x) for x in batch["id"].tolist()]
    documents = [str(x) if x is not None else "" for x in batch["text"].tolist()]
    embeddings = [list(map(float, e)) for e in batch["emb"].tolist()]

    metadata_cols = [c for c in METADATA_COLUMNS if c in batch.columns]
    metadatas: List[dict] = []
    for row in batch[metadata_cols].itertuples(index=False, name=None):
        metadatas.append(
            {col: _coerce_metadata_value(val) for col, val in zip(metadata_cols, row)}
        )
    return ids, embeddings, documents, metadatas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--parquet", required=True, help="Path to the input .parquet file")
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Directory where the persistent Chroma database is stored (default: ./chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default="wiki",
        help="Chroma collection name (default: wiki)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Rows per add() call. Chroma caps single-batch size around 5k; 1000-2000 is safe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only load the first N rows (useful for smoke tests).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection first if it already exists.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.is_file():
        raise SystemExit(f"Parquet file not found: {parquet_path}")

    persist_dir = Path(args.persist_dir).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    df = load_parquet(parquet_path, limit=args.limit)

    client = chromadb.PersistentClient(path=str(persist_dir))

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"[chroma-load] Deleted existing collection '{args.collection}'")
        except Exception:  # noqa: BLE001 — collection may not exist
            pass

    # embedding_function=None -> we always supply our own precomputed vectors
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    print(
        f"[chroma-load] Using collection '{args.collection}' at {persist_dir} "
        f"(current count: {collection.count():,})"
    )

    total = len(df)
    added = 0
    with tqdm(total=total, unit="row", desc="[chroma-load] upserting") as bar:
        for batch in iter_batches(df, args.batch_size):
            ids, embeddings, documents, metadatas = build_batch(batch)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas if metadatas and metadatas[0] else None,
            )
            added += len(ids)
            bar.update(len(ids))

    print(
        f"[chroma-load] Done. Upserted {added:,} rows. "
        f"Collection '{args.collection}' now has {collection.count():,} items."
    )


if __name__ == "__main__":
    main()
