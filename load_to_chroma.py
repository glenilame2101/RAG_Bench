"""Load a Wikipedia parquet file (with pre-computed embeddings) into ChromaDB.

Usage:
    python load_to_chroma.py --parquet data/wiki.parquet --collection wikipedia

The parquet schema expected:
    id           int    – row id (used as Chroma string id)
    title        str    – article title
    text         str    – paragraph text  (stored as Chroma document)
    url          str    – Wikipedia URL
    wiki_id      int
    views        float
    paragraph_id int
    langs        int
    emb          list[float] – pre-computed embedding (NOT recomputed)
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Optional tqdm
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)

except ImportError:
    _tqdm = None

    def _progress(iterable, total=None, desc="", **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a parquet file with pre-computed embeddings into ChromaDB."
    )
    parser.add_argument("--parquet", required=True, help="Path to the parquet file.")
    parser.add_argument(
        "--collection", default="wikipedia", help="ChromaDB collection name (default: wikipedia)."
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Directory for ChromaDB persistence (default: ./chroma_db).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows per collection.add() call (default: 5000; Chroma max ≈ 5461).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: stop after this many rows (useful for testing).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_native(value: Any) -> Any:
    """Convert numpy scalars to native Python types so Chroma accepts them."""
    # Import numpy lazily to keep the function usable without it.
    try:
        import numpy as np  # noqa: F401

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return value


def _row_to_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": str(row["title"]),
        "url": str(row["url"]),
        "wiki_id": _to_native(row["wiki_id"]),
        "views": _to_native(row["views"]),
        "paragraph_id": _to_native(row["paragraph_id"]),
        "langs": _to_native(row["langs"]),
    }


def _emb_to_list(emb: Any) -> List[float]:
    """Convert an embedding (ndarray, list, or other sequence) to list[float]."""
    try:
        import numpy as np

        if isinstance(emb, np.ndarray):
            return list(map(float, emb))
    except ImportError:
        pass
    return list(map(float, emb))


def _insert_batch(collection: Any, df: Any) -> int:
    """Build Chroma-compatible lists from a DataFrame slice and call collection.add().

    Returns the number of rows inserted.
    """
    ids = [str(v) for v in df["id"]]
    documents = df["text"].tolist()
    embeddings = [_emb_to_list(e) for e in df["emb"]]
    metadatas = [_row_to_metadata(row) for row in df.to_dict("records")]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_parquet_to_chroma(
    parquet_path: str,
    collection_name: str,
    persist_dir: str,
    batch_size: int,
    limit: int | None,
) -> None:
    # --- ChromaDB client & collection -----------------------------------
    try:
        import chromadb
    except ImportError:
        sys.exit("chromadb is not installed. Run: pip install chromadb")

    client = chromadb.PersistentClient(path=persist_dir)
    # embedding_function=None tells Chroma we supply embeddings ourselves.
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
    )
    print(f"[chroma] Collection '{collection_name}' ready (current count: {collection.count()}).")

    # --- Open parquet (prefer pyarrow, fall back to pandas) -------------
    pq = None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pass

    total_inserted = 0

    if pq is not None:
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        if limit is not None:
            total_rows = min(total_rows, limit)

        batch_iter = pf.iter_batches(batch_size=batch_size)
        progress = _progress(
            batch_iter,
            total=((total_rows + batch_size - 1) // batch_size),
            desc="Loading batches",
            unit="batch",
        )

        for arrow_batch in progress:
            df = arrow_batch.to_pandas()

            # Respect --limit
            remaining = limit - total_inserted if limit is not None else None
            if remaining is not None:
                if remaining <= 0:
                    break
                df = df.iloc[:remaining]

            total_inserted += _insert_batch(collection, df)

            if limit is not None and total_inserted >= limit:
                break

    else:
        # pandas fallback (loads entire file into memory)
        try:
            import pandas as pd
        except ImportError:
            sys.exit("Neither pyarrow nor pandas is installed. Run: pip install pyarrow")

        print("[load] pyarrow not found — loading full parquet with pandas (memory-heavy).")
        df_full = pd.read_parquet(parquet_path)
        if limit is not None:
            df_full = df_full.iloc[:limit]

        total_rows = len(df_full)
        num_batches = (total_rows + batch_size - 1) // batch_size

        for batch_idx in _progress(range(num_batches), desc="Loading batches", unit="batch"):
            start = batch_idx * batch_size
            df = df_full.iloc[start : start + batch_size]
            total_inserted += _insert_batch(collection, df)

    print(f"\n[done] Inserted {total_inserted} rows.")
    print(f"[done] collection.count() = {collection.count()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    load_parquet_to_chroma(
        parquet_path=args.parquet,
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
