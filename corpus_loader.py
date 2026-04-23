"""Shared corpus loader for all build_*_index.py scripts.

Accepts:
- Directory of `*.txt` files (one document per file; id = file stem)
- `.jsonl` file (one JSON document per line)
- `.parquet` file (one document per row)

Text is read from the first of these fields that's present and non-empty:
`contents`, `text`, `content`, `document`, `body`. Falls back to
`question + " " + answer` for QA-style corpora.

Returns a list of `{"id": str, "text": str}` dicts. Each caller adapts
the shape it needs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm


TEXT_FIELDS = ("contents", "text", "content", "document", "body")


def _extract_text(row: dict) -> Optional[str]:
    for field in TEXT_FIELDS:
        val = row.get(field)
        if val is None:
            continue
        if isinstance(val, list):
            val = " ".join(str(t) for t in val)
        text = str(val).strip()
        if text:
            return text
    text = (str(row.get("question", "")) + " " + str(row.get("answer", ""))).strip()
    return text or None


def _extract_id(row: dict, fallback_index: int) -> str:
    for key in ("id", "_id", "doc_id", "document_id"):
        val = row.get(key)
        if val is not None and str(val) != "":
            return str(val)
    return str(fallback_index)


def load_corpus(
    corpus_path: str,
    partial_pct: Optional[float] = None,
    label: str = "Corpus",
) -> List[dict]:
    """Load a corpus as a list of `{"id": str, "text": str}` dicts.

    `partial_pct`, if given, takes the first N% of the corpus in the
    natural iteration order (sorted file listing for directories, file
    order for JSONL and Parquet).
    """
    path = Path(corpus_path)
    if path.is_dir():
        return _load_txt_dir(path, partial_pct, label)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _load_parquet(path, partial_pct, label)
    return _load_jsonl(path, partial_pct, label)


def _load_txt_dir(path: Path, partial_pct: Optional[float], label: str) -> List[dict]:
    files = sorted(path.glob("*.txt"))
    if partial_pct is not None:
        n = max(1, int(len(files) * partial_pct / 100))
        print(f"[{label}] --partial-index {partial_pct}%: using {n} of {len(files)} files")
        files = files[:n]
    docs: List[dict] = []
    for file_path in tqdm(files, desc=f"[{label}] Loading files", unit="file"):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            docs.append({"id": file_path.stem, "text": content})
    return docs


def _load_jsonl(path: Path, partial_pct: Optional[float], label: str) -> List[dict]:
    target_lines: Optional[int] = None
    if partial_pct is not None:
        print(f"[{label}] --partial-index {partial_pct}%: counting lines in {path}...")
        with path.open("r", encoding="utf-8") as fh:
            total = sum(1 for _ in fh)
        target_lines = max(1, int(total * partial_pct / 100))
        print(f"[{label}] Loading first {target_lines} of {total} lines")

    docs: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        bar = tqdm(fh, desc=f"[{label}] Loading corpus", unit=" line", total=target_lines)
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
            text = _extract_text(doc)
            if text is None:
                continue
            docs.append({"id": _extract_id(doc, len(docs)), "text": text})
    return docs


def _load_parquet(path: Path, partial_pct: Optional[float], label: str) -> List[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            f"Parquet corpus requires pyarrow (listed in requirements.txt). "
            f"Install with: pip install pyarrow. Original error: {exc}"
        )
    pf = pq.ParquetFile(str(path))
    total_rows = pf.metadata.num_rows
    target_rows: Optional[int] = None
    if partial_pct is not None:
        target_rows = max(1, int(total_rows * partial_pct / 100))
        print(f"[{label}] --partial-index {partial_pct}%: loading first {target_rows} of {total_rows} rows")
    else:
        print(f"[{label}] Loading {total_rows} rows")

    want = target_rows if target_rows is not None else total_rows
    docs: List[dict] = []
    read = 0
    bar = tqdm(total=want, desc=f"[{label}] Loading corpus", unit=" row")
    try:
        for batch in pf.iter_batches(batch_size=10_000):
            if read >= want:
                break
            pydict = batch.to_pydict()
            if not pydict:
                continue
            n_in_batch = len(next(iter(pydict.values())))
            for i in range(n_in_batch):
                if read >= want:
                    break
                row: dict[str, Any] = {k: pydict[k][i] for k in pydict}
                read += 1
                bar.update(1)
                text = _extract_text(row)
                if text is None:
                    continue
                docs.append({"id": _extract_id(row, len(docs)), "text": text})
    finally:
        bar.close()
    return docs
