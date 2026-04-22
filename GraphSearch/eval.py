"""Simple RAG retrieval evaluation.

The retriever is reached via the `RETRIEVER_URL` env var (set by
`run_benchmark.py`, or set manually). The corpus path is supplied with
`--corpus`; no dataset names are hardcoded here.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests


def load_dataset(corpus_path: str, limit: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(corpus_path):
        raise SystemExit(f"Corpus path not found: {corpus_path}")

    ext = corpus_path.lower().rsplit(".", 1)[-1]
    if ext == "jsonl":
        return _load_jsonl(corpus_path, limit)
    if ext == "json":
        return _load_json(corpus_path, limit)
    if ext == "parquet":
        return _load_parquet(corpus_path, limit)
    # try jsonl as the default
    return _load_jsonl(corpus_path, limit)


def _load_jsonl(path: str, limit: Optional[int]) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_json(path: str, limit: Optional[int]) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data[:limit] if limit else data
    if isinstance(data, dict):
        return [data]
    return []


def _load_parquet(path: str, limit: Optional[int]) -> List[Dict]:
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required for Parquet support: pip install pandas pyarrow")
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    return rows[:limit] if limit else rows


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def compute_exact_match(pred: str, gold: List[str]) -> float:
    pred_norm = normalize_answer(pred)
    return 1.0 if any(pred_norm == normalize_answer(g) for g in gold) else 0.0


def compute_f1(pred: str, gold: List[str]) -> float:
    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return 0.0
    best = 0.0
    pred_tokens = set(pred_norm.split())
    for g in gold:
        gold_norm = normalize_answer(g)
        if not gold_norm:
            continue
        gold_tokens = set(gold_norm.split())
        overlap = len(pred_tokens & gold_tokens)
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def call_retriever(url: str, query: str) -> str:
    try:
        resp = requests.post(url, json={"queries": [query]}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"[Eval] Error calling retriever: {exc}")
        return ""

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("results", "")
        return str(first)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return ""


def evaluate_single(question_data: Dict, retriever_url: str) -> Dict[str, Any]:
    question = question_data.get("question") or question_data.get("query") or ""
    gold = question_data.get("answer") or question_data.get("answers") or []
    if isinstance(gold, str):
        gold = [gold]
    retrieved = call_retriever(retriever_url, question)
    return {
        "question": question,
        "gold": gold,
        "retrieved": retrieved[:200] if retrieved else "",
        "em": compute_exact_match(retrieved, gold),
        "f1": compute_f1(retrieved, gold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--retriever", default="unknown", help="Retriever name (used for output filename)")
    parser.add_argument("--corpus", required=True, help="Path to JSONL/JSON/Parquet evaluation file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    retriever_url = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8306/search")

    data = load_dataset(args.corpus, args.limit if args.limit > 0 else None)
    print(f"[Eval] Loaded {len(data)} questions from {args.corpus}")
    if not data:
        return
    print(f"[Eval] Retriever: {retriever_url}")
    print(f"[Eval] Top-K: {args.top_k}  Concurrency: {args.concurrency}")

    results: List[Dict[str, Any]] = []
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(evaluate_single, item, retriever_url): item for item in data}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i, item in enumerate(data):
            results.append(evaluate_single(item, retriever_url))
            if (i + 1) % 10 == 0:
                print(f"[Eval] Processed {i + 1}/{len(data)}")

    em = sum(r["em"] for r in results) / max(1, len(results))
    f1 = sum(r["f1"] for r in results) / max(1, len(results))
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total questions: {len(results)}")
    print(f"Exact Match:     {em:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print("=" * 50)

    output_file = f"eval_results_{args.retriever}.json"
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "retriever": args.retriever,
                "corpus": args.corpus,
                "metrics": {"em": em, "f1": f1},
                "details": results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Eval] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
