"""
Simple RAG evaluation script.

This script evaluates retrieval performance by:
1. Loading questions from a dataset
2. Calling the retriever API
3. Computing retrieval metrics (EM, F1, Recall@K)
"""
import argparse
import json
import os
import requests
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_dataset(dataset_path: str, dataset_name: str, limit: int = None) -> List[Dict]:
    """Load dataset from JSONL file."""
    if dataset_path and os.path.exists(dataset_path):
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data.append(json.loads(line))
        return data

    search_o1_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Search-o1", "data"
    )

    dataset_paths = {
        "bamboogle": os.path.join(search_o1_path, "FlashRAG_datasets", "bamboogle", "test.jsonl"),
        "hotpotqa": os.path.join(search_o1_path, "hotpotqa"),
        "musique": os.path.join(search_o1_path, "musique"),
        "nq": os.path.join(search_o1_path, "nq"),
        "2wikimultihopqa": os.path.join(search_o1_path, "2wikimultihopqa"),
        "triviaqa": os.path.join(search_o1_path, "triviaqa"),
        "popqa": os.path.join(search_o1_path, "popqa"),
    }

    path = dataset_paths.get(dataset_name.lower())
    if not path or not os.path.exists(path):
        print(f"[Eval] Dataset path not found: {path}")
        return []

    data = []
    if path.endswith(".jsonl"):
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = " ".join(text.strip().split())
    return text


def compute_exact_match(pred: str, gold: List[str]) -> float:
    """Compute exact match score."""
    pred_norm = normalize_answer(pred)
    for g in gold:
        if pred_norm == normalize_answer(g):
            return 1.0
    return 0.0


def compute_f1(pred: str, gold: List[str]) -> float:
    """Compute token-level F1 score."""
    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return 0.0

    best_f1 = 0.0
    for g in gold:
        gold_norm = normalize_answer(g)
        if not gold_norm:
            continue

        pred_tokens = set(pred_norm.split())
        gold_tokens = set(gold_norm.split())

        overlap = len(pred_tokens & gold_tokens)
        if overlap == 0:
            continue

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def call_retriever(url: str, query: str, top_k: int) -> List[str]:
    """Call retriever API and return results."""
    try:
        response = requests.post(
            url,
            json={"queries": [query]},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "results" in result[0]:
                return [r["results"] for r in result]
            return result
        elif isinstance(result, dict) and "results" in result:
            return [result["results"]]
        return []
    except Exception as e:
        print(f"[Eval] Error calling retriever: {e}")
        return []


def evaluate_single(question_data: Dict, retriever_url: str, top_k: int) -> Dict:
    """Evaluate a single question."""
    question = question_data.get("question", question_data.get("query", ""))
    gold_answers = question_data.get("answer", question_data.get("answers", []))
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    retrieved = call_retriever(retriever_url, question, top_k)
    retrieved_text = retrieved[0] if retrieved else ""

    em = compute_exact_match(retrieved_text, gold_answers)
    f1 = compute_f1(retrieved_text, gold_answers)

    return {
        "question": question,
        "gold": gold_answers,
        "retrieved": retrieved_text[:200] if retrieved_text else "",
        "em": em,
        "f1": f1
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--graphrag", default=None, help="Retriever type (unused)")
    parser.add_argument("--method", default="graphsearch", help="Method (unused)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions")
    args = parser.parse_args()

    retriever_url = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/search")

    print(f"[Eval] Loading dataset: {args.dataset}")
    data = load_dataset(args.data_path, args.dataset, args.limit if args.limit > 0 else None)
    print(f"[Eval] Loaded {len(data)} questions")

    if not data:
        print("[Eval] No data loaded, exiting")
        return

    print(f"[Eval] Evaluating with retriever at: {retriever_url}")
    print(f"[Eval] Top-K: {args.top_k}, Concurrency: {args.concurrency}")

    results = []
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(evaluate_single, item, retriever_url, args.top_k): item
                for item in data
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for item in data:
            result = evaluate_single(item, retriever_url, args.top_k)
            results.append(result)
            if len(results) % 10 == 0:
                print(f"[Eval] Processed {len(results)}/{len(data)} questions")

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total questions: {len(results)}")
    print(f"Exact Match (EM): {sum(em_scores)/len(em_scores):.4f}")
    print(f"F1 Score:         {sum(f1_scores)/len(f1_scores):.4f}")
    print("=" * 50)

    output_file = f"eval_results_{args.dataset}_{args.graphrag or 'unknown'}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": args.dataset,
            "retriever": args.graphrag,
            "metrics": {
                "em": sum(em_scores)/len(em_scores),
                "f1": sum(f1_scores)/len(f1_scores),
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
