#!/usr/bin/env python3
"""
Unified benchmark launcher for RAGSearch.

Usage:
    python run_benchmark.py --retriever dense --dataset bamboogle --method vanilla
    python run_benchmark.py --retriever graphrag --dataset hotpotqa --method graphsearch

This script:
1. Starts the appropriate retriever server
2. Waits for it to be ready
3. Runs evaluation
4. Cleans up
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[0]
GRAPHSEARCH_DIR = REPO_ROOT / "GraphSearch"
GRAPHR1_DIR = REPO_ROOT / "GraphR1"
SEARCH_R1_DIR = REPO_ROOT / "Search-R1"

RETRIEVER_CONFIG = {
    "dense": {
        "port": 8306,
        "endpoint": "/retrieve",
        "script": REPO_ROOT / "serve_dense.py",
        "method": "dense",
        "venv": ".venv",
        "index_path": REPO_ROOT / "dense_index" / "dense_index.faiss",
        "corpus_path": REPO_ROOT / "dense_index" / "corpus.jsonl",
    },
    "graphrag": {
        "port": 8326,
        "endpoint": "/search",
        "script": GRAPHR1_DIR / "script_api_GraphRAG.py",
        "method": "graphsearch",
        "venv": ".venv",
    },
    "hipporag2": {
        "port": 8316,
        "endpoint": "/search",
        "script": GRAPHR1_DIR / "script_api_HippoRAG.py",
        "method": "graphsearch",
        "venv": "venvs/hipporag",
    },
    "hipporag": {
        "port": 8316,
        "endpoint": "/search",
        "script": GRAPHR1_DIR / "script_api_HippoRAG.py",
        "method": "graphsearch",
        "venv": "venvs/hipporag",
    },
    "raptor": {
        "port": 8346,
        "endpoint": "/search",
        "script": GRAPHR1_DIR / "script_api_RAPTOR.py",
        "method": "graphsearch",
        "venv": ".venv",
        "tree_path": REPO_ROOT / "raptor_index" / "tree.pkl",
    },
    "linearrag": {
        "port": 8356,
        "endpoint": "/search",
        "script": GRAPHR1_DIR / "script_api_LinearRAG.py",
        "method": "graphsearch",
        "venv": "venvs/linearrag",
    },
    "hypergraphrag": {
        "port": 8336,
        "endpoint": "/search",
        "script": GRAPHR1_DIR / "script_api_HypergraphRAG.py",
        "method": "graphsearch",
        "venv": ".venv",
    },
}

DATASET_PATHS = {
    "bamboogle": REPO_ROOT / "Search-o1" / "data" / "FlashRAG_datasets" / "bamboogle" / "test.jsonl",
    "hotpotqa": REPO_ROOT / "Search-o1" / "data" / "hotpotqa",
    "musique": REPO_ROOT / "Search-o1" / "data" / "musique",
    "nq": REPO_ROOT / "Search-o1" / "data" / "nq",
    "2wikimultihopqa": REPO_ROOT / "Search-o1" / "data" / "2wikimultihopqa",
    "triviaqa": REPO_ROOT / "Search-o1" / "data" / "triviaqa",
    "popqa": REPO_ROOT / "Search-o1" / "data" / "popqa",
    "aime": REPO_ROOT / "Search-o1" / "data" / "AIME" / "original_data" / "train-00000-of-00001.parquet",
    "math500": REPO_ROOT / "Search-o1" / "data" / "MATH500" / "original_data" / "test.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified benchmark launcher for RAGSearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_benchmark.py --retriever dense --data_path ./my_corpus.jsonl --method vanilla
    python run_benchmark.py --retriever hypergraphrag --data_path ./my_corpus.jsonl --limit 10
    python run_benchmark.py --retriever raptor --data_path ./my_corpus.jsonl --top_k 5

Supported retrievers: dense, graphrag, hipporag2, raptor, linearrag, hypergraphrag
Supported datasets: bamboogle, hotpotqa, musique, nq, 2wikimultihopqa, triviaqa, popqa, aime, math500
Supported methods: vanilla (no retrieval), naive, dense, graphsearch
        """
    )
    parser.add_argument(
        "--retriever", "-r",
        required=True,
        choices=list(RETRIEVER_CONFIG.keys()),
        help="Retriever backend to use"
    )
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="Dataset name (bamboogle, hotpotqa, musique, nq, etc.)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset JSONL file. If provided, overrides --dataset."
    )
    parser.add_argument(
        "--method", "-m",
        default="graphsearch",
        choices=["vanilla", "naive", "dense", "graphsearch"],
        help="Evaluation method (default: graphsearch)"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=5,
        help="Top-k for retrieval (default: 5)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=-1,
        help="Limit number of samples (-1 for all, default: -1)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Concurrency for evaluation (default: 10)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Override default retriever port"
    )
    parser.add_argument(
        "--embedding_url",
        default=os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8080/v1"),
        help="Embedding server URL (default: from EMBEDDING_BASE_URL env)"
    )
    parser.add_argument(
        "--embedding_model",
        default=os.environ.get("EMBEDDING_MODEL_NAME", "bge-m3-Q8_0"),
        help="Embedding model name (default: from EMBEDDING_MODEL_NAME env)"
    )
    return parser.parse_args()


def get_dataset_path(dataset_name: str) -> Optional[Path]:
    """Get path to dataset file/folder."""
    if dataset_name in DATASET_PATHS:
        path = DATASET_PATHS[dataset_name]
        if path.exists():
            return path
        # Try as raw path
    path = Path(dataset_name)
    if path.exists():
        return path
    return None


def check_server_ready(url: str, timeout: int = 30) -> bool:
    """Check if server is ready by polling the health endpoint."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url.replace("/search", "/status").replace("/retrieve", "/status"), timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def get_venv_python(venv_path: str) -> Path:
    """Get the Python executable for a venv."""
    venv_path = Path(venv_path)
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def start_retriever(args) -> tuple:
    """Start retriever server and return (process, port)."""
    config = RETRIEVER_CONFIG[args.retriever]
    port = args.port or config["port"]
    endpoint = config["endpoint"]
    venv_path = config.get("venv", ".venv")

    print(f"[Launcher] Starting {args.retriever} on port {port}...")
    print(f"[Launcher] Using venv: {venv_path}")

    # Get the Python executable from the venv
    python_exe = get_venv_python(venv_path)
    if not python_exe.exists():
        print(f"[Launcher] ERROR: Python not found at {python_exe}")
        print(f"[Launcher] Create venv with: python -m venv {venv_path}")
        return None, port

    # Set environment
    env = os.environ.copy()
    env["EMBEDDING_BASE_URL"] = args.embedding_url
    env["EMBEDDING_MODEL_NAME"] = args.embedding_model
    env["RETRIEVER_URL"] = f"http://127.0.0.1:{port}{endpoint}"

    if args.retriever == "dense":
        # Dense uses our serve_dense.py with pre-built FAISS index
        script = config["script"]
        index_path = config.get("index_path")
        corpus_path = config.get("corpus_path")

        if not index_path or not index_path.exists():
            print(f"[Launcher] ERROR: Dense index not found at {index_path}")
            print("[Launcher] Run: python build_dense_index.py --input_dir <dir> --output_dir dense_index")
            return None, port

        cmd = [
            str(python_exe),
            str(script),
            "--index_path", str(index_path),
            "--corpus_path", str(corpus_path),
            "--port", str(port),
            "--embedding_url", args.embedding_url,
            "--embedding_model", args.embedding_model,
        ]
    else:
        # Start retriever script (generic case)
        script = config["script"]
        if not script.exists():
            print(f"[Launcher] ERROR: Script not found: {script}")
            return None, port

        # Determine data_source from dataset
        data_source = args.dataset
        if args.dataset == "bamboogle":
            data_source = "bamboogle"

        cmd = [
            str(python_exe),
            str(script),
            "--port", str(port),
            "--embedding_url", args.embedding_url,
            "--embedding_model", args.embedding_model,
        ]

        if args.retriever in ["graphrag"]:
            project_dir = REPO_ROOT / "GraphR1" / "GraphRAG" / "inputs"
            cmd = [
                str(python_exe),
                str(script),
                "--project_dir", str(project_dir),
                "--port", str(port),
            ]
        elif args.retriever == "raptor":
            tree_path = config.get("tree_path")
            if tree_path:
                cmd.insert(2, "--tree_path")
                cmd.insert(3, str(tree_path))
            cmd.insert(4, "--node_scale")
            cmd.insert(5, "5000")
        elif args.retriever in ["hipporag2", "hipporag", "linearrag", "hypergraphrag"]:
            cmd.insert(2, "--data_source")
            cmd.insert(3, data_source)
            cmd.insert(4, "--node_scale")
            cmd.insert(5, "5000")

    print(f"[Launcher] Running: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    print(f"[Launcher] Waiting for server to be ready...")
    ready = check_server_ready(f"http://127.0.0.1:{port}{endpoint}", timeout=60)
    if not ready:
        print(f"[Launcher] WARNING: Server may not be ready. Proceeding anyway...")

    return proc, port


def run_evaluation(args, port):
    """Run the evaluation."""
    config = RETRIEVER_CONFIG[args.retriever]
    # Use --data_path if provided, otherwise use --dataset to find the path
    if args.data_path:
        dataset_path = Path(args.data_path) if Path(args.data_path).exists() else None
    else:
        dataset_path = get_dataset_path(args.dataset) if args.dataset else None

    # Set environment for eval
    env = os.environ.copy()
    env["RETRIEVER_URL"] = f"http://127.0.0.1:{port}{config['endpoint']}"
    env["EMBEDDING_BASE_URL"] = args.embedding_url
    env["EMBEDDING_MODEL_NAME"] = args.embedding_model

    eval_script = GRAPHSEARCH_DIR / "eval.py"

    cmd = [
        sys.executable,
        str(eval_script),
        "--dataset", args.dataset,
        "--graphrag", args.retriever,
        "--method", args.method,
        "--top_k", str(args.top_k),
        "--concurrency", str(args.concurrency),
    ]

    if dataset_path and dataset_path.exists():
        cmd.extend(["--data_path", str(dataset_path)])

    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    print(f"\n[Launcher] Running evaluation: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(GRAPHSEARCH_DIR), env=env)
    return result.returncode


def main():
    args = parse_args()

    print("=" * 60)
    print("RAGSearch Unified Benchmark Launcher")
    print("=" * 60)
    print(f"Retriever: {args.retriever}")
    print(f"Dataset:   {args.data_path or args.dataset or 'none'}")
    print(f"Method:    {args.method}")
    print(f"Top-K:    {args.top_k}")
    print(f"Limit:    {args.limit if args.limit > 0 else 'all'}")
    print("=" * 60)

    # Check dataset
    if args.data_path:
        dataset_path = Path(args.data_path) if Path(args.data_path).exists() else None
        if not dataset_path:
            print(f"[Launcher] ERROR: Dataset file not found: {args.data_path}")
            sys.exit(1)
        print(f"[Launcher] Dataset: {dataset_path}")
    elif args.dataset:
        dataset_path = get_dataset_path(args.dataset)
        if not dataset_path:
            print(f"[Launcher] WARNING: Dataset path not found for '{args.dataset}'")
        else:
            print(f"[Launcher] Dataset: {dataset_path}")
    else:
        dataset_path = None

    # Start retriever (skip for vanilla method)
    proc = None
    port = args.port or RETRIEVER_CONFIG[args.retriever]["port"]

    if args.method != "vanilla":
        proc, port = start_retriever(args)
        if proc is None:
            print("[Launcher] Failed to start retriever. Exiting.")
            sys.exit(1)
        print(f"[Launcher] Retriever started on port {port}")

    # Run evaluation
    print(f"\n[Launcher] Starting evaluation...")
    exit_code = run_evaluation(args, port)

    # Cleanup
    if proc:
        print(f"\n[Launcher] Cleaning up retriever (PID: {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n[Launcher] Done!")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
