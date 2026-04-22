#!/usr/bin/env python3
"""Unified benchmark launcher for RAGSearch.

Boots the requested retriever's HTTP server (using the canonical
serve_<name>.py at the repo root), waits for /status, runs the eval against
the supplied corpus, then shuts the server down.

Usage:
    python run_benchmark.py --retriever dense --corpus <path> --index-dir <dir> --port 8306

Five supported retrievers: dense, hipporag, raptor, hypergraph, linear.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from rag_clients import load_env

REPO_ROOT = Path(__file__).resolve().parent
GRAPHSEARCH_DIR = REPO_ROOT / "GraphSearch"


RETRIEVER_CONFIG = {
    "dense": {
        "default_port": 8306,
        "endpoint": "/search",
        "script": "serve_dense.py",
        "needs_name": False,
    },
    "raptor": {
        "default_port": 8346,
        "endpoint": "/search",
        "script": "serve_raptor.py",
        "needs_name": False,
    },
    "hypergraph": {
        "default_port": 8336,
        "endpoint": "/search",
        "script": "serve_hypergraph.py",
        "needs_name": False,
    },
    "hipporag": {
        "default_port": 8316,
        "endpoint": "/search",
        "script": "serve_hipporag.py",
        "needs_name": False,
    },
    "linear": {
        "default_port": 8356,
        "endpoint": "/search",
        "script": "serve_linear.py",
        "needs_name": True,
    },
    "graphrag": {
        "default_port": 8326,
        "endpoint": "/search",
        "script": "serve_graphrag.py",
        "needs_name": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--retriever", "-r", required=True, choices=list(RETRIEVER_CONFIG))
    parser.add_argument("--corpus", required=True, help="Path to JSONL evaluation corpus")
    parser.add_argument("--index-dir", required=True, help="Directory the retriever's builder wrote to")
    parser.add_argument("--name", default=None, help="Dataset subdirectory (required for linear)")
    parser.add_argument("--port", "-p", type=int, default=None)
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--limit", "-l", type=int, default=-1)
    parser.add_argument("--concurrency", "-c", type=int, default=1)
    return parser.parse_args()


def wait_for_status(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def start_server(args: argparse.Namespace, port: int) -> subprocess.Popen:
    config = RETRIEVER_CONFIG[args.retriever]
    if config["needs_name"] and not args.name:
        raise SystemExit(f"Retriever '{args.retriever}' requires --name")

    cmd = [
        sys.executable,
        str(REPO_ROOT / config["script"]),
        "--index-dir",
        args.index_dir,
        "--port",
        str(port),
    ]
    if config["needs_name"]:
        cmd.extend(["--name", args.name])

    print(f"[Launcher] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT))


def run_evaluation(args: argparse.Namespace, port: int) -> int:
    config = RETRIEVER_CONFIG[args.retriever]
    env = os.environ.copy()
    env["RETRIEVER_URL"] = f"http://127.0.0.1:{port}{config['endpoint']}"

    cmd = [
        sys.executable,
        str(GRAPHSEARCH_DIR / "eval.py"),
        "--retriever",
        args.retriever,
        "--corpus",
        str(Path(args.corpus).resolve()),
        "--top-k",
        str(args.top_k),
        "--concurrency",
        str(args.concurrency),
    ]
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    print(f"[Launcher] Eval: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(GRAPHSEARCH_DIR), env=env)


def main() -> None:
    load_env()
    args = parse_args()
    port = args.port or RETRIEVER_CONFIG[args.retriever]["default_port"]
    endpoint = RETRIEVER_CONFIG[args.retriever]["endpoint"]

    print("=" * 60)
    print(f"Retriever: {args.retriever}")
    print(f"Corpus:    {args.corpus}")
    print(f"Index dir: {args.index_dir}")
    print(f"Port:      {port}")
    print("=" * 60)

    if not Path(args.corpus).is_file():
        raise SystemExit(f"Corpus file not found: {args.corpus}")
    if not Path(args.index_dir).exists():
        raise SystemExit(f"Index directory not found: {args.index_dir}")

    proc = start_server(args, port)
    try:
        if not wait_for_status(f"http://127.0.0.1:{port}/status", timeout=120):
            print("[Launcher] WARNING: server /status did not come up in time, continuing...")
        exit_code = run_evaluation(args, port)
    finally:
        print(f"[Launcher] Stopping server (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
