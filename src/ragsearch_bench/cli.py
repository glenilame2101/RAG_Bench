"""``ragbench`` -- single entry-point CLI for the RAGSearch benchmark.

This is a thin wrapper around the existing ``Search-o1/run_search_o1*.sh`` and
``GraphSearch/run_universal.sh`` shell scripts. It does not re-implement
their logic; it just gives users one command to remember and one place to
pass common flags (retriever, dataset, port, model).

Examples::

    ragbench run search-o1  --retriever hipporag2 --dataset HotpotQA --port 8205
    ragbench run graphsearch --retriever hipporag2 --dataset hotpotqa --port 8205
    ragbench eval            --predictions out.json --dataset hotpotqa
    ragbench env-check

The CLI defaults to the **remote OpenAI-compatible endpoint** defined by the
repo-root ``.env`` file (``URL`` / ``MODEL_NAME`` / ``OPENAI_API_KEY``). Pass
``--local-llm`` to run a local vLLM server instead; this requires the ``train``
extra (``pip install -e '.[train]'``) and a GPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """Load repo-root .env (if python-dotenv is available)."""
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        return
    candidate = REPO_ROOT / ".env"
    if candidate.is_file():
        load_dotenv(candidate, override=False)
        return
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=False)


def _run(cmd: List[str], cwd: Optional[Path] = None, env=None) -> int:
    print(f"[ragbench] $ {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    try:
        if cwd and sys.platform == "win32":
            import os
            os.chdir(str(cwd))
            cwd = None
        return subprocess.call(cmd, cwd=str(cwd) if cwd else None, env=env)
    except FileNotFoundError as exc:
        print(f"[ragbench] error: {exc}", file=sys.stderr)
        return 127


# ---------------------------------------------------------------------------
# subcommand: env-check
# ---------------------------------------------------------------------------
def cmd_env_check(_: argparse.Namespace) -> int:
    _load_env()
    url = os.environ.get("URL", "")
    model = os.environ.get("MODEL_NAME", "")
    key = os.environ.get("OPENAI_API_KEY", "")
    embed_url = os.environ.get("EMBEDDING_BASE_URL", "")
    embed_model = os.environ.get("EMBEDDING_MODEL_NAME", "")
    print(f"Repo root:            {REPO_ROOT}")
    print(f"URL:                  {url or '<unset>'}")
    print(f"MODEL_NAME:           {model or '<unset>'}")
    print(f"OPENAI_API_KEY:       {'set (%d chars)' % len(key) if key else '<unset>'}")
    print(f"EMBEDDING_BASE_URL:   {embed_url or '<unset>'}")
    print(f"EMBEDDING_MODEL_NAME: {embed_model or '<unset>'}")
    try:
        import vllm  # noqa: F401
        print("vllm:                installed (local backend available)")
    except ImportError:
        print("vllm:                not installed (OpenAI backend only -- this is fine)")
    if not url:
        print("\n[ragbench] No URL configured. Copy .env.example to .env and fill it in,")
        print("           or pass --local-llm (requires the 'train' extra).")
        return 1
    return 0


# ---------------------------------------------------------------------------
# subcommand: run <agent>
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    _load_env()
    if args.embedding_url:
        os.environ["EMBEDDING_BASE_URL"] = args.embedding_url
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL_NAME"] = args.embedding_model

    import subprocess
    env = os.environ.copy()

    agent = args.agent
    if agent == "search-o1":
        script = REPO_ROOT / "Search-o1" / "run_search_o1_graphrag.sh"
        cmd = [
            "bash", str(script),
            "-g", args.retriever,
            "-d", args.dataset,
            "-m", args.model_path,
            "-p", str(args.port),
        ]
        if args.split:
            cmd += ["--split", args.split]
        if args.tokenizer_path:
            cmd += ["-t", args.tokenizer_path]
        if args.local_llm:
            cmd += ["-b", "vllm"]
        else:
            cmd += ["-b", "openai"]
        return _run(cmd, cwd=REPO_ROOT / "Search-o1", env=env)

    if agent == "graphsearch":
        script = REPO_ROOT / "GraphSearch" / "eval.py"
        retriever_url = f"http://127.0.0.1:{args.port}/search"
        cmd = [
            sys.executable, str(script),
            "--dataset", args.dataset,
            "--graphrag", args.retriever,
            "--method", args.eval_method,
            "--top_k", str(args.top_k),
            "--concurrency", str(args.concurrency),
        ]
        if args.start is not None:
            cmd += ["--start", str(args.start)]
        if args.end is not None:
            cmd += ["--end", str(args.end)]
        env["RETRIEVER_URL"] = retriever_url
        return _run(cmd, cwd=REPO_ROOT / "GraphSearch", env=env)

    print(f"[ragbench] unknown agent: {agent}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# subcommand: eval
# ---------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> int:
    _load_env()
    # GraphSearch ships the general-purpose evaluator (eval.py). Delegate.
    eval_py = REPO_ROOT / "GraphSearch" / "eval.py"
    if not eval_py.is_file():
        print(f"[ragbench] eval.py not found at {eval_py}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(eval_py),
           "--dataset", args.dataset,
           "--method", args.method,
           "--top_k", str(args.top_k)]
    if args.predictions:
        cmd += ["--predictions", args.predictions]
    if args.graphrag:
        cmd += ["--graphrag", args.graphrag]
    return _run(cmd, cwd=REPO_ROOT / "GraphSearch")


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragbench",
        description="Unified CLI for the RAGSearch benchmark (single venv, "
                    "remote LLM endpoint by default).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env-check", help="Print resolved .env config and exit.").set_defaults(
        func=cmd_env_check
    )

    # run
    run_p = sub.add_parser("run", help="Run a benchmark agent end-to-end.")
    run_sub = run_p.add_subparsers(dest="agent", required=True)

    for agent_name in ("search-o1", "graphsearch"):
        a = run_sub.add_parser(agent_name)
        a.add_argument("--retriever", required=True,
                       help="Retriever backend (hipporag2, graphrag, linearrag, "
                            "raptor, hypergraphrag, dense).")
        a.add_argument("--dataset", required=True,
                       help="Dataset name (e.g. HotpotQA, hotpotqa, NQ, PopQA).")
        a.add_argument("--port", type=int, default=8205,
                       help="Retriever server port (default: 8205).")
        a.add_argument("--local-llm", action="store_true",
                       help="Launch a local vLLM server instead of using the "
                            "remote endpoint (requires the 'train' extra + GPU).")
        a.add_argument("--embedding-url", dest="embedding_url", default=None,
                       help="Embedding server base URL (e.g. http://127.0.0.1:8080/v1).")
        a.add_argument("--embedding-model", dest="embedding_model", default=None,
                       help="Embedding model name served by llama.cpp (e.g. bge-m3-Q8_0).")
        if agent_name == "search-o1":
            a.add_argument("--model-path", dest="model_path",
                           default="Qwen/Qwen2.5-7B-Instruct")
            a.add_argument("--split", default=None)
            a.add_argument("--tokenizer-path", dest="tokenizer_path", default=None)
        else:  # graphsearch
            a.add_argument("--top-k", dest="top_k", type=int, default=5)
            a.add_argument("--concurrency", type=int, default=1)
            a.add_argument("--eval-method", dest="eval_method",
                           default="graphsearch",
                           choices=["graphsearch", "naive", "dense"])
            a.add_argument("--start", type=int, default=None)
            a.add_argument("--end", type=int, default=None)
        a.set_defaults(func=cmd_run)

    # eval
    ev = sub.add_parser("eval", help="Score a predictions file against a dataset.")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--predictions", default=None,
                    help="Path to predictions JSON (defaults to GraphSearch's convention).")
    ev.add_argument("--method", default="graphsearch",
                    choices=["graphsearch", "naive", "dense"])
    ev.add_argument("--graphrag", default=None,
                    help="Retriever name used to produce predictions (optional).")
    ev.add_argument("--top-k", dest="top_k", type=int, default=5)
    ev.set_defaults(func=cmd_eval)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
