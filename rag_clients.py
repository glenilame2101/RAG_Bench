"""Shared HTTP clients for the RAGSearch refactored, OpenAI-compatible-only stack.

Every builder, server, and agent script imports from this module so that:

  * the canonical .env contract is read in exactly one place
  * the EmbeddingClient / RerankerClient / LLMClient classes are defined once
  * URL normalization (trailing /v1) is consistent

The module has no hard dependency on python-dotenv. If dotenv is installed it
is used; otherwise a minimal parser walks the directory tree looking for a
.env file.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests
from tqdm import tqdm


def _parse_env_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_env(start: Optional[Path] = None) -> Optional[Path]:
    """Load .env into os.environ. Returns the path actually loaded, or None.

    Search order:
      1. python-dotenv's find_dotenv (if installed) starting from cwd
      2. Walk upward from `start` (defaults to this file's directory) looking
         for a `.env` sibling.

    Existing environment variables win over .env values.
    """
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore

        discovered = find_dotenv(usecwd=True)
        if discovered:
            load_dotenv(discovered, override=False)
            return Path(discovered)
    except ImportError:
        pass

    base = (start or Path(__file__).resolve().parent).resolve()
    for parent in (base, *base.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            _parse_env_file(candidate)
            return candidate
    return None


def _normalize_base_url(url: str) -> str:
    url = (url or "").rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return url + "/v1"


def _resolve(value: Optional[str], env_key: str) -> str:
    if value:
        return value
    return os.getenv(env_key, "") or ""


class EmbeddingClient:
    """Client for an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = _normalize_base_url(_resolve(base_url, "EMBEDDING_BASE_URL"))
        self.model = _resolve(model, "EMBEDDING_MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout

        if not self.base_url:
            raise ValueError("EMBEDDING_BASE_URL is not set (and no override given)")
        if not self.model:
            raise ValueError("EMBEDDING_MODEL is not set (and no override given)")

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
        **_unused,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out: List[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + batch_size]
            data = self._post({"model": self.model, "input": batch})
            for item in data["data"]:
                vec = np.asarray(item["embedding"], dtype=np.float32)
                if normalize:
                    norm = float(np.linalg.norm(vec))
                    if norm > 0.0:
                        vec = vec / norm
                out.append(vec)
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def __call__(self, text):
        if isinstance(text, str):
            return self.encode([text])[0]
        return self.encode(text)

    def _fingerprint(self, texts: Sequence[str]) -> str:
        """Stable hash of the input set; used to refuse mismatched resumes."""
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(str(len(texts)).encode("utf-8"))
        if texts:
            h.update(texts[0][:1000].encode("utf-8", errors="replace"))
            h.update(texts[-1][:1000].encode("utf-8", errors="replace"))
            mid = texts[len(texts) // 2]
            h.update(mid[:1000].encode("utf-8", errors="replace"))
        return h.hexdigest()

    def encode_with_checkpoint(
        self,
        texts: Sequence[str],
        checkpoint_dir: Union[str, Path],
        batch_size: int = 32,
        normalize: bool = True,
        save_every_pct: float = 1.0,
    ) -> np.ndarray:
        """Embed `texts`, saving partial progress to `checkpoint_dir` every
        `save_every_pct` percent of total work.

        On re-invocation with the same input set, picks up where it left off.
        Refuses to resume if model name or input fingerprint differ.

        Layout under `checkpoint_dir`:
            state.json       — {model, total, completed, fingerprint, ...}
            emb_000000.npy   — first chunk of saved embeddings
            emb_000001.npy   — ...
            (concatenated in order to produce the final array)
        """
        texts = list(texts)
        n_total = len(texts)
        if n_total == 0:
            return np.zeros((0, 0), dtype=np.float32)

        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_path = ckpt_dir / "state.json"

        fingerprint = self._fingerprint(texts)

        completed = 0
        loaded_chunks: List[np.ndarray] = []
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Checkpoint state at {state_path} is corrupt ({exc}). "
                    f"Delete the directory to start over."
                )
            if state.get("fingerprint") != fingerprint:
                raise SystemExit(
                    f"Checkpoint at {ckpt_dir} was created from a different input set "
                    f"(model/count/text fingerprint mismatch). Delete the directory or "
                    f"use a different --output-dir."
                )
            completed = int(state.get("completed", 0))
            for f in sorted(ckpt_dir.glob("emb_*.npy")):
                loaded_chunks.append(np.load(f))
            loaded_count = sum(c.shape[0] for c in loaded_chunks)
            if loaded_count != completed:
                raise SystemExit(
                    f"Checkpoint inconsistent: state.json says {completed} done but "
                    f"found {loaded_count} embeddings on disk in {ckpt_dir}."
                )
            if completed >= n_total:
                print(f"[Embed] Checkpoint already complete ({completed}/{n_total})")
                return np.vstack(loaded_chunks)
            print(f"[Embed] Resuming from checkpoint: {completed}/{n_total} done")

        save_every = max(batch_size, int(n_total * save_every_pct / 100))
        next_chunk_index = len(loaded_chunks)
        new_chunks: List[np.ndarray] = []
        pending: List[np.ndarray] = []
        pending_count = 0

        bar = tqdm(
            total=n_total,
            initial=completed,
            desc="[Embed] Embedding",
            unit="chunk",
        )
        try:
            for start in range(completed, n_total, batch_size):
                batch = texts[start : start + batch_size]
                data = self._post({"model": self.model, "input": batch})
                for item in data["data"]:
                    vec = np.asarray(item["embedding"], dtype=np.float32)
                    if normalize:
                        norm = float(np.linalg.norm(vec))
                        if norm > 0.0:
                            vec = vec / norm
                    pending.append(vec)
                pending_count += len(batch)
                bar.update(len(batch))

                if pending_count >= save_every:
                    arr = np.vstack(pending)
                    np.save(ckpt_dir / f"emb_{next_chunk_index:06d}.npy", arr)
                    new_chunks.append(arr)
                    next_chunk_index += 1
                    completed += pending_count
                    state_path.write_text(
                        json.dumps(
                            {
                                "model": self.model,
                                "total": n_total,
                                "completed": completed,
                                "batch_size": batch_size,
                                "normalize": normalize,
                                "fingerprint": fingerprint,
                            }
                        ),
                        encoding="utf-8",
                    )
                    pending = []
                    pending_count = 0

            if pending:
                arr = np.vstack(pending)
                np.save(ckpt_dir / f"emb_{next_chunk_index:06d}.npy", arr)
                new_chunks.append(arr)
                completed += pending_count
                state_path.write_text(
                    json.dumps(
                        {
                            "model": self.model,
                            "total": n_total,
                            "completed": completed,
                            "batch_size": batch_size,
                            "normalize": normalize,
                            "fingerprint": fingerprint,
                        }
                    ),
                    encoding="utf-8",
                )
        finally:
            bar.close()

        all_chunks = loaded_chunks + new_chunks
        return np.vstack(all_chunks)


class RerankerClient:
    """Client for an OpenAI-compatible /v1/rerank endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = _normalize_base_url(_resolve(base_url, "RERANKER_BASE_URL"))
        self.model = _resolve(model, "RERANKER_MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout

        if not self.base_url:
            raise ValueError("RERANKER_BASE_URL is not set (and no override given)")
        if not self.model:
            raise ValueError("RERANKER_MODEL is not set (and no override given)")

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
        }
        if top_n is not None:
            payload["top_n"] = int(top_n)
        resp = requests.post(
            f"{self.base_url}/rerank",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [(int(item["index"]), float(item["relevance_score"])) for item in data.get("results", [])]
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results


class LLMClient:
    """Thin wrapper over the OpenAI SDK pointed at OPENAI_BASE_URL."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        from openai import OpenAI  # local import keeps module light

        self.base_url = _normalize_base_url(_resolve(base_url, "OPENAI_BASE_URL"))
        self.api_key = _resolve(api_key, "OPENAI_API_KEY") or "EMPTY"
        self.model = _resolve(model, "OPENAI_MODEL")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is not set (and no override given)")
        if not self.model:
            raise ValueError("OPENAI_MODEL is not set (and no override given)")
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)

    def chat(self, messages: Iterable[dict], **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=list(messages),
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    @property
    def raw(self):
        """Underlying openai.OpenAI client for callers that need it."""
        return self._client


__all__ = [
    "EmbeddingClient",
    "LLMClient",
    "RerankerClient",
    "load_env",
]
