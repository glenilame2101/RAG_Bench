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


def _configure_ca_bundle() -> Optional[str]:
    """Configure a company CA bundle for TLS-inspecting corporate proxies.

    Resolution order:
      1. ``$COMPANY_CA_CERT`` — explicit path override
      2. ``cert/knapp.pem`` found by walking upward from the cwd

    When a bundle is located, sets ``REQUESTS_CA_BUNDLE`` and
    ``SSL_CERT_FILE`` in ``os.environ`` (with ``setdefault`` so any
    preexisting user value wins). Both ``requests`` and ``httpx`` honor
    these, so every HTTP call in the stack picks up the bundle automatically.

    Returns the absolute path to the bundle, or ``None`` if none was found.
    """
    override = os.environ.get("COMPANY_CA_CERT", "").strip()
    candidates: List[str] = []
    if override:
        candidates.append(override)

    current = os.getcwd()
    while True:
        candidates.append(os.path.join(current, "cert", "knapp.pem"))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            abs_path = os.path.abspath(candidate)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", abs_path)
            os.environ.setdefault("SSL_CERT_FILE", abs_path)
            print(f"[ca-bundle] Using company CA bundle: {abs_path}")
            return abs_path
    return None


def load_env(start: Optional[Path] = None) -> Optional[Path]:
    """Load .env into os.environ. Returns the path actually loaded, or None.

    Search order:
      1. python-dotenv's find_dotenv (if installed) starting from cwd
      2. Walk upward from `start` (defaults to this file's directory) looking
         for a `.env` sibling.

    Existing environment variables win over .env values.

    After loading, also probes for a company CA bundle at ``cert/knapp.pem``
    (or ``$COMPANY_CA_CERT``) and wires it into the HTTP client stack if
    present.
    """
    loaded: Optional[Path] = None
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore

        discovered = find_dotenv(usecwd=True)
        if discovered:
            load_dotenv(discovered, override=False)
            loaded = Path(discovered)
    except ImportError:
        pass

    if loaded is None:
        base = (start or Path(__file__).resolve().parent).resolve()
        for parent in (base, *base.parents):
            candidate = parent / ".env"
            if candidate.is_file():
                _parse_env_file(candidate)
                loaded = candidate
                break

    _configure_ca_bundle()
    return loaded


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
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.base_url = _normalize_base_url(_resolve(base_url, "EMBEDDING_BASE_URL"))
        self.model = _resolve(model, "EMBEDDING_MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.base_url:
            raise ValueError("EMBEDDING_BASE_URL is not set (and no override given)")
        if not self.model:
            raise ValueError("EMBEDDING_MODEL is not set (and no override given)")

    def _post(self, payload: dict) -> dict:
        # Retries cover transient network failures (read timeout, connection
        # reset) — NOT 5xx responses, which on llama.cpp typically mean a
        # permanent condition like oversized input that retry won't fix.
        from tenacity import (
            Retrying,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type(
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                )
            ),
        )
        for attempt in retrying:
            with attempt:
                resp = requests.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
        raise RuntimeError("unreachable")  # retrying loop always returns or raises

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
        max_chars: Optional[int] = None,
        **_unused,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if max_chars and max_chars > 0:
            n_truncated = sum(1 for t in texts if len(t) > max_chars)
            if n_truncated > 0:
                print(f"[Embed] Truncating {n_truncated}/{len(texts)} texts to {max_chars} chars")
            texts = [t[:max_chars] for t in texts]

        out: List[np.ndarray] = []
        bar = tqdm(total=len(texts), desc="[Embed] Embedding", unit="chunk")
        try:
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
                bar.update(len(batch))
        finally:
            bar.close()
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def __call__(self, text):
        if isinstance(text, str):
            return self.encode([text])[0]
        return self.encode(text)

    def _fingerprint(
        self,
        texts: Sequence[str],
        prefix_len: Optional[int] = None,
        normalize: bool = True,
        max_chars: Optional[int] = None,
    ) -> str:
        """Stable hash over the first `prefix_len` texts (or all, if None).

        Covers (model, normalize flag, max_chars truncation setting, prefix
        length, first/middle/last text of the prefix). Used to validate
        resumable checkpoints: two runs that share the same prefix AND
        the same truncation setting produce the same hash and may resume.
        """
        n = len(texts) if prefix_len is None else prefix_len
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(b"norm=1" if normalize else b"norm=0")
        h.update(f"max_chars={max_chars or 0}".encode("utf-8"))
        h.update(str(n).encode("utf-8"))
        if n > 0:
            h.update(texts[0][:1000].encode("utf-8", errors="replace"))
            h.update(texts[n - 1][:1000].encode("utf-8", errors="replace"))
            h.update(texts[n // 2][:1000].encode("utf-8", errors="replace"))
        return h.hexdigest()

    def encode_with_checkpoint(
        self,
        texts: Sequence[str],
        checkpoint_dir: Union[str, Path],
        batch_size: int = 32,
        normalize: bool = True,
        save_every_pct: float = 1.0,
        max_chars: Optional[int] = None,
    ) -> np.ndarray:
        """Embed `texts`, saving partial progress to `checkpoint_dir` every
        `save_every_pct` percent of total work.

        The checkpoint is a persistent prefix cache: if a previous run embedded
        the first K texts and the current run starts with those same K texts
        (prefix-fingerprint match), resume from K instead of recomputing.
        If the new run requests fewer texts than are saved, returns the
        first n_total from the saved embeddings.

        Layout under `checkpoint_dir`:
            state.json       — {model, completed, prefix_fingerprint, ...}
            emb_000000.npy   — first chunk of saved embeddings
            emb_000001.npy   — ...
            (concatenated in order to produce the final array)
        """
        texts = list(texts)
        n_total = len(texts)
        if n_total == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if max_chars and max_chars > 0:
            n_truncated = sum(1 for t in texts if len(t) > max_chars)
            if n_truncated > 0:
                print(f"[Embed] Truncating {n_truncated}/{n_total} texts to {max_chars} chars")
            texts = [t[:max_chars] for t in texts]

        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state_path = ckpt_dir / "state.json"

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
            completed = int(state.get("completed", 0))
            for f in sorted(ckpt_dir.glob("emb_*.npy")):
                loaded_chunks.append(np.load(f))
            loaded_count = sum(c.shape[0] for c in loaded_chunks)
            if loaded_count != completed:
                raise SystemExit(
                    f"Checkpoint inconsistent: state.json says {completed} done but "
                    f"found {loaded_count} embeddings on disk in {ckpt_dir}."
                )
            if completed > 0:
                if n_total < completed:
                    raise SystemExit(
                        f"Checkpoint at {ckpt_dir} already holds {completed} embeddings "
                        f"but this run only needs {n_total}. Refusing to truncate — the "
                        f"stored fingerprint covers {completed} items and can't verify "
                        f"a shorter prefix. Use --no-checkpoint, delete {ckpt_dir}, or "
                        f"point at a different checkpoint dir."
                    )
                expected_fp = self._fingerprint(
                    texts, prefix_len=completed, normalize=normalize, max_chars=max_chars
                )
                saved_fp = state.get("prefix_fingerprint")
                if saved_fp != expected_fp:
                    raise SystemExit(
                        f"Checkpoint at {ckpt_dir} was built from a different input "
                        f"prefix or different settings (model / normalize / max_chars / "
                        f"text mismatch over first {completed} items). Delete the "
                        f"directory or use a different checkpoint dir."
                    )
            if completed == n_total:
                print(f"[Embed] Checkpoint covers request ({completed}/{n_total})")
                return np.vstack(loaded_chunks)
            print(f"[Embed] Resuming from checkpoint: {completed}/{n_total} done")

        save_every = max(batch_size, int(n_total * save_every_pct / 100))
        next_chunk_index = len(loaded_chunks)
        new_chunks: List[np.ndarray] = []
        pending: List[np.ndarray] = []
        pending_count = 0

        def _write_state() -> None:
            state_path.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "completed": completed,
                        "batch_size": batch_size,
                        "normalize": normalize,
                        "max_chars": max_chars or 0,
                        "prefix_fingerprint": self._fingerprint(
                            texts, prefix_len=completed, normalize=normalize,
                            max_chars=max_chars,
                        ),
                    }
                ),
                encoding="utf-8",
            )

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
                    _write_state()
                    pending = []
                    pending_count = 0

            if pending:
                arr = np.vstack(pending)
                np.save(ckpt_dir / f"emb_{next_chunk_index:06d}.npy", arr)
                new_chunks.append(arr)
                completed += pending_count
                _write_state()
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

        # Honor a company CA bundle if one was configured by load_env().
        ca_bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
        http_client = None
        if ca_bundle and os.path.isfile(ca_bundle):
            import httpx
            http_client = httpx.Client(verify=ca_bundle, timeout=timeout)

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            http_client=http_client,
        )

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
