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
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests
from tqdm import tqdm


class EmbeddingCache:
    """Persistent, content-addressed cache for embedding vectors.

    Keys are `sha256(model || normalize-flag || text)` so that:

      * Re-running a build reuses previously-computed embeddings, even if the
        input set changes (e.g. widening `--partial-index 0.1` -> `0.2`).
      * Different models / normalization settings live side by side without
        collisions.

    Backed by SQLite (WAL mode) so the cache is a single portable file and is
    safe to open from multiple processes. All values are stored as raw
    float32 blobs.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key        TEXT PRIMARY KEY,
                model      TEXT NOT NULL,
                dim        INTEGER NOT NULL,
                normalized INTEGER NOT NULL,
                vec        BLOB NOT NULL
            )
            """
        )
        self._conn.commit()
        self._lock = threading.Lock()

    @staticmethod
    def make_key(model: str, text: str, normalized: bool) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(b"N" if normalized else b"R")
        h.update(b"\x00")
        h.update(text.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict:
        """Return {key: np.ndarray} for all keys that are present."""
        if not keys:
            return {}
        out: dict = {}
        with self._lock:
            cur = self._conn.cursor()
            # SQLite parameter limit is ~999; chunk conservatively.
            for i in range(0, len(keys), 500):
                chunk = list(keys[i : i + 500])
                placeholders = ",".join("?" * len(chunk))
                rows = cur.execute(
                    f"SELECT key, dim, vec FROM embeddings WHERE key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for key, dim, blob in rows:
                    out[key] = np.frombuffer(blob, dtype=np.float32).reshape(int(dim))
        return out

    def put_many(self, items: Iterable[Tuple[str, str, int, bool, np.ndarray]]) -> None:
        """Upsert a batch of (key, model, dim, normalized, vec) rows."""
        rows = [
            (k, m, int(d), 1 if n else 0, np.asarray(v, dtype=np.float32).tobytes())
            for (k, m, d, n, v) in items
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, model, dim, normalized, vec) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


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
        cache: Optional[EmbeddingCache] = None,
    ):
        self.base_url = _normalize_base_url(_resolve(base_url, "EMBEDDING_BASE_URL"))
        self.model = _resolve(model, "EMBEDDING_MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.cache = cache

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

        # If a cache is wired in at construction time, transparently dedupe
        # against it so repeat calls (e.g. from LinearRAG) are free.
        if self.cache is not None:
            return self.encode_with_cache(
                texts,
                cache=self.cache,
                batch_size=batch_size,
                normalize=normalize,
            )

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

    def encode_with_cache(
        self,
        texts: Sequence[str],
        cache: EmbeddingCache,
        batch_size: int = 32,
        normalize: bool = True,
        save_every: Optional[int] = None,
    ) -> np.ndarray:
        """Embed `texts`, consulting `cache` first and only calling the
        embeddings API for keys that aren't present.

        New vectors are persisted in chunks of `save_every` so an interrupted
        run can resume with no lost work. Unlike the previous chunk-file
        checkpoint scheme, the cache is content-addressed: changing the input
        order or widening `--partial-index` still produces cache hits.
        """
        texts = list(texts)
        n_total = len(texts)
        if n_total == 0:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [EmbeddingCache.make_key(self.model, t, normalize) for t in texts]
        cached = cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in cached]

        if not missing:
            print(f"[Embed] Cache hit: {n_total}/{n_total} (no API calls needed)")
            return np.vstack([cached[k] for k in keys])

        print(
            f"[Embed] Cache hit: {n_total - len(missing)}/{n_total}; "
            f"embedding {len(missing)} new texts via {self.base_url} ({self.model})"
        )

        if save_every is None:
            # Flush roughly every 1% of the work, but never less often than
            # one batch (so interrupts lose at most one batch of progress).
            save_every = max(batch_size, n_total // 100)

        pending: List[Tuple[str, str, int, bool, np.ndarray]] = []
        bar = tqdm(
            total=len(missing),
            desc="[Embed] Embedding",
            unit="chunk",
        )
        try:
            for start in range(0, len(missing), batch_size):
                idx_batch = missing[start : start + batch_size]
                batch_texts = [texts[i] for i in idx_batch]
                data = self._post({"model": self.model, "input": batch_texts})
                items = data.get("data") or []
                if len(items) != len(idx_batch):
                    raise RuntimeError(
                        f"Embeddings endpoint returned {len(items)} vectors for "
                        f"{len(idx_batch)} inputs"
                    )
                for j, item in enumerate(items):
                    vec = np.asarray(item["embedding"], dtype=np.float32)
                    if normalize:
                        norm = float(np.linalg.norm(vec))
                        if norm > 0.0:
                            vec = vec / norm
                    i_orig = idx_batch[j]
                    cached[keys[i_orig]] = vec
                    pending.append(
                        (keys[i_orig], self.model, int(vec.shape[0]), normalize, vec)
                    )
                bar.update(len(idx_batch))

                if len(pending) >= save_every:
                    cache.put_many(pending)
                    pending = []
            if pending:
                cache.put_many(pending)
        finally:
            bar.close()

        return np.vstack([cached[k] for k in keys])

    def encode_with_checkpoint(
        self,
        texts: Sequence[str],
        checkpoint_dir: Union[str, Path],
        batch_size: int = 32,
        normalize: bool = True,
        save_every_pct: float = 1.0,
    ) -> np.ndarray:
        """Embed `texts`, persisting progress to an on-disk cache so repeat /
        interrupted runs reuse already-computed embeddings.

        The cache file is `<checkpoint_dir>/embeddings.sqlite`. It is
        content-addressed by (model, normalize-flag, text), so re-runs with a
        different `--partial-index`, different document ordering, or an
        overlapping corpus all produce cache hits.
        """
        texts = list(texts)
        n_total = len(texts)
        if n_total == 0:
            return np.zeros((0, 0), dtype=np.float32)

        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_every = max(batch_size, int(n_total * save_every_pct / 100))
        with EmbeddingCache(ckpt_dir / "embeddings.sqlite") as cache:
            return self.encode_with_cache(
                texts,
                cache=cache,
                batch_size=batch_size,
                normalize=normalize,
                save_every=save_every,
            )


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
    "EmbeddingCache",
    "EmbeddingClient",
    "LLMClient",
    "RerankerClient",
    "load_env",
]
