"""OpenAI-compatible backend for Search-o1 client scripts.

This module exposes an ``OpenAILLM`` class whose ``generate`` method matches
the shape the run scripts originally expected from a local generation engine:
prompts in, list of objects with ``.outputs[0].text`` out. After the
OpenAI-only refactor the only supported backend is the OpenAI HTTP API
(via OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL).

Usage::

    from openai_llm import build_llm, add_backend_args, load_env_file

    load_env_file()  # optional, picks up repo-root .env
    add_backend_args(parser)  # adds --backend / --base_url / --api_key / ...
    args = parser.parse_args()
    llm = build_llm(args, model=model_path)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
def load_env_file(path: Optional[str] = None) -> None:
    """Load ``.env`` from *path* or walk upward from the cwd to find one.

    Silently no-ops if ``python-dotenv`` is not installed or no file is found.
    The three variables this project cares about are ``URL``, ``MODEL_NAME``
    and ``OPENAI_API_KEY``.
    """
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        return
    if path and os.path.isfile(path):
        load_dotenv(path, override=False)
        return
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=False)


def _normalize_base_url(url: str) -> str:
    """Ensure the URL ends with ``/v1`` (required by the OpenAI SDK)."""
    url = url.rstrip("/")
    if not url:
        return url
    if url.endswith("/v1") or url.endswith("/v1/"):
        return url.rstrip("/")
    return url + "/v1"


# ---------------------------------------------------------------------------
# SamplingParams shim
# ---------------------------------------------------------------------------
# The Search-o1 scripts instantiate ``SamplingParams(...)`` directly. We
# expose a duck-typed dataclass with the same attributes so the scripts can
# keep using ``SamplingParams(max_tokens=..., temperature=..., ...)`` without
# pulling in any local-inference framework.
@dataclass
class SamplingParams:  # type: ignore[no-redef]
    """Duck-typed sampling-params dataclass for the OpenAI-only client."""

    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    include_stop_str_in_output: bool = False
    n: int = 1
    seed: Optional[int] = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True


# ---------------------------------------------------------------------------
# Fake RequestOutput shapes (the run scripts expect .outputs[0].text)
# ---------------------------------------------------------------------------
@dataclass
class _FakeCompletionOutput:
    text: str
    token_ids: List[int]
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class _FakeRequestOutput:
    prompt: str
    outputs: List[_FakeCompletionOutput]


# ---------------------------------------------------------------------------
# OpenAI-backed LLM
# ---------------------------------------------------------------------------
class OpenAILLM:
    """Client that hits an OpenAI-compatible chat-completions API.

    Only the ``generate(prompts, sampling_params)`` method is implemented,
    since that is all the Search-o1 scripts use.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        max_workers: int = 16,
        timeout: float = 600.0,
        **_ignored: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ImportError(
                "The 'openai' package is required for the OpenAI backend. "
                "Install it with `pip install openai`."
            ) from exc

        self.model_name = model_name
        self.base_url = _normalize_base_url(base_url)
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
        )
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prompt_to_messages(prompt: Any) -> List[dict]:
        """Convert a prompt (string or already-structured messages) to OpenAI
        chat-completions ``messages`` format.

        The Search-o1 scripts pre-format prompts via the local tokenizer
        stub, then hand the result here. The OpenAI-compatible server
        re-applies its chat template, so we wrap the formatted text as a
        single user message — the outer template contribution is tiny and
        the model has seen the inner format during pre-training.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return prompt  # already messages
        raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")

    @staticmethod
    def _sampling_to_kwargs(sp: Any) -> dict:
        """Translate a ``SamplingParams``-shaped object into chat.completions kwargs.

        ``top_k`` and ``repetition_penalty`` are not part of the standard
        OpenAI API but are accepted by some OpenAI-compatible servers via
        ``extra_body``; we forward them when set.
        """
        kwargs: dict = {}
        extra_body: dict = {}

        def _get(name, default=None):
            return getattr(sp, name, default) if sp is not None else default

        max_tokens = _get("max_tokens")
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)

        temperature = _get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        top_p = _get("top_p")
        if top_p is not None:
            kwargs["top_p"] = float(top_p)

        stop = _get("stop")
        if stop is not None:
            # Normalize to list and remove empty strings (OpenAI rejects empty stop strings)
            if isinstance(stop, str):
                stop_list = [stop]
            else:
                stop_list = list(stop)
            stop_list = [s for s in stop_list if s]
            if stop_list:
                kwargs["stop"] = stop_list

        presence_penalty = _get("presence_penalty")
        if presence_penalty:
            kwargs["presence_penalty"] = float(presence_penalty)

        frequency_penalty = _get("frequency_penalty")
        if frequency_penalty:
            kwargs["frequency_penalty"] = float(frequency_penalty)

        # vLLM-specific extensions
        top_k = _get("top_k")
        if top_k is not None and top_k != -1:
            extra_body["top_k"] = int(top_k)

        repetition_penalty = _get("repetition_penalty")
        if repetition_penalty is not None and repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = float(repetition_penalty)

        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    def _single_generate(
        self,
        prompt: Any,
        sampling_params: Any,
    ) -> _FakeRequestOutput:
        messages = self._prompt_to_messages(prompt)
        kwargs = self._sampling_to_kwargs(sampling_params)

        include_stop = bool(getattr(sampling_params, "include_stop_str_in_output", False))

        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            stop_reason = getattr(choice, "stop_reason", None)

            # vLLM's include_stop_str_in_output re-appends the matched stop
            # string to the generated text. Mirror that here so the caller's
            # regex / string-matching logic keeps working.
            if include_stop and finish_reason == "stop":
                stop_list = kwargs.get("stop") or []
                if isinstance(stop_reason, str) and stop_reason:
                    text += stop_reason
                elif stop_list:
                    # Best-effort: if exactly one stop was configured, use it.
                    if len(stop_list) == 1:
                        text += stop_list[0]
        except Exception as exc:  # pragma: no cover - network failure path
            prompt_preview = prompt if isinstance(prompt, str) else str(prompt)
            prompt_preview = prompt_preview[:80].replace("\n", " ")
            print(
                f"[OpenAILLM] request failed (prompt='{prompt_preview}...'): {exc}"
            )
            text = ""
            finish_reason = "error"
            stop_reason = None

        return _FakeRequestOutput(
            prompt=prompt if isinstance(prompt, str) else "",
            outputs=[
                _FakeCompletionOutput(
                    text=text,
                    token_ids=[],
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )
            ],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Union[str, Sequence[Any]],
        sampling_params: Any = None,
        **_ignored: Any,
    ) -> List[_FakeRequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if not prompts:
            return []

        workers = max(1, min(self._max_workers, len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(lambda p: self._single_generate(p, sampling_params), prompts)
            )
        return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_backend_args(parser) -> None:
    """Attach --backend / remote-endpoint flags to an argparse parser.

    Defaults are read from environment variables (``URL``, ``MODEL_NAME``,
    ``OPENAI_API_KEY``) so that ``.env`` values are picked up automatically.
    """
    default_url = os.environ.get("OPENAI_BASE_URL") or ""
    default_model = os.environ.get("OPENAI_MODEL") or ""
    default_key = os.environ.get("OPENAI_API_KEY") or ""
    # The OpenAI-compatible endpoint is the only supported backend.
    default_backend = "openai"

    parser.add_argument(
        "--backend",
        choices=["openai"],
        default="openai",
        help="LLM backend (only 'openai' remains after the OpenAI-only refactor).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=default_url,
        help="Base URL of the OpenAI-compatible endpoint (reads $OPENAI_BASE_URL by default).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=default_key,
        help="API key for the OpenAI-compatible endpoint (reads "
             "$OPENAI_API_KEY by default).",
    )
    parser.add_argument(
        "--remote_model_name",
        type=str,
        default=default_model,
        help="Model name sent in chat.completions requests (reads $OPENAI_MODEL by default).",
    )
    # Default CA bundle: prefer env, else try repo-relative `certs/knapp.pem`.
    default_ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("OPENAI_CA_BUNDLE") or ""
    if not default_ca:
        repo_cert = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "certs", "knapp.pem")
        )
        if os.path.exists(repo_cert):
            default_ca = repo_cert

    parser.add_argument(
        "--ca_bundle",
        type=str,
        default=default_ca,
        help="Path to a CA bundle file to verify TLS for the OpenAI endpoint (sets REQUESTS_CA_BUNDLE and SSL_CERT_FILE).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer path/name to use instead of --model_path. "
             "Useful when --model_path is a remote model id not present "
             "locally but a matching tokenizer is available.",
    )
    parser.add_argument(
        "--openai_max_workers",
        type=int,
        default=16,
        help="Maximum number of concurrent OpenAI chat.completions requests.",
    )


def build_llm(args, model: str, **_unused: Any):
    """Return an :class:`OpenAILLM` configured from args / env."""
    backend = getattr(args, "backend", "openai")
    if backend == "openai":
        base_url = getattr(args, "base_url", "") or os.environ.get("OPENAI_BASE_URL", "")
        if not base_url:
            raise ValueError(
                "OPENAI_BASE_URL is not set. Configure it in .env or pass --base_url."
            )
        api_key = getattr(args, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        model_name = (
            getattr(args, "remote_model_name", "")
            or os.environ.get("OPENAI_MODEL", "")
            or model
        )
        # Configure CA bundle / certs: prefer CLI/env, else fallback to repo certs.
        ca_bundle = getattr(args, "ca_bundle", "") or os.environ.get("REQUESTS_CA_BUNDLE", "") or os.environ.get("OPENAI_CA_BUNDLE", "")
        if ca_bundle:
            os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
            os.environ["SSL_CERT_FILE"] = ca_bundle
            print(f"[build_llm] Using CA bundle: {ca_bundle}")
        else:
            repo_cert = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "certs", "knapp.pem"))
            repo_certs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "certs"))
            if os.path.exists(repo_cert):
                os.environ["REQUESTS_CA_BUNDLE"] = repo_cert
                os.environ["SSL_CERT_FILE"] = repo_cert
                print(f"[build_llm] Using repo CA bundle: {repo_cert}")
            elif os.path.isdir(repo_certs_dir):
                os.environ["SSL_CERT_DIR"] = repo_certs_dir
                print(f"[build_llm] Using repo certs directory as SSL_CERT_DIR: {repo_certs_dir}")
        max_workers = int(getattr(args, "openai_max_workers", 16) or 16)
        print(
            f"[build_llm] Using OpenAI-compatible backend: "
            f"base_url={_normalize_base_url(base_url)} model={model_name}"
        )
        return OpenAILLM(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            max_workers=max_workers,
        )

    raise ValueError(
        "Local vLLM backend has been removed in the OpenAI-only refactor. "
        "Use --backend=openai (the default) with OPENAI_BASE_URL configured."
    )


def resolve_tokenizer_path(args, model_path: str) -> str:
    """Return ``args.tokenizer_path`` if set, else ``model_path``."""
    override = getattr(args, "tokenizer_path", None)
    return override if override else model_path
