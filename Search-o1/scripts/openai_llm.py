"""OpenAI-compatible backend for Search-o1 client scripts.

This module provides an ``OpenAILLM`` class whose ``generate`` method mimics
``vllm.LLM.generate`` closely enough that the existing Search-o1 scripts work
without modification beyond swapping the constructor.

Usage::

    from openai_llm import build_llm, add_backend_args, load_env_file

    load_env_file()  # optional, picks up repo-root .env
    add_backend_args(parser)  # adds --backend / --base_url / --api_key / ...
    args = parser.parse_args()
    llm = build_llm(args, model=model_path, gpu_memory_utilization=0.9)

When ``args.backend == "openai"`` the returned object is an :class:`OpenAILLM`
that calls an OpenAI-compatible chat-completions endpoint (e.g. a remote
``vllm.entrypoints.openai.api_server``). When ``args.backend == "vllm"`` the
function falls back to instantiating a real ``vllm.LLM``.
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
# The Search-o1 scripts instantiate ``SamplingParams(...)`` directly. When
# vllm is not installed (single-venv / remote-endpoint deployments) we fall
# back to this dataclass, which is duck-compatible with the attributes
# ``_sampling_to_kwargs`` below reads.
@dataclass
class SamplingParams:  # type: ignore[no-redef]
    """Duck-typed fallback for ``vllm.SamplingParams``.

    Used when vllm is not installed (single-venv / remote-endpoint setup).
    Exposes the same attributes the Search-o1 scripts set and the ones
    :meth:`OpenAILLM._sampling_to_kwargs` below reads, so callers can keep
    writing ``SamplingParams(max_tokens=..., temperature=..., ...)`` without
    caring whether the real vllm class is available.
    """

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
# Fake RequestOutput shapes (mimic vLLM)
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
    """Drop-in replacement for ``vllm.LLM`` that hits an OpenAI-compatible API.

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

        The Search-o1 scripts already apply the tokenizer chat template and
        pass the resulting string in. Those strings include special tokens
        like ``<|im_start|>user\n...<|im_end|>``; the remote vLLM server will
        re-apply its chat template to whatever we send, so we wrap the raw
        string as a single user message. The server then treats the
        already-templated text as user content -- slightly wasteful but
        semantically identical because the model has seen that format in
        pre-training and the outer template contributes only a thin wrapper.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return prompt  # already messages
        raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")

    @staticmethod
    def _sampling_to_kwargs(sp: Any) -> dict:
        """Translate a ``vllm.SamplingParams`` (or anything with matching
        attributes) into chat.completions kwargs.

        ``top_k`` and ``repetition_penalty`` are not part of the standard
        OpenAI API but are accepted by the vLLM OpenAI server via
        ``extra_body``.
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
        if stop:
            kwargs["stop"] = list(stop) if not isinstance(stop, str) else [stop]

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
    default_url = os.environ.get("URL") or os.environ.get("LLM_BASE_URL") or ""
    default_model = os.environ.get("MODEL_NAME") or ""
    default_key = os.environ.get("OPENAI_API_KEY") or ""
    # Default backend: remote OpenAI-compatible endpoint. The user must set
    # URL (via .env or --base_url) for it to work; fall back to local vllm
    # only with --backend vllm explicitly.
    default_backend = "openai"

    parser.add_argument(
        "--backend",
        choices=["vllm", "openai"],
        default=default_backend,
        help="LLM backend: 'openai' (remote OpenAI-compatible endpoint, "
             "default) or 'vllm' (local GPU vLLM; requires the vllm package "
             "and a GPU).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=default_url,
        help="Base URL of the OpenAI-compatible endpoint (reads $URL or "
             "$LLM_BASE_URL by default).",
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
        help="Model name sent in chat.completions requests (reads "
             "$MODEL_NAME by default). Only used when --backend=openai.",
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


def build_llm(args, model: str, **vllm_kwargs: Any):
    """Return a backend LLM: either ``vllm.LLM`` or :class:`OpenAILLM`."""
    backend = getattr(args, "backend", "openai")
    if backend == "openai":
        base_url = getattr(args, "base_url", "") or os.environ.get("URL", "")
        if not base_url:
            raise ValueError(
                "--backend=openai requires --base_url or the URL env var. "
                "Set URL in .env (see .env.example) or pass --base_url, or "
                "use --backend vllm for a local GPU run."
            )
        api_key = getattr(args, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        model_name = (
            getattr(args, "remote_model_name", "")
            or os.environ.get("MODEL_NAME", "")
            or model
        )
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

    # Explicit opt-in: real vLLM (local GPU).
    try:
        from vllm import LLM  # lazy import so the OpenAI backend doesn't need it
    except ImportError as exc:
        raise ImportError(
            "--backend=vllm requires the 'vllm' package. Install with the "
            "optional 'train' extra (pip install -e '.[train]'), or switch "
            "to --backend openai against a remote endpoint."
        ) from exc
    print(f"[build_llm] Using local vLLM backend: model={model}")
    return LLM(model=model, **vllm_kwargs)


def resolve_tokenizer_path(args, model_path: str) -> str:
    """Return ``args.tokenizer_path`` if set, else ``model_path``."""
    override = getattr(args, "tokenizer_path", None)
    return override if override else model_path
