"""HippoRAG fact rerank filter.

Originally backed by DSPy / OpenAI chat-completions. After the OpenAI-only
refactor this is a thin adapter around the canonical ``RerankerClient`` from
``rag_clients``: if RERANKER_BASE_URL + RERANKER_MODEL are configured, facts
are reranked via that endpoint; otherwise the filter is a no-op pass-through.

The class is named ``DSPyFilter`` for compatibility with the call-site in
``HippoRAG.rerank_facts``.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .utils.logging_utils import get_logger

logger = get_logger(__name__)


def _fact_to_string(fact: Iterable) -> str:
    """Render a HippoRAG fact (subject, relation, object) as a single string."""
    if isinstance(fact, str):
        return fact
    try:
        return " ".join(str(part) for part in fact)
    except TypeError:
        return str(fact)


class DSPyFilter:
    """Optional reranker wrapping the configured OpenAI-compatible /rerank endpoint."""

    def __init__(self, hipporag_instance) -> None:
        self.hipporag = hipporag_instance
        self._client = None
        self._tried_init = False

    def _get_client(self):
        if self._tried_init:
            return self._client
        self._tried_init = True
        try:
            # Local import to avoid a hard dependency cycle: rag_clients lives
            # at the repo root and HippoRAG is vendored beneath it.
            from rag_clients import RerankerClient  # type: ignore

            self._client = RerankerClient()
            logger.info(
                f"DSPyFilter rerank enabled via {self._client.base_url} ({self._client.model})"
            )
        except Exception as exc:  # missing env vars or import error
            logger.info(f"DSPyFilter rerank disabled (passthrough): {exc}")
            self._client = None
        return self._client

    def __call__(
        self,
        query: str,
        candidate_facts: Sequence,
        candidate_fact_indices: Sequence[int],
        len_after_rerank: int,
    ) -> Tuple[List[int], List, dict]:
        if not candidate_facts:
            return [], [], {"facts_before_rerank": [], "facts_after_rerank": []}

        client = self._get_client()
        if client is None:
            top_k = min(len_after_rerank, len(candidate_facts))
            return (
                list(candidate_fact_indices[:top_k]),
                list(candidate_facts[:top_k]),
                {
                    "facts_before_rerank": list(candidate_facts),
                    "facts_after_rerank": list(candidate_facts[:top_k]),
                },
            )

        documents = [_fact_to_string(fact) for fact in candidate_facts]
        ranked = client.rerank(query, documents, top_n=len_after_rerank)
        kept_indices = [candidate_fact_indices[idx] for idx, _ in ranked]
        kept_facts = [candidate_facts[idx] for idx, _ in ranked]
        return (
            kept_indices,
            kept_facts,
            {
                "facts_before_rerank": list(candidate_facts),
                "facts_after_rerank": list(kept_facts),
            },
        )
