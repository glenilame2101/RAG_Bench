# OpenAI-Compatible Only Refactor — Design Spec

**Date:** 2026-04-22
**Scope:** RAGSearch (entire repo)

## Goal

Strip every vLLM, HuggingFace, sentence-transformers, and "pull-from-the-internet"
code path from RAGSearch. Every retriever — index builder, server, and query
client — must talk to the user's local OpenAI-compatible HTTP endpoints
(LLM, embeddings, reranker) via plain `requests` / `openai` SDK calls
configured from one `.env` file.

The codebase must be runnable with a single Python virtualenv, a single flat
`requirements.txt`, and no per-retriever sub-environments or vendored
"package" tricks.

## Non-Goals

- Re-implementing the retrieval algorithms themselves. Existing logic stays.
- Optimizing performance. We are removing optionality, not tuning.
- Re-doing the agent layer (`Search-o1/scripts/run_*.py`) end-to-end. Only the
  `vllm` and `transformers` import boundaries get touched there.

## The Single `.env` Contract

All Python entry points load `.env` from the repo root. The variables are:

| Variable | Purpose | Required by |
|---|---|---|
| `OPENAI_BASE_URL` | OpenAI-compatible LLM endpoint base URL | LLM-using builders, agents |
| `OPENAI_API_KEY`  | API key for `OPENAI_BASE_URL` | All |
| `OPENAI_MODEL`    | LLM model name | LLM-using builders, agents |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embeddings endpoint | All builders, all servers |
| `EMBEDDING_MODEL`    | Embedding model name | All builders, all servers |
| `RERANKER_BASE_URL`  | OpenAI-compatible reranker endpoint | Servers that rerank |
| `RERANKER_MODEL`     | Reranker model name | Servers that rerank |

`/v1` is appended to a base URL if missing. No other variables are read.
The legacy `URL`, `MODEL_NAME`, `LLM_BASE_URL`, `EMBEDDING_URL`,
`EMBEDDING_MODEL_NAME`, `REMOTE_MODEL_NAME` aliases are dropped — one name
per concept.

## Shared HTTP Clients

A single new file `RAGSearch/rag_clients.py` (sits at the repo root, no
package, importable via `from rag_clients import ...`) provides:

- `load_env()` — reads `.env` from the repo root (walks up if needed). No
  hard dependency on `python-dotenv`; falls back to a 10-line parser.
- `EmbeddingClient(base_url=None, model=None)` — `.encode(texts, batch_size,
  normalize) -> np.ndarray`. Defaults read env vars.
- `RerankerClient(base_url=None, model=None)` — `.rerank(query, documents,
  top_n) -> list[(idx, score)]`. Defaults read env vars.
- `LLMClient(base_url=None, api_key=None, model=None)` — `.chat(messages,
  **kwargs) -> str`. Thin wrapper over `openai.OpenAI`.

Every other file imports these instead of redefining its own client class.
The current ~20 duplicate `EmbeddingClient` / `HTTPEmbeddingModel` definitions
all go away.

## Builders (CLI Convention)

All five builders converge on the same flag set:

```
python build_<retriever>_index.py \
  --corpus <path>                # required: dir of .txt OR .jsonl file
  --output-dir <path>            # required: where to write index files
  [--name <str>]                 # optional: subdir name under output-dir
  [--batch-size <int>]
  [--embedding-base-url <url>]   # overrides .env
  [--embedding-model <str>]      # overrides .env
  [--reranker-base-url <url>]    # only retrievers that rerank
  [--reranker-model <str>]
```

Rules:

- No hardcoded paths. `--corpus` and `--output-dir` are required.
- `.jsonl` is detected by extension; otherwise `--corpus` is treated as a
  directory of `*.txt` files.
- Builders never call out to HuggingFace or download anything.
- Hippo / Linear builders no longer prepend `VLLM/` to model names.

## Servers (CLI Convention)

All servers converge on:

```
python serve_<retriever>.py \
  --index-dir <path>            # required: directory produced by the builder
  --port <int>                  # required (no default that masks conflicts)
  [--embedding-base-url ...]    # overrides .env
  [--embedding-model ...]
  [--reranker-base-url ...]
  [--reranker-model ...]
```

Five server scripts at the repo root, named symmetrically with the builders:

- `serve_dense.py`
- `serve_hipporag.py` (replaces `GraphR1/script_api_HippoRAG.py`)
- `serve_raptor.py`  (replaces `GraphR1/script_api_RAPTOR.py`)
- `serve_hypergraph.py` (replaces `GraphR1/script_api_HypergraphRAG.py`)
- `serve_linear.py`  (replaces `GraphR1/script_api_LinearRAG.py`)

GraphRAG (`graphrag` PyPI package) is not on the local-only path — it pulls
its own world of HF / Azure deps. We **drop GraphRAG** from the supported
retrievers and remove `script_api_GraphRAG.py` and `GraphR1/GraphRAG/`. The
README will list five retrievers, not six.

All servers expose:

- `POST /search`  — `{"queries": [...]}` -> `[{"results": "..."}, ...]`
- `GET  /status`  — `{"status": "ok", "retriever": "<name>", "index_size": N}`

(`serve_dense.py` keeps its existing `/retrieve` endpoint for Search-o1
compatibility, but also exposes `/search` as the standard.)

## Vendored Library Surgery

### `GraphR1/HippoRAG/src/hipporag/`

Keep:
- `HippoRAG.py` (edited to drop the offline/vllm branches)
- `embedding_store.py`
- `embedding_model/{__init__.py, base.py, OpenAI.py}` — `OpenAI.py`
  becomes the single HTTP-backed implementation. `__init__.py` returns it
  unconditionally.
- `llm/{__init__.py, base.py, openai_gpt.py}`
- `information_extraction/{__init__.py, openie_openai.py, openie_litellm_offline.py}`
- `evaluation/`, `prompts/`, `utils/`, `rerank.py`

Delete:
- `embedding_model/{VLLM,Cohere,GritLM,NVEmbedV2,Contriever,Transformers}.py`
- `llm/{vllm_offline,transformers_llm,transformers_offline,bedrock_llm}.py`
- `information_extraction/{openie_vllm_offline,openie_transformers_offline}.py`
- `StandardRAG.py` (imports vllm-only OpenIE, unused by our entry points)

`_get_embedding_model_class` collapses to "always return the OpenAI HTTP
class". The class no longer treats a `VLLM/` prefix as a sentinel.

### `GraphR1/raptor/raptor/`

Keep:
- `cluster_tree_builder.py`, `cluster_utils.py`
- `tree_builder.py`, `tree_retriever.py`, `tree_structures.py`
- `RetrievalAugmentation.py`, `Retrievers.py`, `FaissRetriever.py`
- `EmbeddingModels.py` — keep only `BaseEmbeddingModel` and
  `HTTPEmbeddingModel`. Delete `OpenAIEmbeddingModel`, `SBertEmbeddingModel`.
- `utils.py`
- `__init__.py` — re-export only the kept names.

Delete:
- `QAModels.py` and `SummarizationModels.py` (only the Dummy subclass is
  used; we move that into `serve_raptor.py` directly).

### `GraphR1/LinearRAG/src/`

Keep:
- `LinearRAG.py`, `embedding_store.py`, `ner.py`, `utils.py`, `config.py`,
  `evaluate.py`

`utils.py` already uses `openai.OpenAI` — drop the `httpx` import (use the
default sync transport so we have one fewer dependency to pin) and read from
the canonical env names.

## Files to Delete Outright

- `RAGSearch/build/` (build artifact)
- `RAGSearch/venvs/` (per-retriever venvs — single-venv only)
- `RAGSearch/rebuild_linear_index.py` (duplicate of `build_linear_index.py`)
- `RAGSearch/serve_dense.py` is rewritten in place; stale variants gone
- `RAGSearch/GraphR1/agent/vllm_infer/` (vllm batch runner)
- `RAGSearch/GraphR1/agent/tool/tool_env_old.py`
- `RAGSearch/GraphR1/script_api_*.py` (replaced by `serve_*.py` at root)
- `RAGSearch/GraphR1/GraphRAG/` (drop GraphRAG retriever entirely)
- `RAGSearch/GraphR1/HippoRAG/build_hipporag_increment.py` (uses old VLLM/
  hack and the deleted offline OpenIE)
- `RAGSearch/Search-o1/scripts/lcb_runner/` (LiveCodeBench evaluator —
  unused by every entry point we keep)
- `RAGSearch/GraphR1/raptor/raptor/QAModels.py`
- `RAGSearch/GraphR1/raptor/raptor/SummarizationModels.py`
- `RAGSearch/GraphR1/HippoRAG/src/hipporag/StandardRAG.py`
- `RAGSearch/GraphR1/HippoRAG/src/hipporag/embedding_model/{VLLM,Cohere,GritLM,NVEmbedV2,Contriever,Transformers}.py`
- `RAGSearch/GraphR1/HippoRAG/src/hipporag/llm/{vllm_offline,transformers_llm,transformers_offline,bedrock_llm}.py`
- `RAGSearch/GraphR1/HippoRAG/src/hipporag/information_extraction/openie_vllm_offline.py`
- `RAGSearch/GraphR1/HippoRAG/src/hipporag/information_extraction/openie_transformers_offline.py`
- All per-subdir `requirements.txt` (`Search-o1/`, `GraphR1/HippoRAG/`,
  `GraphR1/LinearRAG/`, `GraphR1/raptor/`)
- `RAGSearch/Search-o1/run_search_o1*.sh` (shell runners that hardwire vllm flags)
- `__pycache__` everywhere (caches; will be regenerated)

## Dependencies — Single `requirements.txt`

A single repo-root `requirements.txt`. No optional extras, no per-retriever
files. All versions chosen for mutual compatibility (Python 3.10+):

```
openai>=1.40
python-dotenv>=1.0
requests>=2.31
numpy>=1.26,<2.0
fastapi>=0.110
uvicorn>=0.29
pydantic>=2.6
faiss-cpu>=1.7.4
scikit-learn>=1.3
pandas>=2.0
pyarrow>=14.0
python-igraph>=0.11
networkx>=3.2
spacy>=3.7
tqdm>=4.66
```

Notably absent (and all forbidden): `vllm`, `torch`, `transformers`,
`sentence-transformers`, `huggingface-hub`, `gritlm`, `dspy`, `tiktoken`,
`tenacity`, `umap-learn`, `boto3`, `litellm`, `graphrag`. If a retriever
genuinely needs one of these, that retriever gets cut.

`spacy` requires the user to download the English model once
(`python -m spacy download en_core_web_sm`); README documents this.

## Search-o1 Agent Surgery

The `Search-o1/scripts/run_*.py` family already routes through
`openai_llm.py` for inference. We:

- Delete the `from vllm import ...` try/except in every `run_*.py`. Replace
  with `from openai_llm import SamplingParams`. There is no fallback to
  vllm.
- Delete the `--backend vllm` branch in `openai_llm.py`. The shim is
  OpenAI-only.
- Delete `from transformers import AutoTokenizer` — the agent doesn't need
  to pre-tokenize when the server handles the chat template. Replace any
  remaining tokenizer-length checks with character-count heuristics.
- Drop `lcb_runner/` (LiveCodeBench), `evaluate.py`'s `vllm` branch, and
  any code path gated on `args.backend == "vllm"`.

## Updated Project Layout

```
RAGSearch/
├── .env                          # the only config file
├── .env.example
├── README.md                     # rewritten
├── requirements.txt              # single, flat
├── rag_clients.py                # shared HTTP clients
├── build_dense_index.py
├── build_hipporag_index.py       # renamed from build_hippo_index.py
├── build_hypergraph_index.py
├── build_linear_index.py
├── build_raptor_index.py
├── serve_dense.py
├── serve_hipporag.py
├── serve_hypergraph.py
├── serve_linear.py
├── serve_raptor.py
├── run_benchmark.py              # rewritten — five retrievers, --corpus required
├── GraphR1/
│   ├── HippoRAG/src/hipporag/    # trimmed
│   ├── LinearRAG/src/            # trimmed
│   ├── raptor/raptor/            # trimmed
│   └── agent/                    # only `tool/` survives, vllm_infer/ deleted
├── GraphSearch/
│   └── eval.py                   # unchanged, already env-driven
└── Search-o1/
    ├── data/                     # corpora, untouched
    └── scripts/                  # vllm imports stripped
```

The `src/` empty directory at repo root and the `__pycache__` folders go.

## README

The README is rewritten with one quick-start path:

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm`
3. Fill in `.env` (seven variables; example provided)
4. Build an index: `python build_<retriever>_index.py --corpus <path> --output-dir <path>`
5. Serve it: `python serve_<retriever>.py --index-dir <path> --port <port>`
6. Evaluate: `python run_benchmark.py --retriever <name> --corpus <path>`

No mention of vllm, conda, GPUs, per-retriever venvs, llama.cpp setup, or
Hugging Face downloads.

## Testing / Verification

After the refactor, the following must succeed from a clean checkout with
the single venv:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# .env points at user's endpoints
python build_dense_index.py     --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir dense_index
python build_raptor_index.py    --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir raptor_index
python build_linear_index.py    --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir linear_index --name bamboogle
python build_hipporag_index.py  --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir hippo_index --name bamboogle
python build_hypergraph_index.py --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir hypergraph_index

# Each server boots and answers /status:
python serve_dense.py     --index-dir dense_index     --port 8306 &  curl localhost:8306/status
python serve_raptor.py    --index-dir raptor_index    --port 8346 &  curl localhost:8346/status
python serve_linear.py    --index-dir linear_index/bamboogle --port 8356 &  curl localhost:8356/status
python serve_hipporag.py  --index-dir hippo_index/bamboogle  --port 8316 &  curl localhost:8316/status
python serve_hypergraph.py --index-dir hypergraph_index --port 8336 &  curl localhost:8336/status
```

Static checks:

- `grep -RIn 'vllm\|VLLM\|huggingface\|sentence_transformers\|from transformers'
  RAGSearch/` returns zero matches in source files (excluding `.venv`,
  `__pycache__`, deleted dirs).
- `python -c "import rag_clients"` succeeds.
- Every `serve_*.py` and `build_*.py` runs `--help` without import errors.

## Rollout

Single PR. No flag-gating: this is a hard cut from "many partial paths" to
"one local-OpenAI path". The user already operates this way; the legacy
paths are dead weight.
