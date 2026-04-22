# Implementation Plan — OpenAI-Compatible Only Refactor

Drives off `docs/superpowers/specs/2026-04-22-openai-only-refactor-design.md`.

The order is chosen so each step leaves the repo in a still-runnable state:
the shared client lands first; builders/servers are migrated one at a time;
the heavy deletes happen at the end so half-removed dependencies never
break a parallel step.

---

## Step 1 — Drop in `rag_clients.py` and `.env.example`

**Files**

- New: `RAGSearch/rag_clients.py`
- New: `RAGSearch/.env.example`
- Edit: `RAGSearch/.env` (rename keys to canonical set)

`rag_clients.py` exports `load_env`, `EmbeddingClient`, `RerankerClient`,
`LLMClient`. `load_env` walks up from cwd (and from the file's own dir) to
find `.env`; parses it with the 10-line fallback if `python-dotenv` is
absent. Each client class accepts explicit overrides; otherwise reads
`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` /
`EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `RERANKER_BASE_URL` /
`RERANKER_MODEL`. URL is normalized to end in `/v1`.

**Verify**

```
python -c "from rag_clients import load_env, EmbeddingClient, RerankerClient, LLMClient; load_env(); print('ok')"
```

---

## Step 2 — New `requirements.txt` at repo root

**Files**

- Edit: `RAGSearch/requirements.txt` (overwrite or create)

Use the exact list from the spec. Delete the per-retriever requirements
files in the cleanup step (Step 11) so they don't shadow this one yet.

**Verify**

```
pip install -r requirements.txt   # runs clean in a fresh venv
```

---

## Step 3 — Rewrite all five build scripts on top of `rag_clients`

**Files**

- Edit: `RAGSearch/build_dense_index.py`
- Edit: `RAGSearch/build_hippo_index.py` → rename to `build_hipporag_index.py`
- Edit: `RAGSearch/build_hypergraph_index.py`
- Edit: `RAGSearch/build_linear_index.py`
- Edit: `RAGSearch/build_raptor_index.py`

For each:

- Strip the inline `EmbeddingClient` / `RerankerClient` / `load_dotenv`
  copies. Import from `rag_clients`.
- Replace `--input_dir` / `--input_jsonl` / `--output_dir` with the
  spec's `--corpus` / `--output-dir` (and `--name` where a per-dataset
  subdirectory is meaningful).
- Auto-detect corpus type by extension (`.jsonl` → load_jsonl, else dir
  scan). No defaults that point at hardcoded paths.
- HippoRAG builder: stop prepending `VLLM/`. Pass the bare embedding model
  name to the trimmed `OpenAI` embedding class.
- Linear builder: use `rag_clients.EmbeddingClient` directly as the
  `embedding_model` for `LinearRAGConfig` (it already exposes `.encode`).
- Drop unused `RerankerClient` instantiations from builders that don't
  rerank during indexing.

**Verify (per builder, against the bamboogle corpus)**

```
python build_dense_index.py     --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir _tmp/dense
python build_raptor_index.py    --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir _tmp/raptor
python build_linear_index.py    --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir _tmp/linear --name bamboogle
python build_hipporag_index.py  --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir _tmp/hippo --name bamboogle
python build_hypergraph_index.py --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --output-dir _tmp/hyper
```

For builders that need the LLM endpoint (HippoRAG, Linear), the user
provides a working `OPENAI_BASE_URL` in `.env`. We don't simulate it.

`--help` on each builder must succeed with no import errors before the
network calls are even attempted; that's the smoke gate at minimum.

---

## Step 4 — Trim vendored HippoRAG to OpenAI-only

**Files**

- Delete: `GraphR1/HippoRAG/src/hipporag/embedding_model/{VLLM,Cohere,GritLM,NVEmbedV2,Contriever,Transformers}.py`
- Delete: `GraphR1/HippoRAG/src/hipporag/llm/{vllm_offline,transformers_llm,transformers_offline,bedrock_llm}.py`
- Delete: `GraphR1/HippoRAG/src/hipporag/information_extraction/{openie_vllm_offline,openie_transformers_offline}.py`
- Delete: `GraphR1/HippoRAG/src/hipporag/StandardRAG.py`
- Edit: `GraphR1/HippoRAG/src/hipporag/embedding_model/__init__.py`
  - drop deleted imports
  - `_get_embedding_model_class` returns the `OpenAIEmbeddingModel` class
    unconditionally
- Edit: `GraphR1/HippoRAG/src/hipporag/embedding_model/OpenAI.py`
  - point at `OPENAI_BASE_URL` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`
    via `rag_clients` rather than hardcoded `os.environ` / OpenAI defaults
  - accept the bare model name; no `VLLM/` prefix handling
- Edit: `GraphR1/HippoRAG/src/hipporag/llm/__init__.py`
  - drop bedrock + transformers branches; `_get_llm_class` returns
    `CacheOpenAI.from_experiment_config(config)` directly
- Edit: `GraphR1/HippoRAG/src/hipporag/HippoRAG.py`
  - remove the `openie_mode == 'offline'` and `'Transformers-offline'`
    branches and their imports; keep `openie_mode == 'online'` (the only
    one we ship)
  - drop the `from transformers import HfArgumentParser` import (not used
    on the kept code paths)

**Verify**

```
python -c "import sys, pathlib; sys.path.insert(0, str(pathlib.Path('GraphR1/HippoRAG/src').resolve())); from hipporag import HippoRAG; print('ok')"
```

---

## Step 5 — Trim vendored RAPTOR to HTTP-only

**Files**

- Delete: `GraphR1/raptor/raptor/QAModels.py`
- Delete: `GraphR1/raptor/raptor/SummarizationModels.py`
- Edit: `GraphR1/raptor/raptor/EmbeddingModels.py`
  - drop `OpenAIEmbeddingModel`, `SBertEmbeddingModel`, the
    `sentence_transformers` and `tenacity` imports
  - keep `BaseEmbeddingModel`, `HTTPEmbeddingModel`
- Edit: `GraphR1/raptor/raptor/__init__.py`
  - re-export only kept names
- Edit: `GraphR1/raptor/raptor/tree_builder.py` and
  `RetrievalAugmentation.py`
  - if either imports the deleted QA / summarization classes at module
    level, drop or guard those imports

**Verify**

```
python -c "import sys, pathlib; sys.path.insert(0, str(pathlib.Path('GraphR1/raptor').resolve())); from raptor import HTTPEmbeddingModel, RetrievalAugmentation, RetrievalAugmentationConfig; print('ok')"
```

---

## Step 6 — Trim vendored LinearRAG

**Files**

- Edit: `GraphR1/LinearRAG/src/utils.py`
  - drop the `httpx` import; pass nothing for `http_client` to `OpenAI(...)`
  - read from `OPENAI_BASE_URL` / `OPENAI_API_KEY` (canonical names)
- `GraphR1/LinearRAG/src/LinearRAG.py`, `embedding_store.py`, `ner.py`,
  `config.py`, `evaluate.py` — keep, no source edits beyond the env-name
  cleanup if they reference the legacy names.

**Verify**

```
python -c "import sys, pathlib; sys.path.insert(0, str(pathlib.Path('GraphR1/LinearRAG').resolve())); from LinearRAG.src.LinearRAG import LinearRAG; print('ok')"
```

---

## Step 7 — Write the five `serve_*.py` scripts at repo root

**Files (all new at repo root)**

- `serve_dense.py`       (rewrite of existing root file)
- `serve_hipporag.py`    (replaces `GraphR1/script_api_HippoRAG.py`)
- `serve_raptor.py`      (replaces `GraphR1/script_api_RAPTOR.py`)
- `serve_hypergraph.py`  (replaces `GraphR1/script_api_HypergraphRAG.py`)
- `serve_linear.py`      (replaces `GraphR1/script_api_LinearRAG.py`)

Common skeleton:

```python
from rag_clients import load_env, EmbeddingClient, RerankerClient
load_env()
parser = argparse.ArgumentParser()
parser.add_argument("--index-dir", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--embedding-base-url")
parser.add_argument("--embedding-model")
# + --reranker-* for retrievers that rerank
args = parser.parse_args()
# load index from args.index_dir
# instantiate embedding/reranker from args overriding env
# FastAPI: POST /search, GET /status
uvicorn.run(app, host="0.0.0.0", port=args.port)
```

Per-server specifics:

- `serve_dense.py`: keeps `/retrieve` for back-compat with Search-o1
  scripts; `--index-dir` is expected to contain `dense_index.faiss` and
  `corpus.jsonl`.
- `serve_hipporag.py`: `--index-dir` is the directory the builder wrote
  to (containing `chunk_embeddings/`, `entity_embeddings/`,
  `fact_embeddings/`, the graph file). Constructs `HippoRAG(save_dir=...)`.
- `serve_raptor.py`: `--index-dir` contains `tree.pkl`. Inlines the
  `DummyQAModel` that's the only thing the existing script needed from
  the deleted `QAModels.py`.
- `serve_hypergraph.py`: rewrites the current sentence-transformers-based
  server to use `EmbeddingClient.encode` for the query.
  `--index-dir` contains `index_entity.bin`, `index_hyperedge.bin`,
  `kv_store_entities.json`, `kv_store_hyperedges.json`.
- `serve_linear.py`: instantiates `LinearRAG` with
  `EmbeddingClient` as the `embedding_model`. `--index-dir` contains the
  parquet/json files produced by the linear builder.

**Verify**

For each server:

```
python serve_<name>.py --help
python serve_<name>.py --index-dir _tmp/<name>[/<name>] --port <port> &
sleep 2
curl -s http://127.0.0.1:<port>/status
curl -s -X POST http://127.0.0.1:<port>/search -H 'content-type: application/json' \
  -d '{"queries": ["test"]}'
kill %1
```

---

## Step 8 — Rewrite `run_benchmark.py`

**Files**

- Edit: `RAGSearch/run_benchmark.py`

Changes:

- Drop the `RETRIEVER_CONFIG` venv field; everything is the single venv.
- Drop the `graphrag` retriever entry.
- Drop the `DATASET_PATHS` table (hardcoded paths). `--corpus <path>` is
  the only way to point at data.
- Each retriever entry references the new `serve_*.py` script at repo
  root and the standard CLI (`--index-dir`, `--port`).
- Drop the legacy `--data_path` alias (we already renamed to `--corpus`).
- The launcher passes `RETRIEVER_URL=http://127.0.0.1:<port>/search` (or
  `/retrieve` for dense) into the env it forwards to `eval.py`.

**Verify**

```
python run_benchmark.py --help
python run_benchmark.py --retriever dense --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --index-dir _tmp/dense --limit 2
```

---

## Step 9 — Strip vllm/transformers from Search-o1 scripts

**Files**

- Edit: `Search-o1/scripts/openai_llm.py`
  - delete `--backend vllm` branch and the `vllm` lazy import
  - default `--backend openai`; the flag becomes a no-op kept for back-compat
- Edit (in each `Search-o1/scripts/run_*.py`):
  - replace the `try: from vllm import LLM, SamplingParams` /
    `except ImportError: from openai_llm import SamplingParams` block
    with a single `from openai_llm import SamplingParams`
  - delete `from transformers import AutoTokenizer` and any tokenizer-
    based length/template logic; if a length cap is needed, use
    `len(text)` as a coarse character cap
- Delete: `Search-o1/scripts/lcb_runner/`
- Delete: `Search-o1/scripts/openai_llm.py` references to `vllm.LLM` in
  comments / docstrings
- Edit: `Search-o1/scripts/evaluate.py` — drop any `vllm` branch (search
  the file; the audit showed it imports `vllm`)
- Delete: `Search-o1/run_search_o1*.sh` shell scripts (they hardcode
  `--backend vllm`)
- Delete: `Search-o1/requirements.txt` (covered by repo-root requirements)

**Verify**

```
grep -RIn 'from vllm\|import vllm\|from transformers\|sentence_transformers\|huggingface_hub' Search-o1/scripts/
# (must be empty)
python -c "from Search-o1.scripts.openai_llm import build_llm, SamplingParams" 2>/dev/null || true
python Search-o1/scripts/run_search_o1.py --help    # must not ImportError
```

---

## Step 10 — Update `eval.py` for new convention

**Files**

- Edit: `GraphSearch/eval.py`
  - drop the hardcoded `dataset_paths` dict; the only way to pick a
    dataset is `--corpus <path>` (rename from `--data_path`)
  - `--graphrag` rename → `--retriever` (it's only used for the output
    filename anyway); keep the old name as a hidden alias to not break
    `run_benchmark.py` immediately

**Verify**

```
python GraphSearch/eval.py --help
RETRIEVER_URL=http://127.0.0.1:8306/retrieve python GraphSearch/eval.py \
  --retriever dense --corpus Search-o1/data/FlashRAG_datasets/bamboogle/test.jsonl --top_k 3 --limit 2
```

---

## Step 11 — Hard-delete dead trees

Only after Steps 1–10 leave a working repo:

```
rm -rf RAGSearch/build/
rm -rf RAGSearch/venvs/
rm -rf RAGSearch/src/
rm -rf RAGSearch/__pycache__/
rm    RAGSearch/rebuild_linear_index.py
rm -rf RAGSearch/GraphR1/agent/vllm_infer/
rm    RAGSearch/GraphR1/agent/tool/tool_env_old.py
rm    RAGSearch/GraphR1/script_api_GraphRAG.py
rm    RAGSearch/GraphR1/script_api_HippoRAG.py
rm    RAGSearch/GraphR1/script_api_RAPTOR.py
rm    RAGSearch/GraphR1/script_api_HypergraphRAG.py
rm    RAGSearch/GraphR1/script_api_LinearRAG.py
rm    RAGSearch/GraphR1/openai_client.py        # superseded by rag_clients
rm -rf RAGSearch/GraphR1/GraphRAG/
rm -rf RAGSearch/GraphR1/graphr1/               # vllm-only Search-R1 trainer
rm    RAGSearch/GraphR1/HippoRAG/build_hipporag_increment.py
rm    RAGSearch/GraphR1/HippoRAG/requirements.txt
rm    RAGSearch/GraphR1/LinearRAG/requirements.txt
rm    RAGSearch/GraphR1/raptor/requirements.txt
rm    RAGSearch/GraphR1/LinearRAG/build_nq_5000.sh
```

(`graphr1/` is a vllm-coupled Search-R1 GraphR1 trainer; not used by any
surviving entry point — confirmed during the audit.)

**Verify**

```
grep -RIn --exclude-dir={.venv,venvs,build,__pycache__,.git} \
  'vllm\|VLLM\|huggingface\|sentence_transformers\|^from transformers\|hf_hub' RAGSearch/
# (must return zero hits)

find RAGSearch -name __pycache__ -type d -exec rm -rf {} +
```

---

## Step 12 — Rewrite `README.md`

**Files**

- Edit: `RAGSearch/README.md`

Reflect the spec's "Updated Project Layout", the seven-variable `.env`,
the unified CLI for builders/servers, the single `requirements.txt`
install, and the five supported retrievers (dense, hipporag, raptor,
hypergraph, linear). No mention of vllm, conda, GraphRAG, llama.cpp setup
recipes, or HuggingFace.

Add a short **Migrating from the old layout** section so a returning user
knows their `URL` env var is now `OPENAI_BASE_URL`, etc.

**Verify**

Manual read.

---

## Step 13 — Final smoke test

Run the full verification sequence from the spec's **Testing /
Verification** section against `.env` populated with the user's real
endpoints. Capture pass/fail per command.
