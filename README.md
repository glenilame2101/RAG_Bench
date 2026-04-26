# RAGSearch

A small set of retrieval-only RAG backends — **Dense (FAISS)**,
**HippoRAG**, **RAPTOR**, **HypergraphRAG**, **LinearRAG**, and
**GraphRAG** — that all talk to your own OpenAI-compatible HTTP
endpoints (LLM, embeddings, optional reranker).

> One Python venv. One `requirements.txt`. One `.env`. No vLLM, no
> HuggingFace downloads, no transformers/sentence-transformers, no
> per-retriever virtualenv.

---

## Table of contents

- [Quick start](#quick-start)
- [The `.env` contract](#the-env-contract)
- [Corpus formats](#corpus-formats)
- [Available retrievers](#available-retrievers)
- [Building large indexes](#building-large-indexes)
  - [Partial indexing](#partial-indexing)
  - [Batch size](#batch-size)
  - [Truncation (`--max-chars`)](#truncation---max-chars)
  - [Checkpointing](#checkpointing)
- [Optional retriever extras](#optional-retriever-extras)
- [Reranker support](#reranker-support)
- [Project layout](#project-layout)
- [Further reading](#further-reading)
- [Migrating from older versions](#migrating-from-older-versions)

---

## Quick start

```bash
# 1. Create a venv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate           # Linux / Mac
.venv\Scripts\activate              # Windows

# 2. Install
pip install -r requirements.txt

# 3. spaCy English model (used by LinearRAG NER)
python -m spacy download en_core_web_sm

# 4. Configure endpoints
cp .env.example .env                # then edit .env
```

Build and serve any retriever:

```bash
# Dense (FAISS) — example
python build_dense_index.py --corpus ./mycorpus.jsonl --output-dir ./indexes/dense
python serve_dense.py      --index-dir ./indexes/dense --port 8306
```

Or do build + serve + evaluate in one shot:

```bash
python run_benchmark.py --retriever dense \
    --corpus ./mycorpus.jsonl --index-dir ./indexes/dense
```

Every retriever follows the same shape — see the [retrievers table](#available-retrievers).

---

## The `.env` contract

Seven variables. No aliases, no fallbacks:

| Variable             | Purpose                                          |
|----------------------|--------------------------------------------------|
| `OPENAI_BASE_URL`    | Base URL of your OpenAI-compatible LLM endpoint  |
| `OPENAI_API_KEY`     | API key for that endpoint                        |
| `OPENAI_MODEL`       | LLM model name                                   |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embeddings endpoint            |
| `EMBEDDING_MODEL`    | Embedding model name                             |
| `RERANKER_BASE_URL`  | OpenAI-compatible `/v1/rerank` endpoint (opt.)   |
| `RERANKER_MODEL`     | Reranker model name (opt.)                       |

Notes:

- Trailing `/v1` is added automatically if missing.
- Override any of these per command with the matching `--*-base-url` /
  `--*-model` flag.
- `.env.example` ships a working template against a local llama.cpp
  instance plus a remote MiniMax LLM.

> **Self-hosting embeddings?** See [docs/TEI.md](docs/TEI.md) for the
> recommended setup with HuggingFace's Text Embeddings Inference server.
>
> **Behind a corporate TLS-inspecting proxy?** See
> [docs/CERTIFICATES.md](docs/CERTIFICATES.md).

---

## Corpus formats

`--corpus` accepts three formats, auto-detected from the path:

- **JSONL file** — one JSON document per line.
- **Parquet file** — one document per row (requires `pyarrow`, already
  pinned).
- **Directory of `*.txt` files** — one document per file; id = file
  stem.

For JSONL and Parquet, the text is read from the first of these fields
that's present and non-empty: `contents`, `text`, `content`, `document`,
`body`. As a last resort it concatenates `question` + `answer` for
QA-style corpora.

Document ids come from `id`, `_id`, `doc_id`, or `document_id` (falling
back to row index). Nothing is hardcoded — every path comes in via the
CLI, and all six builders share the same loader (`corpus_loader.py`).

---

## Available retrievers

| Retriever  | Default port | Builder                       | Server                  | Extras                              |
|------------|--------------|-------------------------------|-------------------------|-------------------------------------|
| dense      | 8306         | `build_dense_index.py`        | `serve_dense.py`        | —                                   |
| raptor     | 8346         | `build_raptor_index.py`       | `serve_raptor.py`       | —                                   |
| hypergraph | 8336         | `build_hypergraph_index.py`   | `serve_hypergraph.py`   | —                                   |
| hipporag   | 8316         | `build_hipporag_index.py`     | `serve_hipporag.py`     | `pip install torch`                 |
| linear     | 8356         | `build_linear_index.py`       | `serve_linear.py`       | —                                   |
| graphrag   | 8326         | `build_graphrag_index.py`     | `serve_graphrag.py`     | `pip install graphrag lancedb`      |

Every server exposes:

- `POST /search` — body `{"queries": ["..."]}`, returns `[{"results": "..."}, ...]`
- `GET  /status` — `{"status": "ok", "retriever": "<name>", ...}`

`serve_dense.py` additionally keeps `POST /retrieve` for compatibility
with the Search-o1 client scripts.

---

## Building large indexes

Every builder shows tqdm progress bars during corpus loading and (where
applicable) per-document loops. For multi-GB corpora the flags below
matter most.

### Partial indexing

```bash
python build_raptor_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/raptor --partial-index 10
```

`--partial-index N` indexes only the first **N%** of the corpus
(`0 < N ≤ 100`). Useful for smoke-tests, parameter sweeps, or staging a
small index before committing to a multi-hour run.

How it slices, by format:

- **JSONL** — fast streaming line-count, then loads the first N% of lines.
- **Parquet** — reads `num_rows` from file metadata (O(1)), streams the
  first N% of rows in row-group batches.
- **Directory** — first N% of `*.txt` files, sorted alphabetically.

### Batch size

`--batch-size` (default **32**) controls how many texts go in each
embeddings request. Bigger batches = fewer round-trips = faster.

| Backend                                     | Recommended start | Notes                                                                 |
|---------------------------------------------|-------------------|-----------------------------------------------------------------------|
| Hosted (OpenAI, Cohere, Voyage, Together)   | 128               | OpenAI ≤ 2048 / ~300k tokens; Cohere ≤ 96; Voyage ≤ 128.              |
| Self-hosted (TEI, llama.cpp, vLLM, Ollama)  | 128               | Bound by GPU memory and the server's own `--max-batch-size`.          |
| llama.cpp specifically                      | tune `--batch-size` | `-np N` controls parallel slots, NOT batch capacity.                |

`EmbeddingClient` uses a 300-second timeout and retries up to 3 times on
`Timeout` / `ConnectionError` (exponential backoff). 5xx is **not**
retried — on llama.cpp those usually mean a permanent condition like
oversized input, which `--max-chars` is the right fix for.

### Truncation (`--max-chars`)

Available on **dense, raptor, hypergraph**. Truncates each text to N
characters before sending it to the embeddings endpoint. **Default:
8000.** Pass `--max-chars 0` to disable.

Why this exists: if a single text exceeds the embedding server's context
window, the whole batch fails (e.g. llama.cpp:
`input (13918 tokens) is too large to process`). Client-side truncation
avoids that.

Tuning guidance:

- 8000 chars stays under an 8192-token context for any script.
- English prose ≈ 4 chars/token. CJK, code, and mixed text can be
  1–2 chars/token.
- Monolingual English wiki? Raise to e.g. `--max-chars 24000`.
- Mixed/CJK/code corpora? Keep at 8000 or lower.

> The full document is still stored in the output artifacts
> (`corpus.jsonl`, `tree.pkl`, hypergraph KV stores). Only the text sent
> to the embedding endpoint is truncated.

`max_chars` is part of the checkpoint fingerprint, so changing it
between runs invalidates the existing `.checkpoint/`. This is
intentional — mixing embeddings computed at different truncation
points in the same index would be silently wrong.

### Checkpointing

Available on **dense, raptor, hypergraph**. Embeddings are persisted to
disk **every 1%** of total work. The checkpoint serves two purposes:

1. **Crash recovery.** A crashed or Ctrl-C'd run loses at most the last
   <1% of embeddings; rerun the same command to resume.
2. **Prefix cache across `--partial-index` runs.** A run that embedded
   the first K items leaves them on disk. A later run asking for the
   same first K (or a prefix-extending superset) skips them and only
   embeds the new items.

Example — staging a dense index in growing slices without re-embedding
overlap:

```bash
# Run 1: first 0.1% of the corpus (e.g., 200 docs).
python build_dense_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/dense --partial-index 0.1

# Run 2: first 0.2% (400 docs). The first 200 are reused.
python build_dense_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/dense --partial-index 0.2

# Run 3: full corpus. Reuses everything cached; embeds the rest.
python build_dense_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/dense
```

Each run overwrites the final artifact (`dense_index.faiss`, `tree.pkl`,
hypergraph `index_*.bin`) to reflect the current `--partial-index`
slice, but the checkpoint directory persists.

Layout under `<output-dir>/.checkpoint/`:

```
state.json           # {model, completed, prefix_fingerprint, ...}
emb_000000.npy       # first slice of saved embeddings (float32, normalized)
emb_000001.npy
...
```

Hypergraph has two subdirs (`.checkpoint/entities/` and
`.checkpoint/hyperedges/`) since it runs two independent embedding
passes.

**Safety guards:**

- A SHA-256 **prefix fingerprint** of `(model name, normalize flag,
  completed count, first/middle/last text of the completed prefix)` is
  stored in `state.json`. Reruns against the same prefix match;
  different model / corpus / reordered prefix produces a clean
  `SystemExit`.
- **Going backwards is refused.** If the cache holds 400 items but your
  new `--partial-index` asks for only 100, the script errors out
  rather than silently truncating. Use `--no-checkpoint`, delete
  `.checkpoint/`, or pick a different `--output-dir`.
- If the on-disk count of embeddings disagrees with `state.json`
  (interrupted mid-flush), the script errors instead of producing a
  corrupt array.

**Storage cost:** roughly `4 KB × num_items` for a 1024-dim model like
bge-m3 (≈ 4 GB per million items).

**Disabling:** pass `--no-checkpoint`.

> The checkpoint is **not** auto-deleted after the final artifact is
> written — it stays so the next `--partial-index` run can reuse it.
> Delete `.checkpoint/` manually once you're done growing the index.

The linear, hipporag, and graphrag builders don't have checkpointing:
they delegate the embedding loop to vendored library code, so adding
it there means modifying their internals rather than a one-flag change.

---

## Optional retriever extras

Two retrievers need packages deliberately kept out of `requirements.txt`
so the base venv stays light. Install them only when you actually want
that retriever.

### HippoRAG

The vendored HippoRAG calls `torch` in one place
(`utils/embed_utils.retrieve_knn`) for entity-entity KNN:

```bash
pip install torch
```

CPU-only is fine — the function falls back to CPU when CUDA isn't
present.

### GraphRAG

GraphRAG uses Microsoft's `graphrag` package and its LanceDB vector
store. Both pull in heavy transitive deps (dspy, fastlite, etc.) that
the rest of the stack doesn't need:

```bash
pip install graphrag lancedb
```

`build_graphrag_index.py` writes a `settings.yaml` into your
`--output-dir` that uses `${VAR}` substitution against the same `.env`
contract. The LLM hits `OPENAI_BASE_URL` / `OPENAI_MODEL` and the
embedder hits `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`. No
GraphRAG-specific config files to maintain.

> **Heads-up on cost.** GraphRAG is the most LLM-expensive retriever
> here: every chunk costs entity + relationship extraction, every
> community costs a summarization pass, and every `/search` call invokes
> the LLM at retrieval time. Plan to use it against small corpora
> (HotpotQA-distractor, sample wikis) unless you have a local LLM with
> serious throughput.

---

## Reranker support

If `RERANKER_BASE_URL` and `RERANKER_MODEL` are set, retrievers that
benefit from query-time reranking (currently HypergraphRAG and HippoRAG
fact reranking) will use the endpoint via `POST /v1/rerank`.

If those vars are unset, retrievers degrade gracefully to passthrough
ranking — no errors, no warnings beyond a single startup log line.

> TEI can serve a reranker model on the same OpenAI-compatible
> protocol — see [docs/TEI.md](docs/TEI.md#use-tei-as-a-reranker-too).

---

## Project layout

```
RAG_Bench/
├── .env / .env.example              # canonical config
├── requirements.txt                 # single, flat
├── rag_clients.py                   # shared HTTP clients (Embedding/Reranker/LLM)
├── corpus_loader.py                 # shared corpus loader for all builders
├── build_<retriever>_index.py       # six builders (CLI: --corpus, --output-dir)
├── serve_<retriever>.py             # six servers   (CLI: --index-dir, --port)
├── run_benchmark.py                 # build/serve/eval orchestrator
├── docs/
│   ├── TEI.md                       # Text Embeddings Inference setup
│   └── CERTIFICATES.md              # Corporate TLS / CA bundle setup
├── GraphR1/
│   ├── HippoRAG/src/hipporag/       # vendored HippoRAG, OpenAI-only
│   ├── LinearRAG/src/               # vendored LinearRAG
│   └── raptor/raptor/               # vendored RAPTOR, HTTP embeddings only
├── GraphSearch/
│   └── eval.py                      # standalone EM/F1 evaluator
└── Search-o1/
    ├── data/                        # corpora (gitignored)
    └── scripts/                     # agent client (run_search_o1*.py + helpers)
```

---

## Further reading

- **[docs/TEI.md](docs/TEI.md)** — Run HuggingFace's Text Embeddings
  Inference server and point the stack at it.
- **[docs/CERTIFICATES.md](docs/CERTIFICATES.md)** — Corporate
  TLS-inspecting proxy / CA bundle setup.

---

## Migrating from older versions

The legacy environment variable names (`URL`, `MODEL_NAME`,
`LLM_BASE_URL`, `EMBEDDING_URL`, `EMBEDDING_MODEL_NAME`,
`REMOTE_MODEL_NAME`) have been collapsed into the seven names above.
Update your `.env` accordingly.

The per-retriever `requirements.txt` files, the `venvs/` per-backend
virtualenvs, the `script_api_*.py` server scripts under `GraphR1/`, the
`vllm_infer/` runner, the `lcb_runner/` LiveCodeBench evaluator, and the
`graphr1/` Search-R1 trainer have all been removed. If you depended on
any of those, pin the previous tag.

The GraphRAG retriever was originally removed in the same refactor (its
`graphrag` PyPI package pulled in dspy/litellm and other deps the base
stack avoids). It has since been added back as an **opt-in extra** —
`build_graphrag_index.py` and `serve_graphrag.py` are first-class
scripts matching the other retrievers' shape, but you must `pip install
graphrag lancedb` separately. See
[Optional retriever extras](#optional-retriever-extras).
