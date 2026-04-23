# RAGSearch

A small set of retrieval-only RAG backends — Dense (FAISS), HippoRAG,
RAPTOR, HypergraphRAG, LinearRAG, and GraphRAG — that all talk to your own
OpenAI-compatible HTTP endpoints (LLM, embeddings, optional reranker).

There is **no vLLM, no HuggingFace download, no transformers, no
sentence-transformers, no per-retriever virtual environment**. One Python
venv, one `requirements.txt`, one `.env` file with seven variables.

Two of the six retrievers (HippoRAG, GraphRAG) need extra packages that are
not pinned in `requirements.txt` because they pull in heavy deps the rest of
the stack avoids — see [Optional retriever extras](#optional-retriever-extras).

## Quick start

```bash
# 1. Create and activate a single venv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate           # Linux / Mac
.venv\Scripts\activate              # Windows

# 2. Install everything
pip install -r requirements.txt

# 3. spaCy needs the English model once (used by LinearRAG NER)
python -m spacy download en_core_web_sm

# 4. Configure your endpoints
cp .env.example .env                # then edit .env

# 5. Build an index for the retriever you want
python build_dense_index.py      --corpus ./mycorpus.jsonl --output-dir ./indexes/dense
python build_raptor_index.py     --corpus ./mycorpus.jsonl --output-dir ./indexes/raptor
python build_hypergraph_index.py --corpus ./mycorpus.jsonl --output-dir ./indexes/hyper
python build_hipporag_index.py   --corpus ./mycorpus.jsonl --output-dir ./indexes/hippo     # needs torch
python build_linear_index.py     --corpus ./mycorpus.jsonl --output-dir ./indexes/linear
python build_graphrag_index.py   --corpus ./mycorpus.jsonl --output-dir ./indexes/graphrag  # needs graphrag + lancedb

# 6. Serve the index (same path you built into)
python serve_dense.py      --index-dir ./indexes/dense    --port 8306
python serve_raptor.py     --index-dir ./indexes/raptor   --port 8346
python serve_hypergraph.py --index-dir ./indexes/hyper    --port 8336
python serve_hipporag.py   --index-dir ./indexes/hippo    --port 8316
python serve_linear.py     --index-dir ./indexes/linear   --port 8356
python serve_graphrag.py   --index-dir ./indexes/graphrag --port 8326

# 7. Or build + serve + evaluate in one command
python run_benchmark.py --retriever dense \
    --corpus ./mycorpus.jsonl --index-dir ./indexes/dense
```

`--corpus` accepts three formats, auto-detected from the path:

- **JSONL file** (one JSON document per line)
- **Parquet file** (one document per row; columns follow the same field
  priority — requires `pyarrow`, already in `requirements.txt`)
- **Directory of `*.txt` files** (one document per file; id = file stem)

For JSONL and Parquet, the text is read from the first of these fields
that's present and non-empty: `contents`, `text`, `content`, `document`,
`body`. As a last resort it concatenates `question` + `answer` for
QA-style corpora. Document ids come from `id`, `_id`, `doc_id`, or
`document_id` (falling back to row index). Nothing is hardcoded — every
path comes in via the CLI, and all six builders share the same loader
(`corpus_loader.py`).

## `.env` contract

The seven variables every entry point reads (no aliases, no fallbacks):

| Variable             | Purpose                                          |
|----------------------|--------------------------------------------------|
| `OPENAI_BASE_URL`    | Base URL of your OpenAI-compatible LLM endpoint  |
| `OPENAI_API_KEY`     | API key for that endpoint                        |
| `OPENAI_MODEL`       | LLM model name                                   |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embeddings endpoint            |
| `EMBEDDING_MODEL`    | Embedding model name                             |
| `RERANKER_BASE_URL`  | OpenAI-compatible `/v1/rerank` endpoint (opt.)   |
| `RERANKER_MODEL`     | Reranker model name (opt.)                       |

Trailing `/v1` is added automatically if missing. You can override any of
these per command with the matching `--*-base-url` / `--*-model` flag.

`.env.example` ships a working template against a local llama.cpp instance
plus a remote MiniMax LLM.

## Corporate TLS-inspecting proxies

If your network sits behind a TLS-inspecting proxy (Zscaler, Netskope,
Palo Alto, etc.) the public CAs in your system trust store will fail to
verify the proxy's substituted certificate, producing
`SSLError: CERTIFICATE_VERIFY_FAILED` on every outbound HTTPS call.

To avoid hand-patching every HTTP client, drop your company CA bundle at
the repo root and the stack will pick it up automatically:

```
RAGSearch/
├── cert/
│   └── knapp.pem        # your corporate CA bundle (not committed)
├── .env
└── ...
```

Both `rag_clients.load_env()` (used by builders/servers) and
`openai_llm.load_env_file()` (used by Search-o1 client scripts) probe for
this file at startup. When found they:

- Set `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` in the process environment
  (so `requests` and `httpx` honor it transparently)
- Pass `verify=<bundle>` explicitly to the OpenAI SDK's underlying
  `httpx.Client`

You'll see one log line on startup confirming the bundle was loaded:

```
[ca-bundle] Using company CA bundle: /abs/path/to/cert/knapp.pem
```

If `cert/knapp.pem` is absent the helper is a silent no-op and the system
default CA store is used — fine for non-corporate machines.

**Override paths:** set `COMPANY_CA_CERT=/some/other/path.pem` in `.env`
or the shell environment to point at a bundle outside the default
`cert/knapp.pem` location.

`cert/` is gitignored by convention — keep it out of version control.

## Available retrievers

| Retriever  | Default port | Builder                       | Server                  | Extras                  |
|------------|--------------|-------------------------------|-------------------------|-------------------------|
| dense      | 8306         | `build_dense_index.py`        | `serve_dense.py`        | —                       |
| raptor     | 8346         | `build_raptor_index.py`       | `serve_raptor.py`       | —                       |
| hypergraph | 8336         | `build_hypergraph_index.py`   | `serve_hypergraph.py`   | —                       |
| hipporag   | 8316         | `build_hipporag_index.py`     | `serve_hipporag.py`     | `pip install torch`     |
| linear     | 8356         | `build_linear_index.py`       | `serve_linear.py`       | —                       |
| graphrag   | 8326         | `build_graphrag_index.py`     | `serve_graphrag.py`     | `pip install graphrag lancedb` |

Every server exposes:

- `POST /search`  — body `{"queries": ["..."]}`, returns `[{"results": "..."}, ...]`
- `GET  /status`  — `{"status": "ok", "retriever": "<name>", ...}`

`serve_dense.py` additionally keeps `POST /retrieve` for compatibility with
the Search-o1 client scripts.

## Building large indexes

Every builder shows tqdm progress bars during corpus loading and (where
applicable) per-document loops. For multi-GB corpora the following flags
matter:

### `--partial-index N` (all six builders)

Index only the first `N%` of the corpus. `N` is a number in `(0, 100]`.

```bash
python build_raptor_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/raptor --partial-index 10
```

Useful for smoke-tests, parameter sweeps, or staging a small index
before committing to a full multi-hour run. For JSONL the builder does
a fast streaming line-count first (no JSON parsing), then loads only
the first `N%` of lines. For Parquet it reads `num_rows` from the file
metadata (O(1)) and then streams the first `N%` of rows in row-group
batches. For directories it takes the first `N%` of `*.txt` files
sorted alphabetically.

### `--batch-size`

Default is 32. This is the number of texts sent per request to your
embeddings endpoint. Each request has fixed network/server overhead, so
bigger batches = fewer round-trips = faster indexing.

- **Hosted endpoints** (OpenAI, Cohere, Voyage, Together, …): start at 128.
  OpenAI accepts up to 2048 inputs / ~300k tokens per request; Cohere caps
  at 96; Voyage at 128. Read your provider's docs.
- **Self-hosted** (llama.cpp, vLLM, TEI, Ollama): the only caps are GPU
  memory and the server's own `--max-batch-size` (or equivalent). Try 128;
  if the server returns 4xx, raise its limit. Note that the
  `EmbeddingClient` has a 60-second request timeout — on slow hardware,
  very large batches can time out even when the server would eventually
  reply.
- **llama.cpp specifically:** `-np N` controls server-side parallel slots,
  not batch capacity. Concurrent client requests against `-np 1` queue
  rather than parallelize, so the only useful lever is `--batch-size`.

### Checkpointing (dense, raptor, hypergraph)

Three builders — `build_dense_index.py`, `build_raptor_index.py`, and
`build_hypergraph_index.py` — checkpoint embeddings to disk **every 1%**
of total work by default. The checkpoint serves two purposes:

1. **Crash recovery.** A crashed or Ctrl-C'd run loses at most the last
   <1% of embeddings; re-run the same command to resume.
2. **Prefix cache across `--partial-index` runs.** A run that embedded
   the first `K` items leaves those embeddings on disk. A later run
   asking for the same first `K` items, *or any prefix-extending
   superset*, skips those `K` and embeds only the new items.

Example — staging a dense index in growing slices without re-embedding
the overlap:

```bash
# Run 1: embed the first 0.1% of the corpus (e.g., 200 docs).
python build_dense_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/dense --partial-index 0.1

# Run 2: embed the first 0.2% (400 docs). The first 200 are reused
# from the checkpoint; only the new 200 hit the embedding endpoint.
python build_dense_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/dense --partial-index 0.2

# Run 3: full corpus. Reuses everything already cached; embeds the rest.
python build_dense_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/dense
```

Each run overwrites the final artifact (`dense_index.faiss`,
`tree.pkl`, or the hypergraph `index_*.bin`) to reflect the current
`--partial-index` slice, but the checkpoint directory persists.

Layout under `<output-dir>/.checkpoint/`:

```
state.json           # {model, completed, prefix_fingerprint, ...}
emb_000000.npy       # one slice of saved embeddings (float32, normalized)
emb_000001.npy
...
```

Hypergraph has two subdirs (`.checkpoint/entities/` and
`.checkpoint/hyperedges/`) since it runs two independent embedding
passes.

Safety guards:

- A SHA-256 **prefix fingerprint** of `(model name, normalize flag,
  completed count, first/middle/last text of the completed prefix)`
  is stored in `state.json`. Re-running against the same corpus prefix
  matches; re-running with a different model, a different corpus, or a
  reordered prefix produces a clean `SystemExit` telling you to delete
  the directory or point at a different one.
- **Going backwards** is refused: if the cache holds 400 items but your
  new `--partial-index` asks for only 100, the script errors out rather
  than silently truncating. Use `--no-checkpoint`, delete
  `.checkpoint/`, or use a separate `--output-dir`.
- If the on-disk count of embeddings disagrees with `state.json`
  (interrupted mid-flush), the script errors out instead of producing a
  corrupt array.

Pass `--no-checkpoint` to disable. Storage cost is roughly
`4 KB × num_items` for a 1024-dim model like bge-m3 (≈ 4 GB per million
items). Unlike earlier versions, the checkpoint is **not** auto-deleted
after the final artifact is written — it stays so the next
`--partial-index` run can reuse it. Delete `.checkpoint/` manually
once you're done growing the index.

The linear, hipporag, and graphrag builders still don't have
checkpointing: they delegate the embedding loop to vendored library
code, so adding it there means modifying their internals rather than a
one-flag change.

## Project layout

```
RAGSearch/
├── .env / .env.example              # canonical config
├── requirements.txt                 # single, flat
├── rag_clients.py                   # shared HTTP clients (Embedding/Reranker/LLM)
├── build_<retriever>_index.py       # five builders (CLI: --corpus, --output-dir)
├── serve_<retriever>.py             # five servers   (CLI: --index-dir, --port)
├── run_benchmark.py                 # build/serve/eval orchestrator
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

## Optional retriever extras

Two retrievers need packages that are deliberately kept out of
`requirements.txt` so the base venv stays light. Install them only when you
actually want that retriever.

### HippoRAG

The vendored HippoRAG still calls `torch` in one place
(`utils/embed_utils.retrieve_knn`) for entity-entity KNN. Install once into
the same venv:

```bash
pip install torch
```

CPU-only is fine — the function falls back to CPU when CUDA isn't present.

### GraphRAG

GraphRAG uses Microsoft's `graphrag` package and its LanceDB vector store.
Both pull in heavy transitive deps (dspy, fastlite, etc.) that the rest of
the stack doesn't need:

```bash
pip install graphrag lancedb
```

`build_graphrag_index.py` writes a `settings.yaml` into your `--output-dir`
that uses `${VAR}` substitution against the same `.env` contract — the LLM
hits `OPENAI_BASE_URL` / `OPENAI_MODEL` and the embedder hits
`EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`. No GraphRAG-specific config files
to maintain.

GraphRAG is the most LLM-expensive retriever in the set: every chunk costs
entity + relationship extraction, every community costs a summarization
pass, and every `/search` call invokes the LLM once at retrieval time.
Plan to use it against small corpora (HotpotQA-distractor, sample wikis)
unless you have a local LLM with serious throughput.

## Reranker support

If you set `RERANKER_BASE_URL` and `RERANKER_MODEL` in `.env`, retrievers
that benefit from query-time reranking (currently HypergraphRAG and
HippoRAG fact reranking) will use that endpoint via `POST /v1/rerank`.
If those vars are unset, retrievers degrade gracefully to a passthrough
ranking — no errors, no warnings beyond a single startup log line.

## Migrating from older versions

The legacy environment variable names (`URL`, `MODEL_NAME`, `LLM_BASE_URL`,
`EMBEDDING_URL`, `EMBEDDING_MODEL_NAME`, `REMOTE_MODEL_NAME`) have been
collapsed into the seven names above. Update your `.env` accordingly.

The per-retriever `requirements.txt` files, the `venvs/` per-backend
virtualenvs, the `script_api_*.py` server scripts under `GraphR1/`, the
`vllm_infer/` runner, the `lcb_runner/` LiveCodeBench evaluator, and the
`graphr1/` Search-R1 trainer have all been removed. If you depended on any
of those, pin the previous tag.

The GraphRAG retriever was originally removed in the same refactor (its
`graphrag` PyPI package pulled in dspy/litellm and other deps the base
stack avoids). It has since been added back as an **opt-in extra** —
`build_graphrag_index.py` and `serve_graphrag.py` are first-class scripts
matching the other retrievers' shape, but you must `pip install graphrag
lancedb` separately. See [Optional retriever extras](#optional-retriever-extras).
