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

`--corpus` accepts either a JSONL file (one JSON document per line, with any
of `contents`, `text`, `content`, `document`, `body`) or a directory of
`*.txt` files. Nothing is hardcoded — every path comes in via the CLI.

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

Useful for smoke-tests, parameter sweeps, or staging a small index before
committing to a full multi-hour run. For JSONL the builder does a fast
streaming line-count first (no JSON parsing), then loads only the first
`N%` of lines. For directories it takes the first `N%` of `*.txt` files
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

### Checkpointing (raptor)

`build_raptor_index.py` checkpoints embeddings to disk **every 1%** of
total chunks by default, so a crashed or interrupted run does not throw
away the embeddings already computed. Re-run the same command to resume.

```bash
# First run — crashes at 47%.
python build_raptor_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/raptor --batch-size 64

# Same command again — resumes from the last 1% boundary, then finishes.
python build_raptor_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/raptor --batch-size 64
```

Layout under `<output-dir>/.checkpoint/`:

```
state.json           # {model, total, completed, fingerprint, ...}
emb_000000.npy       # one slice of saved embeddings (float32, normalized)
emb_000001.npy
...
```

After `tree.pkl` is written successfully the `.checkpoint/` directory is
auto-deleted.

Safety guards:

- A SHA-256 fingerprint of `(model name, total count, first/middle/last
  text)` is stored in `state.json`. If you re-run with a different model
  or a different corpus, the script refuses to mix incompatible vectors
  and tells you to delete the directory.
- If the on-disk count of embeddings disagrees with `state.json`
  (interrupted mid-flush), the script errors out instead of producing a
  corrupt array.
- A SIGINT/Ctrl-C between flushes loses at most the last <1% of work.

Pass `--no-checkpoint` to disable. Storage cost is roughly
`4 KB × num_chunks` for a 1024-dim model like bge-m3 (≈ 4 GB per million
chunks) at peak, freed once `tree.pkl` is written. Make sure
`--output-dir` lives on a drive with room.

The dense, hypergraph, linear, hipporag, and graphrag builders do not yet
have checkpointing — linear/hipporag/graphrag delegate the embedding loop
to vendored library code, so adding it there means modifying their
internals rather than a one-flag change.

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
