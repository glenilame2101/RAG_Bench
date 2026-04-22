# RAGSearch

A small set of retrieval-only RAG backends — Dense (FAISS), HippoRAG,
RAPTOR, HypergraphRAG, and LinearRAG — that all talk to your own
OpenAI-compatible HTTP endpoints (LLM, embeddings, optional reranker).

There is **no vLLM, no HuggingFace download, no transformers, no
sentence-transformers, no per-retriever virtual environment**. One Python
venv, one `requirements.txt`, one `.env` file with seven variables.

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
python build_dense_index.py     --corpus ./mycorpus.jsonl --output-dir ./indexes/dense
python build_raptor_index.py    --corpus ./mycorpus.jsonl --output-dir ./indexes/raptor
python build_hypergraph_index.py --corpus ./mycorpus.jsonl --output-dir ./indexes/hyper
python build_hipporag_index.py  --corpus ./mycorpus.jsonl --output-dir ./indexes/hippo
python build_linear_index.py    --corpus ./mycorpus.jsonl --output-dir ./indexes/linear --name mydata

# 6. Serve the index (same path you built into)
python serve_dense.py     --index-dir ./indexes/dense  --port 8306
python serve_raptor.py    --index-dir ./indexes/raptor --port 8346
python serve_hypergraph.py --index-dir ./indexes/hyper --port 8336
python serve_hipporag.py  --index-dir ./indexes/hippo  --port 8316
python serve_linear.py    --index-dir ./indexes/linear --name mydata --port 8356

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

## Available retrievers

| Retriever  | Default port | Builder                       | Server                  |
|------------|--------------|-------------------------------|-------------------------|
| dense      | 8306         | `build_dense_index.py`        | `serve_dense.py`        |
| raptor     | 8346         | `build_raptor_index.py`       | `serve_raptor.py`       |
| hypergraph | 8336         | `build_hypergraph_index.py`   | `serve_hypergraph.py`   |
| hipporag   | 8316         | `build_hipporag_index.py`     | `serve_hipporag.py`     |
| linear     | 8356         | `build_linear_index.py`       | `serve_linear.py`       |

Every server exposes:

- `POST /search`  — body `{"queries": ["..."]}`, returns `[{"results": "..."}, ...]`
- `GET  /status`  — `{"status": "ok", "retriever": "<name>", ...}`

`serve_dense.py` additionally keeps `POST /retrieve` for compatibility with
the Search-o1 client scripts.

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
`vllm_infer/` runner, the `lcb_runner/` LiveCodeBench evaluator, the
`graphr1/` Search-R1 trainer, and the GraphRAG retriever (the `graphrag`
PyPI package) have all been removed. If you depended on any of those, pin
the previous tag.
