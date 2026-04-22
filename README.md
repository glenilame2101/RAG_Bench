# RAGSearch

Unified benchmark for retrieval-augmented search agents (Search-o1,
GraphSearch, and friends) against a suite of graph/linear/dense retriever
backends.

## Architecture Overview

```
                    +------------------+
                    |  MiniMax API     |
                    | (LLM Endpoint)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  run_benchmark.py |
                    |  (Unified CLI)   |
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
+-------+--------+   +--------+--------+   +-----+----+
|  Dense (FAISS)|   |   RAPTOR        |   | GraphRAG |
|  Port: 8306   |   |   Port: 8346    |   | Port: 8326|
+---------------+   +-----------------+   +----------+

Embedding Server: localhost:8080 (llama.cpp + BGE model)
```

## Current Status (April 2026)

### Working Retrievers ✅

| Retriever | Port | Index Status | Notes |
|-----------|------|--------------|-------|
| Vanilla | N/A | N/A | LLM-only, no retrieval |
| Dense | 8306 | ✅ Built | FAISS index, works fully |
| RAPTOR | 8346 | ✅ Built | Tree-based index, works fully |
| LinearRAG | 8356 | ✅ Rebuilt | Graph index rebuilt with `rebuild_linear_index.py` |

### Retrievers Needing Attention 🔧

| Retriever | Status | Issue |
|-----------|--------|-------|
| HippoRAG | Blocked | Requires `vllm` package in `venvs/hipporag` for index building |
| HypergraphRAG | No index | No build script exists - uses FlagEmbedding + pre-computed FAISS |
| GraphRAG | No index | Entity extraction fails with MiniMax (LLM doesn't follow structured output) |

### Per-Retriever Virtual Environments

Due to dependency conflicts, each retriever may use its own venv:

```
venvs/
├── linearrag/     # For LinearRAG (spacy, torch, faiss, sentence-transformers)
└── hipporag/      # For HippoRAG (vllm, litellm, torch, openai, transformers)
```

## Installation

**One virtualenv, one `pip install`, one `.env` file.** The benchmark
clients talk to a remote OpenAI-compatible LLM endpoint (e.g. a hosted vLLM
server) by default — no local CUDA, no local vLLM, no conda.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Base benchmark (Search-o1 + GraphSearch clients, no retriever extras)
pip install -e .

# Add the retriever backends you actually need:
pip install -e ".[hipporag]"             # HippoRAG2
pip install -e ".[graphrag]"            # Microsoft GraphRAG
pip install -e ".[linearrag]"           # LinearRAG (spaCy, igraph)
pip install -e ".[raptor]"              # RAPTOR
pip install -e ".[dense]"               # Dense retriever (faiss, sbert, pyserini)
# ...or combine them:
pip install -e ".[hipporag,graphrag,dense]"

cp .env.example .env
# Edit .env and fill in URL, MODEL_NAME, OPENAI_API_KEY
```

### RL training (optional, GPU-only)

Only needed to **reproduce the trained checkpoints** of Search-R1 /
Graph-R1. The benchmark itself does not require any of this.

```bash
pip install -e ".[train]"   # pulls vllm, flash-attn, ray, accelerate, wandb
```

The legacy 8-conda-env recipe is preserved in
[`docs/legacy-conda.md`](docs/legacy-conda.md).

## Configure the remote LLM endpoint

Create `.env` in the repo root (see `.env.example`):

```bash
URL=https://api.minimax.io/v1
MODEL_NAME=MiniMax-M2.7
OPENAI_API_KEY=your-key-here

# Embedding server (llama.cpp + BGE model)
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1
EMBEDDING_MODEL_NAME=bge-m3-Q8_0
```

## Benchmark CLI Usage

Use `run_benchmark.py` to run comparisons across all retrievers:

```bash
# Vanilla (no retrieval - LLM only baseline)
python run_benchmark.py --retriever vanilla --dataset bamboogle --method vanilla --limit 10

# Dense retriever
python run_benchmark.py --retriever dense --dataset bamboogle --method naive --limit 10

# RAPTOR (working)
python run_benchmark.py --retriever raptor --dataset bamboogle --method graphsearch --limit 10

# LinearRAG (working - graph index rebuilt)
python run_benchmark.py --retriever linearrag --dataset bamboogle --method graphsearch --limit 10
```

### Arguments

- `--retriever, -r`: dense, graphrag, hipporag, raptor, linearrag, hypergraphrag, vanilla
- `--dataset, -d`: bamboogle, hotpotqa, musique, nq, 2wikimultihopqa, triviaqa, popqa, aime, math500
- `--method, -m`: vanilla (no retrieval), naive (simple retrieval), graphsearch (graph-based)
- `--limit, -l`: Number of samples to test (-1 for all, default: -1)
- `--top_k, -k`: Top-k for retrieval (default: 5)
- `--concurrency, -c`: Concurrent requests (default: 10)

## Index Builders

Build indexes before running benchmarks:

```bash
# Dense FAISS index
python build_dense_index.py --input_dir test_graphrag/input --output_dir dense_index

# RAPTOR tree index
python build_raptor_index.py --input_dir test_graphrag/input --output_dir raptor_index

# LinearRAG graph index (rebuilt successfully)
python rebuild_linear_index.py --input_jsonl dense_index/corpus.jsonl --output_dir linear_index --dataset_name bamboogle

# HippoRAG index (blocked - needs vllm dependency)
python build_hippo_index.py --input_jsonl dense_index/corpus.jsonl --output_dir hippo_index --dataset_name bamboogle
```

## Retriever Server Scripts

Start servers manually if needed:

```bash
# Dense (port 8306)
python serve_dense.py --index_path dense_index/dense_index.faiss --corpus_path dense_index/corpus.jsonl --port 8306

# RAPTOR (port 8346)
python GraphR1/script_api_RAPTOR.py --tree_path raptor_index/tree.pkl --port 8346 --embedding_url http://127.0.0.1:8080/v1 --embedding_model bge-m3-Q8_0

# LinearRAG (port 8356) - requires venv
venvs/linearrag/Scripts/python.exe GraphR1/script_api_LinearRAG.py --data_source bamboogle --port 8356 --working_dir linear_index --embedding_url http://127.0.0.1:8080/v1 --embedding_model bge-m3-Q8_0 --spacy_model en_core_web_sm
```

## Known Issues & Fixes Applied (April 2026)

### LinearRAG - Index Rebuilt Successfully ✅

**Problem:** Original index build failed due to embedding server 500 errors during batch requests.

**Solution:** Created `rebuild_linear_index.py` with:
- Local `HTTPEmbeddingModel` class (no raptor import dependency)
- Single-request encoding to avoid batch errors
- Proper NER processing with spacy `en_core_web_sm`

**Files now present in `linear_index/bamboogle/`:**
- `entity_embedding.parquet`
- `entity_to_sentence.json`
- `LinearRAG.graphml`
- `ner_results.json`
- `passage_embedding.parquet`
- `passage_node_indices.json`
- `sentence_embedding.parquet`
- `sentence_to_entity.json`

**Additional fixes in `GraphR1/script_api_LinearRAG.py`:**
- Added local `HTTPEmbeddingModel` class instead of importing from non-installed `raptor.raptor`
- Replaced `🚀` emoji with text to avoid Windows Unicode encoding error
- Added `--spacy_model en_core_web_sm` since `en_core_web_trf` wasn't installed

**Dependencies installed in `venvs/linearrag`:**
- tiktoken, umap-learn, tenacity, spacy (en_core_web_sm)

### HippoRAG - Windows Multiprocessing Bug Fixed ✅ (but dependency issues remain)

**Problem:** `hipporag/embedding_model/base.py` uses `multiprocessing.Manager()` at class definition time, which fails on Windows.

**Solution:** Changed `EmbeddingCache` class to use lazy initialization:

```python
class EmbeddingCache:
    _manager = None
    _cache = None
    _lock = None

    @classmethod
    def _init_manager(cls):
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
            cls._cache = cls._manager.dict()
            cls._lock = threading.Lock()
    # ... rest of methods call _init_manager() first
```

**Remaining issue:** Building HippoRAG index requires `vllm` package which has complex dependencies. The `build_hippo_index.py` fails with `ModuleNotFoundError: No module named 'vllm'` when run with `venvs/hipporag`.

### HypergraphRAG - No Build Script Exists ⚠️

**Problem:** The `script_api_HypergraphRAG.py` loads pre-computed FAISS indexes from `./graphrags/{data_source}/hypergraphrag/`:
- `index_entity.bin`
- `index_hyperedge.bin`
- `kv_store_entities.json`
- `kv_store_hyperedges.json`

There is no build script in the repository to create these indexes. The script uses `FlagEmbedding` (BAAI/bge-large-en-v1.5) directly, not the HTTP embedding server.

### GraphRAG - Entity Extraction Fails with MiniMax ❌

**Problem:** GraphRAG's entity extraction prompt expects structured JSON output but MiniMax doesn't follow the format reliably.

**Status:** Would need either:
- Different LLM that follows structured output better
- Modified prompt engineering
- Pre-built GraphRAG index provided externally

## What Still Needs Doing

1. **HippoRAG index build** - Install `vllm` in `venvs/hipporag` or find alternative approach
2. **HypergraphRAG index build** - Create build script that uses FlagEmbedding to compute entity/hyperedge indexes
3. **GraphRAG index build** - Either use a different LLM or provide pre-built index
4. **Larger corpus** - Current test corpus is tiny (2 documents: Einstein/Curie). Need wiki-18 or similar for meaningful benchmarks

## Test Corpus

Current corpus for testing (`dense_index/corpus.jsonl`):
- `curie.txt` - Marie Curie biography
- `einstein.txt` - Albert Einstein biography

This is sufficient for basic functionality testing but too small for meaningful benchmark comparisons.

## Remote-endpoint notes and limitations

* **Tokenizer is still loaded locally** by Search-o1 (for `apply_chat_template`
  and `eos_token` stop-sequence handling). Only the tokenizer files (~5 MB)
  are needed — no model weights. If `--model-path` is not available locally,
  pass `--tokenizer-path` pointing at an equivalent Hugging Face tokenizer.
* **vLLM-specific sampling knobs** (`top_k`, `repetition_penalty`) are
  forwarded via the OpenAI SDK's `extra_body`, which the vLLM OpenAI server
  honors. Other OpenAI-compatible servers may silently ignore them.
* **RL training is unchanged.** The Search-R1 / Graph-R1 training scripts
  still need local vLLM + GPU; they live under the `train` extra.