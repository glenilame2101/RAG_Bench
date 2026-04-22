# RAGSearch

Unified benchmark for retrieval-augmented search agents against graph/linear/dense retriever backends.

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

### Working Retrievers

| Retriever | Port | Status | Notes |
|-----------|------|--------|-------|
| Dense | 8306 | WORKS | FAISS index, full benchmark support |
| RAPTOR | 8346 | WORKS | Server starts, eval may timeout with tiny corpus |
| HypergraphRAG | 8336 | WORKS | Uses pre-computed FAISS indexes |
| HippoRAG | 8316 | WORKS | Server starts after fixes, eval may timeout |

### Retrievers Needing Attention

| Retriever | Status | Issue |
|-----------|--------|-------|
| LinearRAG | BROKEN | Missing spaCy model `en_core_web_trf`, entity embeddings empty |
| GraphRAG | BROKEN | MiniMax doesn't follow structured JSON output for entity extraction |

## Quick Start

```bash
# Test all retrievers with tiny test dataset
python run_benchmark.py --retriever dense --dataset bamboogle --limit 5
python run_benchmark.py --retriever raptor --dataset bamboogle --limit 5
python run_benchmark.py --retriever hypergraphrag --dataset bamboogle --limit 5
python run_benchmark.py --retriever hipporag --dataset bamboogle --limit 5
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -U pip
pip install -e .

# Per-retriever venvs (for dependency isolation)
python -m venv venvs/linearrag
python -m venv venvs/hipporag

# Install retriever-specific deps
pip install -e ".[hipporag]"   # HippoRAG
pip install -e ".[linearrag]"   # LinearRAG
pip install -e ".[raptor]"      # RAPTOR

# Embedding server (llama.cpp + BGE model)
# Start before running benchmarks
```

## Configure Environment

Create `.env` in repo root:

```bash
URL=https://api.minimax.io/v1
MODEL_NAME=MiniMax-M2.7
OPENAI_API_KEY=your-key-here

# Embedding server (llama.cpp + BGE model)
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1
EMBEDDING_MODEL_NAME=bge-m3-Q8_0
```

## Benchmark CLI

```bash
# Dense retriever
python run_benchmark.py --retriever dense --dataset bamboogle --limit 10

# RAPTOR
python run_benchmark.py --retriever raptor --dataset bamboogle --limit 10

# LinearRAG
python run_benchmark.py --retriever linearrag --dataset bamboogle --limit 10

# HypergraphRAG
python run_benchmark.py --retriever hypergraphrag --dataset bamboogle --limit 10

# HippoRAG
python run_benchmark.py --retriever hipporag --dataset bamboogle --limit 10
```

### Arguments

- `--retriever, -r`: dense, graphrag, hipporag, raptor, linearrag, hypergraphrag
- `--dataset, -d`: bamboogle, hotpotqa, musique, nq, 2wikimultihopqa, triviaqa, popqa, aime, math500
- `--method, -m`: vanilla (no retrieval), naive (simple retrieval), graphsearch (graph-based)
- `--limit, -l`: Number of samples (-1 for all, default: -1)
- `--top_k, -k`: Top-k for retrieval (default: 5)
- `--concurrency, -c`: Concurrent requests (default: 10)

## Index Builders

```bash
# Dense FAISS index
python build_dense_index.py --input_dir <corpus_dir> --output_dir dense_index

# RAPTOR tree index
python build_raptor_index.py --input_dir <corpus_dir> --output_dir raptor_index

# LinearRAG graph index
python rebuild_linear_index.py --input_jsonl <corpus.jsonl> --output_dir linear_index --dataset_name bamboogle

# HippoRAG index
python build_hippo_index.py --input_jsonl <corpus.jsonl> --output_dir hippo_index --dataset_name bamboogle
```

## Retriever Servers

Start servers manually:

```bash
# Dense (port 8306)
python serve_dense.py --index_path dense_index/dense_index.faiss --corpus_path dense_index/corpus.jsonl --port 8306

# RAPTOR (port 8346)
python GraphR1/script_api_RAPTOR.py --tree_path raptor_index/tree.pkl --port 8346 --embedding_url http://127.0.0.1:8080/v1 --embedding_model bge-m3-Q8_0

# LinearRAG (port 8356)
venvs/linearrag/Scripts/python.exe GraphR1/script_api_LinearRAG.py --data_source bamboogle --port 8356 --working_dir linear_index

# HypergraphRAG (port 8336)
python GraphR1/script_api_HypergraphRAG.py --data_source bamboogle --port 8336

# HippoRAG (port 8316)
venvs/hipporag/Scripts/python.exe GraphR1/script_api_HippoRAG.py --data_source bamboogle --port 8316
```

## Directory Structure

```
RAGSearch/
├── GraphR1/                    # Retriever implementations
│   ├── HippoRAG/              # HippoRAG source
│   ├── LinearRAG/             # LinearRAG source
│   └── script_api_*.py        # API server scripts
├── GraphSearch/               # Evaluation script
├── Search-o1/                # Benchmark datasets
├── dense_index/              # Dense FAISS index
├── linear_index/             # LinearRAG index
├── hypergraph_index/         # HypergraphRAG index (backup)
├── graphrags/bamboogle/      # HypergraphRAG index (canonical)
├── hippo_index/              # HippoRAG index
├── raptor_index/             # RAPTOR index
├── venvs/                    # Per-retriever virtual environments
├── build_*.py                # Index builders
├── rebuild_linear_index.py    # LinearRAG index rebuild
├── run_benchmark.py          # Unified benchmark CLI
└── serve_dense.py            # Dense retriever server
```

## Fixes Applied (April 2026)

### HippoRAG Module Name Bug Fixed
**Problem:** `prompt_template_manager.py` had hardcoded `Hipporag` instead of `HippoRAG`.

**Fix:** `GraphR1/HippoRAG/src/hipporag/prompts/prompt_template_manager.py:69`
```python
# Before
module = importlib.import_module(module_name, 'Hipporag.src.hipporag')
# After
module = importlib.import_module(module_name, 'HippoRAG.src.hipporag')
```

### HippoRAG Index Path Fixed
**Problem:** Server looked at `HippoRAG/Graphrags/{data_source}/hipporag` instead of `hippo_index/{data_source}/hipporag`.

**Fix:** `GraphR1/script_api_HippoRAG.py`
```python
# Before
save_dir=f"./HippoRAG/Graphrags/{data_source}/hipporag"
# After
save_dir=f"{REPO_ROOT}/hippo_index/{data_source}/hipporag"
```

### LinearRAG Index Path Fixed
**Problem:** Server looked at `LinearRAG/graphrags` instead of `linear_index`.

**Fix:** `GraphR1/script_api_LinearRAG.py`
```python
# Before
parser.add_argument('--working_dir', default="./LinearRAG/graphrags")
# After
parser.add_argument('--working_dir', default=f"{REPO_ROOT}/linear_index")
```

### HypergraphRAG Venv Fixed
**Problem:** `run_benchmark.py` referenced non-existent `venvs/hypergraphrag`.

**Fix:** Changed to use `.venv` in `run_benchmark.py` RETRIEVER_CONFIG.

### Windows Unicode Fix
**Problem:** Emoji characters (🚀, 💾) in print statements caused `UnicodeEncodeError` on Windows.

**Fix:** Replaced with ASCII text or set `PYTHONIOENCODING=utf-8`.

### Evaluation Script Created
**Problem:** `GraphSearch/eval.py` was deleted during cleanup.

**Fix:** Created minimal `GraphSearch/eval.py` for benchmark evaluation.

## Known Issues

1. **LinearRAG**: Requires spaCy model `en_core_web_trf` (not installed), entity embeddings are empty
2. **GraphRAG**: MiniMax doesn't follow structured JSON output for entity extraction
3. **Eval timeouts**: RAPTOR and HippoRAG may timeout during evaluation with tiny corpus due to slow LLM calls
4. **Small corpus**: Test corpus has only 2 documents (Einstein, Curie) - too small for meaningful benchmarks

## Legacy Cleanup

Removed the following legacy directories:
- `Search-R1/` - Unused research project
- `test_graphrag/` - Test artifacts
- `docs/` - Legacy documentation
- `src/ragsearch_bench/` - Old CLI wrapper (superseded by `run_benchmark.py`)
- `HippoRAG/Graphrags/` - Duplicate index location
