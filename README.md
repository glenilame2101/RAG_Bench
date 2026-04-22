# RAGSearch

Graph-based retrieval and search system using GraphRAG, HippoRAG, RAPTOR, and other retrieval methods.

## Overview

This is a simplified, focused version of RAGSearch that provides:
- **Graph search** using GraphRAG, HippoRAG, HypergraphRAG
- **Retrieval** using RAPTOR, Dense (FAISS), LinearRAG
- **Search agents** via Search-o1

No RL training or benchmark infrastructure included.

## Quick Start

### 1. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Upgrade pip
pip install -U pip
```

### 2. Install Dependencies

**Core dependencies:**
```bash
pip install openai python-dotenv httpx aiohttp requests tqdm numpy fastapi uvicorn pydantic torch
```

**For specific retrievers:**
```bash
# HippoRAG
pip install hipporag>=2.0.0a4 python-igraph networkx scipy

# RAPTOR
pip install faiss-cpu scikit-learn umap-learn

# Dense retrieval
pip install faiss-cpu datasets pyserini

# GraphRAG
pip install graphrag==1.0.1
```

### 3. Configure Environment

Create a `.env` file in the RAGSearch directory:

```bash
# LLM API (OpenAI-compatible)
URL=https://api.minimax.io/v1
MODEL_NAME=MiniMax-M2.7
OPENAI_API_KEY=your-api-key-here

# Embedding server (optional)
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1
EMBEDDING_MODEL_NAME=bge-m3-Q8_0
```

### 4. Run a Retriever Server

Start a retriever server in one terminal:

**Dense (FAISS):**
```bash
python serve_dense.py --index_path dense_index/dense_index.faiss --corpus_path dense_index/corpus.jsonl --port 8306
```

**HippoRAG:**
```bash
python GraphR1/script_api_HippoRAG.py --data_source bamboogle --port 8316
```

**RAPTOR:**
```bash
python GraphR1/script_api_RAPTOR.py --tree_path raptor_index/tree.pkl --port 8346
```

**HypergraphRAG:**
```bash
python GraphR1/script_api_HypergraphRAG.py --data_source bamboogle --port 8336
```

**GraphRAG:**
```bash
python GraphR1/script_api_GraphRAG.py --project_dir GraphR1/GraphRAG/inputs --port 8326
```

### 5. Run Evaluation

With a retriever server running, run evaluation:

```bash
python run_benchmark.py --retriever hypergraphrag --dataset bamboogle --method graphsearch --limit 10
```

Or use the GraphSearch eval directly:

```bash
cd GraphSearch
python eval.py --dataset bamboogle --method graphsearch --top_k 5
```

## Project Structure

```
RAGSearch/
├── GraphR1/                           # Retriever implementations
│   ├── script_api_*.py                # API server scripts
│   ├── GraphRAG/                      # GraphRAG implementation
│   ├── HippoRAG/                      # HippoRAG implementation
│   ├── raptor/                        # RAPTOR implementation
│   └── LinearRAG/                     # LinearRAG implementation
├── GraphSearch/                       # Evaluation scripts
│   └── eval.py                        # Retrieval evaluation
├── Search-o1/                         # Search agent
│   └── scripts/                       # Search scripts
├── dense_index/                       # Dense FAISS index
├── hippo_index/                       # HippoRAG index
├── raptor_index/                      # RAPTOR tree index
├── run_benchmark.py                   # Benchmark launcher
└── serve_dense.py                    # Dense retriever server
```

## Available Retrievers

| Retriever | Port | Description |
|-----------|------|-------------|
| dense | 8306 | FAISS dense retrieval |
| graphrag | 8326 | Graph-based retrieval |
| hipporag | 8316 | Hippocampus RAG |
| raptor | 8346 | Tree-structured retrieval |
| hypergraphrag | 8336 | Hypergraph RAG |
| linearrag | 8356 | Linear graph RAG |

## Building Indexes

**Dense index:**
```bash
python build_dense_index.py --input_dir <corpus_dir> --output_dir dense_index
```

**RAPTOR tree:**
```bash
python build_raptor_index.py --input_dir <corpus_dir> --output_dir raptor_index
```

**HippoRAG index:**
```bash
python build_hippo_index.py --input_jsonl <corpus.jsonl> --output_dir hippo_index --dataset_name <name>
```

## Adding Datasets

Place corpus files in `Search-o1/data/`:

```
Search-o1/data/mycorpus/
├── doc1.txt
├── doc2.txt
```

Or as JSONL:
```
Search-o1/data/mycorpus.jsonl   # {"id": "...", "contents": "..."}
```

## Architecture

```
                    +------------------+
                    |  LLM API         |
                    | (OpenAI-compat) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Retriever API   |
                    |  (per backend)  |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v------+   +--------v--------+   +-----v-----+
|  Dense FAISS |   |  GraphRAG       |   |  RAPTOR   |
|  Port: 8306  |   |  Port: 8326     |   |  Port:8346|
+--------------+   +-----------------+   +-----------+
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `URL` | LLM API base URL | Required |
| `MODEL_NAME` | LLM model name | Required |
| `OPENAI_API_KEY` | API key | Required |
| `EMBEDDING_BASE_URL` | Embedding server URL | Optional |
| `EMBEDDING_MODEL_NAME` | Embedding model | Optional |
