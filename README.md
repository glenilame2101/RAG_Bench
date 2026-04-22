# RAGSearch

Graph-based retrieval and search system using GraphRAG, HippoRAG, RAPTOR, and other retrieval methods.

## Quick Start

### 1. Install Dependencies

Create a virtual environment and install common dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install base dependencies
pip install -U pip
pip install openai python-dotenv httpx aiohttp requests tqdm numpy fastapi uvicorn pydantic torch
```

For specific retriever backends:

```bash
# HippoRAG
pip install hipporag>=2.0.0a4

# GraphRAG
pip install graphrag==1.0.1

# RAPTOR
pip install faiss-cpu scikit-learn umap-learn

# Dense retrieval
pip install faiss-cpu datasets pyserini
```

### 2. Configure Environment

Create a `.env` file in the RAGSearch directory:

```bash
# LLM API (OpenAI-compatible)
URL=https://api.minimax.io/v1
MODEL_NAME=MiniMax-M2.7
OPENAI_API_KEY=your-api-key-here

# Embedding server (optional, for some retrievers)
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1
EMBEDDING_MODEL_NAME=bge-m3-Q8_0
```

### 3. Start a Retriever Server

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

### 4. Run Evaluation

```bash
# Using the benchmark script
python run_benchmark.py --retriever dense --dataset bamboogle --method graphsearch --limit 10

# Direct evaluation
cd GraphSearch
python eval.py --dataset bamboogle --method graphsearch --top_k 5
```

## Project Structure

```
RAGSearch/
├── GraphR1/                    # Retriever implementations
│   ├── script_api_*.py        # API server scripts for each retriever
│   ├── GraphRAG/              # GraphRAG implementation
│   ├── HippoRAG/              # HippoRAG implementation
│   └── raptor/                # RAPTOR implementation
├── GraphSearch/               # Evaluation scripts
│   └── eval.py                # Retrieval evaluation
├── Search-o1/                # Search agent implementation
│   └── scripts/               # Search scripts
├── dense_index/              # Dense FAISS index
├── hippo_index/              # HippoRAG index
├── raptor_index/             # RAPTOR tree index
└── run_benchmark.py          # Unified benchmark launcher
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

```bash
# Dense index
python build_dense_index.py --input_dir <corpus_dir> --output_dir dense_index

# RAPTOR tree
python build_raptor_index.py --input_dir <corpus_dir> --output_dir raptor_index

# HippoRAG
python build_hippo_index.py --input_jsonl <corpus.jsonl> --output_dir hippo_index --dataset_name <name>
```

## Running Search

Search-o1 provides an interactive search agent:

```bash
cd Search-o1
python scripts/run_search_o1.py --dataset_name bamboogle --split test --model_path <model> --bing_subscription_key <key>
```
