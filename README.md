# RAGSearch

Unified benchmark for retrieval-augmented search agents (Search-o1,
GraphSearch, and friends) against a suite of graph/linear/dense retriever
backends.

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
pip install -e ".[graphrag]"             # Microsoft GraphRAG
pip install -e ".[linearrag]"            # LinearRAG (spaCy, igraph)
pip install -e ".[raptor]"               # RAPTOR
pip install -e ".[dense]"                # Dense retriever (faiss, sbert, pyserini)
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
URL=https://your.endpoint.example.com     # /v1 appended automatically
MODEL_NAME=Qwen2.5-7B-Instruct
OPENAI_API_KEY=sk-...
```

Shell scripts auto-discover `.env` by walking up from the script directory;
the Python side uses `python-dotenv`'s `find_dotenv(usecwd=True)`. Verify
with:

```bash
ragbench env-check
```

## Quick start

One command per agent:

```bash
# Search-o1 + HippoRAG2 on HotpotQA, remote LLM endpoint
ragbench run search-o1 --retriever hipporag2 --dataset HotpotQA --port 8205

# GraphSearch + GraphRAG on HotpotQA, remote LLM endpoint
ragbench run graphsearch --retriever graphrag --dataset hotpotqa --port 8205

# Score a predictions file
ragbench eval --dataset hotpotqa --predictions out.json
```

Add `--local-llm` to either `run` subcommand to launch a local GPU vLLM
server instead of calling the remote endpoint (requires the `train` extra).

The existing shell wrappers still work for anyone who prefers them; they
now activate `.venv` first and fall back to conda only if `.venv` is
missing:

```bash
cd Search-o1
bash run_search_o1_graphrag.sh -g hipporag -d HotpotQA \
    -m Qwen/Qwen2.5-7B-Instruct -p 8205

cd GraphSearch
bash run_universal.sh -m hipporag2 -d hotpotqa -p 8205
```

## Offline GraphRAG construction

This is an example of constructing GraphRAG on HippoRAG2; other retrieve
backends follow a similar process.

- Step 1 — set your OpenAI API key in `.env` (see above).
- Step 2 — build the GraphRAG with `build_hipporag_increment.py`.

## RL training quick-start (optional)

### Graph-R1
```bash
cd GraphR1
# -a is the retriever server port number; -d is the training dataset;
# -p is the backbone path; -m is the backbone name;
# -r is the retriever backend; -b is the batch size; -g is num GPUs.
bash run_graphr1.sh -a 8007 -d NQ-HotpotQA -p Qwen/Qwen2.5-7B-Instruct \
    -m Qwen2.5-7B-Instruct -r hipporag2 -b 128 -g 4
# inference
bash infer_graphr1.sh -a 8503 -t NQ-HotpotQA -d PopQA -r hipporag2 -g 1 \
    -m your/pre-trained-model-path
```

### Search-R1
```bash
cd Search-R1
nohup bash ANN_retrieval_launch.sh -p 8605 > retriever_8605.log 2>&1 &
bash train_grpo.sh -p 8605 -d nq-hotpotqa
# inference
python run_eval_vllm.py \
    --model_id /your/pre-trained-model-path \
    --dataset_path /path-to-questions.json \
    --output_file /path-to-answers.json \
    --max_turns 5 \
    --retriever_port 8205 \
    --tensor_parallel_size 1
```

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
