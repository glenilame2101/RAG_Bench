# Text Embeddings Inference (TEI)

[Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
(TEI) is HuggingFace's high-performance embedding server. It's a strong
self-hosted choice for this stack because it:

- Exposes an **OpenAI-compatible `/v1/embeddings` endpoint**, so it drops
  straight into the `EMBEDDING_BASE_URL` contract — no client changes.
- Runs on CPU or GPU (CUDA, ROCm, Metal).
- Supports the popular open embedding models out of the box: `bge-m3`,
  `bge-large`, `e5`, `gte`, `nomic-embed-text`, `qwen3-embedding`, etc.
- Also exposes `/rerank`, so the same container can serve the reranker
  endpoint (`RERANKER_BASE_URL`) for HypergraphRAG and HippoRAG.

## Run TEI with Docker

The simplest way to get started — pick the image that matches your hardware.

### CPU

```bash
docker run -p 8080:80 \
    -v $PWD/tei-data:/data \
    --pull always \
    ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
    --model-id BAAI/bge-m3
```

### NVIDIA GPU

```bash
docker run --gpus all -p 8080:80 \
    -v $PWD/tei-data:/data \
    --pull always \
    ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id BAAI/bge-m3
```

The `-v $PWD/tei-data:/data` mount caches the model weights so subsequent
starts skip the download.

## Point the stack at TEI

Set these two variables in `.env`:

```bash
EMBEDDING_BASE_URL=http://localhost:8080
EMBEDDING_MODEL=BAAI/bge-m3
```

The model name must match what TEI was launched with (`--model-id`).

That's all. Every builder and server reads these from the shared `.env`
contract. Verify the wiring with a one-liner:

```bash
curl http://localhost:8080/embed \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "hello world"}'
```

## Use TEI as a reranker too

TEI can also host a reranker model and exposes a compatible `/rerank`
endpoint. Run a second container on a different port:

```bash
docker run --gpus all -p 8081:80 \
    -v $PWD/tei-rerank-data:/data \
    ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id BAAI/bge-reranker-v2-m3
```

Then add to `.env`:

```bash
RERANKER_BASE_URL=http://localhost:8081
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

HypergraphRAG and HippoRAG will pick this up automatically. Without
these vars, they degrade gracefully to passthrough ranking — no errors.

## Tuning batch size for TEI

TEI is fast and accepts large batches. For indexing a multi-GB corpus,
raise `--batch-size` on any builder:

```bash
python build_dense_index.py --corpus ./corpus.jsonl \
    --output-dir ./indexes/dense \
    --batch-size 128
```

If TEI returns 4xx, lower the value or raise TEI's own
`--max-batch-tokens` / `--max-client-batch-size` flags. The
`EmbeddingClient` uses a 300-second timeout and retries up to 3 times on
network errors (with exponential backoff).

## Why not vLLM / llama.cpp / Ollama for embeddings?

All three work — they each expose an OpenAI-compatible
`/v1/embeddings` endpoint and slot into the same `.env` contract. TEI is
the recommended default because it's purpose-built for embeddings (no
generation overhead), supports rerankers natively, and tends to have the
best throughput on the small embedding models this stack uses.

If you already run llama.cpp or vLLM for the LLM side, just spin up an
extra TEI process for embeddings — keeping the two concerns on separate
ports avoids resource contention.
