#!/bin/bash
# filepath: GraphSearch-main/run_universal.sh
# GraphSearch evaluation script (local/interactive run, supports multiple Retrievers + vLLM + eval)

# ================================
# Default parameter settings
# ================================
METHOD=""
DATASET=""
START=0
END=-1
TOP_K=5
CONCURRENCY=1
EVAL_METHOD="graphsearch"   # graphsearch | naive | dense
REMOTE_LLM=""              # "", "1"/"true" to force remote; auto-detected from .env URL

PROJECT_DIR="./"
RETRIEVER_CODE_BASE="../Graph-R1"
# Legacy conda root, used as a fallback when .venv is absent.
CONDA_ROOT="../conda_env"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Preferred activation: the single repo-root .venv. Falls back to conda
# if .venv is not present.
_activate_venv() {
  local root="$1"
  local conda_env="$2"
  if [ -f "${root}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "${root}/.venv/bin/activate"
    return 0
  fi
  if command -v conda &>/dev/null && [ -n "$conda_env" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "$conda_env" 2>/dev/null || return 1
    return 0
  fi
  return 1
}

# ================================
# Load .env from repo root (if present) to pick up URL / MODEL_NAME /
# OPENAI_API_KEY for remote OpenAI-compatible endpoints.
# ================================
_find_dotenv() {
  local dir
  dir="$(cd "$(dirname "$0")" && pwd)"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/.env" ]; then
      echo "$dir/.env"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}
ENV_FILE="$(_find_dotenv || true)"
if [ -n "$ENV_FILE" ]; then
  echo "Loading environment from: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# ================================
# Command line argument parsing
# ================================
usage() {
  echo "Usage: $0 -m METHOD -d DATASET [-p RETRIEVER_PORT] [-s START] [-e END] [-k TOP_K] [-c CONCURRENCY] [-x EVAL_METHOD]"
  echo "  -m: Retriever method (hypergraphrag, hipporag2, linearrag, graphrag, raptor, dense)"
  echo "  -d: Dataset (hotpotqa, musique, 2wikimultihopqa, nq, triviaqa, popqa)"
  echo "  -p: Retriever port (optional; if not set, use method default); vLLM port is retriever+1"
  echo "  -s: Start sample index (default: 0)"
  echo "  -e: End sample index (default: -1 means all)"
  echo "  -k: top_k (default: 5)"
  echo "  -c: Concurrency (default: 1)"
  echo "  -x: Evaluation mode graphsearch|naive|dense (default: graphsearch)"
  echo "  -R: Force remote OpenAI-compatible LLM endpoint (default)."
  echo "      Reads URL / MODEL_NAME / OPENAI_API_KEY from .env."
  echo "  -L: Opt in to launching a local GPU vLLM server instead of remote."
  echo "      Requires the 'train' extra (pip install -e '.[train]') and a GPU."
  exit 1
}

PORT=""
while getopts "m:d:p:s:e:k:c:x:RLh" opt; do
  case $opt in
    m) METHOD=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    p) PORT=$OPTARG ;;
    s) START=$OPTARG ;;
    e) END=$OPTARG ;;
    k) TOP_K=$OPTARG ;;
    c) CONCURRENCY=$OPTARG ;;
    x) EVAL_METHOD=$OPTARG ;;
    R) REMOTE_LLM=1 ;;
    L) LOCAL_LLM=1 ;;
    h) usage ;;
    *) usage ;;
  esac
done

if [ -z "$METHOD" ] || [ -z "$DATASET" ]; then
  echo "❌ Error: -m METHOD and -d DATASET are required"
  usage
fi

# ================================
# Retriever environment and script routing
# ================================
RETRIEVER_EXTRA_ARGS=""
case "$METHOD" in
  hypergraphrag)
    RETRIEVER_ENV_PATH="$CONDA_ROOT/graphr1"
    RETRIEVER_SCRIPT="script_api_HypergraphRAG.py"
    RETRIEVER_PORT=8336
    ;;
  hipporag2)
    RETRIEVER_ENV_PATH="$CONDA_ROOT/hipporag"
    RETRIEVER_SCRIPT="script_api_HippoRAG.py"
    RETRIEVER_PORT=8316
    RETRIEVER_EXTRA_ARGS="--node_scale 5000"
    ;;
  linearrag)
    RETRIEVER_ENV_PATH="$CONDA_ROOT/linearrag"
    RETRIEVER_SCRIPT="script_api_LinearRAG.py"
    RETRIEVER_PORT=8356
    ;;
  graphrag)
    RETRIEVER_ENV_PATH="$CONDA_ROOT/graphrag"
    RETRIEVER_SCRIPT="script_api_GraphRAG.py"
    RETRIEVER_PORT=8326
    ;;
  raptor)
    RETRIEVER_ENV_PATH="$CONDA_ROOT/raptor"
    RETRIEVER_SCRIPT="script_api_RAPTOR.py"
    RETRIEVER_PORT=8346
    ;;
  dense)
    RETRIEVER_ENV_PATH="/scratch/df2362/conda_env/retriever"
    RETRIEVER_LAUNCH_SCRIPT="ANN_retrieval_launch.sh"   # use this script under Search-R1 to start dense search
    RETRIEVER_PORT=8306
    RETRIEVER_CODE_BASE="/scratch/df2362/Search-R1"
    ;;
  *)
    echo "❌ Error: Unsupported method: $METHOD"
    echo "Supported: hypergraphrag, hipporag2, linearrag, graphrag, raptor, dense"
    exit 1
    ;;
esac

# If -p is provided, override the default retriever port for this method
if [ -n "$PORT" ]; then
  RETRIEVER_PORT=$PORT
fi
# vLLM port = Retriever port + 1
VLLM_PORT=$((RETRIEVER_PORT + 1))

# ================================
# Decide whether to run a local vLLM server or use a remote endpoint.
# Default is **remote** (the whole point of the single-venv refactor); pass
# -L / --local-llm to opt in to launching a local GPU vLLM server, or -R
# to force remote (useful when overriding a previously-exported LOCAL flag).
# ================================
if [ -n "$REMOTE_LLM" ]; then
  USE_LOCAL_VLLM=0
elif [ "${LOCAL_LLM:-0}" = "1" ]; then
  USE_LOCAL_VLLM=1
else
  # default: remote. Warn if neither URL nor LOCAL_LLM is set.
  USE_LOCAL_VLLM=0
  if [ -z "${URL:-}" ]; then
    echo "⚠️  No URL in .env and -L not set; remote calls will fail."
    echo "    Copy .env.example to .env and fill in URL / MODEL_NAME / OPENAI_API_KEY,"
    echo "    or re-run with -L to launch a local vLLM server (requires GPU)."
  fi
fi

# ================================
# Dataset name mapping (eval lowercase -> Retriever CamelCase)
# ================================
case "$DATASET" in
  hotpotqa)       SOURCE_NAME="HotpotQA" ;;
  musique)        SOURCE_NAME="Musique" ;;
  2wikimultihopqa) SOURCE_NAME="2WikiMultiHopQA" ;;
  nq)             SOURCE_NAME="NQ" ;;
  triviaqa)       SOURCE_NAME="TriviaQA" ;;
  popqa)          SOURCE_NAME="PopQA" ;;
  *)              SOURCE_NAME="$DATASET" ;;
esac

# ================================
# Python paths and log directory
# ================================
# Prefer the unified .venv; fall back to the legacy per-retriever conda env
# when .venv is absent.
_activate_venv "$REPO_ROOT" "${RETRIEVER_ENV_PATH}" || \
    echo "⚠️  Falling back to current Python -- no .venv and retriever conda env missing."
PYTHON_VLLM="$(command -v python)"
PYTHON_RETRIEVER="$PYTHON_VLLM"
# If the legacy per-retriever conda env exists, override the retriever python
# so heavy retriever deps (spacy/faiss/etc.) resolve when the single venv
# does not have the matching extra installed.
if [ -x "${RETRIEVER_ENV_PATH}/bin/python" ] && [ ! -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  PYTHON_RETRIEVER="${RETRIEVER_ENV_PATH}/bin/python"
fi
# Log directory: the folder where this script resides; log file names include port and method
LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_LOG="$LOG_DIR/vllm_${VLLM_PORT}_${METHOD}.log"
RETRIEVER_LOG="$LOG_DIR/retriever_${RETRIEVER_PORT}_${METHOD}.log"
EVAL_LOG="$LOG_DIR/eval_${METHOD}_start_${START}_end_${END}.log"
echo "📝 Log directory: $LOG_DIR"

# ================================
# Print configuration summary
# ================================
echo "========================================="
echo "GraphSearch evaluation launcher"
echo "========================================="
echo "Method:        $METHOD"
echo "Dataset:       $DATASET -> $SOURCE_NAME"
echo "Eval mode:     $EVAL_METHOD"
echo "Start/End:     $START / $END"
echo "Top-K:         $TOP_K"
echo "Concurrency:   $CONCURRENCY"
echo "Retriever env: $RETRIEVER_ENV_PATH"
echo "Retriever:     ${RETRIEVER_LAUNCH_SCRIPT:-$RETRIEVER_SCRIPT} (port $RETRIEVER_PORT)"
echo "vLLM port:     $VLLM_PORT (retriever+1)"
if [ "$USE_LOCAL_VLLM" = "1" ]; then
  echo "LLM backend:   local vLLM server (will be launched)"
else
  echo "LLM backend:   remote OpenAI-compatible endpoint"
  echo "  URL:         ${URL:-<unset>}"
  echo "  MODEL_NAME:  ${MODEL_NAME:-<unset>}"
fi
# Same as Step 3 export, for display only
if [ "$METHOD" = "dense" ]; then
  echo "RETRIEVER_URL:  http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
else
  echo "RETRIEVER_URL:  http://127.0.0.1:${RETRIEVER_PORT}/search"
fi
echo "========================================="
echo ""

# ================================
# Cleanup function (PID + ports fallback, similar to transfer_graphr1.sh)
# ================================
cleanup() {
  echo
  echo "🧹 Cleaning up services (close port $RETRIEVER_PORT, $VLLM_PORT)..."

  if [ -n "${RETRIEVER_PID:-}" ] && kill -0 "$RETRIEVER_PID" 2>/dev/null; then
    kill "$RETRIEVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$RETRIEVER_PID" 2>/dev/null || true
    echo "  ✓ Retriever (port $RETRIEVER_PORT) stopped"
  fi

  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$VLLM_PID" 2>/dev/null || true
    echo "  ✓ vLLM (port $VLLM_PORT) stopped"
  fi

  fuser -k ${RETRIEVER_PORT}/tcp 2>/dev/null || true
  fuser -k ${VLLM_PORT}/tcp 2>/dev/null || true
  echo "  ✓ Ports cleared"
}
trap cleanup EXIT INT TERM

# ================================
# Load environment (if on cluster)
# ================================
if command -v module &>/dev/null; then
  module purge 2>/dev/null || true
  module load anaconda3/2024.02 2>/dev/null || true
  module load cuda/11.6.2 2>/dev/null || true
  eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

# ================================
# Step 1: Start vLLM
# ================================
echo "=== Step 1: Start vLLM Server ==="
if [ "$USE_LOCAL_VLLM" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=0
  nohup $PYTHON_VLLM -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --trust-remote-code \
    --port $VLLM_PORT \
    --gpu-memory-utilization 0.70 \
    --max-model-len 32768 \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  echo "  vLLM PID: $VLLM_PID"
else
  echo "  Skipping local vLLM launch (using remote endpoint: ${URL:-unset})"
fi

# ================================
# Step 2: Start Retriever
# ================================
echo "=== Step 2: Start Retriever ($METHOD) ==="
cd "$RETRIEVER_CODE_BASE"
if [ "$METHOD" = "dense" ]; then
  # dense: use ANN_retrieval_launch.sh (must ensure python inside uses retriever env)
  nohup env PATH="$RETRIEVER_ENV_PATH/bin:$PATH" bash "$RETRIEVER_LAUNCH_SCRIPT" -p $RETRIEVER_PORT \
    > "$RETRIEVER_LOG" 2>&1 &
else
  RETRIEVER_CMD="--data_source $SOURCE_NAME --port $RETRIEVER_PORT"
  if [ -n "${EMBEDDING_BASE_URL:-}" ]; then
    RETRIEVER_CMD="$RETRIEVER_CMD --embedding_url $EMBEDDING_BASE_URL"
  fi
  if [ -n "${EMBEDDING_MODEL_NAME:-}" ]; then
    RETRIEVER_CMD="$RETRIEVER_CMD --embedding_model $EMBEDDING_MODEL_NAME"
  fi
  nohup $PYTHON_RETRIEVER $RETRIEVER_SCRIPT \
    $RETRIEVER_CMD \
    $RETRIEVER_EXTRA_ARGS \
    > "$RETRIEVER_LOG" 2>&1 &
fi
RETRIEVER_PID=$!
echo "  Retriever PID: $RETRIEVER_PID"
cd "$PROJECT_DIR"

if [ "$USE_LOCAL_VLLM" = "1" ]; then
  echo "⏳ Wait 300s for services to be ready..."
  sleep 300
else
  echo "⏳ Wait 30s for retriever to be ready (skipping vLLM wait)..."
  sleep 30
fi

# ================================
# Step 3: Run evaluation
# ================================
echo "=== Step 3: Run evaluation ==="
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
if [ "$USE_LOCAL_VLLM" = "1" ]; then
  export LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
else
  # Remote endpoint from .env. Normalize: strip trailing slash, then ensure /v1 suffix.
  _BASE_URL="${URL%/}"
  case "$_BASE_URL" in
    */v1) ;;
    *) _BASE_URL="${_BASE_URL}/v1" ;;
  esac
  export LLM_BASE_URL="$_BASE_URL"
  export LLM_API_KEY="${OPENAI_API_KEY:-}"
  export REMOTE_MODEL_NAME="${MODEL_NAME:-}"
fi
# Retriever URL is set dynamically: dense uses /retrieve, others use /search
if [ "$METHOD" = "dense" ]; then
  export RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
else
  export RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}/search"
fi
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-}"
export EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-}"

$PYTHON_VLLM eval.py \
  --dataset "$DATASET" \
  --graphrag "$METHOD" \
  --method "$EVAL_METHOD" \
  --top_k $TOP_K \
  --concurrency $CONCURRENCY \
  --start $START \
  --end $END

EXIT_CODE=$?

# ================================
# Step 4: Cleanup (stop Retriever / vLLM processes and ports)
# ================================
echo "Job finished, exit code: $EXIT_CODE"
cleanup
exit $EXIT_CODE
