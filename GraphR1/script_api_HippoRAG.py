import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import argparse
import os
import threading
import torch
import time
from tqdm import tqdm

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "GraphR1"))
from HippoRAG.src.hipporag import HippoRAG
# =====================================================

# ================== Parameter parsing =========================
parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='2WikiMultiHopQA')
parser.add_argument('--port', type=int, default=8000, help='API service port number')
parser.add_argument('--node_scale', type=int, default=1000)
parser.add_argument('--embedding_url', default=None, help='Embedding server base URL (e.g. http://127.0.0.1:8080/v1)')
parser.add_argument('--embedding_model', default=None, help='Embedding model name (e.g. bge-m3-Q8_0)')
args = parser.parse_args()

data_source = args.data_source
node_scale = args.node_scale

# ================== Environment variable reading =====================
embedding_base_url = args.embedding_url or os.environ.get("EMBEDDING_BASE_URL", "")
embedding_model_name = args.embedding_model or os.environ.get("EMBEDDING_MODEL_NAME", "")
llm_model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
llm_base_url = os.environ.get("LLM_BASE_URL", None)

if embedding_base_url and embedding_model_name:
    embedding_model_name = f"VLLM/{embedding_model_name}"
else:
    embedding_model_name = "nvidia/NV-Embed-v2"
    embedding_base_url = None

print("[DEBUG] HippoRAG2 LOADED")




print("[Load] Initializing HippoRAG... (this may take time)")
rag = HippoRAG(
    save_dir=f"{REPO_ROOT}/hippo_index/{data_source}/hipporag",
    llm_model_name=llm_model_name,
    llm_base_url=llm_base_url,
    embedding_model_name=embedding_model_name,
    embedding_base_url=embedding_base_url
)

print("[Load] HippoRAG model initialized.")

# === 加载完成后关闭 heartbeat ===
# stop_heartbeat()

# =====================================================


# ====================== Batch processing version ======================

def process_query_batch(query_list, rag_instance):
    """Process all queries in batch - single retrieval"""
    results = rag_instance.retrieve(
        queries=query_list,
        num_to_retrieve=5
    )
    return results


def queries_to_results(queries: List[str]) -> List[str]:
    """Perform single retrieval for entire batch of queries"""

    batch_result = process_query_batch(queries, rag)
    results = []

    for i, q in enumerate(queries):
        item = batch_result[i]
        retrieve_result = ""

        # Output top 5 passages
        for passage in item.docs[:5]:
            retrieve_result += f"Wikipedia Title: {passage}\n\n"

        results.append(json.dumps({"results": retrieve_result}))

    return results


# ======================= API service =======================

app = FastAPI(
    title="Search API of HippoRAG (Batch Mode)",
    description="An API for document retrieval using HippoRAG (batch accelerated)."
)

class SearchRequest(BaseModel):
    queries: List[str]

@app.post("/search")
def search(request: SearchRequest):
    return queries_to_results(request.queries)


if __name__ == "__main__":
    print(f"Starting API service, listening on port: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
