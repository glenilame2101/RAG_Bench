import json
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from typing import List
import argparse
import os
import sys
import threading
import torch
import time
import pickle
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", os.environ.get("MINIMAX_API_KEY", "dummy-key-for-init"))

raptor_path = Path(__file__).parent / "raptor"
sys.path.insert(0, str(raptor_path))
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import HTTPEmbeddingModel
from raptor.QAModels import BaseQAModel

class DummyQAModel(BaseQAModel):
    def answer_question(self, context, question, max_tokens=150):
        return "Dummy QA model - not used for retrieval"

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default="HotpotQA", type=str, help='Data source name')
parser.add_argument('--tree_path', default=None, type=str, help='Path to pickled tree file')
parser.add_argument('--port', type=int, default=8002, help='API service port number')
parser.add_argument('--node_scale', type=int, default=1000)
parser.add_argument('--embedding_url', default=None, help='Embedding server base URL (e.g. http://127.0.0.1:8080/v1)')
parser.add_argument('--embedding_model', default=None, help='Embedding model name (e.g. bge-m3-Q8_0)')
args = parser.parse_args()

data_source = args.data_source
tree_path = args.tree_path or f"./raptor/graphrags/{data_source}"
node_scale = args.node_scale

print("[DEBUG] Raptor API LOADED")

embedding_url = args.embedding_url or "http://127.0.0.1:8080/v1"
embedding_model = args.embedding_model or "bge-m3-Q8_0"

print(f"[Load] Loading tree from: {tree_path}")
with open(tree_path, "rb") as f:
    tree = pickle.load(f)
print(f"[Load] Tree loaded successfully: {type(tree)}")

print("[Load] Creating embedding model...")
emb_model = HTTPEmbeddingModel(base_url=embedding_url, model_name=embedding_model)

print("[Load] Creating config...")
config = RetrievalAugmentationConfig(
    embedding_model=emb_model,
    qa_model=DummyQAModel(),
)

print("[Load] Initializing Raptor with tree and config...")
RA = RetrievalAugmentation(config=config, tree=tree)


def process_query_batch(query_list, ra_instance):
    start_time = time.time()
    results = []
    for q in query_list:
        context, _ = ra_instance.retrieve(question=q, top_k=2)
        results.append(context)
    end_time = time.time()
    print(f"[DEBUG] Batch retrieval time: {end_time - start_time:.2f} seconds")
    return results


def queries_to_results(queries: List[str]) -> List[str]:
    batch_result = process_query_batch(queries, RA)
    results = []
    for item in batch_result:
        results.append(json.dumps({"results": item}))
    return results


app = FastAPI(
    title="Search API of Raptor (Batch Style API, Single Query Engine)",
    description="Raptor API that mimics HippoRAG batch API style but uses single-query retrieval."
)

class SearchRequest(BaseModel):
    queries: List[str]

@app.get("/status")
def status():
    return {"status": "ok", "retriever": "raptor"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/search")
def search(request: SearchRequest):
    return queries_to_results(request.queries)


if __name__ == "__main__":
    print(f"Starting Raptor API service, listening on port: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)