"""
HypergraphRAG API Server - Simplified version using pre-computed FAISS indexes.
"""
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='bamboogle')
parser.add_argument('--port', type=int, default=8336)
parser.add_argument('--embedding_model', default='sentence-transformers/all-mpnet-base-v2')
args = parser.parse_args()
data_source = args.data_source

print(f"Loading model: {args.embedding_model}")
model = SentenceTransformer(args.embedding_model, device="cpu")

base_dir = f"./graphrags/{data_source}/hypergraphrag"

print(f"Loading entity index from {base_dir}")
index_entity = faiss.read_index(f"{base_dir}/index_entity.bin")
with open(f"{base_dir}/kv_store_entities.json") as f:
    entities = json.load(f)
corpus_entity = [entities[item]['entity_name'] for item in entities]
print(f"Loaded {len(corpus_entity)} entities")

print(f"Loading hyperedge index from {base_dir}")
index_hyperedge = faiss.read_index(f"{base_dir}/index_hyperedge.bin")
with open(f"{base_dir}/kv_store_hyperedges.json") as f:
    hyperedges = json.load(f)
corpus_hyperedge = [hyperedges[item]['content'] for item in hyperedges]
print(f"Loaded {len(corpus_hyperedge)} hyperedges")


app = FastAPI(title="HypergraphRAG API", description="Document retrieval using HypergraphRAG with pre-computed FAISS indexes.")

class SearchRequest(BaseModel):
    queries: List[str]

@app.post("/search")
def search(request: SearchRequest):
    embeddings = model.encode(request.queries, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    _, entity_ids = index_entity.search(embeddings, 5)
    _, hyperedge_ids = index_hyperedge.search(embeddings, 5)

    results = []
    for i, query in enumerate(request.queries):
        entity_results = [corpus_entity[idx] for idx in entity_ids[i] if idx < len(corpus_entity)]
        hyperedge_results = [corpus_hyperedge[idx] for idx in hyperedge_ids[i] if idx < len(corpus_hyperedge)]

        context = "\n".join(entity_results[:3]) + "\n---\n" + "\n".join(hyperedge_results[:2])
        results.append({"results": context})

    return results

@app.get("/status")
def status():
    return {"status": "ok", "retriever": "hypergraphrag", "entities": len(corpus_entity), "hyperedges": len(corpus_hyperedge)}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print(f"Starting HypergraphRAG API on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)