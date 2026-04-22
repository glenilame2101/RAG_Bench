"""
Build HypergraphRAG index using sentence-transformers.

This creates the required FAISS indexes and KV stores for HypergraphRAG:
- index_entity.bin (FAISS index for entities)
- index_hyperedge.bin (FAISS index for hyperedges)
- kv_store_entities.json (entity data)
- kv_store_hyperedges.json (hyperedge content)
"""
import json
import os
import argparse
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def extract_entities_and_hyperedges(texts: list) -> tuple:
    """Extract entities and hyperedges from texts using simple NLP."""
    entities = []
    hyperedges = []
    entity_texts = set()

    for text in texts:
        words = text.replace(".", " ").replace(",", " ").split()
        words = [w for w in words if len(w) > 3]

        for i in range(len(words)):
            for j in range(i + 1, min(i + 8, len(words) + 1)):
                phrase = " ".join(words[i:j])
                if len(phrase) > 5 and phrase not in entity_texts:
                    entities.append(phrase)
                    entity_texts.add(phrase)

        sentences = text.split(".")
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:
                hyperedges.append(sent)

    return list(set(entities))[:1000], list(set(hyperedges))[:500]


def build_hypergraph_index(
    texts: list,
    output_dir: str,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    device: str = "cpu",
):
    """Build HypergraphRAG index."""
    print(f"[HypergraphRAG] Building index for {len(texts)} texts...")

    print(f"[HypergraphRAG] Loading sentence-transformers model: {embedding_model}")
    model = SentenceTransformer(embedding_model, device=device)

    print("[HypergraphRAG] Extracting entities and hyperedges...")
    entity_list, hyperedge_list = extract_entities_and_hyperedges(texts)

    if not entity_list:
        entity_list = texts[:100]
    if not hyperedge_list:
        hyperedge_list = texts[:50]

    print(f"[HypergraphRAG] {len(entity_list)} entities, {len(hyperedge_list)} hyperedges")

    print("[HypergraphRAG] Encoding entities...")
    entity_embeddings = model.encode(entity_list, batch_size=32, normalize_embeddings=True)

    print("[HypergraphRAG] Encoding hyperedges...")
    hyperedge_embeddings = model.encode(hyperedge_list, batch_size=32, normalize_embeddings=True)

    print("[HypergraphRAG] Building FAISS indexes...")
    dim = entity_embeddings.shape[1]
    entity_index = faiss.IndexFlatIP(dim)
    entity_index.add(entity_embeddings.astype(np.float32))

    hyperedge_index = faiss.IndexFlatIP(dim)
    hyperedge_index.add(hyperedge_embeddings.astype(np.float32))

    print("[HypergraphRAG] Saving indexes and KV stores...")
    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(entity_index, os.path.join(output_dir, "index_entity.bin"))
    faiss.write_index(hyperedge_index, os.path.join(output_dir, "index_hyperedge.bin"))

    kv_entities = {str(i): {"entity_name": e, "embedding": entity_embeddings[i].tolist()} for i, e in enumerate(entity_list)}
    with open(os.path.join(output_dir, "kv_store_entities.json"), "w") as f:
        json.dump(kv_entities, f, ensure_ascii=False)

    kv_hyperedges = {str(i): {"content": h, "embedding": hyperedge_embeddings[i].tolist()} for i, h in enumerate(hyperedge_list)}
    with open(os.path.join(output_dir, "kv_store_hyperedges.json"), "w") as f:
        json.dump(kv_hyperedges, f, ensure_ascii=False)

    print(f"[HypergraphRAG] Index built successfully at: {output_dir}")
    print(f"[HypergraphRAG] Files created:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


def load_corpus_from_jsonl(jsonl_path: str) -> list:
    """Load corpus from JSONL file."""
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                text = doc.get("contents", "").strip()
                if text:
                    texts.append(text)
            except Exception:
                continue
    return texts


def load_corpus_from_dir(input_dir: str) -> list:
    """Load text files from directory."""
    texts = []
    input_path = Path(input_dir)
    for file_path in sorted(input_path.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            texts.append(content)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Build HypergraphRAG index")
    parser.add_argument("--input_dir", type=str, help="Directory with text files")
    parser.add_argument("--input_jsonl", type=str, help="JSONL file with corpus")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5", help="FlagEmbedding model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if args.input_jsonl:
        texts = load_corpus_from_jsonl(args.input_jsonl)
    elif args.input_dir:
        texts = load_corpus_from_dir(args.input_dir)
    else:
        raise ValueError("Must specify --input_jsonl or --input_dir")

    print(f"[HypergraphRAG] Loaded {len(texts)} documents")

    build_hypergraph_index(
        texts=texts,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()