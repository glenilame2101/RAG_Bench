"""Build a HypergraphRAG index using the OpenAI-compatible embeddings endpoint.

Produces (in --output-dir):
    index_entity.bin
    index_hyperedge.bin
    kv_store_entities.json
    kv_store_hyperedges.json

Usage:
    python build_hypergraph_index.py --corpus <path> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from rag_clients import EmbeddingClient, load_env


TEXT_FIELDS = ("contents", "text", "content", "document", "body")


def load_corpus(corpus_path: str) -> List[str]:
    path = Path(corpus_path)
    if path.is_dir():
        out = []
        for file_path in sorted(path.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                out.append(content)
        return out
    if not path.is_file():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    out = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = None
            for field in TEXT_FIELDS:
                if field in doc:
                    text = doc[field]
                    break
            if text is None:
                text = (doc.get("question", "") + " " + doc.get("answer", "")).strip()
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            text = str(text).strip()
            if text:
                out.append(text)
    return out


def extract_entities_and_hyperedges(texts: List[str]) -> Tuple[List[str], List[str]]:
    entities: List[str] = []
    hyperedges: List[str] = []
    seen = set()
    for text in texts:
        words = [w for w in text.replace(".", " ").replace(",", " ").split() if len(w) > 3]
        for i in range(len(words)):
            for j in range(i + 1, min(i + 8, len(words) + 1)):
                phrase = " ".join(words[i:j])
                if len(phrase) > 5 and phrase not in seen:
                    seen.add(phrase)
                    entities.append(phrase)
        for sentence in text.split("."):
            sentence = sentence.strip()
            if len(sentence) > 20:
                hyperedges.append(sentence)
    return list(dict.fromkeys(entities))[:1000], list(dict.fromkeys(hyperedges))[:500]


def build_hypergraph_index(
    texts: List[str],
    output_dir: Path,
    embedder: EmbeddingClient,
    batch_size: int,
) -> None:
    print(f"[Hypergraph] Extracting entities + hyperedges from {len(texts)} docs...")
    entity_list, hyperedge_list = extract_entities_and_hyperedges(texts)
    if not entity_list:
        entity_list = texts[:100]
    if not hyperedge_list:
        hyperedge_list = texts[:50]
    print(f"[Hypergraph] {len(entity_list)} entities, {len(hyperedge_list)} hyperedges")

    print(f"[Hypergraph] Embedding via {embedder.base_url} ({embedder.model})")
    entity_emb = embedder.encode(entity_list, batch_size=batch_size, normalize=True)
    hyper_emb = embedder.encode(hyperedge_list, batch_size=batch_size, normalize=True)
    if entity_emb.size == 0:
        raise RuntimeError("No entity embeddings generated")

    dim = int(entity_emb.shape[1])
    entity_index = faiss.IndexFlatIP(dim)
    entity_index.add(entity_emb.astype(np.float32))
    hyper_index = faiss.IndexFlatIP(dim)
    hyper_index.add(hyper_emb.astype(np.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(entity_index, str(output_dir / "index_entity.bin"))
    faiss.write_index(hyper_index, str(output_dir / "index_hyperedge.bin"))

    kv_entities = {str(i): {"entity_name": e} for i, e in enumerate(entity_list)}
    kv_hyperedges = {str(i): {"content": h} for i, h in enumerate(hyperedge_list)}
    (output_dir / "kv_store_entities.json").write_text(
        json.dumps(kv_entities, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "kv_store_hyperedges.json").write_text(
        json.dumps(kv_hyperedges, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Hypergraph] Wrote index files to {output_dir}")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus or directory of .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory for hypergraph index files")
    parser.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    texts = load_corpus(args.corpus)
    if not texts:
        raise SystemExit(f"No documents loaded from {args.corpus}")
    print(f"[Hypergraph] Loaded {len(texts)} documents from {args.corpus}")

    embedder = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    build_hypergraph_index(
        texts=texts,
        output_dir=Path(args.output_dir),
        embedder=embedder,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
