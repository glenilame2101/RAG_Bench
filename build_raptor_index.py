"""Build a RAPTOR tree index using the OpenAI-compatible embeddings endpoint.

Usage:
    python build_raptor_index.py --corpus <path> --output-dir <dir>

The output directory will contain `tree.pkl` (a pickled RaptorTree).
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

from rag_clients import EmbeddingClient, load_env


TEXT_FIELDS = ("contents", "text", "content", "document", "body")


def load_corpus(corpus_path: str, partial_pct: Optional[float] = None) -> List[str]:
    path = Path(corpus_path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if partial_pct is not None:
            n = max(1, int(len(files) * partial_pct / 100))
            print(f"[Raptor] --partial-index {partial_pct}%: using {n} of {len(files)} files")
            files = files[:n]
        out = []
        for file_path in tqdm(files, desc="[Raptor] Loading files", unit="file"):
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                out.append(content)
        return out
    if not path.is_file():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    target_lines: Optional[int] = None
    if partial_pct is not None:
        print(f"[Raptor] --partial-index {partial_pct}%: counting lines in {path}...")
        with path.open("r", encoding="utf-8") as fh:
            total = sum(1 for _ in fh)
        target_lines = max(1, int(total * partial_pct / 100))
        print(f"[Raptor] Loading first {target_lines} of {total} lines")

    out = []
    with path.open("r", encoding="utf-8") as fh:
        bar = tqdm(fh, desc="[Raptor] Loading corpus", unit=" line", total=target_lines)
        for i, raw in enumerate(bar):
            if target_lines is not None and i >= target_lines:
                break
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


class TreeNode:
    def __init__(self, node_id: str, content: str, level: int = 0, children: Optional[List[str]] = None):
        self.node_id = node_id
        self.content = content
        self.level = level
        self.children = children or []
        self.embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "content": self.content, "level": self.level, "children": self.children}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TreeNode":
        node = cls(d["node_id"], d["content"], d.get("level", 0), d.get("children", []))
        if "embedding" in d:
            node.embedding = np.array(d["embedding"])
        return node


class RaptorTree:
    def __init__(self):
        self.nodes: Dict[str, TreeNode] = {}
        self.root_ids: List[str] = []

    def add_node(self, node: TreeNode):
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self.nodes.get(node_id)

    def set_root(self, node_ids: List[str]):
        self.root_ids = node_ids


def chunk_text(text: str, max_tokens: int = 100) -> List[str]:
    sentences = text.split(".")
    chunks: List[str] = []
    current_chunk = ""
    current_tokens = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent = sent + "."
        tokens = len(sent.split())
        if current_tokens + tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent
            current_tokens = tokens
        else:
            current_chunk += " " + sent
            current_tokens += tokens
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def build_raptor_tree(
    texts: List[str],
    embedder: EmbeddingClient,
    num_layers: int,
    max_tokens: int,
    cluster_size: int,
    batch_size: int,
    checkpoint_dir: Optional[Path] = None,
) -> RaptorTree:
    print(f"[Raptor] Chunking {len(texts)} documents...")
    all_chunks = [chunk for text in texts for chunk in chunk_text(text, max_tokens)]
    if not all_chunks:
        raise ValueError("No chunks created from input texts")

    print(f"[Raptor] Embedding {len(all_chunks)} chunks via {embedder.base_url} ({embedder.model})")
    if checkpoint_dir is not None:
        embeddings = embedder.encode_with_checkpoint(
            all_chunks,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            normalize=True,
            save_every_pct=1.0,
        )
    else:
        embeddings = embedder.encode(all_chunks, batch_size=batch_size, normalize=True)

    node_id_counter = 0
    level_nodes: Dict[int, List[TreeNode]] = defaultdict(list)

    leaf_nodes: List[TreeNode] = []
    for i, chunk in enumerate(all_chunks):
        node = TreeNode(f"node_{node_id_counter}", chunk, level=0)
        node.embedding = embeddings[i]
        node_id_counter += 1
        leaf_nodes.append(node)
        level_nodes[0].append(node)

    current_nodes = leaf_nodes
    current_embeddings = embeddings
    current_level = 0

    while current_level < num_layers - 1 and len(current_nodes) > cluster_size:
        n_clusters = max(1, len(current_nodes) // cluster_size)
        print(f"[Raptor] Clustering level {current_level}: {len(current_nodes)} -> {n_clusters}")
        # Full KMeans is O(n * k * d * iters * n_init) and gets prohibitive past
        # ~10k samples (e.g. 40k chunks at d=1024 with k=4k can take hours).
        # MiniBatchKMeans is the standard sklearn fix for large n — same family
        # of algorithm, processes data in mini-batches, near-identical quality.
        if len(current_nodes) > 10_000:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=3,
                batch_size=4096,
                max_iter=100,
            )
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(current_embeddings)

        new_nodes: List[TreeNode] = []
        new_embeddings: List[np.ndarray] = []
        for cluster_id in range(n_clusters):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) == 0:
                continue
            cluster_node_objs = [current_nodes[i] for i in indices]
            cluster_emb = current_embeddings[indices]
            summary = " ".join(n.content for n in cluster_node_objs[:5])
            if len(summary) > 500:
                summary = summary[:500] + "..."
            summary_emb = cluster_emb.mean(axis=0)
            norm = float(np.linalg.norm(summary_emb))
            if norm > 0:
                summary_emb = summary_emb / norm
            node = TreeNode(f"node_{node_id_counter}", summary, level=current_level + 1)
            node.embedding = summary_emb
            node.children = [n.node_id for n in cluster_node_objs]
            node_id_counter += 1
            for n in cluster_node_objs:
                n.children.append(node.node_id)
            new_nodes.append(node)
            new_embeddings.append(summary_emb)

        level_nodes[current_level + 1].extend(new_nodes)
        current_nodes = new_nodes
        current_embeddings = np.array(new_embeddings)
        current_level += 1

    tree = RaptorTree()
    for nodes in level_nodes.values():
        for node in nodes:
            tree.add_node(node)
    root_nodes = current_nodes if current_nodes else leaf_nodes[: min(10, len(leaf_nodes))]
    tree.set_root([n.node_id for n in root_nodes])
    print(f"[Raptor] Tree built: {len(tree.nodes)} nodes, {len(tree.root_ids)} roots")
    return tree


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus or directory of .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory for tree.pkl")
    parser.add_argument("--embedding-base-url", default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--cluster-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--partial-index",
        type=float,
        default=None,
        help="Index only the first N%% of the corpus (e.g., 10 = first 10%%)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable embedding checkpointing (default: checkpoint every 1%% to <output-dir>/.checkpoint/)",
    )
    args = parser.parse_args()

    if args.partial_index is not None and not (0 < args.partial_index <= 100):
        raise SystemExit("--partial-index must be in (0, 100]")

    texts = load_corpus(args.corpus, partial_pct=args.partial_index)
    if not texts:
        raise SystemExit(f"No documents loaded from {args.corpus}")
    print(f"[Raptor] Loaded {len(texts)} documents from {args.corpus}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = None if args.no_checkpoint else output_dir / ".checkpoint"

    embedder = EmbeddingClient(base_url=args.embedding_base_url, model=args.embedding_model)
    tree = build_raptor_tree(
        texts=texts,
        embedder=embedder,
        num_layers=args.num_layers,
        max_tokens=args.max_tokens,
        cluster_size=args.cluster_size,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
    )

    tree_path = output_dir / "tree.pkl"
    with tree_path.open("wb") as fh:
        pickle.dump(tree, fh)
    print(f"[Raptor] Wrote {tree_path}")

    if checkpoint_dir is not None and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"[Raptor] Cleaned up checkpoint dir {checkpoint_dir}")


if __name__ == "__main__":
    main()
