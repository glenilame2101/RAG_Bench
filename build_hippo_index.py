"""
Build HippoRAG index from text files or JSONL corpus.
"""
import os
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "GraphR1" / "HippoRAG"))

from src.hipporag import HippoRAG


def load_corpus_from_dir(input_dir: str) -> list:
    """Load text files from directory and return list of documents."""
    docs = []
    input_path = Path(input_dir)
    for file_path in sorted(input_path.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            docs.append(content)
    return docs


def load_corpus_from_jsonl(jsonl_path: str) -> list:
    """Load corpus from JSONL file."""
    docs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                text = doc.get("contents", "").strip()
                if text:
                    docs.append(text)
            except Exception as e:
                logger.warning(f"Error reading line: {e}")
                continue
    return docs


def build_hippo_index(
    docs: list,
    output_dir: str,
    embedding_model: str = "VLLM/bge-m3-Q8_0",
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = None,
    embedding_base_url: str = None,
    batch_size: int = 8,
):
    """Build HippoRAG index."""
    logger.info(f"Building HippoRAG index with {len(docs)} documents...")

    hippo = HippoRAG(
        save_dir=output_dir,
        llm_model_name=llm_model,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model,
        embedding_base_url=embedding_base_url,
    )

    logger.info("Indexing documents...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        hippo.index(docs=batch)
        logger.info(f"Indexed {min(i + batch_size, len(docs))}/{len(docs)} docs")

    logger.info(f"✅ Index built successfully at: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build HippoRAG index")
    parser.add_argument("--input_dir", type=str, help="Directory with text files")
    parser.add_argument("--input_jsonl", type=str, help="JSONL file with corpus")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="corpus", help="Dataset name")
    parser.add_argument("--embedding_model", type=str, default="VLLM/bge-m3-Q8_0", help="Embedding model name")
    parser.add_argument("--embedding_url", type=str, default=None, help="Embedding server URL")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--llm_base_url", type=str, default=None, help="LLM base URL")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for indexing")
    args = parser.parse_args()

    if args.input_dir:
        docs = load_corpus_from_dir(args.input_dir)
    elif args.input_jsonl:
        docs = load_corpus_from_jsonl(args.input_jsonl)
    else:
        raise ValueError("Must specify --input_dir or --input_jsonl")

    logger.info(f"Loaded {len(docs)} documents")

    output_dir = os.path.join(args.output_dir, args.dataset_name, "hipporag")
    os.makedirs(output_dir, exist_ok=True)

    build_hippo_index(
        docs=docs,
        output_dir=output_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        embedding_base_url=args.embedding_url,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()