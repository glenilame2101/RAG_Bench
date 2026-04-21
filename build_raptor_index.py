"""
Build RAPTOR tree index using HTTP embedding server.
"""
import json
import os
import argparse
import pickle
from pathlib import Path

import sys
raptor_path = Path(__file__).parent / "GraphR1" / "raptor"
sys.path.insert(0, str(raptor_path))

import raptor
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import HTTPEmbeddingModel
from raptor.QAModels import BaseQAModel
from raptor.SummarizationModels import BaseSummarizationModel


class HTTPQAModel(BaseQAModel):
    """QA model using MiniMax API."""
    def __init__(self, api_key=None, base_url="https://api.minimax.io/v1"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url

    def answer_question(self, context, question, max_tokens=150):
        """Answer a question using MiniMax."""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            response = client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"QA error: {e}")
            return "I don't know."


class HTTPSummarizationModel(BaseSummarizationModel):
    """Summarization model using MiniMax API."""
    def __init__(self, api_key=None, base_url="https://api.minimax.io/v1"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url

    def summarize(self, context, max_tokens=150):
        """Summarize text using MiniMax."""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            response = client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Write a concise summary of the following:\n\n{context}"},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Summarization error: {e}")
            return str(context)[:500]  # Fallback to truncated context


def load_corpus_from_dir(input_dir: str) -> str:
    """Load text files from directory and concatenate."""
    texts = []
    input_path = Path(input_dir)
    for file_path in sorted(input_path.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            texts.append(content)
    return "\n\n".join(texts)


def load_corpus_from_jsonl(jsonl_path: str) -> str:
    """Load corpus from JSONL file and concatenate contents."""
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            if 'contents' in obj:
                texts.append(obj['contents'])
            elif 'text' in obj:
                texts.append(obj['text'])
    return "\n\n".join(texts)


def build_raptor_index(
    corpus_text: str,
    embedding_url: str,
    embedding_model: str,
    output_path: str,
    num_layers: int = 5,
    max_tokens: int = 100,
    summarization_length: int = 100,
):
    """Build RAPTOR tree index."""
    print(f"[RAPTOR Index Builder] Building tree...")

    # Create embedding model
    embedding = HTTPEmbeddingModel(base_url=embedding_url, model_name=embedding_model)

    # Configure
    config = RetrievalAugmentationConfig(
        embedding_model=embedding,
        summarization_model=HTTPSummarizationModel(),
        qa_model=HTTPQAModel(),
        tree_builder_type="cluster",
        tb_num_layers=num_layers,
        tb_max_tokens=max_tokens,
        tb_summarization_length=summarization_length,
        tb_cluster_embedding_model="EMB",
        tr_num_layers=num_layers,
    )

    # Create RetrievalAugmentation
    RA = RetrievalAugmentation(config=config)

    # Add documents
    print(f"[RAPTOR Index Builder] Adding documents...")
    RA.add_documents(corpus_text)

    # Save tree
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(RA.tree, f)

    print(f"[RAPTOR Index Builder] Tree saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build RAPTOR tree index")
    parser.add_argument("--input_dir", type=str, help="Directory with text files")
    parser.add_argument("--input_jsonl", type=str, help="JSONL file with corpus")
    parser.add_argument("--output_path", type=str, required=True, help="Output pickle path")
    parser.add_argument("--embedding_url", type=str, default="http://127.0.0.1:8080/v1", help="Embedding server URL")
    parser.add_argument("--embedding_model", type=str, default="bge-m3-Q8_0", help="Embedding model name")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of tree layers")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens per node")
    parser.add_argument("--summarization_length", type=int, default=100, help="Summarization length")
    args = parser.parse_args()

    # Load corpus
    if args.input_dir:
        corpus_text = load_corpus_from_dir(args.input_dir)
    elif args.input_jsonl:
        corpus_text = load_corpus_from_jsonl(args.input_jsonl)
    else:
        raise ValueError("Must specify either --input_dir or --input_jsonl")

    print(f"[RAPTOR Index Builder] Loaded corpus with {len(corpus_text)} characters")

    # Build index
    build_raptor_index(
        corpus_text=corpus_text,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model,
        output_path=args.output_path,
        num_layers=args.num_layers,
        max_tokens=args.max_tokens,
        summarization_length=args.summarization_length,
    )


if __name__ == "__main__":
    main()
