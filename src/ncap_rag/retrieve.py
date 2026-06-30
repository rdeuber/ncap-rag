from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncap_rag.ingest import CHUNKS_PATH, EMBEDDING_MODEL_NAME, INDEX_PATH


def load_chunks() -> list[dict]:
    """Load chunk metadata and text from disk."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_index() -> faiss.Index:
    """Load FAISS index from disk."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    return faiss.read_index(str(INDEX_PATH))


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most relevant chunks for a query."""
    chunks = load_chunks()
    index = load_index()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"Top {len(results)} retrieved chunks:\n")

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source_file']}, page {result['page_number']}")
        print("-" * 80)
        print(result["text"][:1200])
        print()


if __name__ == "__main__":
    main()
