"""Top-k search over one chunk-size index.

Usage:  python task2_rag/retrieve.py "what does the GOVERN function cover?" --size 150 --k 5
"""
import argparse
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_common import EMBED_MODEL, QUERY_PREFIX, chunks_path, faiss_path


class Retriever:
    def __init__(self, size, model=None):
        self.size = size
        self.model = model or SentenceTransformer(EMBED_MODEL)   # pass a model in to share it across sizes
        self.index = faiss.read_index(str(faiss_path(size)))
        with open(chunks_path(size), encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

    def search(self, query, k=5):
        """Returns [(score, chunk), ...] best first. Score = cosine similarity."""
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(q, k)
        return [(float(s), self.chunks[i]) for s, i in zip(scores[0], ids[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    for rank, (score, c) in enumerate(Retriever(args.size).search(args.query, args.k), start=1):
        print(f"#{rank} score={score:.3f} page={c['page']} chunk={c['id']}\n   {c['text'][:300]}...\n")


if __name__ == "__main__":
    main()
