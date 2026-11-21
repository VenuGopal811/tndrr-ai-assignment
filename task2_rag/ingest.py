"""PDF -> pages -> word chunks (150 and 400 words, 50 overlap) -> bge-small embeddings -> FAISS index.

No RAG framework: pypdf for text, a plain loop for chunking, sentence-transformers for vectors, faiss for search.

Usage:  python task2_rag/ingest.py
Writes  task2_rag/index/chunks_{size}.jsonl, task2_rag/index/faiss_{size}.index, task2_rag/results/ingest_stats.json
"""
import json
import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rag_common import (CHUNK_SIZES, EMBED_MODEL, INDEX_DIR, OVERLAP, PDF_PATH, RESULTS_DIR,
                        chunks_path, faiss_path)

HEADER = "NIST AI 100-1 AI RMF 1.0"   # running header printed at the top of every page


def clean_page(text):
    text = text.replace(HEADER, " ")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)   # "envi-\nronment" -> "environment"
    return " ".join(text.split())


def load_words():
    """All words of the document in order, each tagged with its PDF page number (1-based)."""
    words = []
    for page_no, page in enumerate(PdfReader(PDF_PATH).pages, start=1):
        for w in clean_page(page.extract_text() or "").split():
            words.append((w, page_no))
    return words


def make_chunks(words, size, overlap):
    """Sliding window over the whole document. A chunk can cross a page break, so we keep every page it
    touches; `page` is the page it starts on (used for citations)."""
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        pages = sorted({p for _, p in window})
        chunks.append({"id": len(chunks), "page": pages[0], "pages": pages,
                       "text": " ".join(w for w, _ in window)})
        if start + size >= len(words):   # last window already reaches the end
            break
    return chunks


def main():
    INDEX_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    words = load_words()
    n_pages = len(PdfReader(PDF_PATH).pages)
    print(f"{n_pages} pages, {len(words)} words")

    model = SentenceTransformer(EMBED_MODEL)
    stats = {"pdf": PDF_PATH.name, "pages": n_pages, "words": len(words), "embed_model": EMBED_MODEL,
             "overlap_words": OVERLAP, "chunk_sizes": {}}
    for size in CHUNK_SIZES:
        chunks = make_chunks(words, size, OVERLAP)
        # Normalised vectors + inner product = cosine similarity. Passages get no prefix (per bge model card).
        vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                            batch_size=32, show_progress_bar=False).astype(np.float32)
        index = faiss.IndexFlatIP(vecs.shape[1])   # exact search; the corpus is tiny, no approximation needed
        index.add(vecs)
        faiss.write_index(index, str(faiss_path(size)))
        with open(chunks_path(size), "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        stats["chunk_sizes"][str(size)] = {"chunks": len(chunks), "dim": int(vecs.shape[1])}
        print(f"size {size}: {len(chunks)} chunks, dim {vecs.shape[1]}")

    (RESULTS_DIR / "ingest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
