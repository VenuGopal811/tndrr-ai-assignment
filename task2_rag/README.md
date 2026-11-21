# Task 2: RAG from scratch

Document: NIST AI Risk Management Framework 1.0 (NIST AI 100-1), stored as `data/nist_ai_rmf.pdf`.
Source URL: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

No RAG frameworks (no LangChain, LlamaIndex, Haystack, txtai, DSPy, vector-DB SDKs). Only pypdf, numpy, faiss-cpu, sentence-transformers and plain HTTP for the LLM.

## Files

| File | What it does |
|---|---|
| `rag_common.py` | Shared settings (paths, chunk sizes, model name, query prefix) and text normalisation |
| `ingest.py` | PDF -> pages -> word chunks (150 and 400 words, 50 overlap) -> bge-small embeddings -> FAISS index |
| `retrieve.py` | Top-k search for a query |
| `evaluate.py` | Recall@1/3/5 per chunk size on `eval/questions.json` |
| `answer.py` | Retrieve + LLM answer with cited chunks |
| `eval/questions.json` | 15 questions written by Venu |
| `results/` | `recall.json`, `failures.md` |

## Method

- **Parsing:** pypdf, page by page. The running header "NIST AI 100-1 AI RMF 1.0" is removed and words split by a line-break hyphen are re-joined.
- **Chunking:** sliding window of 150 or 400 words over the whole document, 50-word overlap. A chunk may cross a page break; it keeps every page it touches (`pages`) and cites the page it starts on (`page`).
- **Embeddings:** `BAAI/bge-small-en-v1.5`, normalised. Queries get the model card's prefix "Represent this sentence for searching relevant passages: "; passages get none.
- **Index:** `faiss.IndexFlatIP` (exact inner product = cosine on normalised vectors), one per chunk size.
- **Evaluation:** a top-k chunk is a hit if it contains the gold passage (normalised substring match) or covers the gold page. Recall@1/3/5 are reported for `hit` (either), `passage` only and `page` only. Page numbers are PDF page numbers (1-48), not the printed ones.

## Commands

```
python task2_rag/ingest.py                         # builds task2_rag/index/ (git-ignored) + results/ingest_stats.json
python task2_rag/retrieve.py "your query" --size 150 --k 5
python task2_rag/evaluate.py                       # needs all 15 questions in eval/questions.json
python task2_rag/evaluate.py --allow-partial       # while still writing questions
python task2_rag/evaluate.py --bm25                # optional BM25 keyword baseline on the same chunks
python task2_rag/answer.py "your question"                     # Groq, needs GROQ_API_KEY in .env
python task2_rag/answer.py "your question" --backend ollama    # offline, needs Ollama running
```
