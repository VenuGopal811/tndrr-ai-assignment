# Task 2: retrieval failures

Results from `recall.json` (embedding search, top 5). 3 questions missed at 150 words (q4, q12, q13) and 2 at 400 words (q12, q13).

## q13: "What does the framework expect when an organisation retires an old AI system?"
- Gold: page 28, GOVERN 1.7 ("decommissioning and phasing out AI systems safely...").
- Retrieved (400 words): pages [7-8], [13-14], [47-48], [13], [1-4]. BM25 also missed.
- Why: vocabulary mismatch ("retires" vs "decommissioning / phasing out"), and the passage is one short row in a long table of GOVERN subcategories, so the chunk vector is dominated by other rows.
- Fix: one chunk per Core subcategory; query expansion / LLM query rewriting.

## q12: "When is an AI system supposed to be tested?"
- Gold: page 33 ("AI systems should be tested before their deployment and regularly while in operation.").
- Retrieved (400 words): pages [34-35], [18-19], [19-20], [20-21], [43-44]. BM25 found it at rank 3.
- Why: short, generic question; testing/TEVV is discussed on many pages, so many chunks look similar to the embedding.
- Fix: hybrid BM25 + embedding ranking, then a cross-encoder re-ranker over the top 20.
