"""Recall@1/3/5 for each chunk size on Venu's questions (eval/questions.json).

A retrieved chunk counts as a hit for a question if:
  passage : the normalised gold passage is a substring of the normalised chunk text
  page    : the gold page is one of the pages the chunk covers
  hit     : passage OR page (the main metric, as in DESIGN.md)
recall@k = share of questions with at least one hit in the top k.

Usage:
  python task2_rag/evaluate.py                    # all 15 questions must be filled in
  python task2_rag/evaluate.py --allow-partial    # score only the filled-in ones (while writing them)
  python task2_rag/evaluate.py --bm25             # also score a BM25 keyword baseline on the same chunks
Writes task2_rag/results/recall.json
"""
import argparse
import json
import sys

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest import load_words
from rag_common import CHUNK_SIZES, EMBED_MODEL, HERE, OVERLAP, RESULTS_DIR, normalise
from retrieve import Retriever

KS = [1, 3, 5]
CRITERIA = ["hit", "passage", "page"]


def load_questions(allow_partial):
    qs = json.loads((HERE / "eval" / "questions.json").read_text(encoding="utf-8"))["questions"]
    filled = [q for q in qs if q["question"].strip() and q["gold_passage"].strip() and q["gold_page"]]
    if not filled or (len(filled) < len(qs) and not allow_partial):
        sys.exit(f"{len(filled)}/{len(qs)} questions filled in eval/questions.json - fill them all "
                 f"(or use --allow-partial).")
    return filled


def chunk_hits(chunk, q):
    passage = normalise(q["gold_passage"]) in normalise(chunk["text"])
    page = q["gold_page"] in chunk["pages"]
    return {"hit": passage or page, "passage": passage, "page": page}


def score_rankings(questions, rankings):
    """rankings[i] = retrieved chunks for question i, best first. Returns recall table + per-question detail."""
    recall = {f"recall@{k}": {} for k in KS}
    detail = []
    for q, chunks in zip(questions, rankings):
        hits = [chunk_hits(c, q) for c in chunks]
        first = {c: next((r + 1 for r, h in enumerate(hits) if h[c]), None) for c in CRITERIA}
        detail.append({"id": q["id"], "question": q["question"], "gold_page": q["gold_page"],
                       "first_hit_rank": first,
                       "retrieved": [{"chunk": c["id"], "pages": c["pages"]} for c in chunks]})
    for k in KS:
        for c in CRITERIA:
            n = sum(1 for d in detail if d["first_hit_rank"][c] and d["first_hit_rank"][c] <= k)
            recall[f"recall@{k}"][c] = round(n / len(questions), 3)
    return recall, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--bm25", action="store_true")
    args = ap.parse_args()

    questions = load_questions(args.allow_partial)

    # Sanity check on the gold data itself: the passage must exist in the document text.
    doc = normalise(" ".join(w for w, _ in load_words()))
    missing = [q["id"] for q in questions if normalise(q["gold_passage"]) not in doc]
    if missing:
        print(f"WARNING: gold_passage not found in the PDF text for question ids {missing} "
              f"(copy it exactly; these can only score via the page criterion)")

    model = SentenceTransformer(EMBED_MODEL)
    out = {"n_questions": len(questions), "question_ids": [q["id"] for q in questions],
           "embed_model": EMBED_MODEL, "overlap_words": OVERLAP,
           "gold_passage_not_in_pdf": missing, "by_size": {}}

    for size in CHUNK_SIZES:
        r = Retriever(size, model)
        rankings = [[c for _, c in r.search(q["question"], max(KS))] for q in questions]
        recall, detail = score_rankings(questions, rankings)
        res = {"n_chunks": len(r.chunks), "embedding": {"recall": recall, "per_question": detail}}

        if args.bm25:   # keyword baseline over the same chunks, for comparison only
            bm25 = BM25Okapi([normalise(c["text"]).split() for c in r.chunks])
            bm_rank = []
            for q in questions:
                scores = bm25.get_scores(normalise(q["question"]).split())
                top = sorted(range(len(scores)), key=lambda i: -scores[i])[:max(KS)]
                bm_rank.append([r.chunks[i] for i in top])
            bm_recall, bm_detail = score_rankings(questions, bm_rank)
            res["bm25"] = {"recall": bm_recall, "per_question": bm_detail}
        out["by_size"][str(size)] = res

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "recall.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console table
    print(f"\n{len(questions)} questions")
    for size, res in out["by_size"].items():
        for method in [m for m in ["embedding", "bm25"] if m in res]:
            rec = res[method]["recall"]
            print(f"size {size:>3} {method:<9} " +
                  "  ".join(f"R@{k} hit={rec[f'recall@{k}']['hit']:.3f} passage={rec[f'recall@{k}']['passage']:.3f} "
                            f"page={rec[f'recall@{k}']['page']:.3f}" for k in KS))
    # Misses at k=5 (embedding), to choose real failures for results/failures.md
    for size, res in out["by_size"].items():
        for d in res["embedding"]["per_question"]:
            if not d["first_hit_rank"]["hit"]:
                print(f"\nMISS size {size} q{d['id']}: {d['question']}\n  gold page {d['gold_page']}, "
                      f"retrieved pages {[x['pages'] for x in d['retrieved']]}")
    print(f"\nwrote {RESULTS_DIR / 'recall.json'}")


if __name__ == "__main__":
    main()
