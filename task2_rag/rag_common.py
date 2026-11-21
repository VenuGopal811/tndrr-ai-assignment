"""Shared settings and small helpers for Task 2 (kept in one place so every script agrees)."""
import re
from pathlib import Path

HERE = Path(__file__).parent
PDF_PATH = HERE / "data" / "nist_ai_rmf.pdf"
INDEX_DIR = HERE / "index"            # git-ignored; rebuilt by ingest.py
RESULTS_DIR = HERE / "results"

CHUNK_SIZES = [150, 400]              # words per chunk
OVERLAP = 50                          # words shared by neighbouring chunks
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# The bge model card recommends this instruction in front of short queries (not in front of passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chunks_path(size):
    return INDEX_DIR / f"chunks_{size}.jsonl"


def faiss_path(size):
    return INDEX_DIR / f"faiss_{size}.index"


def normalise(text):
    """Lowercase, re-join words split by a line-break hyphen, keep only letters/digits, single spaces.
    Used to compare a gold passage with chunk text regardless of line breaks, quotes or punctuation."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()
