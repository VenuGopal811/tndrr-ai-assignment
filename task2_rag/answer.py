"""Retrieve the top-5 chunks, ask an LLM to answer ONLY from them, print the answer + the cited chunks.

Backends (plain HTTP, no SDK / framework):
  groq   : Groq's OpenAI-compatible API, key GROQ_API_KEY in .env (default model llama-3.1-8b-instant)
  ollama : local Ollama server, offline fallback (default model qwen2.5:1.5b-instruct-q4_K_M)

Usage:
  python task2_rag/answer.py "What are the four functions of the AI RMF Core?"
  python task2_rag/answer.py "..." --backend ollama --size 400
"""
import argparse
import os

import requests
from dotenv import load_dotenv

from rag_common import HERE
from retrieve import Retriever

SYSTEM = ("You answer questions about the NIST AI Risk Management Framework using ONLY the numbered context "
          "passages given. Cite the passages you used like [1] or [2][4]. If the context does not contain the "
          "answer, say \"The provided context does not answer this.\" Do not use outside knowledge.")


def build_prompt(question, hits):
    ctx = "\n\n".join(f"[{i}] (page {c['page']})\n{c['text']}" for i, (_, c) in enumerate(hits, start=1))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def ask_groq(prompt, model):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY missing: put it in .env at the repo root, or use --backend ollama")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": 0,
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": prompt}]},
                      timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_ollama(prompt, model, host="http://localhost:11434"):
    r = requests.post(f"{host}/api/chat",
                      json={"model": model, "stream": False, "options": {"temperature": 0, "seed": 42},
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": prompt}]},
                      timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    load_dotenv(HERE.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--size", type=int, default=150, help="chunk size index to use (150 or 400)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--backend", choices=["groq", "ollama"], default="groq")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    hits = Retriever(args.size).search(args.question, args.k)
    prompt = build_prompt(args.question, hits)
    if args.backend == "groq":
        answer = ask_groq(prompt, args.model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    else:
        answer = ask_ollama(prompt, args.model or "qwen2.5:1.5b-instruct-q4_K_M")

    print("ANSWER:\n" + answer.strip() + "\n\nCONTEXT CHUNKS:")
    for i, (score, c) in enumerate(hits, start=1):
        print(f"[{i}] page {c['page']} (chunk {c['id']}, score {score:.3f}): {c['text'][:200]}...")


if __name__ == "__main__":
    main()
