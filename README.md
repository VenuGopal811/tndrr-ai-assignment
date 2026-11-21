# tndrr AI Internship Assignment

Author: Venu Gopal

| Folder | Task | Where it runs |
|---|---|---|
| [task1_lora/](task1_lora/) | QLoRA fine-tune of Qwen2.5-0.5B-Instruct: plain English to JSON | Google Colab (free T4) |
| [task2_rag/](task2_rag/) | RAG from scratch over NIST AI RMF 1.0 (no RAG frameworks) | Local CPU |
| [task3_local/](task3_local/) | Ollama qwen2.5:1.5b-instruct, q8_0 vs q4_K_M | Local laptop |
| [report/](report/) | 4-page PDF report | - |

## Setup (local, Tasks 2 and 3)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

Task 1 runs on Colab and installs its own pinned libraries from `task1_lora/requirements-colab.txt` (Colab keeps its own torch/CUDA build).

API keys (Task 2 answer step only) go in a `.env` file at the repo root, which is git-ignored:

```
GROQ_API_KEY=...
```

## Hardware used

TODO(Venu): CPU, RAM, GPU (if any), OS for the local runs; Colab GPU type for Task 1.

## Reproducibility

- Seed 42 everywhere.
- Exact commands are in each task's README.
- Every number in the report comes from a file in a `results/` folder.
