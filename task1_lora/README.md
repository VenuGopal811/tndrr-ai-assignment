# Task 1: QLoRA fine-tuning (plain English to JSON)

Model: `Qwen/Qwen2.5-0.5B-Instruct`, 4-bit NF4 base + LoRA adapters, free Colab T4.

Task: turn a short note like "spent Rs 450 on groceries yesterday" into

```json
{"type": "expense", "amount": 450, "currency": "INR", "category": "food", "date": "2026-11-20", "title": "groceries"}
```

Dates are resolved against a fixed reference date, 2026-11-21.

## Files

| File | What it does |
|---|---|
| `make_data.py` | Generates `data/train.jsonl` and `data/test.jsonl` from templates. Train and test use disjoint templates. Labels come from code. |
| `train_qlora.ipynb` | Colab notebook: QLoRA training, saves adapter + logs to Google Drive. |
| `evaluate.ipynb` | Colab notebook: base vs fine-tuned on `test.jsonl`. |
| `config.json` | All LoRA / training hyperparameters. |
| `requirements-colab.txt` | Pinned libraries installed inside Colab. |
| `results/` | `metrics.json`, `predictions.jsonl`, `failures.md` (from real runs only). |

## Commands

1. Generate data (local, deterministic, seed 42):
   ```
   python task1_lora/make_data.py
   ```
2. Upload to Google Drive at `MyDrive/tndrr/task1_lora/`: `config.json`, `requirements-colab.txt`, `data/train.jsonl`, `data/test.jsonl`.
3. Open `train_qlora.ipynb` in Colab, Runtime -> Change runtime type -> T4 GPU, Run all.
   The adapter lands in `MyDrive/tndrr/task1_lora/adapter/`; logs in `MyDrive/tndrr/task1_lora/results/`.
4. Download `results/train_metrics.json`, `results/train_log.json`, `results/train_loss.png` into this repo's `task1_lora/results/`.
5. Open `evaluate.ipynb` in Colab (T4), Run all. It loads the same 4-bit base + the adapter from Drive and scores
   base (adapter disabled) vs fine-tuned on `data/test.jsonl` with greedy decoding.
6. Download `results/metrics.json` and `results/predictions.jsonl` into this repo's `task1_lora/results/`.
   Pick 3 real failures from the last cell's printout for `results/failures.md`.
