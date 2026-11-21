# Task 3: Local model, two quantisation levels

Same model, two settings, served by Ollama:

- `qwen2.5:1.5b-instruct-q8_0`
- `qwen2.5:1.5b-instruct-q4_K_M`

## Files

| File | What it does |
|---|---|
| `bench.py` | Calls the Ollama HTTP API for each model x prompt; records tokens/s and memory |
| `prompts.json` | 10 fixed prompts |
| `results/` | `bench.json`, `outputs.md` |

## Method

- Every call: `stream: false`, options `{"temperature": 0, "seed": 42, "num_predict": 256, "num_ctx": 2048}`.
  `num_ctx` is fixed so both models reserve the same KV-cache size and memory is comparable.
- Per model: unload everything, 1 untimed warm-up call (loads the weights), then the 10 prompts.
- Generation speed = `eval_count / eval_duration * 1e9` (Ollama's own timers, nanoseconds). Reported pooled
  (all tokens / all time), mean and median over the 10 prompts. Prompt-processing speed is recorded too.
- Memory = `size` (and `size_vram` if a GPU is used) from `/api/ps` while the model is loaded. Only one model is loaded at a time.

## Commands

Install Ollama from https://ollama.com/download, then:

```
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python task3_local/bench.py
```

Close other heavy apps first: both models run on the same machine, and background load changes tokens/s.
Outputs: `results/bench.json` (all numbers + every response), `results/outputs.md` (answers side by side).
