"""Task 3: benchmark one model at two quantisation levels through Ollama's HTTP API.

For each model: unload everything, 1 warm-up call (loads weights, not timed), then the 10 prompts from
prompts.json with temperature 0 and seed 42. Speed comes from Ollama's own timers in each response:
    generation tokens/s = eval_count / eval_duration * 1e9          (durations are in nanoseconds)
    prompt tokens/s     = prompt_eval_count / prompt_eval_duration * 1e9
Memory comes from /api/ps while the model is loaded: `size` (total bytes) and `size_vram` (bytes on GPU).
Only one model is loaded at a time, so /api/ps shows just that model.

Usage (Ollama must be running):
    python bench.py
    python bench.py --models qwen2.5:1.5b-instruct-q4_K_M      # a single model
Writes results/bench.json and results/outputs.md.
"""
import argparse
import json
import platform
import statistics
from pathlib import Path

import requests

MODELS = ["qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b-instruct-q4_K_M"]
# Same options for every call. num_ctx is fixed because the KV cache (and so memory) grows with it.
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 256, "num_ctx": 2048}
HERE = Path(__file__).parent


def post(host, path, body):
    r = requests.post(f"{host}{path}", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def get(host, path):
    r = requests.get(f"{host}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def unload_all(host):
    """keep_alive=0 tells Ollama to drop a model from memory right away."""
    for m in get(host, "/api/ps").get("models", []):
        post(host, "/api/generate", {"model": m["name"], "keep_alive": 0})


def generate(host, model, prompt):
    return post(host, "/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                        "options": OPTIONS, "keep_alive": "10m"})


def per_sec(count, duration_ns):
    return round(count / duration_ns * 1e9, 2) if duration_ns else None


def bench_model(host, model, prompts):
    unload_all(host)
    generate(host, model, "Hello")                       # warm-up: loads weights, not counted

    runs = []
    for p in prompts:
        r = generate(host, model, p["prompt"])
        runs.append({
            "id": p["id"],
            "response": r["response"],
            "eval_count": r["eval_count"],
            "eval_duration_ns": r["eval_duration"],
            "prompt_eval_count": r.get("prompt_eval_count"),
            "prompt_eval_duration_ns": r.get("prompt_eval_duration"),
            "total_duration_ns": r["total_duration"],
            "gen_tokens_per_sec": per_sec(r["eval_count"], r["eval_duration"]),
            "prompt_tokens_per_sec": per_sec(r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0)),
        })

    # Memory while the model is still loaded.
    loaded = [m for m in get(host, "/api/ps").get("models", []) if m["name"] == model]
    mem = loaded[0] if loaded else {}

    speeds = [x["gen_tokens_per_sec"] for x in runs]
    return {
        "model": model,
        "summary": {
            # pooled = all generated tokens / all generation time (long answers weigh more)
            "gen_tokens_per_sec_pooled": per_sec(sum(x["eval_count"] for x in runs),
                                                 sum(x["eval_duration_ns"] for x in runs)),
            "gen_tokens_per_sec_mean": round(statistics.mean(speeds), 2),
            "gen_tokens_per_sec_median": round(statistics.median(speeds), 2),
            "prompt_tokens_per_sec_pooled": per_sec(sum(x["prompt_eval_count"] or 0 for x in runs),
                                                    sum(x["prompt_eval_duration_ns"] or 0 for x in runs)),
            "total_generated_tokens": sum(x["eval_count"] for x in runs),
            "memory_size_bytes": mem.get("size"),
            "memory_size_vram_bytes": mem.get("size_vram"),
            "memory_size_gb": round(mem["size"] / 1024**3, 3) if mem.get("size") else None,
        },
        "runs": runs,
    }


def write_outputs_md(results, prompts, path):
    """Both models' answers to each prompt, one after the other, for side-by-side reading."""
    lines = ["# Task 3: outputs side by side", "",
             f"Options for every call: `{json.dumps(OPTIONS)}`", ""]
    for i, p in enumerate(prompts):
        lines += [f"## {i + 1}. {p['id']}", "", "**Prompt:**", "", "```", p["prompt"], "```", ""]
        for res in results:
            run = res["runs"][i]
            lines += [f"**{res['model']}** ({run['eval_count']} tokens, {run['gen_tokens_per_sec']} tok/s)", "",
                      "```", run["response"].strip(), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    results = []
    for model in args.models:
        print(f"benchmarking {model} ...")
        res = bench_model(args.host, model, prompts)
        print("  ", json.dumps(res["summary"]))
        results.append(res)
    unload_all(args.host)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    bench = {
        "ollama_version": get(args.host, "/api/version").get("version"),
        "machine": {"os": platform.platform(), "processor": platform.processor(), "python": platform.python_version()},
        "options": OPTIONS,
        "n_prompts": len(prompts),
        "results": results,
    }
    (out / "bench.json").write_text(json.dumps(bench, indent=2, ensure_ascii=False), encoding="utf-8")
    write_outputs_md(results, prompts, out / "outputs.md")
    print(f"wrote {out / 'bench.json'} and {out / 'outputs.md'}")


if __name__ == "__main__":
    main()
