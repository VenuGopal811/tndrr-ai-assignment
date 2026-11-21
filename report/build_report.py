"""Fill the AUTO tables in report.md from the results files, then print report.md to report.pdf.

Every number in an AUTO block is read from a file in the repo; if the file is missing the block says
"not run" (rules.md #2). Venu's prose outside the AUTO blocks is never touched.

Usage:
  python report/build_report.py              # fill tables + make report.pdf
  python report/build_report.py --no-pdf     # only refresh the tables in report.md
PDF: markdown -> HTML -> headless Edge/Chrome "print to PDF" (no extra install). Set BROWSER to override.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import markdown
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "report.md"
NOT_RUN = "_not run: `{}` does not exist yet._"
FIELDS = ["type", "amount", "currency", "category", "date", "title"]


def load(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("-" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def t1_data():
    counts = {}
    for split in ["train", "test"]:
        p = ROOT / "task1_lora" / "data" / f"{split}.jsonl"
        if not p.exists():
            return NOT_RUN.format(f"task1_lora/data/{split}.jsonl")
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[split] = (len(rows), sum(r["output"]["type"] == "expense" for r in rows),
                         len({r["template_id"] for r in rows}))
    return table(["Split", "Examples", "Expense", "Reminder", "Templates"],
                 [[s, n, e, n - e, t] for s, (n, e, t) in counts.items()])


def t1_metrics():
    m = load("task1_lora/results/metrics.json")
    if not m:
        return NOT_RUN.format("task1_lora/results/metrics.json")
    b, f = m["base"]["all"], m["finetuned"]["all"]
    rows = [["Valid JSON (strict)", b["valid_json_pct"], f["valid_json_pct"]],
            ["Valid JSON (lenient)", b["valid_json_lenient_pct"], f["valid_json_lenient_pct"]],
            ["Exact match", b["exact_match_pct"], f["exact_match_pct"]]]
    rows += [[f"Field: {k}", b["field_accuracy_pct"][k], f["field_accuracy_pct"][k]] for k in FIELDS]
    for sub in ["expense", "reminder", "seen_words", "unseen_words"]:
        rows.append([f"Exact match, {sub} (n={m['base'][sub]['n']})",
                     m["base"][sub]["exact_match_pct"], m["finetuned"][sub]["exact_match_pct"]])
    return (f"Test set n={b['n']}, greedy decoding. All values are % of examples.\n\n"
            + table(["Metric (%)", "Base", "Fine-tuned"], rows))


def t1_train():
    t = load("task1_lora/results/train_metrics.json")
    if not t:
        return NOT_RUN.format("task1_lora/results/train_metrics.json")
    return (f"Training: {t['train_examples']} examples, {t['total_steps']} steps, "
            f"{t['train_runtime_sec']} s on {t['gpu']}, peak GPU memory {t['peak_gpu_mem_gb']} GB, "
            f"{t['trainable_params']:,} trainable params, logged loss {t['first_logged_loss']} -> "
            f"{t['last_logged_loss']}. Curve: `task1_lora/results/train_loss.png`.")


def t1_lora():
    c = load("task1_lora/config.json")
    if not c:
        return NOT_RUN.format("task1_lora/config.json")
    l, t, q = c["lora"], c["training"], c["quantization"]
    return table(["Setting", "Value"], [
        ["Quantisation", f"{q['bnb_4bit_quant_type']} 4-bit, double quant {q['bnb_4bit_use_double_quant']}, "
                         f"compute {q['bnb_4bit_compute_dtype']}"],
        ["Rank r / alpha / dropout", f"{l['r']} / {l['lora_alpha']} / {l['lora_dropout']}"],
        ["Target modules", ", ".join(l["target_modules"])],
        ["LR / schedule / warmup", f"{t['learning_rate']} / {t['lr_scheduler_type']} / {t['warmup_ratio']}"],
        ["Epochs / batch / max length", f"{t['num_train_epochs']} / {t['per_device_train_batch_size']} / {t['max_length']}"],
        ["Loss", "answer tokens only" if t["completion_only_loss"] else "all tokens"],
        ["Seed", c["seed"]]])


def t2_ingest():
    s = load("task2_rag/results/ingest_stats.json")
    if not s:
        return NOT_RUN.format("task2_rag/results/ingest_stats.json")
    sizes = ", ".join(f"{k} words -> {v['chunks']} chunks" for k, v in s["chunk_sizes"].items())
    return f"Corpus: {s['pages']} pages, {s['words']:,} words; {sizes} ({s['overlap_words']}-word overlap)."


def t2_recall():
    r = load("task2_rag/results/recall.json")
    if not r:
        return NOT_RUN.format("task2_rag/results/recall.json")
    rows = []
    for size, res in r["by_size"].items():
        for method in [m for m in ["embedding", "bm25"] if m in res]:
            rec = res[method]["recall"]
            rows.append([size, method] + [rec[f"recall@{k}"][c] for c in ["hit", "passage"] for k in [1, 3, 5]])
    return (f"n={r['n_questions']} questions. `hit` = passage or page match (main metric); "
            f"`passage` = gold sentence inside the chunk (stricter).\n\n"
            + table(["Chunk size", "Retriever", "hit@1", "hit@3", "hit@5", "passage@1", "passage@3", "passage@5"],
                    rows))


def t3_bench():
    b = load("task3_local/results/bench.json")
    if not b:
        return NOT_RUN.format("task3_local/results/bench.json")
    rows = []
    for res in b["results"]:
        s = res["summary"]
        vram = s["memory_size_vram_bytes"]
        rows.append([res["model"], s["gen_tokens_per_sec_pooled"], s["gen_tokens_per_sec_median"],
                     s["prompt_tokens_per_sec_pooled"], s["memory_size_gb"],
                     round(vram / 1024**3, 3) if vram else vram, s["total_generated_tokens"]])
    return (f"Ollama {b['ollama_version']}, {b['n_prompts']} prompts per model.\n\n"
            + table(["Model", "Gen tok/s (pooled)", "Gen tok/s (median)", "Prompt tok/s", "Memory GB",
                     "VRAM GB", "Tokens generated"], rows))


BLOCKS = {"t1_data": t1_data, "t1_metrics": t1_metrics, "t1_train": t1_train, "t1_lora": t1_lora,
          "t2_ingest": t2_ingest, "t2_recall": t2_recall, "t3_bench": t3_bench}


def fill(text):
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- AUTO:{name} -->\n).*?(<!-- /AUTO:{name} -->)", re.S)
        if not pat.search(text):
            sys.exit(f"marker AUTO:{name} missing from report.md")
        text = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), text)
    return text


CSS = """@page { size: A4; margin: 13mm 14mm; }
body { font-family: Calibri, Arial, sans-serif; font-size: 9.5pt; line-height: 1.3; }
h1 { font-size: 15pt; margin: 0 0 4px; } h2 { font-size: 12pt; margin: 10px 0 4px; border-bottom: 1px solid #999; }
h3 { font-size: 10.5pt; margin: 8px 0 3px; } p { margin: 3px 0; }
table { border-collapse: collapse; margin: 4px 0; font-size: 8.5pt; }
th, td { border: 1px solid #aaa; padding: 1px 5px; } th { background: #eee; }
blockquote { margin: 4px 0; padding: 2px 8px; border-left: 3px solid #c00; background: #fff4f4; }
code { font-size: 8.5pt; }"""


def find_browser():
    cands = [os.getenv("BROWSER"),
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    return next((c for c in cands if c and Path(c).exists()), None)


def make_pdf(text):
    # Drop the AUTO markers and the editing note from the rendered PDF.
    body = re.sub(r"<!-- /?AUTO:\w+ -->\n?", "", text)
    body = re.sub(r"^Everything between .*?\n", "", body, flags=re.M)
    html = markdown.markdown(body, extensions=["tables"])
    html_path = ROOT / "report" / "report.html"
    html_path.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{html}", encoding="utf-8")
    browser = find_browser()
    if not browser:
        sys.exit("No Edge/Chrome found: set BROWSER=path/to/browser, or open report/report.html and print to PDF")
    pdf = ROOT / "report" / "report.pdf"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True, timeout=120)
    pages = len(PdfReader(pdf).pages)
    print(f"wrote {pdf}: {pages} pages" + ("  <-- OVER THE 4-PAGE LIMIT" if pages > 4 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    text = fill(REPORT.read_text(encoding="utf-8"))
    REPORT.write_text(text, encoding="utf-8")
    print(f"filled tables in {REPORT}")
    if not args.no_pdf:
        make_pdf(text)
    todos = len(re.findall(r"TODO\(Venu", text))
    if todos:
        print(f"{todos} TODO(Venu) blocks still open")


if __name__ == "__main__":
    main()
