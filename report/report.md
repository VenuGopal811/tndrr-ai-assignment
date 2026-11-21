# tndrr AI Internship Assignment: Venu Gopal

Repo: TODO(Venu: public GitHub URL)

Everything between `<!-- AUTO:... -->` markers is filled by `report/build_report.py` from the results files.

*Note: I used an AI assistant to help draft the written explanations and the Task 2 test questions. All code was run by me, and every number in this report comes from my own runs (result files in the repo).*

## Task 1: QLoRA fine-tune, plain English to JSON

Qwen2.5-0.5B-Instruct, 4-bit NF4 QLoRA. Task: turn a short personal note into `{type, amount, currency, category, date, title}`. The test set uses templates that are never seen in training.

<!-- AUTO:t1_data -->
| Split | Examples | Expense | Reminder | Templates |
|---|---|---|---|---|
| train | 600 | 375 | 225 | 17 |
| test | 60 | 40 | 20 | 9 |
<!-- /AUTO:t1_data -->

<!-- AUTO:t1_metrics -->
_not run: `task1_lora/results/metrics.json` does not exist yet._
<!-- /AUTO:t1_metrics -->

<!-- AUTO:t1_train -->
_not run: `task1_lora/results/train_metrics.json` does not exist yet._
<!-- /AUTO:t1_train -->

**LoRA settings** (from `task1_lora/config.json`)

<!-- AUTO:t1_lora -->
| Setting | Value |
|---|---|
| Quantisation | nf4 4-bit, double quant True, compute float16 |
| Rank r / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LR / schedule / warmup | 0.0002 / cosine / 0.05 |
| Epochs / batch / max length | 3 / 8 / 256 |
| Loss | answer tokens only |
| Seed | 42 |
<!-- /AUTO:t1_lora -->

**Why these settings.**
- **Rank 16, alpha 32:** the task is narrow (one fixed JSON schema), so a small rank is enough; 16 gives some headroom over 8 for the date and title rules without adding much memory (8.8M trainable params, 1.7% of the model). Alpha = 2 x r is the common default and keeps the effective scaling at 2.
- **All attention and MLP projections:** the model has to learn a new output format, not just re-weight attention, so adapting the MLP layers too helps. On a 0.5B model the extra cost is small.
- **LR 2e-4, cosine, 3 epochs:** 2e-4 is the usual LoRA learning rate (LoRA tolerates a higher LR than full fine-tuning because only the small adapters move). 3 epochs over 600 examples was enough: loss fell from 0.273 to about 0.0001.
- **QLoRA (4-bit NF4 + fp16 compute):** it fits easily on a free T4 (peak 4.21 GB) and it is the setup I would need for a 7B model, so I wanted to practise it end to end.
- **Loss on answer tokens only:** the system prompt and note are the same kind of text every time; training on them would waste capacity. Only the JSON is learned.

**3 failure cases** (full detail in `task1_lora/results/failures.md`)

The fine-tuned model got 52/60 exactly right; these are 3 of the 8 misses.

1. **"yesterday shelled out 1220 rupees on tea"** (test #33). Expected title `tea`, got `shelled out`. "shelled out" only appears in a test template. In training the item usually comes right after the verb or date ("bought {item}"), so the model seems to have learned "the words after the date are the title" and copied the unfamiliar verb phrase instead. The same mistake happens on #56 (jeans). Fix: more varied verb phrasings in training.
2. **"note to self: return library books in 10 days"** (test #37). Expected date `2026-12-01`, got `2026-11-22` (tomorrow). Every other field is right. "in N days" is in training (N from 2 to 10), but adding N days to a date is arithmetic, not a pattern, and a 0.5B model does it unreliably, especially across a month boundary. Fix: let the model output the relative phrase and compute the date in code.
3. **"had to pay 1790 rupees for mobile recharge last Saturday"** (test #13). Expected category `bills`, got `shopping`. "mobile recharge" is a held-out word. In training every bills item ends in "bill", and "recharge" only appeared as a reminder task, so without the word "bill" the model fell back to the closest spending category. Fix: add bill-like items that don't contain the word "bill".

### Memory for a 7B model: full fine-tune vs LoRA vs QLoRA

Assumptions: 7B parameters, Llama-style shape (32 layers, hidden 4096, MLP 11008), AdamW, mixed precision. Activations depend on batch and sequence length; with gradient checkpointing and a small batch I budget about 2-4 GB for them.

**(a) Full fine-tuning.** Every weight is trained, so per parameter: fp16 weight 2 B + fp16 gradient 2 B + fp32 master copy 4 B + Adam m and v 4 B + 4 B = **16 bytes**. 7B x 16 B = **112 GB**, plus activations, so roughly **115-120 GB** (two 80 GB GPUs or DeepSpeed/FSDP sharding).

**(b) LoRA (r = 16 on all 7 projections, base frozen in fp16).**
- LoRA params per layer: attention 4 x 16 x (4096 + 4096) = 524k; MLP 3 x 16 x (4096 + 11008) = 725k; total about 1.25M per layer x 32 layers = **~40M trainable**.
- Frozen base: 7B x 2 B = **14 GB** (no gradients or optimiser state).
- Trainable part: 40M x 16 B = **0.64 GB**.
- Total about 14 + 0.64 + 2-4 activations = **~17-19 GB** (fits a 24 GB GPU).

**(c) QLoRA (base in 4-bit NF4).**
- Frozen base: 7B x 0.5 B = 3.5 GB, plus quantisation constants (~0.1-0.2 GB with double quant) = **~3.7 GB**.
- Same LoRA adapters and optimiser: **0.64 GB**.
- Total about 3.7 + 0.64 + 2-4 activations = **~7-9 GB** (fits a free 16 GB T4).

For scale: my 0.5B QLoRA run peaked at 4.21 GB, and most of that was activations (batch 8, length 256), which is why activations can't be ignored in these estimates.

## Task 2: RAG from scratch over NIST AI RMF 1.0

pypdf, word-window chunks (50-word overlap), `BAAI/bge-small-en-v1.5`, `faiss.IndexFlatIP`. No RAG framework. 15 questions written by Venu, each with a gold page and passage. A hit = a top-k chunk contains the gold passage or covers the gold page.

<!-- AUTO:t2_ingest -->
Corpus: 48 pages, 15,417 words; 150 words -> 154 chunks, 400 words -> 44 chunks (50-word overlap).
<!-- /AUTO:t2_ingest -->

<!-- AUTO:t2_recall -->
_not run: `task2_rag/results/recall.json` does not exist yet._
<!-- /AUTO:t2_recall -->

**2 retrieval failures and fixes** (full detail in `task2_rag/results/failures.md`)

> **TODO(Venu): failure 1.** Question / what was retrieved / why it missed / fix.
>
> **TODO(Venu): failure 2.** Question / what was retrieved / why it missed / fix.

> **TODO(Venu): chunk size.** What the 150 vs 400 numbers above show.

## Task 3: Local model, two quantisation levels

Ollama, `qwen2.5:1.5b-instruct` at q8_0 and q4_K_M. Same 10 prompts, temperature 0, seed 42, 256 max new tokens, 1 warm-up call per model. Tokens/s = `eval_count / eval_duration`; memory = `size` from `/api/ps`.

<!-- AUTO:t3_bench -->
Ollama 0.40.17, 10 prompts per model.

| Model | Gen tok/s (pooled) | Gen tok/s (median) | Prompt tok/s | Memory GB | VRAM GB | Tokens generated |
|---|---|---|---|---|---|---|
| qwen2.5:1.5b-instruct-q8_0 | 53.69 | 55.11 | 1228.83 | 1.648 | 1.648 | 1388 |
| qwen2.5:1.5b-instruct-q4_K_M | 82.0 | 81.08 | 1075.27 | 1.032 | 1.032 | 1412 |
<!-- /AUTO:t3_bench -->

**Hardware:** Intel Core i5-11260H, NVIDIA RTX 2050 (4 GB VRAM), 15.7 GB RAM, Windows 11. Both models fitted fully in VRAM.

**Quality observations** (full side-by-side in `task3_local/results/outputs.md`):
- **Same answers on simple tasks.** JSON output was identical; the maths answer (17% of 2,450 = 416.5) and the palindrome code were the same logic with small wording differences.
- **Reasoning got wordier, not wrong.** On the train-time question both reach 12:40 PM, but q4 took a longer route (192 vs 256 tokens; q4 hit the 256-token limit before finishing its summary).
- **Both failed the same hard prompts.** Both wrongly concluded "some bloops are lazzies" (the syllogism does not follow), and both produced a broken Hindi translation that mentions French. These are limits of a 1.5B model, not of quantisation.
- **Conclusion:** on these 10 prompts q4_K_M is 1.53x faster and uses 38% less memory, with no failure that q8_0 didn't also have. For this model size I would deploy q4_K_M; I would re-check on longer reasoning tasks, where q4 drifted most.

## Written answers

### BM25 vs embedding retrieval, and what a re-ranker does

Embedding search compares meaning: the question and each chunk become vectors, so "attacks on AI" can match "adversarial examples, data poisoning". Its weak spot is exact terms: rare names, codes and numbers ("GOVERN 1.7", "ISO 26000") get blurred into general meaning. **BM25** is keyword scoring: it rewards chunks that contain the question's exact words, weighting rare words more (IDF) and not over-rewarding repeats. Adding BM25 to embeddings (hybrid search, e.g. merging both rankings) catches the exact-term questions that embeddings miss, while embeddings catch the paraphrased ones BM25 misses.

A **re-ranker** is a second, slower stage. Retrieval (BM25 / embeddings) quickly gets maybe the top 20-50 chunks; then a cross-encoder reads the question and each chunk *together* and scores how well that chunk answers the question. Because it sees both texts at once, it is more accurate than comparing two separately computed vectors, but too slow to run on the whole corpus, so it only re-orders the shortlist and we keep the top 5.

### KV cache, and why quantisation saves memory

When a transformer generates text, each new token attends to all previous tokens, which needs each earlier token's **key and value vectors** in every layer. Instead of recomputing them for the whole sequence at every step, the model stores them once: that store is the **KV cache**. It makes generation fast (each step only computes the new token), but it costs memory that grows with sequence length and with the number of users: about 2 (K and V) x layers x KV heads x head dim x bytes per value, per token.

**Quantisation** stores each weight in fewer bits: 16-bit uses 2 bytes per weight, 8-bit about 1, 4-bit about 0.5. Same number of weights, fewer bytes each, so the model takes less memory. My bench shows it: the same qwen2.5 1.5B model used 1.65 GB at q8_0 and 1.03 GB at q4_K_M. It is also faster, because generating each token means reading all the weights from memory, and on a small GPU that memory bandwidth is the bottleneck; fewer bytes to read means more tokens per second (82.0 vs 53.7 tok/s). The cost is a small loss of precision, which can show up as slightly different or wordier answers.

### Serving 20 concurrent users on one 24 GB GPU

**Assumptions:** model Qwen2.5-7B-Instruct (28 layers, grouped-query attention with 4 KV heads x head dim 128); each user needs up to 4,096 tokens of context (prompt + answer); usable GPU memory ~90% of 24 GB = ~21.6 GB.

1. **Weights:** fp16 would be 7.6B x 2 B = ~15.2 GB, which leaves little room. I would use a **4-bit AWQ (or 8-bit) quantised** version: ~7.6B x 0.5 B = **~5 GB** (call it 5.5 GB with overhead).
2. **KV cache per token:** 2 x 28 layers x 4 heads x 128 dims x 2 bytes (fp16) = 57,344 B = **~56 KB/token**. Per user at 4,096 tokens: **~0.23 GB**. For 20 users: 20 x 4,096 = 81,920 tokens x 56 KB = **~4.6 GB**.
3. **Total:** 5.5 GB weights + 4.6 GB KV cache + ~1.5 GB runtime/activations = **~11.6 GB**, well inside 21.6 GB. The spare ~10 GB lets the server keep more cache, handle longer prompts, or go above 20 users. (Even fp16 weights would just fit: 15.2 + 4.6 + 1.5 = 21.3 GB, but with no headroom.)
4. **Serving engine: vLLM.** **Continuous batching** puts all active users' next-token steps into one GPU batch, so 20 users share each weight read instead of queueing; **PagedAttention** allocates KV cache in small blocks only as each conversation grows, so short chats don't reserve the full 4,096 tokens. Settings: `max_num_seqs=20` (or a bit more), `max_model_len=4096`, `gpu_memory_utilization=0.9`.
5. **What I would measure under a 20-user load test:** time to first token (p50/p95), tokens/s per user and total, GPU memory, and queueing time. If p95 latency is too high, cut max context or move to a smaller model; if memory is spare, raise `max_num_seqs`.
