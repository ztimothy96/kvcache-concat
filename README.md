# kvcache-concat

Benchmark for **Parallel Context Encoding (PCE)**: split a prompt into chunks, encode each independently, concatenate the KV caches, and measure the quality/latency tradeoff.

Reference papers: [CacheBlend (arXiv:2405.16444)](https://arxiv.org/abs/2405.16444), [KVLink (arXiv:2502.16002)](https://arxiv.org/abs/2502.16002)

---

## Methods

| Method | Description |
|---|---|
| `sequential` | Single forward pass over the full prompt (quality upper bound) |
| `direct_concat` | Encode chunks independently, naively concatenate KV caches |
| `rope_adjusted` | Same as direct concat, but re-apply correct RoPE angles to K tensors after concatenation |

## Models

- **Primary:** `meta-llama/Llama-3.1-8B-Instruct`
- **Secondary:** `Qwen/Qwen2.5-7B-Instruct`

## Dataset

[LongBench](https://huggingface.co/datasets/THUDM/LongBench) tasks:

| Task | Type | Metric |
|---|---|---|
| `2wikimqa` | QA | F1 + substring match |
| `musique` | QA | F1 + substring match |
| `samsum` | Summarization | ROUGE-L |
| `multinews` | Summarization | ROUGE-L |

Efficiency metrics tracked for all tasks: **TTFT** (time-to-first-token) and **recomputation ratio** (attention FLOPs fraction relative to sequential).

---

## Setup

Designed to run on a single **RTX 4090 (24 GB)**. LLaMA-3.1-8B in bfloat16 uses ~16 GB, leaving ~8 GB for KV caches.

### 1. HuggingFace authentication

LLaMA-3.1-8B is a gated model. Before running:

1. Request access at [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (agree to Meta's license — approval is usually instant).
2. Create a read token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. On the RunPod pod, run:

```bash
hf auth login   # paste your token when prompted
```

Or export it as an environment variable:

```bash
export HF_TOKEN=hf_...
```

Qwen2.5-7B is ungated and only requires the login step (no access request).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the benchmark

```bash
python scripts/run_benchmark.py \
    --model llama \
    --tasks 2wikimqa musique samsum multinews \
    --methods sequential direct_concat rope_adjusted \
    --num-chunks 4 \
    --max-samples 200 \
    --max-new-tokens 100 \
    --output outputs/results.jsonl
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `llama` | `llama` or `qwen` |
| `--tasks` | all four | Space-separated list of tasks |
| `--methods` | all three | Space-separated list of methods |
| `--num-chunks` | `4` | Number of chunks for PCE methods |
| `--max-samples` | `200` | Examples per task (reduce for quick runs) |
| `--max-new-tokens` | `100` | Max generated tokens per example |
| `--output` | `outputs/results.jsonl` | Results file (appended, not overwritten) |

Results are streamed to `--output` as newline-delimited JSON, one record per example. The script prints an aggregate table when finished.

### Quick smoke test (10 examples, sequential only)

```bash
python scripts/run_benchmark.py \
    --model llama \
    --tasks 2wikimqa \
    --methods sequential \
    --max-samples 10
```

---

## Generating plots

```bash
python scripts/plot_results.py \
    --input outputs/results.jsonl \
    --output-dir outputs/figures
```

Produces:

- `quality_table.csv` — mean ± std per (method, task)
- `ttft_vs_context.png` — TTFT vs. context length scatter with regression lines
- `pareto.png` — quality vs. recomputation ratio Pareto curve
- `speedup.png` — TTFT speedup normalized to sequential baseline

---

## Project structure

```
src/
  models/
    loader.py        # Model + tokenizer loading (always uses eager attention)
    rope_utils.py    # RoPE inverse-rotate / re-rotate for K correction
  encoding/
    sequential.py    # Single-pass baseline
    direct_concat.py # Naive KV concatenation
    rope_adjusted.py # PCE with RoPE correction
  data/
    longbench.py     # LongBench loading + prompt formatting
    templates.py     # Zero-shot prompt templates
  metrics/
    qa_metrics.py    # F1 + substring match
    summarization.py # ROUGE-L
  benchmark/
    runner.py        # Per-example orchestration
    timer.py         # TTFT timing + recomputation ratio
    results.py       # JSONL I/O + aggregation
scripts/
  run_benchmark.py   # CLI entry point
  plot_results.py    # Figures and tables
```
