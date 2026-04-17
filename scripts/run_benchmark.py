#!/usr/bin/env python3
"""
Main benchmark entry point.

Example usage:
    python scripts/run_benchmark.py \\
        --model llama \\
        --tasks 2wikimqa musique samsum multinews \\
        --methods sequential direct_concat rope_adjusted \\
        --num-chunks 4 \\
        --max-samples 200 \\
        --max-new-tokens 100 \\
        --output outputs/results.jsonl
"""
import argparse
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from tqdm import tqdm
from src.models.loader import load_model, SUPPORTED
from src.data.longbench import load_task, ALL_TASKS
from src.benchmark.runner import run_single
from src.benchmark.results import save_result, load_results, aggregate


def parse_args():
    p = argparse.ArgumentParser(description="KV cache concatenation benchmark")
    p.add_argument("--model", choices=list(SUPPORTED.keys()), default="llama")
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["sequential", "direct_concat", "rope_adjusted"],
        choices=["sequential", "direct_concat", "rope_adjusted"],
    )
    p.add_argument("--num-chunks", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--output", default="outputs/results.jsonl")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device)
    print("Model loaded.\n")

    total_runs = len(args.tasks) * len(args.methods)
    run_idx = 0

    for task_name in args.tasks:
        print(f"Loading task: {task_name}")
        dataset = load_task(task_name, max_samples=args.max_samples)
        print(f"  {len(dataset)} examples\n")

        for method in args.methods:
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] method={method}  task={task_name}")

            for example in tqdm(dataset,
                                desc=f"{method}/{task_name}",
                                leave=False):
                try:
                    record = run_single(
                        method=method,
                        task_name=task_name,
                        model_name=args.model,
                        model=model,
                        tokenizer=tokenizer,
                        example=example,
                        n_chunks=args.num_chunks,
                        max_new_tokens=args.max_new_tokens,
                    )
                    save_result(record, args.output)
                except Exception as e:
                    print(f"\n  [ERROR] {e}")
                    continue

    print(f"\nResults saved to {args.output}")

    # Print aggregate table
    try:
        df = load_results(args.output)
        print("\n=== Aggregate Results ===")
        print(aggregate(df).to_string())
    except Exception as e:
        print(f"Could not aggregate results: {e}")


if __name__ == "__main__":
    main()
