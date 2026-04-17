#!/usr/bin/env python3
"""
Generate tables and figures from a results.jsonl file.

Example usage:
    python scripts/plot_results.py --input outputs/results.jsonl --output-dir outputs/figures
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.benchmark.results import load_results, aggregate

METHOD_ORDER = ["sequential", "direct_concat", "rope_adjusted"]
METHOD_LABELS = {
    "sequential": "Sequential",
    "direct_concat": "Direct Concat",
    "rope_adjusted": "RoPE Adjusted",
}
COLORS = {
    "sequential": "#2196F3",
    "direct_concat": "#F44336",
    "rope_adjusted": "#4CAF50",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/results.jsonl")
    p.add_argument("--output-dir", default="outputs/figures")
    return p.parse_args()


def primary_metric(task: str) -> str:
    if task in ("2wikimqa", "musique"):
        return "f1"
    return "rouge_l"


def plot_quality_table(df: pd.DataFrame, output_dir: str):
    """Print and save a quality table (method × task)."""
    rows = []
    for task in df["task"].unique():
        col = primary_metric(task)
        if col not in df.columns:
            continue
        task_df = df[df["task"] == task]
        for method in METHOD_ORDER:
            mdf = task_df[task_df["method"] == method]
            if mdf.empty:
                continue
            rows.append({
                "task": task,
                "method": METHOD_LABELS.get(method, method),
                "metric": col,
                "mean": mdf[col].mean(),
                "std": mdf[col].std(),
            })
    tbl = pd.DataFrame(rows)
    tbl["score"] = tbl.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    pivot = tbl.pivot_table(index="method", columns="task", values="score", aggfunc="first")
    print("\n=== Quality Table ===")
    print(pivot.to_string())
    pivot.to_csv(os.path.join(output_dir, "quality_table.csv"))


def plot_ttft_vs_context(df: pd.DataFrame, output_dir: str):
    """Scatter + regression: TTFT vs. context length, one line per method."""
    if "ttft_ms" not in df.columns or "context_len" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHOD_ORDER:
        mdf = df[df["method"] == method].dropna(subset=["ttft_ms", "context_len"])
        if mdf.empty:
            continue
        ax.scatter(
            mdf["context_len"], mdf["ttft_ms"],
            label=METHOD_LABELS[method], color=COLORS[method], alpha=0.4, s=15,
        )
        # Regression line
        z = np.polyfit(mdf["context_len"], mdf["ttft_ms"], 1)
        x_range = np.linspace(mdf["context_len"].min(), mdf["context_len"].max(), 200)
        ax.plot(x_range, np.polyval(z, x_range), color=COLORS[method], linewidth=1.5)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time-to-First-Token vs. Context Length")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "ttft_vs_context.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_pareto(df: pd.DataFrame, output_dir: str):
    """Accuracy vs. recomputation ratio Pareto curve."""
    if "recomputation_ratio" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for task in df["task"].unique():
        col = primary_metric(task)
        if col not in df.columns:
            continue
        for method in METHOD_ORDER:
            mdf = df[(df["task"] == task) & (df["method"] == method)]
            if mdf.empty:
                continue
            x = mdf["recomputation_ratio"].mean()
            y = mdf[col].mean()
            ax.scatter(x, y, color=COLORS[method], s=80, zorder=3)
            ax.annotate(
                f"{task[:5]}/{METHOD_LABELS[method][:3]}",
                (x, y), textcoords="offset points", xytext=(4, 2), fontsize=6,
            )

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[m], label=METHOD_LABELS[m]) for m in METHOD_ORDER
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Recomputation Ratio (attention FLOPs / sequential FLOPs)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Accuracy vs. Compute Pareto Curve")
    fig.tight_layout()
    path = os.path.join(output_dir, "pareto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_speedup(df: pd.DataFrame, output_dir: str):
    """TTFT speedup bar chart normalized to sequential baseline."""
    if "ttft_ms" not in df.columns:
        return

    tasks = df["task"].unique()
    baseline_ttft = {}
    for task in tasks:
        seq_df = df[(df["task"] == task) & (df["method"] == "sequential")]
        if not seq_df.empty:
            baseline_ttft[task] = seq_df["ttft_ms"].mean()

    rows = []
    for task in tasks:
        if task not in baseline_ttft:
            continue
        for method in METHOD_ORDER:
            mdf = df[(df["task"] == task) & (df["method"] == method)]
            if mdf.empty:
                continue
            speedup = baseline_ttft[task] / mdf["ttft_ms"].mean()
            rows.append({"task": task, "method": method, "speedup": speedup})

    if not rows:
        return

    speedup_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tasks))
    width = 0.25
    for i, method in enumerate(METHOD_ORDER):
        mdf = speedup_df[speedup_df["method"] == method]
        vals = [mdf[mdf["task"] == t]["speedup"].values[0] if not mdf[mdf["task"] == t].empty else 0 for t in tasks]
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS[method], color=COLORS[method])

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("TTFT Speedup (× sequential)")
    ax.set_title("TTFT Speedup vs. Sequential Baseline")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "speedup.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_results(args.input)
    print(f"Loaded {len(df)} records from {args.input}")

    plot_quality_table(df, args.output_dir)
    plot_ttft_vs_context(df, args.output_dir)
    plot_pareto(df, args.output_dir)
    plot_speedup(df, args.output_dir)

    print("\n=== Aggregate Stats ===")
    print(aggregate(df).to_string())


if __name__ == "__main__":
    main()
