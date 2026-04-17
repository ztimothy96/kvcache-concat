import json
import pathlib
import pandas as pd


def save_result(record: dict, path: str = "outputs/results.jsonl"):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_results(path: str = "outputs/results.jsonl") -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path)]
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c in ("f1", "substring", "rouge_l")]
    agg_cols = metric_cols + ["ttft_ms", "recomputation_ratio"]
    agg_cols = [c for c in agg_cols if c in df.columns]
    return (
        df.groupby(["model", "method", "task"])[agg_cols]
        .agg(["mean", "std"])
        .round(4)
    )
