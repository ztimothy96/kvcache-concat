from datasets import load_dataset
from .templates import QA_TEMPLATE, SUMMARIZATION_TEMPLATE

TASKS = {
    "2wikimqa": ("THUDM/LongBench", "2wikimqa_e"),
    "musique": ("THUDM/LongBench", "musique_e"),
    "samsum": ("THUDM/LongBench", "samsum_e"),
    "multinews": ("THUDM/LongBench", "multi_news_e"),
}

TASK_TYPE = {
    "2wikimqa": "qa",
    "musique": "qa",
    "samsum": "summarization",
    "multinews": "summarization",
}

ALL_TASKS = list(TASKS.keys())


def load_task(task_name: str, split: str = "test", max_samples: int = 200):
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}. Choose from {ALL_TASKS}")
    repo, subset = TASKS[task_name]
    ds = load_dataset(repo, subset, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def format_prompt(example: dict, task_name: str) -> str:
    context = example.get("context", "")
    question = example.get("input", "")
    if TASK_TYPE[task_name] == "qa":
        return QA_TEMPLATE.format(context=context, question=question)
    else:
        return SUMMARIZATION_TEMPLATE.format(context=context)


def get_ground_truths(example: dict) -> list[str]:
    # LongBench uses 'answers' for QA and 'answers' for summarization too,
    # but some subsets use different field names.
    for field in ("answers", "all_labels"):
        val = example.get(field)
        if val:
            return val if isinstance(val, list) else [val]
    return []
