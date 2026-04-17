import re
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truths: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(gt_tokens) if gt_tokens else 0.0
        if precision + recall == 0:
            continue
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def substring_match(prediction: str, ground_truths: list[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return float(any(normalize_answer(gt) in norm_pred for gt in ground_truths))
