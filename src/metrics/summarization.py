from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, reference: str) -> float:
    if not reference.strip():
        return 0.0
    scores = _scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure
