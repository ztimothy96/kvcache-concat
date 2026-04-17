import torch
from ..data.longbench import format_prompt, get_ground_truths, TASK_TYPE
from ..metrics.qa_metrics import f1_score, substring_match
from ..metrics.summarization import rouge_l
from ..encoding.sequential import encode_sequential
from ..encoding.direct_concat import encode_direct_concat, compute_chunk_boundaries
from ..encoding.rope_adjusted import encode_rope_adjusted
from .timer import TTFTTimer, recomputation_ratio


def run_single(
    method: str,
    task_name: str,
    model_name: str,
    model,
    tokenizer,
    example: dict,
    n_chunks: int = 4,
    max_new_tokens: int = 100,
) -> dict:
    """
    Run one (method, task, example) combination and return a result record.
    method: "sequential" | "direct_concat" | "rope_adjusted"
    """
    prompt = format_prompt(example, task_name)
    ground_truths = get_ground_truths(example)

    timer = TTFTTimer()
    with timer:
        if method == "sequential":
            output_text, _, input_ids = encode_sequential(
                model, tokenizer, prompt, max_new_tokens
            )
            # sequential uses the full sequence as one chunk
            total_len = input_ids.shape[1]
            boundaries = [0, total_len]
        elif method == "direct_concat":
            output_text, _, input_ids, boundaries = encode_direct_concat(
                model, tokenizer, prompt, n_chunks, max_new_tokens
            )
        elif method == "rope_adjusted":
            output_text, _, input_ids, boundaries = encode_rope_adjusted(
                model, tokenizer, prompt, n_chunks, max_new_tokens
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

    context_len = input_ids.shape[1]
    ratio = recomputation_ratio(boundaries)

    task_type = TASK_TYPE[task_name]
    scores: dict = {}
    if task_type == "qa":
        scores["f1"] = f1_score(output_text, ground_truths)
        scores["substring"] = substring_match(output_text, ground_truths)
    else:
        ref = ground_truths[0] if ground_truths else ""
        scores["rouge_l"] = rouge_l(output_text, ref)

    return {
        "model": model_name,
        "method": method,
        "task": task_name,
        "n_chunks": n_chunks,
        "context_len": context_len,
        "ttft_ms": round(timer.elapsed_ms, 2),
        "recomputation_ratio": round(ratio, 6),
        "output": output_text,
        **scores,
    }
