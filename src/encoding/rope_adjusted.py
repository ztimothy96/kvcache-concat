import torch
from .direct_concat import compute_chunk_boundaries, _encode_chunks, generate_from_kv


@torch.no_grad()
def encode_rope_adjusted(
    model,
    tokenizer,
    prompt: str,
    n_chunks: int = 4,
    max_new_tokens: int = 100,
) -> tuple:
    """
    Encode chunks independently, apply per-chunk RoPE correction to K tensors,
    concatenate, then generate.
    Returns (output_text, concat_kv, input_ids, boundaries).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    boundaries = compute_chunk_boundaries(input_ids, n_chunks)
    concat_kv = _encode_chunks(model, input_ids, boundaries, rope_fix=True)
    output_text = generate_from_kv(model, tokenizer, input_ids, concat_kv, max_new_tokens)
    return output_text, concat_kv, input_ids, boundaries
