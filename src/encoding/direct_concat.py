import torch
from transformers import DynamicCache


def compute_chunk_boundaries(input_ids: torch.Tensor, n_chunks: int) -> list[int]:
    """Return token-level start indices [0, L1, L1+L2, ..., total] for n_chunks equal slices."""
    total = input_ids.shape[1]
    chunk_size = total // n_chunks
    boundaries = [i * chunk_size for i in range(n_chunks)] + [total]
    return boundaries


def _encode_chunks(model, input_ids: torch.Tensor, boundaries: list[int], rope_fix: bool):
    """
    Encode each chunk independently, optionally applying RoPE correction.
    Returns a concatenated KV tuple ready to wrap in DynamicCache.
    """
    from ..models.rope_utils import correct_chunk_rope

    chunks = [
        input_ids[:, s:e]
        for s, e in zip(boundaries[:-1], boundaries[1:])
    ]

    per_chunk_kvs = []
    global_start = 0

    with torch.no_grad():
        for chunk_ids in chunks:
            chunk_len = chunk_ids.shape[1]
            out = model(chunk_ids, use_cache=True)
            # past_key_values: tuple of (K, V) per layer
            kv = list(out.past_key_values)

            if rope_fix:
                kv = [
                    (correct_chunk_rope(k, global_start, chunk_len, model), v)
                    for k, v in kv
                ]

            per_chunk_kvs.append(kv)
            global_start += chunk_len

    num_layers = len(per_chunk_kvs[0])
    concat_kv = tuple(
        (
            torch.cat([c[layer][0] for c in per_chunk_kvs], dim=2),
            torch.cat([c[layer][1] for c in per_chunk_kvs], dim=2),
        )
        for layer in range(num_layers)
    )
    return concat_kv


def generate_from_kv(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    kv_tuple: tuple,
    max_new_tokens: int = 100,
) -> str:
    """Wrap KV tuple in DynamicCache and run model.generate."""
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_tuple):
        cache.update(k, v, layer_idx)

    context_len = kv_tuple[0][0].shape[2]
    attention_mask = torch.ones(1, context_len + 1, dtype=torch.long, device=model.device)

    # Pass only the last token as input_ids (everything else is in the cache)
    last_token = input_ids[:, -1:]

    gen_ids = model.generate(
        input_ids=last_token,
        attention_mask=attention_mask,
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return tokenizer.decode(gen_ids[0][1:], skip_special_tokens=True)


@torch.no_grad()
def encode_direct_concat(
    model,
    tokenizer,
    prompt: str,
    n_chunks: int = 4,
    max_new_tokens: int = 100,
) -> tuple:
    """
    Encode chunks independently, naively concatenate KV caches, then generate.
    Returns (output_text, concat_kv, input_ids).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    boundaries = compute_chunk_boundaries(input_ids, n_chunks)
    concat_kv = _encode_chunks(model, input_ids, boundaries, rope_fix=False)
    output_text = generate_from_kv(model, tokenizer, input_ids, concat_kv, max_new_tokens)
    return output_text, concat_kv, input_ids, boundaries
