import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., seq, head_dim), cos/sin: (1, 1, seq, head_dim) or broadcastable."""
    return x * cos + rotate_half(x) * sin


def apply_inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse of apply_rope: R(-θ) = cos*x - sin*rotate_half(x)."""
    return x * cos + rotate_half(x) * (-sin)


def get_rope_cos_sin(model, position_ids: torch.Tensor, head_dim: int):
    """
    Get RoPE cos/sin for given position_ids using the model's own rotary embedding.

    position_ids: (1, seq_len) LongTensor
    Returns: cos, sin each shaped (1, 1, seq_len, head_dim) — broadcast-ready.
    """
    rope = model.model.layers[0].self_attn.rotary_emb
    dummy = torch.zeros(
        1, position_ids.shape[-1], head_dim,
        dtype=model.dtype, device=model.device,
    )
    cos, sin = rope(dummy, position_ids)
    # cos/sin come out as (1, seq_len, head_dim); add heads dim for broadcasting
    return cos.unsqueeze(1), sin.unsqueeze(1)


def correct_chunk_rope(
    k_chunk: torch.Tensor,
    global_start: int,
    chunk_len: int,
    model,
) -> torch.Tensor:
    """
    Re-rotate cached K tensors from chunk-local positions to global positions.

    When a chunk is encoded independently starting at position 0, each token j
    gets RoPE angle j*θ. But token j's true global position is (global_start + j),
    so its K should have angle (global_start + j)*θ.

    We fix this by: inverse-rotate (remove local angle) then re-rotate (apply global angle).

    k_chunk: (batch, heads, chunk_len, head_dim)
    Returns: corrected K of the same shape.
    """
    head_dim = k_chunk.shape[-1]
    device = k_chunk.device

    local_ids = torch.arange(chunk_len, device=device, dtype=torch.long).unsqueeze(0)
    global_ids = local_ids + global_start

    cos_local, sin_local = get_rope_cos_sin(model, local_ids, head_dim)
    cos_global, sin_global = get_rope_cos_sin(model, global_ids, head_dim)

    k_unrotated = apply_inverse_rope(k_chunk, cos_local, sin_local)
    return apply_rope(k_unrotated, cos_global, sin_global)
