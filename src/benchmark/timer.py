import time
import torch


class TTFTTimer:
    """Context manager that measures Time-To-First-Token in milliseconds."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def recomputation_ratio(boundaries: list[int]) -> float:
    """
    Compute attention-FLOPs ratio relative to a single full-sequence pass.

    For a sequence of total T tokens split into chunks of sizes L_c:
      ratio = sum(L_c^2) / T^2

    For N equal chunks: ratio = 1/N.
    For sequential (single chunk equal to full sequence): ratio = 1.0.
    """
    chunk_sizes = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    total = sum(chunk_sizes)
    if total == 0:
        return 1.0
    return sum(l**2 for l in chunk_sizes) / (total**2)
