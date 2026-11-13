"""Read-only contiguous-token shards used for comparable language-model evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def load_token_shard(path: str | Path) -> np.ndarray:
    """Open a one-dimensional integer ``.npy`` token shard without copying it."""
    tokens = np.load(Path(path), mmap_mode="r")
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("A token shard must be a one-dimensional integer NumPy array.")
    if len(tokens) < 2:
        raise ValueError("A token shard needs at least two tokens.")
    return tokens


def contiguous_batches(
    tokens: np.ndarray,
    sequence_length: int,
    batch_size: int,
    max_batches: int | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield deterministic next-token prediction batches from a fixed token stream."""
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("sequence_length and batch_size must be positive.")

    available_sequences = (len(tokens) - 1) // sequence_length
    full_batches = available_sequences // batch_size
    if max_batches is not None:
        full_batches = min(full_batches, max_batches)
    if full_batches == 0:
        raise ValueError("Token shard is too short for one complete evaluation batch.")

    for batch_index in range(full_batches):
        start = batch_index * batch_size * sequence_length
        stop = start + batch_size * sequence_length + 1
        # Copying the small active slice avoids a warning for read-only mmap data.
        window = torch.from_numpy(np.array(tokens[start:stop], dtype=np.int64, copy=True))
        inputs = window[:-1].view(batch_size, sequence_length)
        targets = window[1:].view(batch_size, sequence_length)
        yield inputs, targets
