"""Inference-only language-model evaluation independent of optimizer choice."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EvaluationResult:
    mean_nll: float
    perplexity: float
    tokens_evaluated: int
    elapsed_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def evaluate_next_token_prediction(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> EvaluationResult:
    """Evaluate exact next-token NLL without gradients or parameter updates."""
    was_training = model.training
    model.eval()
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    token_weighted_nll = 0.0
    tokens_evaluated = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in batches:
            _, loss = model(inputs.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("Language model did not return a loss.")
            token_count = targets.numel()
            token_weighted_nll += float(loss) * token_count
            tokens_evaluated += token_count

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes: int | None = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory_bytes = None
    elapsed_seconds = time.perf_counter() - start
    if tokens_evaluated == 0:
        raise ValueError("Evaluation received no tokens.")

    mean_nll = token_weighted_nll / tokens_evaluated
    model.train(was_training)
    return EvaluationResult(
        mean_nll=mean_nll,
        perplexity=math.exp(mean_nll),
        tokens_evaluated=tokens_evaluated,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_evaluated / elapsed_seconds,
        peak_memory_bytes=peak_memory_bytes,
    )
