"""Deterministic fixed-token training used by the optimizer experiments."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from soap_muon.training.optimizers import OptimizerBundle


@dataclass(frozen=True)
class TrainingRunResult:
    updates: int
    tokens_processed: int
    mean_loss: float
    final_loss: float
    elapsed_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int | None
    data_passes: int

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def cosine_learning_rate(
    base_learning_rate: float,
    min_lr_fraction: float,
    warmup_steps: int,
    total_steps: int,
    step: int,
) -> float:
    """Return a warmup-plus-cosine rate indexed by zero-based update step."""
    if not 0.0 < min_lr_fraction <= 1.0:
        raise ValueError("min_lr_fraction must be in (0, 1].")
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("step must index a positive total number of steps.")
    if warmup_steps and step < warmup_steps:
        return base_learning_rate * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * (min_lr_fraction + (1.0 - min_lr_fraction) * cosine)


def train_fixed_token_budget(
    model: nn.Module,
    optimizer: OptimizerBundle,
    batch_factory: Callable[[], Iterator[tuple[Tensor, Tensor]]],
    *,
    global_batch_tokens: int,
    max_tokens: int,
    learning_rate: float,
    warmup_fraction: float,
    min_lr_fraction: float,
    gradient_clip_norm: float,
    gradient_accumulation_steps: int,
    device: torch.device,
    use_bf16_autocast: bool = True,
) -> TrainingRunResult:
    """Train whole updates only, never exceeding the requested token budget."""
    if global_batch_tokens <= 0 or max_tokens <= 0:
        raise ValueError("global_batch_tokens and max_tokens must be positive.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1).")
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")

    total_steps = max_tokens // global_batch_tokens
    if total_steps == 0:
        raise ValueError("max_tokens must contain at least one global batch.")
    warmup_steps = int(total_steps * warmup_fraction)
    autocast_enabled = use_bf16_autocast and device.type == "cuda"

    model.to(device)
    model.train()
    base_learning_rates = [
        (inner_optimizer, tuple(float(group["lr"]) for group in inner_optimizer.param_groups))
        for inner_optimizer in optimizer.optimizers
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    batches = batch_factory()
    data_passes = 1
    loss_sum = 0.0
    final_loss = 0.0

    for step in range(total_steps):
        rate_multiplier = cosine_learning_rate(
            1.0, min_lr_fraction, warmup_steps, total_steps, step
        )
        for inner_optimizer, initial_rates in base_learning_rates:
            for group, initial_rate in zip(inner_optimizer.param_groups, initial_rates, strict=True):
                group["lr"] = initial_rate * rate_multiplier

        optimizer.zero_grad()
        step_loss = 0.0
        step_tokens = 0
        for _ in range(gradient_accumulation_steps):
            try:
                inputs, targets = next(batches)
            except StopIteration:
                batches = batch_factory()
                data_passes += 1
                inputs, targets = next(batches)
            batch_tokens = inputs.numel()
            step_tokens += batch_tokens
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                _, loss = model(inputs.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("Model did not return a training loss.")
            step_loss += float(loss.detach())
            (loss / gradient_accumulation_steps).backward()

        if step_tokens != global_batch_tokens:
            raise ValueError(
                "Microbatches do not add up to global_batch_tokens: "
                f"expected {global_batch_tokens}, got {step_tokens}."
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        loss_sum += step_loss / gradient_accumulation_steps
        final_loss = step_loss / gradient_accumulation_steps

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes: int | None = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory_bytes = None
    elapsed_seconds = time.perf_counter() - start
    tokens_processed = total_steps * global_batch_tokens
    return TrainingRunResult(
        updates=total_steps,
        tokens_processed=tokens_processed,
        mean_loss=loss_sum / total_steps,
        final_loss=final_loss,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_processed / elapsed_seconds,
        peak_memory_bytes=peak_memory_bytes,
        data_passes=data_passes,
    )
