"""Bounded fixed-shape training benchmark for fair GPU selection."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from soap_muon.training.optimizers import OptimizerBundle


@dataclass(frozen=True)
class TrainingBenchmarkResult:
    method: str
    tokens_processed: int
    elapsed_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int | None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def benchmark_training(
    model: nn.Module,
    optimizer: OptimizerBundle,
    *,
    method: str,
    batch_size: int,
    sequence_length: int,
    steps: int,
    warmup_steps: int,
    device: torch.device,
    use_bf16_autocast: bool = True,
) -> TrainingBenchmarkResult:
    """Measure full forward/backward/optimizer time on a constant synthetic batch."""
    if batch_size <= 0 or sequence_length <= 0 or steps <= warmup_steps:
        raise ValueError("batch_size and sequence_length must be positive; steps must exceed warmup_steps.")
    model.to(device)
    model.train()
    input_ids = torch.arange(
        batch_size * sequence_length, device=device, dtype=torch.long
    ).reshape(batch_size, sequence_length) % model.config.vocab_size
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    autocast_enabled = use_bf16_autocast and device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("Model did not return a training loss.")
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps - warmup_steps):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("Model did not return a training loss.")
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes: int | None = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory_bytes = None

    elapsed_seconds = time.perf_counter() - start
    tokens_processed = (steps - warmup_steps) * batch_size * sequence_length
    return TrainingBenchmarkResult(
        method=method,
        tokens_processed=tokens_processed,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_processed / elapsed_seconds,
        peak_memory_bytes=peak_memory_bytes,
    )
