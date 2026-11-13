"""Versioned checkpoint contract shared by every training method and evaluator."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from soap_muon.config import ModelConfig


CHECKPOINT_FORMAT_VERSION = 1


def save_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_config: ModelConfig,
    *,
    tokens_seen: int,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model-only state suitable for fair, inference-only evaluation.

    Optimizer state belongs in resumable training checkpoints, not in the
    portable ``best.pt`` artifacts discovered by the evaluation driver.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": asdict(model_config),
            "model_state_dict": model.state_dict(),
            "tokens_seen": tokens_seen,
            "step": step,
            "metadata": metadata or {},
        },
        target,
    )


def load_model_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load and minimally validate a portable model-only checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    required = {"format_version", "model_config", "model_state_dict", "tokens_seen", "step"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing fields: {sorted(missing)}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {payload['format_version']}")
    return payload
