"""Validated, serializable experiment configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    n_layers: int
    d_model: int
    n_heads: int
    mlp_ratio: float
    dropout: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    tokenizer: str
    train_tokens: int
    validation_tokens: int
    test_tokens: int
    sequence_length: int
    revision: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    global_batch_tokens: int
    max_tokens: int
    learning_rate: float
    weight_decay: float
    warmup_fraction: float
    min_lr_fraction: float
    seed: int
    precision: str = "bf16"
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    optimizer_name: str
    optimizer: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable snapshot to save with each run."""
        return asdict(self)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or malformed configuration files."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(raw).__name__}.")
    return raw


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load one self-contained experiment YAML file.

    Component YAML files are intentionally referenced by path rather than merged
    through hidden defaults. Every resolved value is saved into the run folder.
    """
    experiment_path = Path(path).resolve()
    raw = load_yaml(experiment_path)
    root = experiment_path.parent

    def component(key: str) -> dict[str, Any]:
        reference = raw.get(key)
        if not isinstance(reference, str):
            raise ValueError(f"{key!r} must be a relative YAML path in {experiment_path}.")
        return load_yaml((root / reference).resolve())

    model = ModelConfig(**component("model"))
    data = DataConfig(**component("data"))
    optimizer = component("optimizer")
    training_raw = raw.get("training")
    if not isinstance(training_raw, dict):
        raise ValueError("'training' must be a mapping.")

    return ExperimentConfig(
        name=str(raw["name"]),
        model=model,
        data=data,
        training=TrainingConfig(**training_raw),
        optimizer_name=str(optimizer.pop("name")),
        optimizer=optimizer,
    )
