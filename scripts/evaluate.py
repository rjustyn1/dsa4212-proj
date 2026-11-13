"""Evaluate every completed model checkpoint against one fixed validation shard.

Example:
    uv run python scripts/evaluate.py \
        --runs-root runs \
        --validation-shard data/processed/fineweb_edu_validation.npy \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from soap_muon.config import ModelConfig
from soap_muon.data.token_shards import contiguous_batches, load_token_shard
from soap_muon.models import GPTLanguageModel
from soap_muon.training.checkpointing import load_model_checkpoint
from soap_muon.training.evaluation import evaluate_next_token_prediction


def discover_checkpoints(runs_root: Path, checkpoint_name: str) -> list[Path]:
    checkpoints = sorted(runs_root.rglob(checkpoint_name))
    if not checkpoints:
        raise FileNotFoundError(f"No {checkpoint_name!r} files found beneath {runs_root}.")
    return checkpoints


def evaluate_checkpoint(
    checkpoint_path: Path,
    validation_tokens_path: Path,
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
) -> dict[str, object]:
    payload = load_model_checkpoint(checkpoint_path, device)
    model_config = ModelConfig(**payload["model_config"])
    model = GPTLanguageModel(model_config)
    model.load_state_dict(payload["model_state_dict"])
    tokens = load_token_shard(validation_tokens_path)
    result = evaluate_next_token_prediction(
        model,
        contiguous_batches(tokens, model_config.context_length, batch_size, max_batches),
        device,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model_parameters": model.parameter_count,
        "tokens_seen_at_checkpoint": payload["tokens_seen"],
        "training_step_at_checkpoint": payload["step"],
        **result.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation_summary.jsonl"))
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    checkpoints = discover_checkpoints(args.runs_root, args.checkpoint_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for checkpoint in checkpoints:
            record = evaluate_checkpoint(
                checkpoint,
                args.validation_shard,
                device,
                args.batch_size,
                args.max_batches,
            )
            output.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
