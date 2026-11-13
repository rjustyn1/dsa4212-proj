"""Verify the exact-A10, shared-rate confirmation artifacts used in the report."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "shared_rate_confirmation_metrics"
SEEDS = (301, 302, 303, 304, 305)
METHODS = ("baseline", "soap")
EXPECTED_TOKENS = 199_983_104
EXPECTED_UPDATES = 6_103
EXPECTED_LR = 4.8e-3


def load_metric(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        metric = json.load(handle)
    if not isinstance(metric, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return metric


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    for seed in SEEDS:
        for method in METHODS:
            metric = load_metric(METRICS_DIR / f"seed{seed}-{method}.json")
            optimizer = metric["optimizer"]
            training = metric["training"]
            validation = metric["validation"]
            prefix = f"seed {seed} {method}"

            require(metric["seed"] == seed, f"{prefix}: seed mismatch.")
            require(metric["device"] == "NVIDIA A10", f"{prefix}: device is not NVIDIA A10.")
            require(metric["global_batch_tokens"] == 32_768, f"{prefix}: batch mismatch.")
            require(training["tokens_processed"] == EXPECTED_TOKENS, f"{prefix}: token mismatch.")
            require(training["updates"] == EXPECTED_UPDATES, f"{prefix}: update mismatch.")
            require(validation["tokens_evaluated"] == 4_997_120, f"{prefix}: endpoint token mismatch.")
            require(math.isfinite(validation["mean_nll"]), f"{prefix}: non-finite endpoint NLL.")
            require(math.isfinite(training["tokens_per_second"]), f"{prefix}: non-finite throughput.")
            require(math.isfinite(training["peak_memory_bytes"]), f"{prefix}: non-finite memory.")
            require(optimizer["adamw_learning_rate"] == EXPECTED_LR, f"{prefix}: AdamW learning rate mismatch.")
            if method == "baseline":
                require(
                    optimizer["moonlight_muon_learning_rate"] == EXPECTED_LR,
                    f"{prefix}: Muon learning rate mismatch.",
                )
            else:
                require(
                    optimizer["hybrid_learning_rate"] == EXPECTED_LR,
                    f"{prefix}: SOAP--Muon learning rate mismatch.",
                )
                require(optimizer["route"] == "both", f"{prefix}: SOAP route mismatch.")

    print("Shared-rate confirmation audit passed: 10 exact-A10 metrics across 5 paired seeds.")


if __name__ == "__main__":
    main()
