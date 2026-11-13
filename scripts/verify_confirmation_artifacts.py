"""Verify the frozen confirmation record against saved data and run metrics.

The confirmation YAML predates the component-path schema used by the current
training CLI.  This verifier intentionally reads its compact schema directly;
it checks the evidence used in the report without mutating configurations or
metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (201, 202, 203, 204, 205)
METHODS = ("baseline", "soap")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return result


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON mapping in {path}.")
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    experiment = load_yaml(ROOT / "configs/experiments/confirm_moonlight_soap_200m.yaml")
    data = load_yaml(ROOT / "configs/data/fineweb_edu_75m.yaml")
    model = load_yaml(ROOT / "configs/model/gpt_35m.yaml")
    manifest = load_json(ROOT / "data/manifests/fineweb_edu_75m_gpt2.json")

    batch_tokens = experiment["global_batch_tokens"]
    nominal_tokens = experiment["max_tokens"]
    completed_tokens = (nominal_tokens // batch_tokens) * batch_tokens
    expected_updates = completed_tokens // batch_tokens

    require(tuple(experiment["seeds"]) == SEEDS, "Confirmation seeds do not match.")
    require(data["revision"] == manifest["revision"], "Dataset revisions do not match.")
    require(data["tokenizer"] == manifest["tokenizer"], "Tokenizers do not match.")
    require(data["train_tokens"] == manifest["shards"]["train"]["tokens"], "Train shard differs.")
    require(data["validation_tokens"] == manifest["shards"]["validation"]["tokens"], "Validation shard differs.")
    require(data["test_tokens"] == manifest["shards"]["test"]["tokens"], "Test shard differs.")
    require(model["context_length"] == data["sequence_length"], "Model/data sequence lengths differ.")

    for seed in SEEDS:
        device_labels: set[str] = set()
        for method in METHODS:
            metric = load_json(ROOT / "results/confirmation_metrics" / f"seed{seed}-{method}.json")
            optimizer = metric["optimizer"]
            training = metric["training"]
            validation = metric["validation"]
            prefix = f"seed {seed} {method}"

            require(metric["seed"] == seed, f"{prefix}: seed mismatch.")
            require(metric["device"] in {"NVIDIA A10", "NVIDIA A10G"}, f"{prefix}: unsupported device label.")
            device_labels.add(metric["device"])
            require(metric["global_batch_tokens"] == batch_tokens, f"{prefix}: batch mismatch.")
            require(metric["gradient_accumulation_steps"] == 2, f"{prefix}: accumulation mismatch.")
            require(metric["microbatch_size"] * metric["gradient_accumulation_steps"] * model["context_length"] == batch_tokens, f"{prefix}: microbatch accounting mismatch.")
            require(training["tokens_processed"] == completed_tokens, f"{prefix}: completed tokens mismatch.")
            require(training["updates"] == expected_updates, f"{prefix}: update count mismatch.")
            require(validation["tokens_evaluated"] == 4_997_120, f"{prefix}: endpoint tokens mismatch.")
            require(math.isfinite(validation["mean_nll"]), f"{prefix}: non-finite endpoint NLL.")
            require(math.isfinite(training["tokens_per_second"]), f"{prefix}: non-finite throughput.")
            require(math.isfinite(training["peak_memory_bytes"]), f"{prefix}: non-finite memory.")
            require(optimizer["weight_decay"] == experiment["weight_decay"], f"{prefix}: decay mismatch.")
            require(optimizer["momentum"] == 0.95, f"{prefix}: momentum mismatch.")
            require(optimizer["newton_schulz_steps"] == 5, f"{prefix}: Newton--Schulz mismatch.")
            require(optimizer["rms_coefficient"] == 0.2, f"{prefix}: RMS coefficient mismatch.")
            require(optimizer["adamw_learning_rate"] == experiment["frozen_learning_rates"]["moonlight_muon"], f"{prefix}: AdamW LR mismatch.")
            require(len(training["curve"]) >= 2, f"{prefix}: insufficient curve points.")
            require(all(point["tokens_evaluated"] == 999_424 for point in training["curve"]), f"{prefix}: curve validation mismatch.")

            if method == "baseline":
                require(optimizer["implementation"] == "moonlight_muon", f"{prefix}: baseline implementation mismatch.")
                require(optimizer["moonlight_muon_learning_rate"] == experiment["frozen_learning_rates"]["moonlight_muon"], f"{prefix}: baseline LR mismatch.")
            else:
                require(optimizer["implementation"] == "moonlight_one_sided_soap_muon", f"{prefix}: SOAP implementation mismatch.")
                require(optimizer["hybrid_learning_rate"] == experiment["frozen_learning_rates"]["moonlight_soap_muon_both"], f"{prefix}: SOAP LR mismatch.")
                require(optimizer["route"] == "both", f"{prefix}: SOAP route mismatch.")
                require(optimizer["beta2"] == 0.95, f"{prefix}: SOAP beta2 mismatch.")
                require(optimizer["precondition_frequency"] == 10, f"{prefix}: SOAP refresh mismatch.")

        require(len(device_labels) == 1, f"seed {seed}: paired methods used different device labels.")

    print(
        "Confirmation artifact audit passed: 10 metric files, 5 paired seeds, "
        f"{completed_tokens:,} completed tokens, and {expected_updates:,} updates per run."
    )


if __name__ == "__main__":
    main()
