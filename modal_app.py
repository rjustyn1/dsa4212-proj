"""Bounded Modal entry points for the DSA4212 optimizer experiments.

Run the inexpensive environment check with:
    uv run modal run modal_app.py

No credentials are stored in this repository. ``modal setup`` must be completed
interactively by the account owner before any remote invocation can run.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "dsa4212-soap-muon"
GPU_TYPE = "L4"
PROBE_TIMEOUT_SECONDS = 60 * 60
SMOKE_TIMEOUT_SECONDS = 2 * 60 * 60
BENCHMARK_TIMEOUT_SECONDS = 20 * 60
BENCHMARK_BATCH_SIZE = 32
BENCHMARK_SEQUENCE_LENGTH = 512
BENCHMARK_WARMUP_STEPS = 20
BENCHMARK_STEPS = 80
TRAINING_MICROBATCH_SIZE = 32
TRAINING_GRADIENT_ACCUMULATION_STEPS = 2
FROZEN_DATA_DIRECTORY = "data/processed/fineweb_edu_75m_gpt2"
DIRECT_BASELINE_TOKENS = 200_000_000
RESULTS_VOLUME_NAME = "dsa4212-optimizer-results"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

# Keep the remote environment separate from the user's local virtualenv. The
# training stack will be pinned once the reference implementation is chosen.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyyaml", "torch")
    .add_local_python_source("soap_muon", copy=True)
    .add_local_dir(FROZEN_DATA_DIRECTORY, "/data", copy=True)
)


@app.function(gpu=GPU_TYPE, timeout=PROBE_TIMEOUT_SECONDS, image=image)
def gpu_probe() -> str:
    """Return enough information to validate the selected remote GPU."""
    import torch

    available = torch.cuda.is_available()
    # Return JSON rather than a Python object containing a torch-specific
    # version type: the local client deliberately does not install PyTorch.
    result = json.dumps({
        "cuda_available": available,
        "device": torch.cuda.get_device_name(0) if available else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    })
    print(result)
    return result


def _run_baseline(
    *,
    method: str,
    max_tokens: int,
    optimizer_learning_rate: float,
    seed: int = 1,
    adamw_learning_rate: float | None = None,
    muon_learning_rate: float | None = None,
    hybrid_learning_rate: float | None = None,
    adamw_weight_decay: float | None = None,
    muon_weight_decay: float | None = None,
    hybrid_weight_decay: float | None = None,
    hybrid_options: dict[str, object] | None = None,
) -> tuple[object, object, dict[str, object]]:
    """Train and validate one configured optimizer against the frozen corpus."""
    import torch

    from soap_muon.config import ModelConfig
    from soap_muon.data import contiguous_batches, load_token_shard
    from soap_muon.models import GPTLanguageModel
    from soap_muon.training.evaluation import evaluate_next_token_prediction
    from soap_muon.training.optimizers import build_optimizer_bundle
    from soap_muon.training.runner import train_fixed_token_budget

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = ModelConfig(
        vocab_size=50257,
        context_length=512,
        n_layers=8,
        d_model=384,
        n_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    model = GPTLanguageModel(config)
    optimizer = build_optimizer_bundle(
        model,
        method=method,
        lr=optimizer_learning_rate,
        adamw_lr=adamw_learning_rate,
        muon_lr=muon_learning_rate,
        hybrid_lr=hybrid_learning_rate,
        weight_decay=0.1,
        adamw_weight_decay=adamw_weight_decay,
        muon_weight_decay=muon_weight_decay,
        hybrid_weight_decay=hybrid_weight_decay,
        hybrid_options=hybrid_options,
    )
    device = torch.device("cuda")
    train_result = train_fixed_token_budget(
        model,
        optimizer,
        lambda: contiguous_batches(
            load_token_shard("/data/train.npy"),
            sequence_length=config.context_length,
            batch_size=TRAINING_MICROBATCH_SIZE,
        ),
        global_batch_tokens=32768,
        max_tokens=max_tokens,
        learning_rate=optimizer_learning_rate,
        warmup_fraction=0.05,
        min_lr_fraction=0.1,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=TRAINING_GRADIENT_ACCUMULATION_STEPS,
        device=device,
    )
    validation_result = evaluate_next_token_prediction(
        model,
        contiguous_batches(
            load_token_shard("/data/validation.npy"),
            sequence_length=config.context_length,
            batch_size=TRAINING_MICROBATCH_SIZE,
        ),
        device,
    )
    return model, config, {
        "method": method,
        "seed": seed,
        "device": torch.cuda.get_device_name(0),
        "microbatch_size": TRAINING_MICROBATCH_SIZE,
        "gradient_accumulation_steps": TRAINING_GRADIENT_ACCUMULATION_STEPS,
        "global_batch_tokens": 32768,
        "training": train_result.as_dict(),
        "validation": validation_result.as_dict(),
    }


@app.function(gpu="A10", timeout=SMOKE_TIMEOUT_SECONDS, image=image)
def smoke_train() -> str:
    """Run the explicit 2M-token AdamW smoke experiment on frozen data."""
    _, _, result = _run_baseline(
        method="adamw", max_tokens=2_000_000, optimizer_learning_rate=6e-4
    )
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


@app.function(
    gpu="A10",
    timeout=SMOKE_TIMEOUT_SECONDS,
    image=image,
    volumes={"/runs": results_volume},
)
def baseline_adamw_200m() -> str:
    """Run and persist the direct 200M-token AdamW baseline."""
    from soap_muon.training.checkpointing import save_model_checkpoint

    model, config, result = _run_baseline(
        method="adamw",
        max_tokens=DIRECT_BASELINE_TOKENS,
        optimizer_learning_rate=6e-4,
    )
    training = result["training"]
    if not isinstance(training, dict):
        raise RuntimeError("Training result is malformed.")
    target = Path("/runs/adamw-baseline-a10-seed1-200m")
    target.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        target / "final.pt",
        model,
        config,
        tokens_seen=int(training["tokens_processed"]),
        step=int(training["updates"]),
        metadata={
            "experiment": "baseline-adamw-200m",
            "global_batch_tokens": 32768,
            "microbatch_size": TRAINING_MICROBATCH_SIZE,
            "gradient_accumulation_steps": TRAINING_GRADIENT_ACCUMULATION_STEPS,
        },
    )
    (target / "metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    results_volume.commit()
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


@app.function(
    gpu="A10",
    timeout=SMOKE_TIMEOUT_SECONDS,
    image=image,
    volumes={"/runs": results_volume},
)
def baseline_muon_200m() -> str:
    """Run and persist the matched direct 200M-token Muon baseline."""
    from soap_muon.training.checkpointing import save_model_checkpoint

    model, config, result = _run_baseline(
        method="muon",
        max_tokens=DIRECT_BASELINE_TOKENS,
        optimizer_learning_rate=0.02,
        muon_learning_rate=0.02,
        adamw_learning_rate=3e-4,
    )
    training = result["training"]
    if not isinstance(training, dict):
        raise RuntimeError("Training result is malformed.")
    target = Path("/runs/muon-baseline-a10-seed1-200m")
    target.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        target / "final.pt",
        model,
        config,
        tokens_seen=int(training["tokens_processed"]),
        step=int(training["updates"]),
        metadata={
            "experiment": "baseline-muon-200m",
            "global_batch_tokens": 32768,
            "microbatch_size": TRAINING_MICROBATCH_SIZE,
            "gradient_accumulation_steps": TRAINING_GRADIENT_ACCUMULATION_STEPS,
            "muon_learning_rate": 0.02,
            "adamw_learning_rate": 3e-4,
        },
    )
    (target / "metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    results_volume.commit()
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


@app.function(
    gpu="A10",
    timeout=SMOKE_TIMEOUT_SECONDS,
    image=image,
    volumes={"/runs": results_volume},
)
def reference_muon_200m(seed: int = 1) -> str:
    """Run and persist the reference-scaled 200M-token Muon baseline."""
    from soap_muon.training.checkpointing import save_model_checkpoint

    model, config, result = _run_baseline(
        method="muon",
        max_tokens=DIRECT_BASELINE_TOKENS,
        optimizer_learning_rate=0.02,
        seed=seed,
        muon_learning_rate=0.02,
        adamw_learning_rate=3e-4,
        muon_weight_decay=0.01,
        adamw_weight_decay=0.01,
    )
    training = result["training"]
    if not isinstance(training, dict):
        raise RuntimeError("Training result is malformed.")
    result["optimizer"] = {
        "muon_learning_rate": 0.02,
        "adamw_learning_rate": 3e-4,
        "muon_weight_decay": 0.01,
        "adamw_weight_decay": 0.01,
    }
    target = Path(f"/runs/reference-muon-a10-seed{seed}-200m")
    target.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        target / "final.pt",
        model,
        config,
        tokens_seen=int(training["tokens_processed"]),
        step=int(training["updates"]),
        metadata={"experiment": "reference-muon-200m", "seed": seed, **result["optimizer"]},
    )
    (target / "metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    results_volume.commit()
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


@app.function(
    gpu="A10",
    timeout=SMOKE_TIMEOUT_SECONDS,
    image=image,
    volumes={"/runs": results_volume},
)
def selective_soap_muon_200m() -> str:
    """Run and persist the direct selective SOAP–Muon primary experiment."""
    from soap_muon.training.checkpointing import save_model_checkpoint

    optimizer_settings = {
        "muon_learning_rate": 0.02,
        "hybrid_learning_rate": 6e-4,
        "adamw_learning_rate": 3e-4,
        "muon_weight_decay": 0.01,
        "hybrid_weight_decay": 0.01,
        "adamw_weight_decay": 0.01,
    }
    model, config, result = _run_baseline(
        method="selective_soap_muon",
        max_tokens=DIRECT_BASELINE_TOKENS,
        optimizer_learning_rate=optimizer_settings["hybrid_learning_rate"],
        muon_learning_rate=optimizer_settings["muon_learning_rate"],
        hybrid_learning_rate=optimizer_settings["hybrid_learning_rate"],
        adamw_learning_rate=optimizer_settings["adamw_learning_rate"],
        muon_weight_decay=optimizer_settings["muon_weight_decay"],
        hybrid_weight_decay=optimizer_settings["hybrid_weight_decay"],
        adamw_weight_decay=optimizer_settings["adamw_weight_decay"],
        hybrid_options={
            "betas": (0.95, 0.95),
            "eps": 1e-8,
            "precondition_frequency": 10,
            "newton_schulz_steps": 5,
            "correct_bias": True,
            "nesterov": False,
        },
    )
    training = result["training"]
    if not isinstance(training, dict):
        raise RuntimeError("Training result is malformed.")
    result["optimizer"] = optimizer_settings
    target = Path("/runs/selective-soap-muon-a10-seed1-200m")
    target.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        target / "final.pt",
        model,
        config,
        tokens_seen=int(training["tokens_processed"]),
        step=int(training["updates"]),
        metadata={"experiment": "selective-soap-muon-200m", **optimizer_settings},
    )
    (target / "metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    results_volume.commit()
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


@app.function(
    gpu="A10",
    timeout=SMOKE_TIMEOUT_SECONDS,
    image=image,
    volumes={"/runs": results_volume},
)
def tune_selective_soap_muon_200m(
    hybrid_learning_rate: float = 0.0018, seed: int = 1
) -> str:
    """Run and persist one predeclared selective SOAP–Muon rate candidate."""
    from soap_muon.training.checkpointing import save_model_checkpoint

    allowed_learning_rates = (0.0018, 0.003)
    if hybrid_learning_rate not in allowed_learning_rates:
        raise ValueError(f"hybrid_learning_rate must be one of {allowed_learning_rates}.")
    optimizer_settings = {
        "muon_learning_rate": 0.02,
        "hybrid_learning_rate": hybrid_learning_rate,
        "adamw_learning_rate": 3e-4,
        "muon_weight_decay": 0.01,
        "hybrid_weight_decay": 0.01,
        "adamw_weight_decay": 0.01,
    }
    model, config, result = _run_baseline(
        method="selective_soap_muon",
        max_tokens=DIRECT_BASELINE_TOKENS,
        optimizer_learning_rate=hybrid_learning_rate,
        seed=seed,
        muon_learning_rate=optimizer_settings["muon_learning_rate"],
        hybrid_learning_rate=hybrid_learning_rate,
        adamw_learning_rate=optimizer_settings["adamw_learning_rate"],
        muon_weight_decay=optimizer_settings["muon_weight_decay"],
        hybrid_weight_decay=optimizer_settings["hybrid_weight_decay"],
        adamw_weight_decay=optimizer_settings["adamw_weight_decay"],
        hybrid_options={
            "betas": (0.95, 0.95),
            "eps": 1e-8,
            "precondition_frequency": 10,
            "newton_schulz_steps": 5,
            "correct_bias": True,
            "nesterov": False,
        },
    )
    training = result["training"]
    if not isinstance(training, dict):
        raise RuntimeError("Training result is malformed.")
    result["optimizer"] = optimizer_settings
    candidate = f"lr{int(hybrid_learning_rate * 10000):04d}"
    target = Path(f"/runs/selective-soap-muon-tune-{candidate}-a10-seed{seed}-200m")
    target.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        target / "final.pt",
        model,
        config,
        tokens_seen=int(training["tokens_processed"]),
        step=int(training["updates"]),
        metadata={
            "experiment": f"selective-soap-muon-tune-{candidate}",
            "seed": seed,
            **optimizer_settings,
        },
    )
    (target / "metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    results_volume.commit()
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


@app.function(
    gpu="A10",
    timeout=SMOKE_TIMEOUT_SECONDS,
    image=image,
    volumes={"/runs": results_volume},
)
def tune_muon_200m(muon_learning_rate: float = 0.02) -> str:
    """Run one predeclared direct Muon tuning candidate and persist its result."""
    from soap_muon.training.checkpointing import save_model_checkpoint

    allowed_learning_rates = (0.01, 0.02, 0.03)
    if muon_learning_rate not in allowed_learning_rates:
        raise ValueError(f"muon_learning_rate must be one of {allowed_learning_rates}.")
    model, config, result = _run_baseline(
        method="muon",
        max_tokens=DIRECT_BASELINE_TOKENS,
        optimizer_learning_rate=muon_learning_rate,
        muon_learning_rate=muon_learning_rate,
        adamw_learning_rate=3e-4,
        muon_weight_decay=0.01,
        adamw_weight_decay=0.01,
    )
    training = result["training"]
    if not isinstance(training, dict):
        raise RuntimeError("Training result is malformed.")
    candidate = f"lr{int(muon_learning_rate * 1000):03d}-wd001"
    target = Path(f"/runs/muon-tune-{candidate}-a10-seed1-200m")
    target.mkdir(parents=True, exist_ok=True)
    result["optimizer"] = {
        "muon_learning_rate": muon_learning_rate,
        "adamw_learning_rate": 3e-4,
        "muon_weight_decay": 0.01,
        "adamw_weight_decay": 0.01,
    }
    save_model_checkpoint(
        target / "final.pt",
        model,
        config,
        tokens_seen=int(training["tokens_processed"]),
        step=int(training["updates"]),
        metadata={"experiment": f"muon-tune-{candidate}", **result["optimizer"]},
    )
    (target / "metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    results_volume.commit()
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


def _benchmark_selective_hybrid() -> str:
    """Run the hardware-comparison workload without reading the frozen corpus."""
    import torch

    from soap_muon.config import ModelConfig
    from soap_muon.models import GPTLanguageModel
    from soap_muon.training import benchmark_training, build_optimizer_bundle

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = GPTLanguageModel(
        ModelConfig(
            vocab_size=50257,
            context_length=BENCHMARK_SEQUENCE_LENGTH,
            n_layers=8,
            d_model=384,
            n_heads=6,
            mlp_ratio=4.0,
            dropout=0.0,
        )
    )
    optimizer = build_optimizer_bundle(
        model,
        method="selective_soap_muon",
        lr=6e-4,
        weight_decay=0.1,
        muon_options={"momentum": 0.95, "newton_schulz_steps": 5},
        hybrid_options={
            "betas": (0.95, 0.95),
            "eps": 1e-8,
            "precondition_frequency": 10,
            "newton_schulz_steps": 5,
        },
    )
    device = torch.device("cuda")
    result = benchmark_training(
        model,
        optimizer,
        method="selective_soap_muon",
        batch_size=BENCHMARK_BATCH_SIZE,
        sequence_length=BENCHMARK_SEQUENCE_LENGTH,
        steps=BENCHMARK_STEPS,
        warmup_steps=BENCHMARK_WARMUP_STEPS,
        device=device,
    ).as_dict()
    result.update(
        {
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "measured_tokens": (BENCHMARK_STEPS - BENCHMARK_WARMUP_STEPS)
            * BENCHMARK_BATCH_SIZE
            * BENCHMARK_SEQUENCE_LENGTH,
            "torch_version": str(torch.__version__),
        }
    )
    output = json.dumps(result, sort_keys=True)
    print(output)
    return output


@app.function(gpu="L4", timeout=BENCHMARK_TIMEOUT_SECONDS, image=image)
def benchmark_l4() -> str:
    """Benchmark the fixed selective-hybrid workload on an L4."""
    return _benchmark_selective_hybrid()


@app.function(gpu="A10", timeout=BENCHMARK_TIMEOUT_SECONDS, image=image)
def benchmark_a10() -> str:
    """Benchmark the fixed selective-hybrid workload on an A10."""
    return _benchmark_selective_hybrid()


@app.local_entrypoint()
def main() -> None:
    """Default command: run only the inexpensive GPU probe."""
    print(gpu_probe.remote())
