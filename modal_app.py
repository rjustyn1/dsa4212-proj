"""Bounded Modal entry points for the DSA4212 optimizer experiments.

Run the inexpensive environment check with:
    uv run modal run modal_app.py

No credentials are stored in this repository. ``modal setup`` must be completed
interactively by the account owner before any remote invocation can run.
"""

from __future__ import annotations

import json

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

app = modal.App(APP_NAME)

# Keep the remote environment separate from the user's local virtualenv. The
# training stack will be pinned once the reference implementation is chosen.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyyaml", "torch")
    .add_local_python_source("soap_muon")
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


@app.function(gpu=GPU_TYPE, timeout=SMOKE_TIMEOUT_SECONDS, image=image)
def smoke_train() -> str:
    """Reserved bounded entry point for the later 2M-token smoke experiment.

    This intentionally refuses to start until the training code is implemented;
    it prevents accidentally spending GPU credit on an implicit default run.
    """
    raise RuntimeError(
        "Training is not implemented yet. Run gpu_probe first, then add an "
        "explicit model, dataset revision, token budget, and checkpoint policy."
    )


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
