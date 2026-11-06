import torch

from soap_muon.config import ModelConfig
from soap_muon.models import GPTLanguageModel
from soap_muon.routing import OptimizerRoute
from soap_muon.training import benchmark_training, build_optimizer_bundle


def tiny_model() -> GPTLanguageModel:
    return GPTLanguageModel(
        ModelConfig(
            vocab_size=64,
            context_length=8,
            n_layers=1,
            d_model=16,
            n_heads=4,
            mlp_ratio=4.0,
        )
    )


def test_selective_bundle_routes_and_steps_every_parameter() -> None:
    model = tiny_model()
    bundle = build_optimizer_bundle(
        model,
        method="selective_soap_muon",
        lr=1e-3,
        weight_decay=0.1,
        muon_options={"newton_schulz_steps": 1},
        hybrid_options={"newton_schulz_steps": 1},
    )

    assert bundle.routes["blocks.0.attention.qkv.weight"] is OptimizerRoute.MUON
    assert bundle.routes["blocks.0.mlp.up_proj.weight"] is OptimizerRoute.ONE_SIDED_SOAP_MUON
    assert bundle.routes["token_embedding.weight"] is OptimizerRoute.ADAMW
    inputs = torch.randint(0, 64, (2, 8))
    _, loss = model(inputs, inputs)
    assert loss is not None
    loss.backward()
    bundle.step()


def test_cpu_benchmark_reports_tokens_and_speed() -> None:
    model = tiny_model()
    bundle = build_optimizer_bundle(
        model,
        method="muon",
        lr=1e-3,
        weight_decay=0.1,
        muon_options={"newton_schulz_steps": 1},
    )

    result = benchmark_training(
        model,
        bundle,
        method="muon",
        batch_size=2,
        sequence_length=8,
        steps=3,
        warmup_steps=1,
        device=torch.device("cpu"),
    )

    assert result.tokens_processed == 32
    assert result.tokens_per_second > 0
    assert result.peak_memory_bytes is None
