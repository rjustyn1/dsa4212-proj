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
        muon_lr=0.02,
        hybrid_lr=6e-4,
        adamw_lr=3e-4,
        muon_weight_decay=0.01,
        hybrid_weight_decay=0.01,
        adamw_weight_decay=0.01,
        muon_options={"newton_schulz_steps": 1},
        hybrid_options={"newton_schulz_steps": 1},
    )

    assert bundle.routes["blocks.0.attention.qkv.weight"] is OptimizerRoute.MUON
    assert bundle.routes["blocks.0.mlp.up_proj.weight"] is OptimizerRoute.ONE_SIDED_SOAP_MUON
    assert bundle.routes["token_embedding.weight"] is OptimizerRoute.ADAMW
    optimizer_settings = {
        type(optimizer).__name__: (optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"])
        for optimizer in bundle.optimizers
    }
    assert optimizer_settings["Muon"] == (0.02, 0.01)
    assert optimizer_settings["ReferenceOneSidedSOAPMuon"] == (6e-4, 0.01)
    assert optimizer_settings["AdamW"] == (3e-4, 0.01)
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


def test_muon_can_use_separate_auxiliary_adamw_learning_rate() -> None:
    model = tiny_model()
    bundle = build_optimizer_bundle(
        model,
        method="muon",
        lr=0.02,
        muon_lr=0.02,
        adamw_lr=3e-4,
        weight_decay=0.1,
        muon_weight_decay=0.01,
        adamw_weight_decay=0.01,
    )

    learning_rates = {
        type(optimizer).__name__: optimizer.param_groups[0]["lr"]
        for optimizer in bundle.optimizers
    }
    assert learning_rates["Muon"] == 0.02
    assert learning_rates["AdamW"] == 3e-4
    weight_decays = {
        type(optimizer).__name__: optimizer.param_groups[0]["weight_decay"]
        for optimizer in bundle.optimizers
    }
    assert weight_decays["Muon"] == 0.01
    assert weight_decays["AdamW"] == 0.01
