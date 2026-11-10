import numpy as np
import torch

from soap_muon.config import ModelConfig
from soap_muon.data import contiguous_batches
from soap_muon.models import GPTLanguageModel
from soap_muon.training.optimizers import build_optimizer_bundle
from soap_muon.training.runner import train_fixed_token_budget


def test_fixed_token_training_uses_complete_global_batches() -> None:
    torch.manual_seed(1)
    model = GPTLanguageModel(
        ModelConfig(
            vocab_size=32,
            context_length=4,
            n_layers=1,
            d_model=16,
            n_heads=4,
            mlp_ratio=2.0,
        )
    )
    optimizer = build_optimizer_bundle(
        model, method="adamw", lr=1e-3, weight_decay=0.0
    )
    tokens = (np.arange(129, dtype=np.int64) % 32).copy()
    result = train_fixed_token_budget(
        model,
        optimizer,
        lambda: contiguous_batches(tokens, sequence_length=4, batch_size=2),
        global_batch_tokens=16,
        max_tokens=160,
        learning_rate=1e-3,
        warmup_fraction=0.5,
        min_lr_fraction=0.1,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=2,
        device=torch.device("cpu"),
    )

    assert result.updates == 10
    assert result.tokens_processed == 160
    assert result.peak_memory_bytes is None
    assert result.data_passes == 2
    assert result.mean_loss > 0.0


def test_fixed_token_training_records_validation_curve() -> None:
    torch.manual_seed(2)
    model = GPTLanguageModel(
        ModelConfig(
            vocab_size=32,
            context_length=4,
            n_layers=1,
            d_model=16,
            n_heads=4,
            mlp_ratio=2.0,
        )
    )
    optimizer = build_optimizer_bundle(model, method="adamw", lr=1e-3, weight_decay=0.0)
    tokens = (np.arange(129, dtype=np.int64) % 32).copy()
    result = train_fixed_token_budget(
        model,
        optimizer,
        lambda: contiguous_batches(tokens, sequence_length=4, batch_size=2),
        global_batch_tokens=16,
        max_tokens=70,
        learning_rate=1e-3,
        warmup_fraction=0.0,
        min_lr_fraction=0.1,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=2,
        device=torch.device("cpu"),
        evaluation_interval_tokens=48,
        evaluation_callback=lambda: {"validation_nll": 1.5, "validation_tokens": 16},
    )

    assert [point["tokens_processed"] for point in result.curve] == [48, 64]
    assert all(point["validation_nll"] == 1.5 for point in result.curve)
