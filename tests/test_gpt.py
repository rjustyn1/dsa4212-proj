import torch

from soap_muon.config import ModelConfig
from soap_muon.models import GPTLanguageModel


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        context_length=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        mlp_ratio=4.0,
    )


def test_forward_returns_logits_and_cross_entropy_loss() -> None:
    model = GPTLanguageModel(tiny_config())
    tokens = torch.randint(0, 128, (2, 8))
    logits, loss = model(tokens, tokens)

    assert logits.shape == (2, 8, 128)
    assert loss is not None and torch.isfinite(loss)


def test_embedding_and_output_head_are_tied() -> None:
    model = GPTLanguageModel(tiny_config())
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
