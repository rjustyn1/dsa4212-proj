import numpy as np
import torch

from soap_muon.config import ModelConfig
from soap_muon.data.token_shards import contiguous_batches, load_token_shard
from soap_muon.models import GPTLanguageModel
from soap_muon.training.checkpointing import load_model_checkpoint, save_model_checkpoint
from soap_muon.training.evaluation import evaluate_next_token_prediction


def tiny_config() -> ModelConfig:
    return ModelConfig(128, 8, 1, 16, 4, 2.0)


def test_evaluation_is_inference_only_and_reports_metrics(tmp_path) -> None:
    model = GPTLanguageModel(tiny_config())
    tokens = np.arange(65, dtype=np.int64) % 128
    result = evaluate_next_token_prediction(
        model,
        contiguous_batches(tokens, sequence_length=8, batch_size=2, max_batches=2),
        torch.device("cpu"),
    )

    assert result.tokens_evaluated == 32
    assert result.mean_nll > 0
    assert result.perplexity > 1


def test_checkpoint_contract_is_portable(tmp_path) -> None:
    model = GPTLanguageModel(tiny_config())
    path = tmp_path / "best.pt"
    save_model_checkpoint(path, model, tiny_config(), tokens_seen=100, step=3)
    payload = load_model_checkpoint(path, torch.device("cpu"))

    assert payload["tokens_seen"] == 100
    assert payload["model_config"]["d_model"] == 16


def test_token_shard_loader_reopens_saved_numpy_tokens(tmp_path) -> None:
    path = tmp_path / "validation.npy"
    np.save(path, np.arange(33, dtype=np.uint16))
    loaded = load_token_shard(path)
    batches = list(contiguous_batches(loaded, sequence_length=8, batch_size=2))

    assert len(batches) == 2
    assert batches[0][0].shape == (2, 8)
