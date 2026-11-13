from pathlib import Path

from soap_muon.config import load_experiment


def test_smoke_config_resolves_all_component_files() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment(root / "configs/experiments/smoke.yaml")

    assert config.name == "smoke-selective-soap-muon"
    assert config.model.d_model == 384
    assert config.data.sequence_length == config.model.context_length
    assert config.optimizer_name == "selective_soap_muon"
