import torch

from soap_muon.optim import Muon, ScalableOneSidedSOAPMuon


def _one_step(
    optimizer_type: type[Muon] | type[ScalableOneSidedSOAPMuon], shape: tuple[int, int]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameter = torch.nn.Parameter(torch.ones(shape))
    optimizer = optimizer_type([parameter], lr=0.1, weight_decay=0.2)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return parameter.detach(), optimizer.state[parameter]


def test_scalable_soap_muon_matches_reference_muon_geometry_for_isotropic_gradients() -> None:
    scalable_wide, wide_state = _one_step(ScalableOneSidedSOAPMuon, (2, 3))
    reference_wide, _ = _one_step(Muon, (2, 3))

    assert torch.allclose(scalable_wide, reference_wide)
    assert torch.equal(wide_state["momentum_buffer"], torch.ones(2, 3))
    assert ScalableOneSidedSOAPMuon._reference_muon_scale(torch.Size((2, 3))) == 1.0
    assert ScalableOneSidedSOAPMuon._reference_muon_scale(torch.Size((4, 3))) == (4 / 3) ** 0.5


def test_scalable_soap_muon_applies_decay_before_the_update() -> None:
    scalable, _ = _one_step(ScalableOneSidedSOAPMuon, (2, 3))
    reference, _ = _one_step(Muon, (2, 3))

    assert torch.allclose(scalable, reference)
