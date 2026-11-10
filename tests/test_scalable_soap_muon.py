import pytest
import torch

from soap_muon.optim import (
    Muon,
    ScalableOneSidedSOAPMuon,
    moonlight_update_scale,
    zeropower_via_newton_schulz,
)


def _one_step(
    optimizer_type: type[Muon] | type[ScalableOneSidedSOAPMuon], shape: tuple[int, int]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameter = torch.nn.Parameter(torch.ones(shape))
    optimizer = optimizer_type([parameter], lr=0.1, weight_decay=0.2)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return parameter.detach(), optimizer.state[parameter]


@pytest.mark.parametrize(
    ("shape", "expected_side"),
    [
        ((4, 4), "right"),
        ((16, 4), "right"),
        ((4, 16), "left"),
        ((32, 4), "right"),
        ((4, 32), "left"),
    ],
)
def test_scalable_soap_muon_tracks_the_smaller_side_and_uses_moonlight_scaling(
    shape: tuple[int, int], expected_side: str
) -> None:
    scalable, state = _one_step(ScalableOneSidedSOAPMuon, shape)

    assert torch.isfinite(scalable).all()
    assert state["side"] == expected_side
    assert torch.equal(state["momentum_buffer"], torch.ones(shape))
    assert ScalableOneSidedSOAPMuon._reference_muon_scale(torch.Size(shape)) == moonlight_update_scale(
        torch.Size(shape)
    )


@pytest.mark.parametrize("shape", [(16, 4), (4, 16), (32, 4), (4, 32)])
def test_rotation_only_ablation_is_equivalent_to_moonlight_muon(
    shape: tuple[int, int],
) -> None:
    torch.manual_seed(sum(shape))
    gradient = torch.randn(shape)
    dimension = min(shape)
    basis = torch.linalg.qr(torch.randn(dimension, dimension)).Q

    if shape[0] >= shape[1]:
        rotated = gradient @ basis.mT
        restored_update = zeropower_via_newton_schulz(rotated) @ basis
    else:
        rotated = basis.mT @ gradient
        restored_update = basis @ zeropower_via_newton_schulz(rotated)

    moonlight_update = zeropower_via_newton_schulz(gradient)
    assert torch.allclose(restored_update, moonlight_update, atol=1e-5, rtol=1e-5)


def test_scalable_soap_muon_applies_decay_before_the_update() -> None:
    scalable, _ = _one_step(ScalableOneSidedSOAPMuon, (2, 3))
    reference, _ = _one_step(Muon, (2, 3))

    assert torch.allclose(scalable, reference)


def test_scalable_soap_muon_preconditions_before_the_moonlight_step() -> None:
    initial = torch.tensor([[1.0, -2.0], [0.25, 1.5], [-1.0, 0.5]])
    gradient = torch.tensor([[0.2, -0.4], [0.3, 0.5], [-0.2, 0.7]])
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = ScalableOneSidedSOAPMuon(
        [parameter], lr=0.1, momentum=0.5, beta2=0.5, eps=1e-8, weight_decay=0.2
    )

    parameter.grad = gradient.clone()
    optimizer.step()

    state = optimizer.state[parameter]
    momentum = gradient * 1.5
    basis = state["basis"]
    projected = momentum @ basis.mT
    second_moment = 0.5 * projected.square()
    denominator = second_moment.sqrt() / (1 - 0.5) ** 0.5 + 1e-8
    preconditioned = projected / denominator
    expected_update = zeropower_via_newton_schulz(preconditioned) @ basis
    expected_update *= moonlight_update_scale(gradient.shape)
    expected = initial * (1 - 0.1 * 0.2) - 0.1 * expected_update

    assert state["side"] == "right"
    assert torch.allclose(parameter, expected)
