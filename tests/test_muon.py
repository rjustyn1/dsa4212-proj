import pytest
import torch

from soap_muon.optim import Muon, zeropower_via_newton_schulz


def test_newton_schulz_preserves_shape_and_bounds_singular_values() -> None:
    matrix = torch.tensor(
        [[1.0, 2.0, -1.0, 3.0], [2.0, -1.0, 0.5, 1.0]],
        dtype=torch.float64,
    )

    update = zeropower_via_newton_schulz(matrix)

    assert update.shape == matrix.shape
    singular_values = torch.linalg.svdvals(update)
    assert torch.all(singular_values > 0.5)
    assert torch.all(singular_values < 1.5)


def test_newton_schulz_supports_tall_matrices_and_zero_inputs() -> None:
    tall = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)

    tall_update = zeropower_via_newton_schulz(tall)
    zero_update = zeropower_via_newton_schulz(torch.zeros(2, 3))

    assert tall_update.shape == tall.shape
    assert torch.isfinite(tall_update).all()
    assert torch.equal(zero_update, torch.zeros(2, 3))


def test_muon_matches_manual_nesterov_update_with_decoupled_weight_decay() -> None:
    initial = torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]])
    first_gradient = torch.tensor([[0.2, -0.4, 0.1], [0.3, 0.5, -0.2]])
    second_gradient = torch.tensor([[0.6, 0.1, -0.3], [-0.2, 0.4, 0.7]])
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = Muon(
        [parameter],
        lr=0.1,
        momentum=0.5,
        weight_decay=0.2,
        nesterov=True,
        newton_schulz_steps=5,
    )

    first_momentum = first_gradient
    first_update = zeropower_via_newton_schulz(1.5 * first_gradient)
    first_update *= 0.2 * 3**0.5
    expected_after_first = initial * (1 - 0.1 * 0.2) - 0.1 * first_update
    parameter.grad = first_gradient.clone()
    optimizer.step()
    assert torch.allclose(parameter, expected_after_first)

    second_momentum = 0.5 * first_momentum + second_gradient
    second_input = second_gradient + 0.5 * second_momentum
    second_update = zeropower_via_newton_schulz(second_input)
    second_update *= 0.2 * 3**0.5
    expected_after_second = expected_after_first * (1 - 0.1 * 0.2) - 0.1 * second_update
    parameter.grad = second_gradient.clone()
    optimizer.step()

    assert torch.allclose(parameter, expected_after_second)
    assert torch.allclose(optimizer.state[parameter]["momentum_buffer"], second_momentum)


def test_muon_rejects_vector_parameters() -> None:
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.ones(3)
    optimizer = Muon([parameter], lr=0.1)

    with pytest.raises(ValueError, match="two-dimensional"):
        optimizer.step()
