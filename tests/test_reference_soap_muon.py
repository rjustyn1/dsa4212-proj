import copy

import torch

from soap_muon.optim import ReferenceOneSidedSOAPMuon


def _optimizer(parameter: torch.nn.Parameter) -> ReferenceOneSidedSOAPMuon:
    return ReferenceOneSidedSOAPMuon(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.2,
        correct_bias=True,
        precondition_frequency=10,
        newton_schulz_steps=5,
        nesterov=False,
    )


def test_reference_port_matches_pinned_two_step_fixture() -> None:
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=torch.float32)
    )
    optimizer = _optimizer(parameter)
    first_gradient = torch.tensor([[0.2, -0.4, 0.1], [0.3, 0.5, -0.2]])
    second_gradient = torch.tensor([[0.6, 0.1, -0.3], [-0.2, 0.4, 0.7]])

    parameter.grad = first_gradient
    optimizer.step()
    assert torch.equal(
        parameter, torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]])
    )

    parameter.grad = second_gradient
    optimizer.step()

    expected = torch.tensor(
        [
            [0.78784555, -2.00074911, 0.44925103],
            [0.29287291, 1.30643976, -1.14356017],
        ]
    )
    assert torch.allclose(parameter, expected, atol=1e-6)
    assert optimizer.state[parameter]["step"] == 1
    assert optimizer.state[parameter]["side"] == "left"


def test_reference_port_supports_tall_matrices_and_state_reload() -> None:
    first = torch.nn.Parameter(torch.arange(1, 13, dtype=torch.float32).reshape(4, 3))
    second = torch.nn.Parameter(first.detach().clone())
    first_optimizer = _optimizer(first)
    second_optimizer = _optimizer(second)
    gradients = [
        torch.arange(1, 13, dtype=torch.float32).reshape(4, 3) / 10,
        torch.arange(12, 0, -1, dtype=torch.float32).reshape(4, 3) / 10,
        torch.ones(4, 3),
    ]

    first.grad = gradients[0]
    first_optimizer.step()
    first.grad = gradients[1]
    first_optimizer.step()
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))
    second.data.copy_(first.detach())
    first.grad = gradients[2]
    second.grad = gradients[2].clone()
    first_optimizer.step()
    second_optimizer.step()

    assert first_optimizer.state[first]["side"] == "right"
    assert torch.allclose(first, second)


def test_reference_port_matches_pinned_basis_refresh_fixture() -> None:
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=torch.float32)
    )
    optimizer = ReferenceOneSidedSOAPMuon(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.2,
        precondition_frequency=1,
        newton_schulz_steps=5,
    )
    gradients = [
        torch.tensor([[0.2, -0.4, 0.1], [0.3, 0.5, -0.2]]),
        torch.tensor([[0.6, 0.1, -0.3], [-0.2, 0.4, 0.7]]),
        torch.tensor([[-0.1, 0.2, 0.8], [0.3, -0.5, 0.4]]),
    ]

    for gradient in gradients:
        parameter.grad = gradient
        optimizer.step()

    expected = torch.tensor(
        [
            [0.64512962, -2.07587004, 0.41614023],
            [0.29536724, 1.31652272, -1.26400530],
        ]
    )
    assert torch.allclose(parameter, expected, atol=1e-6)
