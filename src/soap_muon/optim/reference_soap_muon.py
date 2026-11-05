"""Reference-style one-sided SOAP–Muon from the pinned NanoGPT implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from soap_muon.optim.muon import DEFAULT_NS_COEFFICIENTS


def reference_zeropower_via_newton_schulz(
    matrix: Tensor, *, steps: int, eps: float = 1e-7
) -> Tensor:
    """Match the pinned reference's quintic iteration and two cleanup iterations."""
    if matrix.ndim != 2:
        raise ValueError("SOAP–Muon only supports two-dimensional matrices.")
    update = matrix.clone()
    update /= torch.linalg.vector_norm(update) + eps
    transposed = update.size(0) > update.size(1)
    if transposed:
        update = update.mT

    a, b, c = DEFAULT_NS_COEFFICIENTS
    for _ in range(steps):
        gram = update @ update.mT
        gram_update = gram @ update
        update = a * update + b * gram_update + c * (gram @ gram_update)
    for _ in range(2):
        update = 1.5 * update - 0.5 * (update @ update.mT) @ update

    return update.mT if transposed else update


class ReferenceOneSidedSOAPMuon(Optimizer):
    """A restricted, testable port of the selected upstream SOAP–Muon variant.

    This class intentionally implements only the reference's standard
    projection: the covariance and basis are maintained on the smaller matrix
    dimension. The first optimizer step initializes that basis and performs no
    parameter update, matching the pinned reference implementation.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float] = (0.95, 0.98),
        shampoo_beta: float | None = None,
        eps: float = 1e-15,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        precondition_frequency: int = 10,
        newton_schulz_steps: int = 5,
        nesterov: bool = False,
    ) -> None:
        if lr < 0 or weight_decay < 0:
            raise ValueError("lr and weight_decay must be non-negative.")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must be in [0, 1).")
        if shampoo_beta is not None and not 0 <= shampoo_beta < 1:
            raise ValueError("shampoo_beta must be in [0, 1).")
        if eps <= 0 or precondition_frequency <= 0 or newton_schulz_steps < 0:
            raise ValueError("eps and precondition_frequency must be positive; steps non-negative.")
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": betas[1] if shampoo_beta is None else shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "precondition_frequency": precondition_frequency,
            "newton_schulz_steps": newton_schulz_steps,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _side(gradient: Tensor) -> str:
        return "right" if gradient.size(0) >= gradient.size(1) else "left"

    @staticmethod
    def _project(tensor: Tensor, state: dict[str, Any]) -> Tensor:
        basis: Tensor = state["basis"]
        return basis.mT @ tensor if state["side"] == "left" else tensor @ basis.mT

    @staticmethod
    def _project_back(tensor: Tensor, state: dict[str, Any]) -> Tensor:
        basis: Tensor = state["basis"]
        return basis @ tensor if state["side"] == "left" else tensor @ basis

    @staticmethod
    def _gram(gradient: Tensor, side: str) -> Tensor:
        return gradient @ gradient.mT if side == "left" else gradient.mT @ gradient

    @staticmethod
    def _initial_basis(covariance: Tensor, side: str, dtype: torch.dtype) -> Tensor:
        _, eigenvectors = torch.linalg.eigh(covariance.float())
        eigenvectors = torch.flip(eigenvectors, [1]).to(dtype=dtype)
        return eigenvectors if side == "left" else eigenvectors.mT

    def _refresh_basis(self, state: dict[str, Any]) -> None:
        covariance: Tensor = state["covariance"]
        basis: Tensor = state["basis"]
        if state["side"] == "left":
            estimated_eigenvalues = torch.diag(basis.mT @ covariance @ basis)
            indices = torch.argsort(estimated_eigenvalues, descending=True)
            state["exp_avg_sq"] = state["exp_avg_sq"][indices]
            reordered_basis = basis.mT[indices].mT
            power_iteration = (reordered_basis.mT @ covariance).mT
            state["basis"] = torch.linalg.qr(power_iteration).Q.to(dtype=basis.dtype)
        else:
            estimated_eigenvalues = torch.diag(basis @ covariance @ basis.mT)
            indices = torch.argsort(estimated_eigenvalues, descending=True)
            state["exp_avg_sq"] = state["exp_avg_sq"].mT[indices].mT
            reordered_basis = basis[indices].mT
            power_iteration = (reordered_basis.mT @ covariance).mT
            state["basis"] = torch.linalg.qr(power_iteration).Q.mT.to(dtype=basis.dtype)

    def _update_preconditioner(
        self, gradient: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        covariance: Tensor = state["covariance"]
        covariance.lerp_(
            self._gram(gradient, state["side"]).to(dtype=covariance.dtype),
            1 - group["shampoo_beta"],
        )
        if state["basis"] is None:
            state["basis"] = self._initial_basis(covariance, state["side"], gradient.dtype)
        elif state["step"] > 0 and state["step"] % group["precondition_frequency"] == 0:
            self._refresh_basis(state)

    @staticmethod
    def _stabilize(update: Tensor, shape: torch.Size) -> Tensor:
        if not torch.any(update):
            return torch.zeros_like(update)
        update = torch.sqrt(torch.abs(update)) * torch.sign(update)
        update /= torch.linalg.vector_norm(update)
        update *= min(shape) ** 0.5
        update *= max(1.0, shape[0] / shape[1]) ** 0.5
        return update

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.ndim != 2 or gradient.is_sparse:
                    raise ValueError("SOAP–Muon parameters must have dense, two-dimensional gradients.")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["side"] = self._side(gradient)
                    dimension = min(gradient.shape)
                    state["covariance"] = torch.zeros(
                        (dimension, dimension), device=gradient.device, dtype=torch.float32
                    )
                    state["basis"] = None
                    state["exp_avg"] = torch.zeros_like(gradient)
                    state["exp_avg_sq"] = torch.zeros_like(gradient)
                    self._update_preconditioner(gradient, state, group)
                    continue

                projected_gradient = self._project(gradient, state)
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    projected_gradient, projected_gradient, value=1 - beta2
                )
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                update = self._project(exp_avg, state) / denominator

                step_size = group["lr"]
                if group["correct_bias"]:
                    step_size *= (1 - beta2**state["step"]) ** 0.5 / (1 - beta1**state["step"])
                if group["nesterov"]:
                    update = (
                        self._project(exp_avg, state) + (1 - beta1) * projected_gradient
                    ) / denominator

                update = reference_zeropower_via_newton_schulz(
                    update, steps=group["newton_schulz_steps"]
                )
                update = self._project_back(self._stabilize(update, parameter.shape), state)
                parameter.add_(update, alpha=-step_size)
                if group["weight_decay"]:
                    parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])
                self._update_preconditioner(gradient, state, group)
        return loss
