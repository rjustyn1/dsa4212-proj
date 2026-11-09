"""One-sided SOAP preconditioning with the reference Muon update geometry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from soap_muon.optim.muon import zeropower_via_newton_schulz
from soap_muon.optim.reference_soap_muon import ReferenceOneSidedSOAPMuon


class ScalableOneSidedSOAPMuon(ReferenceOneSidedSOAPMuon):
    """Apply one-sided SOAP before an otherwise reference-equivalent Muon step.

    SOAP supplies a smaller-side rotating basis and a coordinate-wise second
    moment.  The final update deliberately follows Muon's momentum, polar
    iteration, aspect-ratio scaling, and decoupled-decay order exactly.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float,
        momentum: float = 0.95,
        beta2: float = 0.95,
        shampoo_beta: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        newton_schulz_steps: int = 5,
    ) -> None:
        if lr < 0 or weight_decay < 0:
            raise ValueError("lr and weight_decay must be non-negative.")
        if not 0 <= momentum < 1 or not 0 <= beta2 < 1:
            raise ValueError("momentum and beta2 must be in [0, 1).")
        if eps <= 0 or precondition_frequency <= 0 or newton_schulz_steps < 0:
            raise ValueError("eps and precondition_frequency must be positive; steps non-negative.")
        Optimizer.__init__(
            self,
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "beta2": beta2,
                "shampoo_beta": beta2 if shampoo_beta is None else shampoo_beta,
                "eps": eps,
                "weight_decay": weight_decay,
                "precondition_frequency": precondition_frequency,
                "newton_schulz_steps": newton_schulz_steps,
            },
        )

    @staticmethod
    def _reference_muon_scale(shape: torch.Size) -> float:
        return max(1.0, shape[0] / shape[1]) ** 0.5

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.ndim != 2 or gradient.is_sparse:
                    raise ValueError("scalable SOAP–Muon requires dense, two-dimensional gradients.")

                state = self.state[parameter]
                initialized = bool(state)
                if not initialized:
                    state["step"] = 0
                    state["side"] = self._side(gradient)
                    dimension = min(gradient.shape)
                    state["covariance"] = torch.zeros(
                        (dimension, dimension), device=gradient.device, dtype=torch.float32
                    )
                    state["basis"] = None
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                    state["exp_avg_sq"] = torch.zeros_like(gradient)
                    self._update_preconditioner(gradient, state, group)

                state["step"] += 1
                momentum_buffer: Tensor = state["momentum_buffer"]
                momentum_buffer.mul_(group["momentum"]).add_(gradient)
                nesterov_input = gradient.add(momentum_buffer, alpha=group["momentum"])
                projected_input = self._project(nesterov_input, state)

                exp_avg_sq: Tensor = state["exp_avg_sq"]
                exp_avg_sq.mul_(group["beta2"]).addcmul_(
                    projected_input, projected_input, value=1 - group["beta2"]
                )
                bias_correction = (1 - group["beta2"] ** state["step"]) ** 0.5
                denominator = exp_avg_sq.sqrt().div_(bias_correction).add_(group["eps"])
                update = zeropower_via_newton_schulz(
                    projected_input / denominator,
                    steps=group["newton_schulz_steps"],
                )
                update = self._project_back(update, state)
                update.mul_(self._reference_muon_scale(parameter.shape))

                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
                if initialized:
                    self._update_preconditioner(gradient, state, group)
        return loss
