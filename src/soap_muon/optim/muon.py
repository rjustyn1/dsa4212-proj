"""Scalable Muon with Moonshot's fixed-RMS update scaling."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newton_schulz(
    matrix: Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
) -> Tensor:
    """Approximate a matrix's semi-orthogonal polar factor with a quintic iteration."""
    if matrix.ndim != 2:
        raise ValueError("Muon only supports two-dimensional matrices.")
    if steps < 0:
        raise ValueError("Newton–Schulz steps must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    original_dtype = matrix.dtype
    work_dtype = torch.float32 if matrix.dtype in (torch.float16, torch.bfloat16) else matrix.dtype
    update = matrix.to(dtype=work_dtype).clone()
    update = update / (torch.linalg.vector_norm(update) + eps)
    transposed = update.size(0) > update.size(1)
    if transposed:
        update = update.mT

    a, b, c = coefficients
    for _ in range(steps):
        gram = update @ update.mT
        gram_update = gram @ update
        update = a * update + b * gram_update + c * (gram @ gram_update)

    if transposed:
        update = update.mT
    return update.to(dtype=original_dtype)


class Muon(Optimizer):
    """Matrix-only Muon using the 0.2 sqrt(max(m, n)) scalable-update rule."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        newton_schulz_steps: int = 5,
        eps: float = 1e-7,
        update_rms_scale: float = 0.2,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative.")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1).")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if newton_schulz_steps < 0:
            raise ValueError("newton_schulz_steps must be non-negative.")
        if eps <= 0 or update_rms_scale <= 0:
            raise ValueError("eps and update_rms_scale must be positive.")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "newton_schulz_steps": newton_schulz_steps,
            "eps": eps,
            "update_rms_scale": update_rms_scale,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.ndim != 2:
                    raise ValueError("Muon parameters must be two-dimensional matrices.")
                if gradient.is_sparse:
                    raise ValueError("Muon does not support sparse gradients.")

                state = self.state[parameter]
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(gradient)
                    state["momentum_buffer"] = momentum_buffer
                momentum_buffer.mul_(group["momentum"]).add_(gradient)
                update_input = (
                    gradient.add(momentum_buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else momentum_buffer
                )
                update = zeropower_via_newton_schulz(
                    update_input,
                    steps=group["newton_schulz_steps"],
                    eps=group["eps"],
                )
                update.mul_(
                    group["update_rms_scale"] * max(parameter.shape[0], parameter.shape[1]) ** 0.5
                )

                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss
