"""Build auditable optimizer bundles from a model and a named experiment method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.optim import AdamW, Optimizer

from soap_muon.optim import Muon, ReferenceOneSidedSOAPMuon
from soap_muon.routing import OptimizerRoute, RoutingPolicy, route_parameter


@dataclass
class OptimizerBundle:
    """A coordinated set of optimizers covering each trainable parameter once."""

    optimizers: tuple[Optimizer, ...]
    routes: dict[str, OptimizerRoute]

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


def _adamw(
    parameters: list[tuple[str, nn.Parameter]], lr: float, weight_decay: float
) -> AdamW | None:
    if not parameters:
        return None
    decay = [parameter for _, parameter in parameters if parameter.ndim > 1]
    no_decay = [parameter for _, parameter in parameters if parameter.ndim == 1]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def build_optimizer_bundle(
    model: nn.Module,
    *,
    method: str,
    lr: float,
    weight_decay: float,
    routing_policy: RoutingPolicy = RoutingPolicy(),
    muon_options: dict[str, Any] | None = None,
    hybrid_options: dict[str, Any] | None = None,
    adamw_lr: float | None = None,
    muon_lr: float | None = None,
    adamw_weight_decay: float | None = None,
    muon_weight_decay: float | None = None,
) -> OptimizerBundle:
    """Assign every trainable parameter to AdamW, Muon, or reference hybrid.

    Supported methods are adamw, muon, and selective_soap_muon. The Muon
    baseline converts every eligible hidden matrix to Muon; selective routing
    retains the layer-role distinctions encoded in routing_policy.
    """
    if method not in {"adamw", "muon", "selective_soap_muon"}:
        raise ValueError(f"Unsupported benchmark method: {method!r}")
    muon_options = muon_options or {}
    hybrid_options = hybrid_options or {}
    assignments: dict[OptimizerRoute, list[tuple[str, nn.Parameter]]] = {
        route: [] for route in OptimizerRoute
    }
    routes: dict[str, OptimizerRoute] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        route = route_parameter(name, tuple(parameter.shape), routing_policy)
        if method == "adamw":
            route = OptimizerRoute.ADAMW
        elif method == "muon" and route is not OptimizerRoute.ADAMW:
            route = OptimizerRoute.MUON
        assignments[route].append((name, parameter))
        routes[name] = route

    assigned = {id(parameter) for group in assignments.values() for _, parameter in group}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned != expected:
        raise RuntimeError("Optimizer routing did not cover every trainable parameter exactly once.")

    optimizers: list[Optimizer] = []
    adamw = _adamw(
        assignments[OptimizerRoute.ADAMW],
        adamw_lr if adamw_lr is not None else lr,
        adamw_weight_decay if adamw_weight_decay is not None else weight_decay,
    )
    if adamw is not None:
        optimizers.append(adamw)
    muon_parameters = [parameter for _, parameter in assignments[OptimizerRoute.MUON]]
    if muon_parameters:
        optimizers.append(
            Muon(
                muon_parameters,
                lr=muon_lr if muon_lr is not None else lr,
                weight_decay=muon_weight_decay if muon_weight_decay is not None else weight_decay,
                **muon_options,
            )
        )
    hybrid_parameters = [
        parameter for _, parameter in assignments[OptimizerRoute.ONE_SIDED_SOAP_MUON]
    ]
    if hybrid_parameters:
        optimizers.append(
            ReferenceOneSidedSOAPMuon(
                hybrid_parameters, lr=lr, weight_decay=weight_decay, **hybrid_options
            )
        )
    return OptimizerBundle(tuple(optimizers), routes)
