"""Explicit, testable parameter-to-optimizer routing policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence


class OptimizerRoute(StrEnum):
    ADAMW = "adamw"
    MUON = "muon"
    ONE_SIDED_SOAP_MUON = "one_sided_soap_muon"


@dataclass(frozen=True)
class RoutingPolicy:
    """Shape thresholds proposed for the primary experiment.

    A ratio is always largest_dimension / smallest_dimension. The name-based
    exclusions prevent tied embeddings and output heads from silently reaching
    a matrix optimizer.
    """

    muon_max_aspect_ratio: float = 2.0
    hybrid_max_aspect_ratio: float = 8.0
    excluded_name_fragments: tuple[str, ...] = ("embed", "embedding", "lm_head")
    hybrid_name_fragments: tuple[str, ...] = (".mlp.",)
    hybrid_mlp_projections: tuple[str, ...] = (".up_proj.", ".down_proj.")
    global_hybrid: bool = False


def route_parameter(
    name: str, shape: Sequence[int], policy: RoutingPolicy = RoutingPolicy()
) -> OptimizerRoute:
    """Choose an optimizer family for one named parameter.

    All non-matrices, embeddings/output heads, and extremely rectangular
    matrices remain on AdamW. In selective mode, only the intended MLP
    projections selected by ``hybrid_mlp_projections`` receive the hybrid;
    moderately rectangular attention weights (including fused QKV) remain on
    Muon. Set global_hybrid for the global hybrid ablation. This function makes
    route selection auditable and supports separate up-only, down-only, and
    both-projection experiments.
    """
    normalized_name = name.lower()
    if len(shape) != 2 or any(dimension <= 0 for dimension in shape):
        return OptimizerRoute.ADAMW
    if any(fragment in normalized_name for fragment in policy.excluded_name_fragments):
        return OptimizerRoute.ADAMW

    smaller, larger = sorted(shape)
    aspect_ratio = larger / smaller
    if aspect_ratio <= policy.muon_max_aspect_ratio:
        return OptimizerRoute.MUON
    if aspect_ratio <= policy.hybrid_max_aspect_ratio:
        is_selected_mlp_projection = any(
            fragment in normalized_name for fragment in policy.hybrid_name_fragments
        ) and any(
            projection in normalized_name for projection in policy.hybrid_mlp_projections
        )
        if policy.global_hybrid or is_selected_mlp_projection:
            return OptimizerRoute.ONE_SIDED_SOAP_MUON
        return OptimizerRoute.MUON
    return OptimizerRoute.ADAMW
