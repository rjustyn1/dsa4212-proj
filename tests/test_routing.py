from soap_muon.routing import OptimizerRoute, RoutingPolicy, route_parameter


def test_non_matrix_parameters_use_adamw() -> None:
    assert route_parameter("blocks.0.ln.weight", (384,)) is OptimizerRoute.ADAMW


def test_near_square_matrix_uses_muon() -> None:
    assert route_parameter("blocks.0.attn.q_proj.weight", (384, 384)) is OptimizerRoute.MUON


def test_mlp_matrix_uses_selective_hybrid() -> None:
    assert (
        route_parameter("blocks.0.mlp.up_proj.weight", (1536, 384))
        is OptimizerRoute.ONE_SIDED_SOAP_MUON
    )


def test_fused_attention_qkv_uses_muon_in_selective_mode() -> None:
    assert (
        route_parameter("blocks.0.attention.qkv.weight", (1152, 384))
        is OptimizerRoute.MUON
    )


def test_global_mode_routes_fused_attention_qkv_to_hybrid() -> None:
    assert (
        route_parameter(
            "blocks.0.attention.qkv.weight",
            (1152, 384),
            RoutingPolicy(global_hybrid=True),
        )
        is OptimizerRoute.ONE_SIDED_SOAP_MUON
    )


def test_extremely_rectangular_and_embedding_matrices_use_adamw() -> None:
    assert route_parameter("lm_head.weight", (50257, 384)) is OptimizerRoute.ADAMW
    assert route_parameter("some.weight", (4096, 64)) is OptimizerRoute.ADAMW
