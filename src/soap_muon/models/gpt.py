"""A deliberately small, conventional decoder-only GPT for optimizer studies."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from soap_muon.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, sequence, channels = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.out_proj(attended)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = int(config.d_model * config.mlp_ratio)
        self.up_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPTLanguageModel(nn.Module):
    """Decoder-only language model with tied token embedding/output weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # GPT-style residual scaling prevents residual variance from growing
        # rapidly with depth while leaving the optimizer experiment unchanged.
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        _, sequence = input_ids.shape
        if sequence > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence} exceeds context length {self.config.context_length}."
            )

        positions = torch.arange(sequence, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids.")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
