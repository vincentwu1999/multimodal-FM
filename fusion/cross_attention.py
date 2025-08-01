"""
Two‑modal fusion via cross attention.

This module defines the `CrossAttentionFusion` class, which fuses
sensor and text tokens using PyTorch's multi‑head attention.  The
implementation is identical to the original script.
"""

from __future__ import annotations

from typing import Optional  # noqa: F401

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse two modalities via multi‑head cross attention.

    Uses PyTorch's `nn.MultiheadAttention` twice: once with
    sensor queries and text keys/values, and once with text queries
    and sensor keys/values.  The pooled outputs are concatenated
    and projected.

    Args:
        embed_dim: dimension of input tokens.
        num_heads: number of attention heads.
        out_dim: desired output dimension.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim * 2, out_dim or embed_dim)

    def forward(self, sensor_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        attn_output1, _ = self.attn(query=sensor_tokens, key=text_tokens, value=text_tokens)
        attn_output2, _ = self.attn(query=text_tokens, key=sensor_tokens, value=sensor_tokens)
        sensor_repr = attn_output1.mean(dim=1)
        text_repr = attn_output2.mean(dim=1)
        fused = torch.cat([sensor_repr, text_repr], dim=-1)
        fused = self.proj(fused)
        return fused


__all__ = ["CrossAttentionFusion"]
