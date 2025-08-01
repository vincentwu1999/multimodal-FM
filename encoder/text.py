"""
Text encoder module.

This module contains a minimal Transformer encoder for tokenised
text, adapted from the original `multimodal_fusion_complete-real_data.py`
script.  The implementation has been copied verbatim with
updated imports.
"""

from __future__ import annotations

import math  # noqa: F401
from typing import Optional  # noqa: F401

import torch
import torch.nn as nn

from ..preprocessing.positional_encodings import get_positional_encoding


class TextTransformerEncoder(nn.Module):
    """Minimal Transformer encoder for tokenised text.

    Args:
        vocab_size: size of vocabulary.
        max_seq_len: maximum sequence length.
        embed_dim: dimension of token embeddings.
        num_layers: number of Transformer layers.
        num_heads: number of attention heads.
        mlp_ratio: expansion ratio for feedforward networks.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.register_buffer("pos_embed", get_positional_encoding(max_seq_len + 1, embed_dim), persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.embed_dim ** -0.5)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, S = token_ids.shape
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max {self.max_seq_len}")
        token_embeds = self.token_embed(token_ids)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, token_embeds], dim=1)
        pos_emb = self.pos_embed[:S + 1, :].unsqueeze(0)
        tokens = tokens + pos_emb
        encoded = self.encoder(tokens)
        return encoded


__all__ = ["TextTransformerEncoder"]
