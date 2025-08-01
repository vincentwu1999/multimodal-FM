"""
Three‑modal fusion and classifier.

This module defines the staged fusion of sensor, text and image
modalities and wraps everything into a classifier.  The code has
been lifted unchanged from the original script, with imports
adjusted to fit the refactored package structure.
"""

from __future__ import annotations

from typing import Optional  # noqa: F401

import torch
import torch.nn as nn

from .cross_attention import CrossAttentionFusion


class MultiModalFusion4(nn.Module):
    """Fuse sensor, text, waveform and image modalities via staged cross attention.

    Stage 1: fuse sensor and text tokens using `CrossAttentionFusion`.
    Stage 2: fuse the result with waveform tokens.
    Stage 3: treat the fused representation as a query and attend over image tokens.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.fusion_st = CrossAttentionFusion(embed_dim, num_heads, out_dim=embed_dim)
        self.fusion_stw = CrossAttentionFusion(embed_dim, num_heads, out_dim=embed_dim)
        self.attn_im = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim * 2, out_dim or embed_dim)

    def forward(self, sensor_tokens: torch.Tensor, text_tokens: torch.Tensor, 
                waveform_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        # Stage 1: fuse sensor and text
        st_fused = self.fusion_st(sensor_tokens, text_tokens)  # (B, D)
        
        # Stage 2: fuse with waveform
        stw_fused = self.fusion_stw(st_fused.unsqueeze(1), waveform_tokens.unsqueeze(1))  # (B, D)
        
        # Stage 3: attend over image tokens
        query = stw_fused.unsqueeze(1)  # (B, 1, D)
        attn_output, _ = self.attn_im(query=query, key=image_tokens, value=image_tokens)
        
        stw_repr = stw_fused
        im_repr = attn_output.squeeze(1)
        fused = torch.cat([stw_repr, im_repr], dim=-1)
        fused = self.proj(fused)
        return fused


class MultiModalClassifier4(nn.Module):
    """End‑to‑end classifier for sensor + text + waveform + image data."""

    def __init__(
        self,
        sensor_encoder: nn.Module,
        text_encoder: nn.Module,
        waveform_encoder: nn.Module,
        image_encoder: nn.Module,
        fusion: MultiModalFusion4,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.sensor_encoder = sensor_encoder
        self.text_encoder = text_encoder
        self.waveform_encoder = waveform_encoder
        self.image_encoder = image_encoder
        self.fusion = fusion
        self.classifier = nn.Linear(fusion.proj.out_features, num_classes)

    def forward(self, sensor_x: torch.Tensor, text_ids: torch.Tensor, 
                waveform_x: torch.Tensor, image_x: torch.Tensor) -> torch.Tensor:
        sensor_out = self.sensor_encoder.forward_encoder(sensor_x)
        sensor_tokens = sensor_out[0] if isinstance(sensor_out, tuple) else sensor_out
        
        text_tokens = self.text_encoder(text_ids)
        waveform_tokens = self.waveform_encoder(waveform_x).unsqueeze(1)  # Add sequence dim
        image_tokens = self.image_encoder(image_x)
        
        fused = self.fusion(sensor_tokens, text_tokens, waveform_tokens, image_tokens)
        logits = self.classifier(fused)
        return logits


__all__ = ["MultiModalFusion4", "MultiModalClassifier4"]
