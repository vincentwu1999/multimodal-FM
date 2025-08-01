"""
Image encoder modules.

This module contains two classes: one that wraps a torchvision ResNet
for chest X‑ray encoding, and another that projects precomputed
embeddings into a common dimension.  The implementations are taken
directly from the original `multimodal_fusion_complete-real_data.py`.
"""

from __future__ import annotations

from typing import Optional  # noqa: F401

import torch
import torch.nn as nn

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None  # type: ignore


class CXRImageEncoder(nn.Module):
    """Encode chest X‑ray images into embeddings.

    In practice you would load Google's CXR Foundation model, which
    produces high‑quality embeddings from chest X‑rays.  Here a
    ResNet‑18 backbone is used as a stand‑in.

    Args:
        embed_dim: output embedding dimension.
        pretrained: load pretrained weights or not.
    """

    def __init__(self, embed_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        if tv_models is None:
            raise ImportError("torchvision is required for the default CXRImageEncoder")
        backbone = tv_models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(backbone.fc.in_features, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        feats = feats.flatten(1)
        embed = self.proj(feats)
        return embed.unsqueeze(1)


class ImageEmbeddingEncoder(nn.Module):
    """Project precomputed CXR embeddings to a common embed_dim.

    Many clinical imaging pipelines (e.g. BiomedCLIP) provide
    per‑image embeddings of fixed dimension (e.g. 512).  This class
    wraps a linear projection so that those embeddings can be used
    interchangeably with tokens from other modalities.  The output
    shape is `(B, 1, embed_dim)` so that it can be consumed directly
    by the cross‑attention fusion module.

    Args:
        input_dim: dimension of the provided embeddings (e.g. 512 for BiomedCLIP).
        embed_dim: desired common embedding dimension for fusion (must match the
            other modalities).
    """

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, input_dim)
        embed = self.proj(embeddings)
        return embed.unsqueeze(1)


__all__ = ["CXRImageEncoder", "ImageEmbeddingEncoder"]
