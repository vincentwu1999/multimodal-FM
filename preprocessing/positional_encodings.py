"""
Positional encoding helpers.

This module contains functions that generate sine–cosine positional
embeddings for one‑dimensional sequences and two‑dimensional grids.
The implementations mirror those found in the original
`multimodal_fusion_complete-real_data.py` script.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple  # noqa: F401

import numpy as np
import torch


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Create 1D sine-cosine positional embedding from positions.

    Args:
        embed_dim: embedding dimension (must be divisible by 2).
        pos: a 1D array of positions.

    Returns:
        An array of shape (len(pos), embed_dim) containing the positional embeddings.
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    omega = 1.0 / (10000 ** (np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim // 2)))
    pos = pos.reshape(-1, 1)
    args = pos * omega[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim: int, length: int, cls_token: bool = False) -> np.ndarray:
    """Generate 1D sin-cos positional embedding.

    Args:
        embed_dim: embedding dimension.
        length: sequence length.
        cls_token: whether to include an extra zero vector for CLS token.

    Returns:
        Positional embedding of shape (length + 1, embed_dim) if cls_token else (length, embed_dim).
    """
    pos = np.arange(length, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size_time: int, grid_size_sensor: int, cls_token: bool = False) -> np.ndarray:
    """Generate 2D sin-cos positional embedding for (time x sensor) grid.

    Args:
        embed_dim: embedding dimension (must be divisible by 2).
        grid_size_time: number of positions along time dimension.
        grid_size_sensor: number of positions along sensor dimension.
        cls_token: whether to include an extra zero vector for CLS token.

    Returns:
        Positional embedding of shape ((grid_size_time * grid_size_sensor) + 1, embed_dim) if cls_token else (grid_size_time * grid_size_sensor, embed_dim).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    grid_time = np.arange(grid_size_time, dtype=np.float32)
    grid_sensor = np.arange(grid_size_sensor, dtype=np.float32)
    grid = np.meshgrid(grid_time, grid_sensor, indexing='ij')
    grid = np.stack(grid, axis=0)  # shape [2, grid_size_time, grid_size_sensor]
    grid = grid.reshape(2, -1)    # shape [2, grid_size_time * grid_size_sensor]
    emb_time = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_sensor = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_time, emb_sensor], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), emb], axis=0)
    return emb


def get_positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
    """Return a (seq_len, dim) positional encoding matrix.

    Uses the sine/cosine formulation from the original Transformer paper.

    Args:
        seq_len: maximum sequence length.
        dim: embedding dimension.

    Returns:
        Tensor of shape (seq_len, dim).
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
