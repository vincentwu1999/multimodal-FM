"""
Synthetic wearable data loader and preprocessing functions.

This module provides functions to generate random wearable sensor
sequences and corresponding labels, along with a normalisation
routine.  The logic is taken directly from the original
`multimodal_fusion_complete-real_data.py` script.
"""

from __future__ import annotations

from typing import Tuple  # noqa: F401

import numpy as np
import torch


def load_wearable_subset(participant_id: str, num_samples: int, seq_len: int, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load a subset of wearable signals from the BIG IDEAs dataset.

    Args:
        participant_id: ID of the participant (e.g. "001").
        num_samples: number of samples to load.
        seq_len: length of the time series to extract per sample.
        num_classes: number of classes for synthetic labels.

    Returns:
        wearable: array of shape (num_samples, num_wearable_features, seq_len).
        labels: array of integers with length num_samples.

    Note: In a real implementation, this function would read the CSV
    files for the selected participant (ACC.csv, BVP.csv, etc.),
    resample them to a common frequency (e.g. 4 Hz), align them,
    and fill missing values.  The seven channels would be ACC‑X,
    ACC‑Y, ACC‑Z, BVP, EDA, HR, TEMP.
    """
    num_channels = 7  # ACC‑x,y,z, BVP, EDA, HR, TEMP
    wearable = np.random.randn(num_samples, num_channels, seq_len).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int64)
    return wearable, labels


def preprocess_wearable(wearable: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """Normalise wearable data per channel and convert to tensor."""
    wearable_norm = (wearable - wearable.mean(axis=-1, keepdims=True)) / (
        wearable.std(axis=-1, keepdims=True) + 1e-6
    )
    return torch.tensor(wearable_norm, dtype=torch.float32, device=device)


__all__ = ["load_wearable_subset", "preprocess_wearable"]
