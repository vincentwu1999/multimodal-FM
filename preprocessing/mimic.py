"""
Synthetic MIMIC‑IV subset loader and preprocessing functions.

This module defines `load_mimic_subset`, which generates synthetic
clinical notes, vital sign sequences and labels, along with
`preprocess_clinical_text` and `preprocess_vitals` for tokenising
and normalising the data.  The implementations are copied directly
from the original script and retain the same signatures and
behaviour.
"""

from __future__ import annotations

from typing import Optional, Tuple, List  # noqa: F401

import numpy as np
import torch


def load_mimic_subset(num_samples: int, seq_len: int, num_classes: int) -> Tuple[list[str], np.ndarray, np.ndarray]:
    """Load a small subset of MIMIC‑IV clinical notes and vital signs.

    Returns a tuple of (clinical_notes, vitals_array, labels).  Each
    clinical note is a string, vitals_array has shape (num_samples,
    num_vital_features, seq_len), and labels are integer class
    labels.  In a real implementation, this function should read
    from the MIMIC‑IV tables (e.g. `noteevents` for text and
    `chartevents`/`vitalsign` for vitals), apply any necessary
    filtering, and align notes with their corresponding vital
    sign windows.  The notes must then be tokenised using a
    corpus‑specific BPE tokenizer.

    Here we return synthetic notes and vitals for demonstration.
    """
    # Synthetic clinical notes as placeholder
    clinical_notes = [
        "Patient is stable. Blood pressure within normal range. Continue monitoring.",
        "Elevated heart rate observed. Consider beta‑blocker therapy.",
        "Glucose levels remain high; adjust insulin dosage accordingly.",
    ]
    # Cycle notes if num_samples > len(clinical_notes)
    clinical_notes = (clinical_notes * ((num_samples // len(clinical_notes)) + 1))[:num_samples]
    # Synthetic vitals: e.g. HR, SBP, DBP, SpO2, Respiratory rate, Temperature
    num_vitals = 6
    vitals = np.random.randn(num_samples, num_vitals, seq_len).astype(np.float32)
    # Synthetic labels (e.g. outcome classes)
    labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int64)
    return clinical_notes, vitals, labels


def preprocess_clinical_text(notes: list[str], vocab_size: int, max_len: int, tokenizer: Optional[object] = None, device: torch.device | str = "cpu") -> torch.Tensor:
    """Tokenise and pad clinical notes using a BPE tokenizer.

    Args:
        notes: list of strings.
        vocab_size: size of the vocabulary (used for hashing when tokenizer is None).
        max_len: maximum sequence length; sequences longer than this are truncated.
        tokenizer: a tokenizer object with an `encode` method returning
            token IDs.  If None, a simple whitespace split is used.
        device: device on which to place the returned tensor.

    Returns:
        A tensor of shape (len(notes), max_len) containing token IDs.
    """
    tokenised = []
    for note in notes:
        if tokenizer is not None:
            ids = tokenizer.encode(note).ids  # type: ignore[attr-defined]
        else:
            ids = [hash(word) % vocab_size for word in note.lower().split()]
        # pad or truncate
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        tokenised.append(ids)
    return torch.tensor(tokenised, dtype=torch.long, device=device)


def preprocess_vitals(vitals: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """Normalise and convert vital signs to a tensor.

    Args:
        vitals: numpy array of shape (N, num_features, seq_len).
        device: device on which to place the returned tensor.

    Returns:
        Tensor of shape (N, num_features, seq_len) suitable for SensorEncoder.
    """
    # Normalize each vital feature to zero mean and unit variance per sample
    vitals_norm = (vitals - vitals.mean(axis=-1, keepdims=True)) / (
        vitals.std(axis=-1, keepdims=True) + 1e-6
    )
    return torch.tensor(vitals_norm, dtype=torch.float32, device=device)


__all__ = ["load_mimic_subset", "preprocess_clinical_text", "preprocess_vitals"]
