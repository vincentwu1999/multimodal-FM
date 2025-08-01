import numpy as np
import torch
from typing import Tuple

def load_waveform_subset(num_samples: int, seq_len: int = 1000, num_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Load synthetic waveform data (e.g., ECG signals)."""
    # Generate synthetic waveform data
    waveforms = np.random.randn(num_samples, seq_len).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int64)
    return waveforms, labels

def preprocess_waveform(waveforms: np.ndarray, device: torch.device = "cpu") -> torch.Tensor:
    """Normalize waveform data and convert to tensor."""
    # Add channel dimension and normalize
    waveforms = waveforms[:, np.newaxis, :]  # (batch, 1, seq_len)
    waveforms_norm = (waveforms - waveforms.mean(axis=-1, keepdims=True)) / (
        waveforms.std(axis=-1, keepdims=True) + 1e-6
    )
    return torch.tensor(waveforms_norm, dtype=torch.float32, device=device)

__all__ = ["load_waveform_subset", "preprocess_waveform"]
