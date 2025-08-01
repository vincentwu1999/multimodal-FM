"""Synthetic dataset loaders and preprocessing utilities.

This package consolidates the helper functions used to generate
synthetic clinical notes, vitals and wearable signals, as well as
precomputed chest X‑ray embeddings.  These functions mirror the
implementations in the original script.
"""

from preprocessing.mimic import load_mimic_subset, preprocess_clinical_text, preprocess_vitals
from preprocessing.wearable import load_wearable_subset, preprocess_wearable
from preprocessing.image import load_chestx_subset, preprocess_image_embeddings
from preprocessing.waveform import load_waveform_subset, preprocess_waveform

__all__ = [
    "load_mimic_subset",
    "preprocess_clinical_text",
    "preprocess_vitals",
    "load_wearable_subset",
    "preprocess_wearable",
    "load_chestx_subset",
    "preprocess_image_embeddings",
    "load_waveform_subset",
    "preprocess_waveform",
]
