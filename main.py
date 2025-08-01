"""
Demonstration script for the multimodal fusion project.

This file reproduces the `main()` function from the original
`multimodal_fusion_complete-real_data.py` script.  It loads synthetic
data, initialises the various encoders and fusion module, pretrains
the sensor encoder and then fine‑tunes the full model.  The code is
unchanged apart from the import statements, which have been updated
to reflect the modular package structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from encoder.sensor import SensorEncoder  # type: ignore
from encoder.text import TextTransformerEncoder
from encoder.image import ImageEmbeddingEncoder
from encoder.waveform import WaveformEncoder
from fusion.multi_modal_fusion import MultiModalFusion3, MultiModalClassifier3
from preprocessing.mimic import load_mimic_subset, preprocess_clinical_text, preprocess_vitals
from preprocessing.wearable import load_wearable_subset, preprocess_wearable
from preprocessing.image import load_chestx_subset, preprocess_image_embeddings
from preprocessing.waveform import load_waveform_subset, preprocess_waveform
from fusion.multi_modal_fusion import MultiModalFusion4, MultiModalClassifier4


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wearable = False

    # Hyperparameters
    batch_size = 4
    # Text settings
    vocab_size = 10000  # size of BPE vocabulary; adjust when training a real tokenizer
    max_seq_len = 64   # maximum length of clinical note sequences
    num_classes = 3
    # Embedding dimension shared across modalities
    embed_dim = 128
    
    # ------------------------------------------------------------------
    # Load and preprocess a small subset of each dataset
    num_samples = batch_size
    # For this demonstration we use the same sequence length for vitals and wearable data
    notes, vitals, labels_mimic = load_mimic_subset(num_samples, seq_len=max_seq_len * 2, num_classes=num_classes)
    waveforms, labels_wave = load_waveform_subset(num_samples, seq_len=max_seq_len * 2, num_classes=num_classes)
    image_embs, labels_img = load_chestx_subset(num_samples, num_classes=num_classes)
    
    text_ids = preprocess_clinical_text(notes, vocab_size=vocab_size, max_len=max_seq_len, tokenizer=None, device=device)
    vitals_tensor = preprocess_vitals(vitals, device=device)
    
    if use_wearable:
        wearable, labels_wear = load_wearable_subset(participant_id="001", num_samples=num_samples, seq_len=max_seq_len * 2, num_classes=num_classes)
        wearable_tensor = preprocess_wearable(wearable, device=device)

    waveform_tensor = preprocess_waveform(waveforms, device=device)
    image_tensor = preprocess_image_embeddings(image_embs, device=device)
    
    if use_wearable:
        # Combine wearable and vital features into a single sensor tensor
        sensor_data = torch.cat([wearable_tensor, vitals_tensor], dim=1)  # shape: (batch, channels, seq_len)
    else:
        sensor_data = vitals_tensor
    
    # Update sensor_channels and sensor_seq_len based on loaded data
    sensor_channels = sensor_data.shape[1]
    sensor_seq_len = sensor_data.shape[2]
    
    if use_wearable:
        # Instantiate encoders and fusion
        sensor_encoder = SensorEncoder(
            sensor_channels=sensor_channels,
            seq_length=sensor_seq_len,
            embed_dim=embed_dim,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            num_classes=num_classes,
            artificial_mask_ratio=0.5,
        ).to(device)
    text_encoder = TextTransformerEncoder(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
    ).to(device)
    waveform_encoder = WaveformEncoder(out_dim=embed_dim).to(device)
    # Use ImageEmbeddingEncoder to project 512‑dim BiomedCLIP features to embed_dim
    image_encoder = ImageEmbeddingEncoder(input_dim=512, embed_dim=embed_dim).to(device)
    # Fusion module; output dimension matches classifier input
    # fusion = MultiModalFusion3(embed_dim=embed_dim, num_heads=4, out_dim=256).to(device)
    # model = MultiModalClassifier3(sensor_encoder, text_encoder, image_encoder, fusion, num_classes).to(device)
    fusion = MultiModalFusion4(embed_dim=embed_dim, num_heads=4, out_dim=256).to(device)
    model = MultiModalClassifier4(sensor_encoder, text_encoder, waveform_encoder, image_encoder, fusion, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # Data loader and preprocessing functions have been moved to module level.

    # Demonstration dataset sizes
    num_samples = batch_size
    # Load data for each modality
    notes, vitals, labels_mimic = load_mimic_subset(num_samples)
    wearable, labels_wear = load_wearable_subset(participant_id="001", num_samples=num_samples)
    image_embs, labels_img = load_chestx_subset(num_samples)
    # Preprocess data
    text_ids = preprocess_clinical_text(notes, tokenizer=None, max_len=max_seq_len)
    vitals_tensor = preprocess_vitals(vitals)
    wearable_tensor = preprocess_wearable(wearable)
    image_tensor = preprocess_image_embeddings(image_embs)

    # For this demonstration, we combine wearable signals and vital signs
    # into a single multi‑channel sensor tensor.  In practice you could
    # encode them separately and fuse later.
    sensor_data = torch.cat([wearable_tensor, vitals_tensor], dim=1)  # shape (batch, num_channels, seq_len)

    # Pretrain the sensor encoder on the combined wearable/vital data
    pretrain_steps = 5
    for i in range(pretrain_steps):
        pretrain_loss, _ = sensor_encoder.forward_pretrain(sensor_data)
        optimizer.zero_grad()
        pretrain_loss.backward()
        optimizer.step()
        print(f"Pretraining iteration {i+1}, loss = {pretrain_loss.item():.4f}")
    # Fine‑tune the full model on the integrated modalities
    finetune_steps = 5
    labels_tensor = torch.tensor(labels_mimic, device=device)
    for step in range(finetune_steps):
        logits = model(sensor_data, text_ids, waveform_tensor, image_tensor)
        loss = criterion(logits, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Finetune iteration {step+1}, loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
