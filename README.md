# Multimodal Fusion Project

This repository is now extended to support
four modalities: sensor data, clinical text, waveform signals, and chest X-ray images.

For clarity and ease of navigation, the logic has been split across several subpackages:

* `preprocessing/` – utilities for positional encodings and waveform preprocessing. Also data loaders and preprocessing functions for clinical text, wearable signals, waveform data and chest X‑ray embeddings.
* `encoder/` – implementations of the sensor, text, waveform and image
  encoders along with the masked autoencoder used during pretraining.
* `fusion/` – cross‑modal fusion modules and the multi‑modal classifier
  supporting four modalities.
* `datasets/` - input data for each modality.
* `main.py` – a demonstration script that instantiates all four encoders,
  fusion module and classifier, generates synthetic data and runs
  both pretraining and fine‑tuning loops.

## Supported Modalities

1. **Sensor Data**: Wearable device signals and vital signs
2. **Clinical Text**: Tokenized clinical notes  
3. **Waveform Data**: ECG or other physiological waveforms
4. **Medical Images**: Chest X-ray embeddings

The fusion architecture uses staged cross-attention to progressively
combine modalities before final classification.

## Getting Started

1. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the demonstration from the repository root:

   ```bash
   python multimodal_fusion_project/main.py
   ```

   This will run the synthetic pretraining and fine‑tuning loops and
   print loss values to the console.

3. Explore the code in your favourite editor.  The modular
   organisation should make it straightforward to locate and modify
   specific components such as encoders, fusion layers or data
   preprocessing routines.

## Directory Structure

```
multimodal_fusion_project/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   ├── positional_encodings.py
│   ├── mimic.py
│   ├── wearable.py
│   ├── image.py
│   └── waveform.py
├── encoder/
│   ├── __init__.py
│   ├── sensor.py
│   ├── text.py
│   ├── image.py
│   └── waveform.py
├── fusion/
│   ├── __init__.py
│   ├── cross_attention.py
│   └── multi_modal_fusion.py
├── datasets/
│   ├── __init__.py
│   ├── EHR
│   ├── images
│   └── waveforms
├── main.py
└── requirements.txt
```
