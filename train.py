# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import MultiModalModel

# Configuration
NUM_SAMPLES = 500       # number of synthetic training samples
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DROP_PROB = 0.2         # probability to drop each modality in a given sample (for modality dropout)

# Simulate synthetic training data. For simplicity, we'll generate random data for each modality:
# Structured data: random floats in [0,1] for 10 features
# Clinical notes: random integers as token IDs (length 50)
# Waveform: random values (length 100)
# Image embeddings: random values (dim 512)
torch.manual_seed(0)  # for reproducibility of random data
struct_data = torch.rand(NUM_SAMPLES, 10)
text_data = torch.randint(0, 10000, (NUM_SAMPLES, 50))
wave_data = torch.randn(NUM_SAMPLES, 100)
img_data = torch.randn(NUM_SAMPLES, 512)

# Create labels with a simple heuristic to introduce a learnable pattern:
# - If first structured feature > 0.5, label "Septic Shock" = 1, else 0.
# - If first waveform value > 0, label "Respiratory Failure" = 1, else 0.
# This means both labels can be 1 (if both conditions met), supporting multi-label scenario.
labels = torch.zeros(NUM_SAMPLES, 2)  # 2 classes: [shock, respiratory_failure]
labels[:, 0] = (struct_data[:, 0] > 0.5).float()
labels[:, 1] = (wave_data[:, 0] > 0).float()

# Prepare model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)
criterion = nn.BCEWithLogitsLoss()  # appropriate for multi-label classification with logits
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
model.train()
num_batches = NUM_SAMPLES // BATCH_SIZE
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct = 0
    total = 0
    for b in range(num_batches):
        # Batch indices
        start = b * BATCH_SIZE
        end = start + BATCH_SIZE
        struct_batch = struct_data[start:end].clone().to(device)   # clone to avoid modifying original data
        text_batch = text_data[start:end].clone().to(device)
        wave_batch = wave_data[start:end].clone().to(device)
        img_batch = img_data[start:end].clone().to(device)
        label_batch = labels[start:end].to(device)
        
        # Apply modality dropout per sample
        # For each modality, decide randomly for each sample whether to drop it
        drop_mask_struct = torch.rand(struct_batch.size(0), device=device) < DROP_PROB
        drop_mask_text   = torch.rand(text_batch.size(0), device=device) < DROP_PROB
        drop_mask_wave   = torch.rand(wave_batch.size(0), device=device) < DROP_PROB
        drop_mask_img    = torch.rand(img_batch.size(0), device=device) < DROP_PROB
        # Zero out the data for dropped modalities
        if drop_mask_struct.any():
            struct_batch[drop_mask_struct] = 0.0
        if drop_mask_text.any():
            # Replace text tokens with 0 (assuming 0 is PAD or a neutral token in vocab)
            text_batch[drop_mask_text] = 0
        if drop_mask_wave.any():
            wave_batch[drop_mask_wave] = 0.0
        if drop_mask_img.any():
            img_batch[drop_mask_img] = 0.0
        
        # Forward pass
        logits = model(struct_batch, text_batch, wave_batch, img_batch)
        loss = criterion(logits, label_batch)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        # Compute simple accuracy for each label (for monitoring):
        # We'll say a prediction is correct if the round(sigmoid) equals the label for both classes (exact match).
        preds = (torch.sigmoid(logits) >= 0.5).float()
        matches = (preds == label_batch).all(dim=1)  # checks if both labels match
        correct += matches.sum().item()
        total += label_batch.size(0)
    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}, Exact-match Accuracy: {accuracy:.2f}%")

# Save the trained model weights
torch.save(model.state_dict(), "kids_fm_core.pth")
print("Model training complete. Saved weights to kids_fm_core.pth.")
