# evaluate.py
import torch
from model import MultiModalModel


# Configuration for test data
NUM_TEST_SAMPLES = 100
BATCH_SIZE = 20

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)
model.load_state_dict(torch.load("kids_fm_core.pth", map_location=device))
model.eval()
print("Loaded trained model weights for evaluation.")

# Generate synthetic test dataset (same distribution and label rule as training)
torch.manual_seed(42)  # different seed for test data
struct_data = torch.rand(NUM_TEST_SAMPLES, 10)
text_data = torch.randint(0, 10000, (NUM_TEST_SAMPLES, 50))
wave_data = torch.randn(NUM_TEST_SAMPLES, 100)
img_data = torch.randn(NUM_TEST_SAMPLES, 512)
labels = torch.zeros(NUM_TEST_SAMPLES, 2)
labels[:, 0] = (struct_data[:, 0] > 0.5).float()
labels[:, 1] = (wave_data[:, 0] > 0).float()

# Evaluate in batches
total = NUM_TEST_SAMPLES
correct_shock = 0
correct_resp = 0
exact_match = 0

with torch.no_grad():
    for i in range(0, NUM_TEST_SAMPLES, BATCH_SIZE):
        end = min(i + BATCH_SIZE, NUM_TEST_SAMPLES)
        struct_batch = struct_data[i:end].to(device)
        text_batch = text_data[i:end].to(device)
        wave_batch = wave_data[i:end].to(device)
        img_batch = img_data[i:end].to(device)
        label_batch = labels[i:end].to(device)
        
        logits = model(struct_batch, text_batch, wave_batch, img_batch)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        
        # Calculate class-wise accuracy
        correct_shock += (preds[:, 0] == label_batch[:, 0]).sum().item()
        correct_resp += (preds[:, 1] == label_batch[:, 1]).sum().item()
        # Exact match (both labels correct)
        match = (preds == label_batch).all(dim=1).sum().item()
        exact_match += match

# Compute percentages
acc_shock = 100.0 * correct_shock / total
acc_resp = 100.0 * correct_resp / total
acc_exact = 100.0 * exact_match / total

print(f"Test Accuracy - Septic Shock: {acc_shock:.2f}%, Respiratory Failure: {acc_resp:.2f}%")
print(f"Exact-match Accuracy (both correct simultaneously): {acc_exact:.2f}%")

