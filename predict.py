# predict.py
import torch
from model import MultiModalModel

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)
model.load_state_dict(torch.load("kids_fm_core.pth", map_location=device))
model.eval()

# Simulate a single new patient input (in practice, replace this with actual data):
# Here we manually set some values to simulate a scenario:
new_struct = torch.rand(1, 10)
new_text = torch.randint(0, 10000, (1, 50))
new_wave = torch.randn(1, 100)
new_img = torch.randn(1, 512)
# For demonstration, let's set values that trigger our label rule:
new_struct[0, 0] = 0.8   # high first vital -> likely shock
new_wave[0, 0] = 1.0    # positive first waveform value -> likely respiratory failure

# (If this were real, you would collect these from EHR/monitoring devices instead of random generation.)

# Run the model on the new data
with torch.no_grad():
    logits = model(new_struct.to(device), new_text.to(device), new_wave.to(device), new_img.to(device))
    probs = torch.sigmoid(logits).cpu().numpy()[0]  # convert to probabilities and take first (only) sample

# Define class names for clarity
class_names = ["Septic Shock", "Respiratory Failure"]
# Determine predicted yes/no by thresholding at 0.5
pred_labels = ["Yes" if p >= 0.5 else "No" for p in probs]

# Print results
print("Input patient data:")
print(f"  Structured features (first two shown): {new_struct[0, :2].tolist()} ...")
print(f"  First waveform value: {new_wave[0,0].item():.3f}")
print(f"Predicted probabilities for acute conditions:")
for cname, p, lab in zip(class_names, probs, pred_labels):
    print(f"  {cname}: {p*100:.1f}% probability -> Predicted {cname}: {lab}")
