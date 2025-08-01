# import torch
# import torch.nn as nn

# class WaveformEncoder(nn.Module):
#     """
#     Input:  (batch, 1, 2500)  # 10-sec window @250 Hz
#     Output: (batch, 512)
#     """
#     def __init__(self, out_dim=512):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(1, 64, 7, stride=2, padding=3), nn.ReLU(),
#             nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1), nn.Flatten(),
#             nn.Linear(128, out_dim)
#         )
#     def forward(self, x):
#         return self.net(x)

# wave_enc = WaveformEncoder()

###################

# -*- coding: utf-8 -*-
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn

# def extract_waveform_from_image(image_path, normalize=True):
#     img = Image.open(image_path).convert('L')
#     arr = np.array(img, dtype=np.float32)
#     arr = 255.0 - arr
#     indices = arr.argmax(axis=0)
#     signal = indices.astype(np.float32)
#     if normalize:
#         signal = signal / (arr.shape[0] - 1)
#         signal = 2.0 * signal - 1.0
#     return signal

# class WaveformEncoder(nn.Module):
#     def __init__(self, out_dim=512):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),
#         )
#         self.fc = nn.Linear(128, out_dim)

#     def forward(self, x):
#         feats = self.features(x)
#         feats = feats.squeeze(-1)
#         return self.fc(feats)

# if __name__ == "__main__":
#     waveform = extract_waveform_from_image('/Users/kaiyuanwu/Downloads/ecg00004.png', normalize=True)
#     waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     encoder = WaveformEncoder(out_dim=512)
#     embedding = encoder(waveform_tensor)
#     print("Embedding shape:", embedding.shape)
#     print("Embedding[:20]:", embedding[:20])


########## 1st shot !


import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def extract_waveform_from_image(image_path, normalize=True):
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    arr = 255.0 - arr
    indices = arr.argmax(axis=0)
    signal = indices.astype(np.float32)
    if normalize:
        signal = signal / (arr.shape[0] - 1)
        signal = 2.0 * signal - 1.0
    return signal

class WaveformEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        feats = self.features(x)
        feats = feats.squeeze(-1)
        return self.fc(feats)

class SimpleWaveformDecoder(nn.Module):
    def __init__(self, embedding_dim=512, output_length=500):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_length),
            nn.Tanh()
        )

    def forward(self, embedding):
        return self.decoder(embedding)

def visualize_waveform(original, reconstructed):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(original, color='blue')
    plt.title('Original Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed, color='red')
    plt.title('Reconstructed Waveform from Embedding')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = '/Users/kaiyuanwu/Downloads/ecg00004.png'
    waveform = extract_waveform_from_image(image_path, normalize=True)
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    encoder = WaveformEncoder(out_dim=512)
    embedding = encoder(waveform_tensor)

    print("Embedding shape:", embedding.shape)
    print("Embedding[:20]:", embedding[0, :20])

    # Decode embedding back to waveform
    decoder = SimpleWaveformDecoder(embedding_dim=512, output_length=waveform.shape[0])
    reconstructed_waveform = decoder(embedding).detach().numpy().squeeze()

    # Visualize original and reconstructed waveforms
    visualize_waveform(waveform, reconstructed_waveform)


####### improved

# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# # Data Extraction
# def extract_waveform_from_image(image_path, normalize=True):
#     img = Image.open(image_path).convert('L')
#     arr = np.array(img, dtype=np.float32)
#     arr = 255.0 - arr
#     indices = arr.argmax(axis=0)
#     signal = indices.astype(np.float32)
#     if normalize:
#         signal = (signal / (arr.shape[0] - 1)) * 2.0 - 1.0
#     return signal

# # Improved Encoder
# class WaveformEncoder(nn.Module):
#     def __init__(self, embedding_dim=512):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),  
#             nn.ReLU(),
#             nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(4),
#             nn.Flatten(),
#             nn.Linear(256 * 4, embedding_dim),
#         )

#     def forward(self, x):
#         return self.encoder(x)

# # Improved Decoder
# class WaveformDecoder(nn.Module):
#     def __init__(self, embedding_dim=512, output_length=1500):
#         super().__init__()
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(embedding_dim, 256 * 4),
#             nn.ReLU(),
#         )
#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),
#             nn.Tanh(),
#         )
#         self.output_length = output_length

#     def forward(self, z):
#         x = self.decoder_fc(z)
#         x = x.view(-1, 256, 4)
#         x = self.decoder_conv(x)
#         x = nn.functional.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
#         return x

# # Visualize original vs reconstructed waveforms
# def visualize_waveform(original, reconstructed):
#     plt.figure(figsize=(12, 6))

#     plt.subplot(2, 1, 1)
#     plt.plot(original, color='blue')
#     plt.title('Original Waveform')

#     plt.subplot(2, 1, 2)
#     plt.plot(reconstructed, color='red')
#     plt.title('Reconstructed Waveform')

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # Data preparation
#     image_path = '/Users/kaiyuanwu/Downloads/ecg00004.png'
#     waveform = extract_waveform_from_image(image_path)
#     waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     # Model initialization
#     embedding_dim = 512
#     encoder = WaveformEncoder(embedding_dim=embedding_dim)
#     decoder = WaveformDecoder(embedding_dim=embedding_dim, output_length=waveform.shape[0])

#     # Training setup
#     epochs = 500
#     optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
#     criterion = nn.MSELoss()

#     encoder.train()
#     decoder.train()

#     # Training loop
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         embedding = encoder(waveform_tensor)
#         reconstructed = decoder(embedding)
#         loss = criterion(reconstructed, waveform_tensor)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 100 == 0 or epoch == 0:
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

#     # Final reconstruction
#     encoder.eval()
#     decoder.eval()
#     with torch.no_grad():
#         final_embedding = encoder(waveform_tensor)
#         reconstructed_waveform = decoder(final_embedding).cpu().numpy().squeeze()

#     # Visualization
#     visualize_waveform(waveform, reconstructed_waveform)


