# model.py
import torch
import torch.nn as nn


class MultiModalModel(nn.Module):
    """KIDS-FM-Core model: integrates EHR structured data, clinical text, waveform, and image embedding."""
    def __init__(
        self,
        num_struct_features=10,   # number of structured numeric features (labs/vitals)
        struct_bins=10,           # number of bins to quantize each continuous feature
        struct_embed_dim=8,       # embedding size for each quantized feature value
        text_vocab_size=10000,    # vocabulary size for note text (after BPE tokenization)
        text_embed_dim=100,       # embedding dimension for token embeddings
        text_hidden_dim=64,       # hidden size for LSTM encoder for text
        wave_channels=1,          # number of channels in waveform (e.g. 1 for single signal)
        wave_length=100,          # length of waveform input signal
        wave_filters=16,          # number of filters for CNN on waveform
        wave_kernel=5,            # kernel size for CNN
        image_embed_dim=512,      # dimension of incoming CXR image embeddings
        hidden_dim=64,            # common hidden dimension for fused features
        num_classes=2             # number of output classes (e.g. 2 acute conditions)
    ):
        super().__init__()
        # --- Structured data encoder ---
        # Embeddings for each feature's quantized value
        self.num_struct_features = num_struct_features
        self.struct_bins = struct_bins
        self.struct_embeddings = nn.ModuleList([
            nn.Embedding(struct_bins, struct_embed_dim) for _ in range(num_struct_features)
        ])
        # Linear projection to common hidden_dim
        self.struct_proj = nn.Linear(num_struct_features * struct_embed_dim, hidden_dim)
        
        # --- Text (clinical notes) encoder ---
        # Token embedding table (could use pre-trained embeddings in practice)
        self.text_embed = nn.Embedding(text_vocab_size, text_embed_dim)
        # Bidirectional LSTM to encode text sequence
        self.text_lstm = nn.LSTM(text_embed_dim, text_hidden_dim, batch_first=True, bidirectional=True)
        # Linear projection to common hidden_dim (bidirectional LSTM outputs 2*hidden_dim)
        self.text_proj = nn.Linear(2 * text_hidden_dim, hidden_dim)
        
        # --- Waveform encoder ---
        # 1D CNN layers to extract features from waveform
        self.wave_conv1 = nn.Conv1d(wave_channels, wave_filters, kernel_size=wave_kernel, stride=2, padding=2)
        self.wave_conv2 = nn.Conv1d(wave_filters, wave_filters * 2, kernel_size=wave_kernel, stride=2, padding=2)
        # Adaptive pooling to get a single value per filter (global average pooling)
        self.wave_pool = nn.AdaptiveAvgPool1d(1)
        # Linear projection to common hidden_dim
        self.wave_proj = nn.Linear(wave_filters * 2, hidden_dim)
        
        # --- Image embedding encoder ---
        # Linear projection from image embedding to common hidden_dim
        self.img_proj = nn.Linear(image_embed_dim, hidden_dim)
        
        # --- Modality Mutual Attention fusion ---
        # Multi-head attention to fuse modalities (queries=keys=values=all modality features)
        # Using 4 heads here; embed_dim=hidden_dim must be divisible by num_heads.
        self.mma = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # --- Classification head ---
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, struct_data, text_tokens, wave_data, img_embed):
        """
        Forward pass through the multimodal model.
        :param struct_data: Tensor of shape (batch, num_struct_features) with continuous vitals/labs.
        :param text_tokens: Tensor of shape (batch, seq_len) with token indices for clinical note.
        :param wave_data:   Tensor of shape (batch, wave_length) with waveform values.
        :param img_embed:   Tensor of shape (batch, image_embed_dim) with CXR image embedding.
        
        :return: Tensor of shape (batch, num_classes) with logits for each class.
        """
        # Encode structured data:
        # Quantize each continuous feature into an integer bin index (0 to struct_bins-1).
        # Here we assume struct_data is normalized [0,1]; adjust quantization as needed for real data.
        quantized = (struct_data * self.struct_bins).clamp(max=self.struct_bins - 1)  # scale up and clip
        quantized = quantized.long()  # convert to integer bin indices
        # Embed each feature's quantized value and concatenate
        struct_emb_list = []
        for i in range(self.num_struct_features):
            feat_ids = quantized[:, i]                 # shape: (batch,)
            feat_emb = self.struct_embeddings[i](feat_ids)  # shape: (batch, struct_embed_dim)
            struct_emb_list.append(feat_emb)
        # Concatenate all feature embeddings for each patient
        struct_emb_concat = torch.cat(struct_emb_list, dim=1)       # shape: (batch, num_struct_features * struct_embed_dim)
        struct_feat = torch.relu(self.struct_proj(struct_emb_concat))  # shape: (batch, hidden_dim)
        
        # Encode text (clinical notes):
        # Get token embeddings for the sequence
        text_emb = self.text_embed(text_tokens)     # shape: (batch, seq_len, text_embed_dim)
        # Pass through BiLSTM
        # out: (batch, seq_len, 2*text_hidden_dim) if bidirectional; h_n: (2, batch, text_hidden_dim) for 1-layer BiLSTM
        lstm_out, (h_n, c_n) = self.text_lstm(text_emb)
        # h_n contains the last hidden state for each direction. For a 1-layer BiLSTM:
        # h_n[0] is last state of forward direction, h_n[1] is last state of backward direction.
        # Concatenate forward and backward final states to get a fixed-length text representation.
        forward_final = h_n[0]        # shape: (batch, text_hidden_dim)
        backward_final = h_n[1]       # shape: (batch, text_hidden_dim)
        text_feat = torch.cat([forward_final, backward_final], dim=1)  # shape: (batch, 2*text_hidden_dim)
        text_feat = torch.relu(self.text_proj(text_feat))              # shape: (batch, hidden_dim)
        
        # Encode waveform:
        # Waveform input is (batch, wave_length); reshape to (batch, 1, wave_length) for CNN
        wave_x = wave_data.unsqueeze(1)                               # shape: (batch, 1, wave_length)
        wave_x = torch.relu(self.wave_conv1(wave_x))                  # shape: (batch, wave_filters, L/2)
        wave_x = torch.relu(self.wave_conv2(wave_x))                  # shape: (batch, wave_filters*2, L/4)
        wave_x = self.wave_pool(wave_x)                               # shape: (batch, wave_filters*2, 1) after global pooling
        wave_x = wave_x.squeeze(-1)                                   # shape: (batch, wave_filters*2)
        wave_feat = torch.relu(self.wave_proj(wave_x))                # shape: (batch, hidden_dim)
        
        # Encode image embedding:
        img_feat = torch.relu(self.img_proj(img_embed))               # shape: (batch, hidden_dim)
        
        # Fuse modalities with mutual attention:
        # Stack modality feature vectors into a sequence for attention: shape (batch, 4, hidden_dim)
        # The sequence order can be arbitrary; here we use [struct, text, wave, image]
        modality_sequence = torch.stack([struct_feat, text_feat, wave_feat, img_feat], dim=1)
        
        # Apply multi-head attention (self-attention across modality features).
        # attn_output has shape (batch, 4, hidden_dim).
        attn_output, _ = self.mma(modality_sequence, modality_sequence, modality_sequence)
        # We now have an attended representation for each modality token. We combine them:
        fused_feat = attn_output.mean(dim=1)   # shape: (batch, hidden_dim), average across modalities
        
        # Classification: map fused feature to class logits
        logits = self.classifier(fused_feat)   # shape: (batch, num_classes)
        return logits
