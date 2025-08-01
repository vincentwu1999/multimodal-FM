"""
Sensor encoders and masked autoencoders.

This module collects all sensor‑related components from the original
`multimodal_fusion_complete-real_data.py` script.  The classes are
copied verbatim; only the import paths have been updated to
accommodate the new package layout.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..preprocessing.positional_encodings import get_2d_sincos_pos_embed, get_positional_encoding


class SharedConvEncoder(nn.Module):
    """Shared convolutional encoder for sensor data.

    Maps multi‑channel 1D sensor sequences to a sequence of patch embeddings.
    The convolutional parameters are chosen so that the number of output
    patches is divisible by common sensor channel counts (e.g. 3),
    avoiding shape mismatches when constructing 2D positional
    embeddings.  For an input sequence length of 128 and two
    convolutional layers, the default parameters produce 30 patches.

    Args:
        in_channels: number of input channels (sensors).
        embed_dim: dimension of the output embeddings.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        # The first convolution reduces the temporal resolution by half
        # (stride=2).  The second convolution uses a kernel of size 5
        # with stride 2 and no padding, resulting in a final length
        # floor((L2 - 5) / 2 + 1).  For an initial length of 128, this
        # yields 30 patches after the second convolution, which is
        # divisible by 3 and works well with 2D positional embeddings.
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=2, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, embed_dim, L_out)
        x = self.encoder(x)
        # transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class LinearProbe(nn.Module):
    """Simple linear classification head for CLS token."""

    def __init__(self, embed_dim: int, num_classes: int = 1) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use the CLS token (first token) for classification
        return self.head(x[:, 0])


class SensorEncoder(nn.Module):
    """
    Sensor encoder inspired by LSM‑2 with adaptive and inherited masking.

    The encoder uses a shared convolutional encoder to produce patch
    embeddings, then applies a Transformer encoder with adaptive masking
    strategies.  Positional embeddings are generated using 2D sin‑cos
    functions along the time and sensor dimensions.

    Args:
        sensor_channels: number of input sensor channels.
        seq_length: length of the input sequence.
        embed_dim: dimension of the embeddings.
        depth: number of Transformer encoder layers.
        num_heads: number of attention heads.
        mlp_ratio: expansion ratio for the feedforward network.
        num_classes: number of output classes (unused here, but kept for compatibility).
        artificial_mask_ratio: ratio of tokens to drop out during AIM.
        pretrain_mask_ratio: fraction of tokens to mask during pretraining.
        decoder_depth: number of decoder layers used for reconstruction.
    """

    def __init__(
        self,
        sensor_channels: int,
        seq_length: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1,
        artificial_mask_ratio: float = 0.5,
        pretrain_mask_ratio: float = 0.5,
        decoder_depth: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sensor_channels = sensor_channels
        self.artificial_mask_ratio = artificial_mask_ratio
        self.pretrain_mask_ratio = pretrain_mask_ratio
        # convolutional embedding
        self.shared_conv_encoder = SharedConvEncoder(sensor_channels, embed_dim)
        # compute number of patches from an example tensor
        example_tensor = torch.zeros(1, sensor_channels, seq_length)
        with torch.no_grad():
            encoded_patches = self.shared_conv_encoder(example_tensor)
            # record the number of patches after potential cropping
            self.num_patches = encoded_patches.shape[1]
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embedding placeholder (not learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        # linear probe for potential classification (not used here)
        self.linear_probe = LinearProbe(embed_dim, num_classes)
        # components for pretraining: decoder and mask token
        if decoder_depth > 0:
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
            self.pretrain_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pretrain_decoder_pred = nn.Linear(embed_dim, embed_dim)
        else:
            self.pretrain_decoder = None
        # initialize ids_restore
        self.ids_restore = None
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize positional embeddings and layer parameters.

        The 2D sin‑cos positional embedding is computed on a grid whose
        dimensions are derived directly from the number of patches and
        the number of sensor channels.  Because ``SharedConvEncoder``
        crops the temporal dimension so that ``num_patches`` is a
        multiple of ``sensor_channels``, the grid dimensions are
        guaranteed to multiply to ``num_patches``.
        """
        grid_size_time = self.num_patches // self.sensor_channels
        grid_size_sensor = self.sensor_channels
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size_time=grid_size_time,
            grid_size_sensor=grid_size_sensor,
            cls_token=True,
        )
        # assign positional embedding (shape: (num_patches + 1, embed_dim))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def aim_masking(self, batch_size: int, num_patches: int, device: torch.device) -> torch.Tensor:
        # choose strategy randomly
        strategy = np.random.choice(['random', 'temporal', 'signal'])
        mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
        prob = {'random': 0.8, 'temporal': 0.5, 'signal': 0.5}[strategy]
        if strategy == 'random':
            mask = torch.rand(batch_size, num_patches, device=device) < prob
        elif strategy == 'temporal':
            num_masked = int(prob * num_patches)
            masked_indices = np.random.choice(num_patches, num_masked, replace=False)
            mask[:, masked_indices] = True
        elif strategy == 'signal':
            num_signals_to_mask = int(prob * self.sensor_channels)
            signals_to_mask = np.random.choice(self.sensor_channels, num_signals_to_mask, replace=False)
            patches_per_channel = num_patches // self.sensor_channels
            for sig in signals_to_mask:
                start_idx, end_idx = sig * patches_per_channel, (sig + 1) * patches_per_channel
                mask[:, start_idx:end_idx] = True
        return mask

    def forward_encoder(self, sensor_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # convolutional encoding
        x = self.shared_conv_encoder(sensor_data)  # (B, num_patches, embed_dim)
        # add positional embedding (skip cls for now)
        x = x + self.pos_embed[:, 1:, :]
        # generate artificial mask
        artificial_mask = self.aim_masking(x.size(0), x.size(1), x.device)
        # dropout removal step
        D = int(self.artificial_mask_ratio * self.num_patches)
        noise = torch.rand(x.size(0), self.num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_drop = ids_shuffle[:, :D]
        ids_keep = ids_shuffle[:, D:]
        # gather kept tokens and corresponding mask
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        attn_mask_kept = torch.gather(artificial_mask, 1, ids_keep)
        # prepend CLS token and its positional embedding
        cls_tokens = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_tokens.expand(x_kept.shape[0], -1, -1)
        x_kept = torch.cat([cls_tokens, x_kept], dim=1)
        attn_mask_kept = torch.cat([
            torch.zeros((x.size(0), 1), device=x.device, dtype=torch.bool),
            attn_mask_kept
        ], dim=1)
        # transformer encoding with attention masking
        x_encoded = self.transformer_encoder(x_kept, src_key_padding_mask=attn_mask_kept)
        x_encoded = self.norm(x_encoded)
        # store restoration indices (not used here)
        self.ids_restore = ids_shuffle.argsort(dim=1)
        return x_encoded, artificial_mask

    def forward(self, sensor_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_encoded, artificial_mask = self.forward_encoder(sensor_data)
        logits = self.linear_probe(x_encoded)
        return logits, artificial_mask

    # ------------------------------------------------------------------
    # Pretraining with reconstruction of masked conv tokens
    def _random_mask_pretrain(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask a fraction of tokens for pretraining.

        Returns the unmasked tokens, the indices of kept tokens and the indices of masked tokens.
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - self.pretrain_mask_ratio))
        rand_indices = torch.rand(B, N, device=x.device).argsort(dim=1)
        keep_indices = rand_indices[:, :num_keep]
        mask_indices = rand_indices[:, num_keep:]
        x_keep = torch.gather(x, 1, keep_indices.unsqueeze(-1).expand(-1, -1, D))
        return x_keep, keep_indices, mask_indices

    def forward_pretrain(self, sensor_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-training forward pass for masked reconstruction.

        The method reconstructs masked conv-token features rather than raw sensor values.
        It randomly masks a fraction of conv tokens, encodes the unmasked tokens, and
        decodes to reconstruct the masked tokens.  The loss is mean squared error
        between the predicted and original conv-token features at the masked positions.
        """
        if self.pretrain_decoder is None:
            raise RuntimeError("Pretrain decoder is not defined")
        # conv token encoding
        x = self.shared_conv_encoder(sensor_data)  # (B, N, D)
        N = x.size(1)
        # add positional embeddings (skip CLS during pretraining)
        x = x + self.pos_embed[:, 1:, :]
        # randomly mask tokens
        tokens_keep, keep_indices, mask_indices = self._random_mask_pretrain(x)
        # encode unmasked tokens
        latent = self.transformer_encoder(tokens_keep)
        # build decoder input: mask tokens for all positions, scatter latent to kept positions
        B, _, D = x.shape
        dec_input = self.mask_token.repeat(B, N, 1)
        dec_input = dec_input.scatter(1, keep_indices.unsqueeze(-1).expand(-1, -1, D), latent)
        dec_input = dec_input + self.pos_embed[:, 1:, :]
        # decode
        dec_out = self.pretrain_decoder(dec_input, latent)
        pred_tokens = self.pretrain_decoder_pred(dec_out)  # (B, N, D)
        # compute loss on masked positions
        target_masked = torch.gather(x, 1, mask_indices.unsqueeze(-1).expand(-1, -1, D))
        pred_masked = torch.gather(pred_tokens, 1, mask_indices.unsqueeze(-1).expand(-1, -1, D))
        loss = F.mse_loss(pred_masked, target_masked)
        return loss, pred_masked


class SensorPatchEmbed(nn.Module):
    """Embed multi‑channel time–series into patch tokens.

    The input (B, C, L) is divided into non‑overlapping patches of
    length `patch_size`.  Each patch is flattened and projected into
    an embedding vector of dimension `embed_dim`.
    """

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        assert L % self.patch_size == 0, "Sequence length must be divisible by patch size"
        num_patches = L // self.patch_size
        # reshape to (B, num_patches, C * patch_size)
        x = x.view(B, C, num_patches, self.patch_size).permute(0, 2, 1, 3).reshape(B, num_patches, C * self.patch_size)
        return self.proj(x)


class SensorMaskedAutoencoder(nn.Module):
    """Masked autoencoder for multi‑channel sensor sequences.

    During pre‑training, random patches are masked and the model learns
    to reconstruct them; during inference, the encoder produces latent
    tokens for full sequences.  Inspired by the Adaptive & Inherited
    Masking (AIM) strategy in LSM‑2.

    Args:
        in_channels: number of sensor channels.
        patch_size: length of each patch.
        embed_dim: token embedding dimension.
        encoder_layers: number of Transformer encoder layers.
        decoder_layers: number of Transformer decoder layers (for reconstruction).
        num_heads: number of attention heads.
        mlp_ratio: expansion ratio for feedforward networks.
        mask_ratio: fraction of patches to mask during pretraining.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch_embed = SensorPatchEmbed(in_channels, patch_size, embed_dim)
        self.mask_ratio = mask_ratio
        # Positional encoding (initialized as buffer)
        self.register_buffer("pos_embed", get_positional_encoding(1024, embed_dim), persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        # Decoder for reconstruction
        if decoder_layers > 0:
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_pred = nn.Linear(embed_dim, in_channels * patch_size)
        else:
            self.decoder = None
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _random_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        num_keep = int(N * (1 - self.mask_ratio))
        rand_indices = torch.rand(B, N, device=x.device).argsort(dim=1)
        keep_indices = rand_indices[:, :num_keep]
        mask_indices = rand_indices[:, num_keep:]
        x_keep = torch.gather(x, 1, keep_indices.unsqueeze(-1).expand(-1, -1, D))
        return x_keep, keep_indices, mask_indices

    def forward_pretrain(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Patch embedding + positional encoding
        patch_tokens = self.patch_embed(x)
        N = patch_tokens.size(1)
        pos_emb = self.pos_embed[:N, :].unsqueeze(0)
        patch_tokens = patch_tokens + pos_emb
        # Random mask
        tokens_keep, keep_indices, mask_indices = self._random_mask(patch_tokens)
        latent = self.encoder(tokens_keep)
        if self.decoder is None:
            raise RuntimeError("Decoder disabled; cannot pretrain")
        # Build decoder input: mask tokens everywhere, then scatter encoded tokens at keep positions
        B, _, D = patch_tokens.shape
        dec_input = self.mask_token.repeat(B, N, 1)
        dec_input = dec_input.scatter(1, keep_indices.unsqueeze(-1).expand(-1, -1, D), latent)
        dec_input = dec_input + pos_emb
        dec_out = self.decoder(dec_input, latent)
        pred = self.decoder_pred(dec_out)
        # Reconstruct original patch values, not embeddings.  Compute raw
        # patches of shape (B, N, C*patch_size).  We flatten each
        # patch without projecting to embed_dim.
        C = x.size(1)
        raw_patches = x.view(B, C, N, self.patch_embed.patch_size).permute(0, 2, 1, 3).reshape(B, N, C * self.patch_embed.patch_size)
        # Select only masked patches for loss computation
        target_masked = torch.gather(raw_patches, 1, mask_indices.unsqueeze(-1).expand(-1, -1, raw_patches.shape[2]))
        pred_masked = torch.gather(pred, 1, mask_indices.unsqueeze(-1).expand(-1, -1, pred.shape[2]))
        loss = F.mse_loss(pred_masked, target_masked)
        return loss, pred_masked

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(x)
        N = patch_tokens.size(1)
        pos_emb = self.pos_embed[:N, :].unsqueeze(0)
        patch_tokens = patch_tokens + pos_emb
        latent = self.encoder(patch_tokens)
        return latent


__all__ = [
    "SharedConvEncoder",
    "LinearProbe",
    "SensorEncoder",
    "SensorPatchEmbed",
    "SensorMaskedAutoencoder",
]
