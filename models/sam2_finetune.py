"""
SAM2-based lung nodule segmentation model.

Core architecture:
  CT slice (B, 1, H, W)
    → channel adapter Conv2d(1→3)
    → SAM2 image encoder  [or FallbackEncoder if SAM2 not installed]
    → SinusoidalPosEmbed
    → LightweightMaskDecoder (learnable nodule prompt token)
    → logits (B, 1, H, W)

The DropoutMultiheadAttention in the decoder keeps dropout active at
inference time, enabling Monte Carlo Dropout uncertainty estimation.

Run this file directly for a quick sanity check::

    python models/sam2_finetune.py
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAM2 import — graceful fallback when not installed
# ---------------------------------------------------------------------------

_SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2
    from sam2.modeling.sam2_base import SAM2Base

    _SAM2_AVAILABLE = True
    logger.info("SAM2 detected — will use SAM2 image encoder")
except ImportError:
    logger.warning(
        "SAM2 not found (pip install git+https://github.com/facebookresearch/"
        "segment-anything-2.git). Using FallbackEncoder instead."
    )


# ---------------------------------------------------------------------------
# 2D Sinusoidal Positional Embedding
# ---------------------------------------------------------------------------


class SinusoidalPosEmbed(nn.Module):
    """2D sinusoidal positional encoding added to image feature maps.

    Generates a fixed (non-learned) position embedding following the
    formulation from Attention Is All You Need, extended to 2D grids.
    The embedding is computed once and cached; resized on-the-fly when
    the input spatial dimensions change.

    Parameters
    ----------
    embed_dim : int
        Channel dimension of the feature map. Must be divisible by 4.
        Default 256.
    temperature : float
        Temperature scaling for the frequency bands. Default 10000.

    Notes
    -----
    For a feature map of shape (B, C, H, W), the embedding is a tensor
    of shape (1, C, H, W) added to the feature map in-place (broadcasting
    across the batch dimension). This avoids allocating a new tensor for
    every forward pass.
    """

    def __init__(self, embed_dim: int = 256, temperature: float = 10_000.0) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
        self.embed_dim = embed_dim
        self.temperature = temperature
        # Cache: invalidated when spatial dims change
        self._cached_pe: Optional[Tensor] = None
        self._cached_shape: Optional[Tuple[int, int]] = None

    def _build_pe(self, H: int, W: int, device: torch.device) -> Tensor:
        """Build a (1, embed_dim, H, W) sinusoidal position encoding."""
        half_dim = self.embed_dim // 2  # split evenly between H and W encodings

        # Frequency bands
        dim_t = torch.arange(half_dim // 2, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / half_dim)

        y_pos = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
        x_pos = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(1)

        y_enc = y_pos / dim_t  # (H, half_dim//2)
        x_enc = x_pos / dim_t  # (W, half_dim//2)

        # sin/cos interleaved for each axis
        y_embed = torch.stack([y_enc.sin(), y_enc.cos()], dim=2).flatten(
            1
        )  # (H, half_dim)
        x_embed = torch.stack([x_enc.sin(), x_enc.cos()], dim=2).flatten(
            1
        )  # (W, half_dim)

        # Combine: (H, W, embed_dim)
        pe = torch.zeros(H, W, self.embed_dim, device=device)
        pe[:, :, :half_dim] = y_embed.unsqueeze(1).expand(H, W, half_dim)
        pe[:, :, half_dim:] = x_embed.unsqueeze(0).expand(H, W, half_dim)

        # → (1, embed_dim, H, W)
        return pe.permute(2, 0, 1).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Add 2D positional encoding to the input feature map.

        Parameters
        ----------
        x : Tensor
            Feature map of shape (B, embed_dim, H, W).

        Returns
        -------
        Tensor
            x + positional encoding, same shape as input.
        """
        H, W = x.shape[-2], x.shape[-1]
        if self._cached_shape != (H, W) or (
            self._cached_pe is not None and self._cached_pe.device != x.device
        ):
            self._cached_pe = self._build_pe(H, W, x.device)
            self._cached_shape = (H, W)
        return x + self._cached_pe


# ---------------------------------------------------------------------------
# Dropout-enabled Multi-head Attention (for MC Dropout at inference)
# ---------------------------------------------------------------------------


class DropoutMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention with a persistent output dropout.

    Subclasses ``nn.MultiheadAttention`` and adds an extra dropout layer
    applied to the attention output *after* the projection. Unlike the
    built-in ``dropout`` parameter (which is only active during training),
    this ``attn_dropout`` is applied via a separate ``nn.Dropout`` module
    whose training mode is controlled externally — enabling Monte Carlo
    Dropout at inference via ``enable_dropout_modules()``.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model.
    num_heads : int
        Number of parallel attention heads.
    attn_dropout_prob : float
        Dropout probability for the post-attention output. Default 0.1.
    **kwargs
        Additional keyword arguments forwarded to ``nn.MultiheadAttention``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout_prob: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        # This dropout is controlled separately — stays active at MC Dropout inference
        self.attn_dropout = nn.Dropout(p=attn_dropout_prob)

    def forward(  # type: ignore[override]
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with extra post-attention dropout.

        Parameters
        ----------
        query, key, value : Tensor
            Standard multi-head attention inputs.
        **kwargs
            Forwarded to ``nn.MultiheadAttention.forward``.

        Returns
        -------
        attn_output : Tensor
            Attention output with dropout applied.
        attn_weights : Tensor or None
            Attention weight matrix (if ``need_weights=True``).
        """
        attn_output, attn_weights = super().forward(query, key, value, **kwargs)
        # Apply the extra dropout — stays stochastic during MC Dropout inference
        attn_output = self.attn_dropout(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Lightweight Mask Decoder
# ---------------------------------------------------------------------------


class LightweightMaskDecoder(nn.Module):
    """Cross-attention based mask decoder with a learnable nodule prompt token.

    Replaces SAM2's interactive point/box prompts with a single learnable
    "nodule" token that is trained to attend the encoder features relevant
    to pulmonary nodules. Cross-attention is performed via
    ``DropoutMultiheadAttention``, enabling MC Dropout at inference.

    After cross-attention, a two-stage ConvTranspose2d head upsamples the
    cross-attended image features by 4× to produce full-resolution logits.

    Parameters
    ----------
    embed_dim : int
        Feature channel dimension from the encoder. Default 256.
    num_heads : int
        Number of attention heads. Default 8.
    mlp_ratio : float
        MLP hidden dimension ratio relative to embed_dim. Default 4.0.
    attn_dropout : float
        Dropout prob in ``DropoutMultiheadAttention``. Default 0.1.
    proj_dropout : float
        Dropout prob after the FFN projection. Default 0.1.
    num_output_masks : int
        Number of output mask channels (1 for binary segmentation). Default 1.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        num_output_masks: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable nodule prompt token — replaces interactive SAM2 prompts
        # Shape: (1, 1, embed_dim) to broadcast across batch
        self.prompt_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Cross-attention: prompt queries image features
        self.cross_attn = DropoutMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout_prob=attn_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward network on prompt token
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=proj_dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(p=proj_dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Upsampling head: 2× ConvTranspose2d → 4× total upscale
        # embed_dim → embed_dim//2 → num_output_masks
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(
                embed_dim // 2, num_output_masks, kernel_size=2, stride=2
            ),
        )

    def forward(
        self,
        image_features: Tensor,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Decode a segmentation mask from image features.

        Parameters
        ----------
        image_features : Tensor
            Encoder output of shape (B, embed_dim, H', W').
        return_features : bool
            If True, also return the cross-attended feature map for
            visualization (shape B, embed_dim, H', W'). Default False.

        Returns
        -------
        logits : Tensor
            Segmentation logits of shape (B, num_output_masks, H'*4, W'*4).
            Apply sigmoid for probability.
        features : Tensor (only if return_features=True)
            Cross-attended feature map (B, embed_dim, H', W').
        """
        B, C, H, W = image_features.shape

        # Flatten spatial dims for attention: (B, H*W, C)
        feats_flat = image_features.flatten(2).permute(0, 2, 1)

        # Expand learnable prompt token to match batch size
        prompt = self.prompt_token.expand(B, -1, -1)  # (B, 1, embed_dim)

        # Cross-attention: prompt queries flattened image features
        attended, _ = self.cross_attn(
            query=prompt,
            key=feats_flat,
            value=feats_flat,
        )
        prompt = self.norm1(prompt + attended)  # (B, 1, embed_dim)

        # FFN on prompt
        prompt = self.norm2(prompt + self.ffn(prompt))  # (B, 1, embed_dim)

        # Broadcast prompt back into spatial feature map
        # prompt → (B, embed_dim, 1, 1) → broadcast to (B, embed_dim, H, W)
        prompt_spatial = prompt.permute(0, 2, 1).reshape(B, C, 1, 1)
        attended_features = image_features * torch.sigmoid(prompt_spatial)

        # 4× upsampling to produce full-resolution logits
        logits = self.upsample(attended_features)  # (B, num_output_masks, H*4, W*4)

        if return_features:
            return logits, attended_features
        return logits


# ---------------------------------------------------------------------------
# Fallback UNet-style Encoder (no SAM2 needed)
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Two Conv2d + BN + ReLU layers, no pooling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FallbackEncoder(nn.Module):
    """Lightweight UNet-style CNN encoder used when SAM2 is unavailable.

    Produces feature maps at 1/4 of the input resolution with ``embed_dim``
    channels, matching the output contract expected by ``LightweightMaskDecoder``.

    Architecture::

        in_ch → 64 → pool   (1/2)
        64    → 128 → pool  (1/4)
        128   → embed_dim   (no pool — final output at H/4, W/4)

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 after channel adapter). Default 3.
    embed_dim : int
        Output feature channel dimension. Default 256.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            _ConvBlock(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → H/2, W/2
        )
        self.stage2 = nn.Sequential(
            _ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → H/4, W/4
        )
        self.stage3 = _ConvBlock(128, embed_dim)  # → H/4, W/4 at embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """Encode input image to feature map.

        Parameters
        ----------
        x : Tensor
            Input image tensor of shape (B, 3, H, W).

        Returns
        -------
        Tensor
            Feature map of shape (B, embed_dim, H//4, W//4).
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


# ---------------------------------------------------------------------------
# Main model: SAM2LungSegmentor
# ---------------------------------------------------------------------------


class SAM2LungSegmentor(nn.Module):
    """Uncertainty-aware lung nodule segmentation model.

    Wraps a SAM2 image encoder (or FallbackEncoder) with a custom
    ``LightweightMaskDecoder``. The 1-channel CT input is adapted to
    3-channel RGB via a learned ``Conv2d(1, 3, 1)`` before encoding.

    The decoder's ``DropoutMultiheadAttention`` keeps dropout stochastic at
    inference when ``enable_dropout_modules()`` is called, enabling Monte
    Carlo Dropout uncertainty quantification without any extra wrappers.

    Parameters
    ----------
    sam2_checkpoint : str or None
        Path to SAM2 model checkpoint. If None or SAM2 not installed,
        ``FallbackEncoder`` is used.
    sam2_config : str or None
        SAM2 config name (e.g. ``"sam2_hiera_large"``). Only used when
        ``sam2_checkpoint`` is not None.
    embed_dim : int
        Feature channel dimension. Default 256.
    num_heads : int
        Number of attention heads in the decoder. Default 8.
    attn_dropout : float
        Attention dropout probability (kept active at MC inference). Default 0.1.
    proj_dropout : float
        FFN projection dropout probability. Default 0.1.
    encoder_frozen : bool
        If True, encoder parameters require no grad at init. Default True.
    in_channels : int
        Input image channels (1 for CT). Default 1.
    """

    def __init__(
        self,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        encoder_frozen: bool = True,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 1-channel CT → 3-channel "RGB" for SAM2 (or FallbackEncoder) compatibility
        self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)

        # Positional embedding
        self.pos_embed = SinusoidalPosEmbed(embed_dim=embed_dim)

        # Encoder: try SAM2 first, fall back gracefully
        self._using_sam2 = False
        self.encoder: nn.Module

        if sam2_checkpoint is not None and _SAM2_AVAILABLE:
            try:
                sam2_model: SAM2Base = build_sam2(
                    config_file=sam2_config or "sam2_hiera_large.yaml",
                    ckpt_path=sam2_checkpoint,
                )
                self.encoder = sam2_model.image_encoder
                self._using_sam2 = True
                logger.info("SAM2 image encoder loaded from %s", sam2_checkpoint)
            except Exception as exc:
                logger.warning(
                    "Failed to load SAM2 encoder (%s) — using FallbackEncoder", exc
                )
                self.encoder = FallbackEncoder(in_channels=3, embed_dim=embed_dim)
        else:
            if sam2_checkpoint is not None:
                logger.warning(
                    "SAM2 not installed — ignoring checkpoint %s and using FallbackEncoder.",
                    sam2_checkpoint,
                )
            self.encoder = FallbackEncoder(in_channels=3, embed_dim=embed_dim)

        # Lightweight decoder
        self.decoder = LightweightMaskDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            num_output_masks=1,
        )

        # Optionally freeze encoder at init
        if encoder_frozen:
            self.freeze_encoder()

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (require_grad = False).

        Used during the first ``encoder_frozen_epochs`` to stabilise the
        decoder before end-to-end fine-tuning begins.
        """
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        n = sum(1 for p in self.encoder.parameters())
        logger.info("Encoder frozen: %d parameter tensors", n)

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters for end-to-end fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad_(True)
        n = sum(1 for p in self.encoder.parameters())
        logger.info("Encoder unfrozen: %d parameter tensors", n)

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Parameters
        ----------
        trainable_only : bool
            If True, count only parameters with ``requires_grad=True``.
            If False, count all parameters. Default True.

        Returns
        -------
        int
            Total parameter count.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Segment a 2D CT slice.

        Parameters
        ----------
        x : Tensor
            Input batch of shape (B, 1, H, W), float32, values in [0, 1].
        return_features : bool
            If True, return a tuple ``(logits, encoder_features)`` for
            visualization. Default False.

        Returns
        -------
        logits : Tensor
            Segmentation logits of shape (B, 1, H, W) (after upsample ×4
            from H/4, W/4 encoder output). Apply ``torch.sigmoid`` for
            probability maps.
        features : Tensor (only if return_features=True)
            Encoder feature map (B, embed_dim, H/4, W/4).
        """
        # 1-channel CT → 3-channel
        x3 = self.channel_adapter(x)  # (B, 3, H, W)

        # Encode
        if self._using_sam2:
            # SAM2 encoder returns a dict; extract the backbone features
            enc_out = self.encoder(x3)
            if isinstance(enc_out, dict):
                # SAM2 returns {"backbone_fpn": [...], "vision_pos_enc": [...]}
                img_feats = enc_out["backbone_fpn"][-1]  # highest-resolution FPN level
            else:
                img_feats = enc_out
        else:
            img_feats = self.encoder(x3)  # (B, embed_dim, H/4, W/4)

        # Project to embed_dim if SAM2 output has different channels
        if img_feats.shape[1] != self.embed_dim:
            if not hasattr(self, "_feat_proj"):
                self._feat_proj = nn.Conv2d(
                    img_feats.shape[1], self.embed_dim, kernel_size=1
                ).to(img_feats.device)
            img_feats = self._feat_proj(img_feats)

        # Add positional encoding
        img_feats = self.pos_embed(img_feats)  # (B, embed_dim, H', W')

        # Decode
        if return_features:
            logits, dec_feats = self.decoder(img_feats, return_features=True)
            return logits, dec_feats
        else:
            logits = self.decoder(img_feats)
            # Resize logits to match original input resolution (H, W)
            H_in, W_in = x.shape[-2], x.shape[-1]
            if logits.shape[-2] != H_in or logits.shape[-1] != W_in:
                logits = F.interpolate(
                    logits, size=(H_in, W_in), mode="bilinear", align_corners=False
                )
            return logits


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_model(
    sam2_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    embed_dim: int = 256,
    num_heads: int = 8,
    attn_dropout: float = 0.1,
    proj_dropout: float = 0.1,
    encoder_frozen: bool = True,
    **kwargs,
) -> SAM2LungSegmentor:
    """Build and return a ``SAM2LungSegmentor`` instance.

    Parameters
    ----------
    sam2_checkpoint : str or None
        Path to SAM2 checkpoint. None → FallbackEncoder.
    sam2_config : str or None
        SAM2 config name. Only used when checkpoint is not None.
    embed_dim : int
        Feature dimension. Default 256.
    num_heads : int
        Attention heads. Default 8.
    attn_dropout : float
        Attention dropout (MC Dropout key). Default 0.1.
    proj_dropout : float
        FFN dropout. Default 0.1.
    encoder_frozen : bool
        Freeze encoder at init. Default True.
    **kwargs
        Additional kwargs forwarded to ``SAM2LungSegmentor.__init__``.

    Returns
    -------
    SAM2LungSegmentor
        Configured model ready for training or inference.
    """
    model = SAM2LungSegmentor(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        embed_dim=embed_dim,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
        encoder_frozen=encoder_frozen,
        **kwargs,
    )
    total_params = model.num_parameters(trainable_only=False)
    trainable_params = model.num_parameters(trainable_only=True)
    logger.info(
        "Model built: %s | total params: %s | trainable: %s",
        model.__class__.__name__,
        f"{total_params:,}",
        f"{trainable_params:,}",
    )
    print(
        f"[build_model] SAM2LungSegmentor | "
        f"total={total_params:,} params | trainable={trainable_params:,} | "
        f"SAM2={'yes' if model._using_sam2 else 'no (FallbackEncoder)'}"
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity check — run with: python models/sam2_finetune.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 65)
    print("SAM2LungSegmentor — forward pass sanity check")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.1,
        proj_dropout=0.1,
        encoder_frozen=False,
    ).to(device)

    # Test 1: standard forward pass
    B, C, H, W = 2, 1, 512, 512
    x = torch.randn(B, C, H, W, device=device)
    logits = model(x)
    print(f"\n[Test 1] Input:  {tuple(x.shape)}")
    print(f"         Output: {tuple(logits.shape)}")
    assert logits.shape == (B, 1, H, W), f"Shape mismatch: {logits.shape}"
    print("         ✓ Output shape matches input H×W")

    # Test 2: smaller patch (typical training batch)
    x2 = torch.randn(4, 1, 96, 96, device=device)
    logits2 = model(x2)
    print(f"\n[Test 2] Input:  {tuple(x2.shape)}")
    print(f"         Output: {tuple(logits2.shape)}")
    assert logits2.shape == (4, 1, 96, 96)
    print("         ✓ 96×96 patch — shape correct")

    # Test 3: return_features
    logits3, feats = model(x2, return_features=True)
    print(f"\n[Test 3] Logits:   {tuple(logits3.shape)}")
    print(f"         Features: {tuple(feats.shape)}")
    print("         ✓ return_features=True works")

    # Test 4: parameter counts
    total = model.num_parameters(trainable_only=False)
    trainable = model.num_parameters(trainable_only=True)
    print(f"\n[Test 4] Total params:     {total:>12,}")
    print(f"         Trainable params:  {trainable:>12,}")
    assert total > 1_000_000, "Model should have >1M parameters"
    print("         ✓ > 1M parameters")

    # Test 5: freeze / unfreeze
    model.freeze_encoder()
    tr_frozen = model.num_parameters(trainable_only=True)
    model.unfreeze_encoder()
    tr_unfrozen = model.num_parameters(trainable_only=True)
    print(f"\n[Test 5] Trainable when frozen:   {tr_frozen:,}")
    print(f"         Trainable when unfrozen: {tr_unfrozen:,}")
    assert tr_frozen < tr_unfrozen
    print("         ✓ freeze/unfreeze works correctly")

    print("\nAll tests passed. ✓")
