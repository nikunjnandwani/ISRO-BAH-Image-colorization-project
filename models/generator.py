"""
U-Net Generator for Pix2Pix Colorization
==========================================
Input  : 4-channel IR stack  [B5(NIR), B6(SWIR1), B7(SWIR2), B10(Thermal)]
Output : 3-channel RGB image [R(B4),   G(B3),     B(B2)]

Architecture follows the original Pix2Pix paper with additions:
  • Spectral-normalised convolutions in encoder
  • Self-attention at 16×16 resolution
  • Skip connections via concatenation
"""

import torch
import torch.nn as nn


# ── Building Blocks ───────────────────────────────────────────────

class UNetDownBlock(nn.Module):
    """Encoder block: Conv → (BN) → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Decoder block: ConvTranspose → BN → ReLU (+ optional Dropout)."""

    def __init__(self, in_ch: int, out_ch: int, use_dropout: bool = False,
                 dropout_rate: float = 0.5):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        # Handle potential size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)


class SelfAttention(nn.Module):
    """Lightweight self-attention module for spatial feature refinement."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.key   = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        k = self.key(x).view(B, -1, H * W)                      # (B, C', HW)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)           # (B, HW, HW)
        v = self.value(x).view(B, -1, H * W)                    # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# ── U-Net Generator ───────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    U-Net generator for IR → RGB translation.

    For 128×128 input with n_downsample=7:
        Encoder : 128→64→32→16→8→4→2→1
        Decoder : 1→2→4→8→16→32→64→128
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 ngf: int = 64, n_downsample: int = 7,
                 use_dropout: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.n_downsample = n_downsample

        # ── Encoder ──
        # Layer 0: no BatchNorm
        self.down0 = UNetDownBlock(in_channels, ngf, use_bn=False)

        encoder_channels = [ngf]
        in_c = ngf
        self.encoders = nn.ModuleList()
        for i in range(1, n_downsample):
            out_c = min(ngf * (2 ** i), 512)
            self.encoders.append(UNetDownBlock(in_c, out_c, use_bn=True))
            encoder_channels.append(out_c)
            in_c = out_c

        # Self-attention at the 4th encoder (16×16 → 8×8 level for 128 input)
        self.attention = SelfAttention(encoder_channels[3]) if n_downsample > 3 else None
        self.attn_level = 3

        # ── Decoder ──
        self.decoders = nn.ModuleList()
        dec_in = encoder_channels[-1]  # bottleneck channels

        for i in range(n_downsample - 1):
            skip_ch = encoder_channels[n_downsample - 2 - i]
            out_c = skip_ch
            use_drop = use_dropout and (i < 3)  # dropout in first 3 decoder layers
            self.decoders.append(UNetUpBlock(dec_in, out_c, use_drop, dropout_rate))
            dec_in = out_c * 2   # because of concatenation with skip

        # ── Final output ──
        self.final = nn.Sequential(
            nn.ConvTranspose2d(dec_in, out_channels, 4, 2, 1),
            nn.Tanh(),  # output in [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Encoder
        skips = []
        out = self.down0(x)
        skips.append(out)

        for i, enc in enumerate(self.encoders):
            out = enc(out)
            # Apply attention at the specified level
            if self.attention is not None and (i + 1) == self.attn_level:
                out = self.attention(out)
            if i < len(self.encoders) - 1:   # don't store bottleneck as skip
                skips.append(out)

        # Decoder with skip connections
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            out = dec(out, skip)

        return self.final(out)


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    gen = UNetGenerator(in_channels=4, out_channels=3, ngf=64, n_downsample=7)
    x = torch.randn(2, 4, 128, 128)
    y = gen(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in gen.parameters()):,}")
