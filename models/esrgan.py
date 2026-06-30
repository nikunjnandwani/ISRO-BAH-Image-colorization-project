"""
ESRGAN — Lightweight Super-Resolution for Thermal Band B10
===========================================================
Architecture:
    Generator  : RRDB-Net (Residual-in-Residual Dense Blocks) + PixelShuffle
    Discriminator : VGG-style for adversarial training

Input  : 1-channel LR thermal patch  (1, H/sf, W/sf)
Output : 1-channel HR thermal patch  (1, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building Blocks ───────────────────────────────────────────────

class DenseBlock(nn.Module):
    """Single dense block: Conv → LeakyReLU, with dense connections."""

    def __init__(self, in_ch: int, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,           growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch + growth,  growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_ch + 2*growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_ch + 3*growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_ch + 4*growth, in_ch,  3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x                       # residual scaling


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (3 cascaded DenseBlocks)."""

    def __init__(self, nf: int, gc: int = 32):
        super().__init__()
        self.db1 = DenseBlock(nf, gc)
        self.db2 = DenseBlock(nf, gc)
        self.db3 = DenseBlock(nf, gc)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x


# ── Generator ─────────────────────────────────────────────────────

class ESRGANGenerator(nn.Module):
    """
    RRDB-Net generator.

    Parameters
    ----------
    in_channels   : input channels (1 for single-band thermal)
    out_channels  : output channels
    nf            : number of feature maps
    nb            : number of RRDB blocks
    gc            : growth channels inside dense blocks
    scale_factor  : upsampling factor (must be power of 2 or 3)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 nf: int = 64, nb: int = 4, gc: int = 32,
                 scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

        # First convolution
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)

        # RRDB trunk
        self.trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling layers (each doubles resolution)
        upsample_layers = []
        num_up = int(math.log2(scale_factor))
        for _ in range(num_up):
            upsample_layers += [
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsampler = nn.Sequential(*upsample_layers)

        # Final output
        self.conv_hr   = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat = feat + trunk                        # global residual

        feat = self.upsampler(feat)
        out  = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ── Discriminator ──────────────────────────────────────────────────

class ESRGANDiscriminator(nn.Module):
    """VGG-style discriminator for ESRGAN adversarial training."""

    def __init__(self, in_channels: int = 1, nf: int = 64):
        super().__init__()

        def block(inc, outc, stride=1, bn=True):
            layers = [nn.Conv2d(inc, outc, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(outc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *block(in_channels, nf, bn=False),
            *block(nf, nf, stride=2),
            *block(nf, nf * 2),
            *block(nf * 2, nf * 2, stride=2),
            *block(nf * 2, nf * 4),
            *block(nf * 4, nf * 4, stride=2),
            *block(nf * 4, nf * 8),
            *block(nf * 8, nf * 8, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    gen  = ESRGANGenerator(1, 1, nf=64, nb=4, scale_factor=4)
    disc = ESRGANDiscriminator(1, 64)

    lr = torch.randn(2, 1, 32, 32)
    hr = gen(lr)
    print(f"Generator  : {lr.shape} → {hr.shape}")

    score = disc(hr)
    print(f"Discriminator : {hr.shape} → {score.shape}")

    total_g = sum(p.numel() for p in gen.parameters())
    total_d = sum(p.numel() for p in disc.parameters())
    print(f"Generator params : {total_g:,}")
    print(f"Discriminator params : {total_d:,}")
