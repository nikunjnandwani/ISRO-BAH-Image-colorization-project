"""
LadleNet — Dual-Branch IR → RGB Generator
==========================================
Drop-in replacement for UNetGenerator in train_colorize.py and evaluate.py.

Interface is IDENTICAL to UNetGenerator:
    gen = LadleNet(in_channels=4, out_channels=3, ngf=64)
    pred_rgb = gen(ir_input)   # ir_input: (B, 4, H, W)  →  pred_rgb: (B, 3, H, W)

Architecture:
    • Structural Branch  — captures edges / textures from IR bands
    • Semantic Branch    — captures land-cover semantics (water, veg, urban)
    • Cross-Attention Fusion ("the ladle") — scoops relevant features from both
    • Shared U-Net Decoder with skip connections from BOTH branches

Why better than plain Pix2Pix U-Net:
    • Structural branch preserves fine IR edges after ESRGAN SR
    • Semantic branch tells the decoder what object class a region is
    • Cross-attention ensures colors are assigned semantically, not just
      pixel-by-pixel → fewer hallucinations, water stays blue, veg stays green

Colab T4 compatibility:
    • Default ngf=64 → ~54M params (slightly larger than the 42M U-Net)
    • Runs comfortably within T4 16 GB VRAM at batch_size=8, 256×256 tiles
    • ngf=32 reduces memory by ~4× if needed

Usage in train_colorize.py — change ONE import line:
    # OLD:  from models.generator import UNetGenerator
    # NEW:  from models.ladlenet import LadleNet as UNetGenerator

Usage in evaluate.py — same single-line swap.
No other changes needed anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic building blocks ──────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → InstanceNorm → LeakyReLU block used throughout both branches."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 use_norm=True, activation='leaky'):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not use_norm)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if activation == 'leaky':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with instance norm — used in the semantic branch."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


# ── Structural Branch ──────────────────────────────────────────────

class StructuralBranch(nn.Module):
    """
    Focuses on edges, textures, and fine spatial structure of IR bands.

    Uses larger kernels in early layers to capture large spatial patterns
    (field boundaries, road networks, building outlines) that are strong
    in IR data.

    Returns a list of feature maps at 4 scales for U-Net skip connections.
    """

    def __init__(self, in_ch, ngf):
        super().__init__()

        # Stem: large kernel to capture coarse IR structure
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ngf, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder — 4 downsampling stages
        self.down1 = ConvBnRelu(ngf,      ngf * 2,  stride=2)   # H/2
        self.down2 = ConvBnRelu(ngf * 2,  ngf * 4,  stride=2)   # H/4
        self.down3 = ConvBnRelu(ngf * 4,  ngf * 8,  stride=2)   # H/8
        self.down4 = ConvBnRelu(ngf * 8,  ngf * 8,  stride=2)   # H/16

        # Edge refinement at full resolution using dilated convs
        self.edge_refine = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, dilation=2, padding=2, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, dilation=4, padding=4, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        s0 = self.stem(x)                   # (B, ngf,    H,    W)
        s0 = s0 + self.edge_refine(s0)      # edge-enhanced full-res features
        s1 = self.down1(s0)                 # (B, ngf*2,  H/2,  W/2)
        s2 = self.down2(s1)                 # (B, ngf*4,  H/4,  W/4)
        s3 = self.down3(s2)                 # (B, ngf*8,  H/8,  W/8)
        s4 = self.down4(s3)                 # (B, ngf*8,  H/16, W/16)
        return s0, s1, s2, s3, s4


# ── Semantic Branch ────────────────────────────────────────────────

class SemanticBranch(nn.Module):
    """
    Focuses on land-cover semantics — water, vegetation, urban, bare soil.

    Uses smaller kernels + residual blocks to build abstract semantic
    representations. The deeper residual stack helps it learn what class
    a region belongs to (critical for correct color assignment).

    Returns features at the same 4 scales as StructuralBranch.
    """

    def __init__(self, in_ch, ngf):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            ConvBnRelu(ngf,     ngf * 2, stride=2, activation='relu'),
            ResidualBlock(ngf * 2),
        )
        self.down2 = nn.Sequential(
            ConvBnRelu(ngf * 2, ngf * 4, stride=2, activation='relu'),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
        )
        self.down3 = nn.Sequential(
            ConvBnRelu(ngf * 4, ngf * 8, stride=2, activation='relu'),
            ResidualBlock(ngf * 8),
            ResidualBlock(ngf * 8),
        )
        self.down4 = nn.Sequential(
            ConvBnRelu(ngf * 8, ngf * 8, stride=2, activation='relu'),
            ResidualBlock(ngf * 8),
            ResidualBlock(ngf * 8),
            ResidualBlock(ngf * 8),
        )

    def forward(self, x):
        s0 = self.stem(x)       # (B, ngf,    H,    W)
        s1 = self.down1(s0)     # (B, ngf*2,  H/2,  W/2)
        s2 = self.down2(s1)     # (B, ngf*4,  H/4,  W/4)
        s3 = self.down3(s2)     # (B, ngf*8,  H/8,  W/8)
        s4 = self.down4(s3)     # (B, ngf*8,  H/16, W/16)
        return s0, s1, s2, s3, s4


# ── Cross-Attention Fusion ("The Ladle") ───────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention using channel attention instead of
    spatial attention — avoids the H*W softmax that causes OOM.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        # Channel squeeze-excitation style cross attention
        mid = max(channels // 8, 16)

        self.gap = nn.AdaptiveAvgPool2d(1)   # global context from semantic

        self.cross_attn = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, struct_feat, sem_feat):
        B, C, H, W = struct_feat.shape

        # Global semantic context
        sem_ctx = self.gap(sem_feat).expand_as(struct_feat)

        # Cross attention weights: how much semantic context to inject
        # at each channel, conditioned on structural features
        attn_w = self.cross_attn(
            torch.cat([struct_feat, sem_ctx], dim=1)
        )                                         # (B, C, H, W)

        # Apply: semantic features gated by attention weights
        attended = sem_feat * attn_w

        # Fuse structural + attended semantic
        fused = self.fuse(torch.cat([struct_feat, attended], dim=1))
        return fused


# ── Bottleneck ─────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """
    Deepest part of the network. Processes fused features at H/16 resolution.
    Uses dilated convolutions to maintain receptive field without losing resolution.
    """

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, dilation=2, padding=2, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, dilation=4, padding=4, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, dilation=2, padding=2, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


# ── Decoder Block ──────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample → concat with BOTH structural and semantic skip connections
    → conv to reduce channels.

    Takes fused skip (from CrossAttentionFusion) so the decoder always
    has both structural and semantic context at every scale.
    """

    def __init__(self, in_ch, skip_ch, out_ch, use_dropout=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.5) if use_dropout else nn.Identity()

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch from odd-sized inputs
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x


# ── LadleNet ──────────────────────────────────────────────────────

class LadleNet(nn.Module):
    """
    LadleNet: Dual-Branch IR → RGB Colorization Generator.

    Drop-in replacement for UNetGenerator. Accepts identical arguments
    (in_channels, out_channels, ngf) — extra kwargs are absorbed silently
    so the call sites in train_colorize.py and evaluate.py need no changes.

    Args:
        in_channels  (int): number of IR input channels. Default 4 matches
                            your dataset: B5, B6, B7 NIR + B10 thermal.
        out_channels (int): 3 for RGB output.
        ngf          (int): base feature count. Default 64.
        use_dropout  (bool): dropout in decoder (training-time regularization).
        **kwargs: absorbs n_downsample, dropout_rate etc. from call sites.
    """

    def __init__(self, in_channels=4, out_channels=3, ngf=64,
                 use_dropout=True, **kwargs):
        super().__init__()
        self.ngf = ngf

        # ── Dual branches ──
        self.struct_branch = StructuralBranch(in_channels, ngf)
        self.sem_branch    = SemanticBranch(in_channels, ngf)

        # ── Cross-attention fusion at each scale ──
        # Scale 4 (deepest): ngf*8 channels — spatial size H/16
        # Scale 3:           ngf*8 channels — spatial size H/8
        # Scale 2:           ngf*4 channels — spatial size H/4
        # Scale 1:           ngf*2 channels — spatial size H/2
        # Scale 0 (full):    ngf   channels — spatial size H
        self.fuse4 = CrossAttentionFusion(ngf * 8, num_heads=4)
        self.fuse3 = CrossAttentionFusion(ngf * 8, num_heads=4)
        self.fuse2 = CrossAttentionFusion(ngf * 4, num_heads=4)
        self.fuse1 = CrossAttentionFusion(ngf * 2, num_heads=2)
        self.fuse0 = CrossAttentionFusion(ngf,     num_heads=1)

        # ── Bottleneck ──
        self.bottleneck = Bottleneck(ngf * 8)

        # ── Decoder (4 upsampling stages) ──
        # After bottleneck: ngf*8 channels
        # Skip from fuse3:  ngf*8 channels
        self.dec4 = DecoderBlock(ngf * 8,  ngf * 8, ngf * 8,
                                 use_dropout=use_dropout)
        self.dec3 = DecoderBlock(ngf * 8,  ngf * 4, ngf * 4,
                                 use_dropout=use_dropout)
        self.dec2 = DecoderBlock(ngf * 4,  ngf * 2, ngf * 2,
                                 use_dropout=False)
        self.dec1 = DecoderBlock(ngf * 2,  ngf,     ngf,
                                 use_dropout=False)

        # ── Output head ──
        self.output_head = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf // 2, out_channels, 1),
            nn.Tanh(),   # output in [-1, 1] matching train_colorize.py convention
        )

        # Weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) — 4-channel IR input
                channels: [B5_NIR, B6_SWIR1, B7_SWIR2, B10_Thermal]
                normalized to [-1, 1] by ColorizationDataset
        Returns:
            (B, 3, H, W) — predicted RGB in [-1, 1]
        """
        # ── Encode through both branches ──
        st0, st1, st2, st3, st4 = self.struct_branch(x)
        se0, se1, se2, se3, se4 = self.sem_branch(x)

        # ── Fuse at each scale with cross-attention ──
        f4 = self.fuse4(st4, se4)   # (B, ngf*8, H/16, W/16)
        f3 = self.fuse3(st3, se3)   # (B, ngf*8, H/8,  W/8)
        f2 = self.fuse2(st2, se2)   # (B, ngf*4, H/4,  W/4)
        f1 = self.fuse1(st1, se1)   # (B, ngf*2, H/2,  W/2)
        f0 = self.fuse0(st0, se0)   # (B, ngf,   H,    W)

        # ── Bottleneck at deepest fused features ──
        b = self.bottleneck(f4)     # (B, ngf*8, H/16, W/16)

        # ── Decode with fused skip connections ──
        d4 = self.dec4(b,  f3)     # (B, ngf*8, H/8,  W/8)
        d3 = self.dec3(d4, f2)     # (B, ngf*4, H/4,  W/4)
        d2 = self.dec2(d3, f1)     # (B, ngf*2, H/2,  W/2)
        d1 = self.dec1(d2, f0)     # (B, ngf,   H,    W)

        # ── Output ──
        return self.output_head(d1)  # (B, 3, H, W) in [-1, 1]


# ── Updated train_colorize.py usage ───────────────────────────────
#
# In train_colorize.py, change:
#
#   from models.generator import UNetGenerator
#   gen = UNetGenerator(
#       in_channels=4, out_channels=3,
#       ngf=args.ngf, n_downsample=7,
#       use_dropout=True, dropout_rate=0.5
#   ).to(device)
#
# To:
#
#   from models.ladlenet import LadleNet
#   gen = LadleNet(
#       in_channels=4, out_channels=3,
#       ngf=args.ngf, use_dropout=True
#   ).to(device)
#
# In evaluate.py, change:
#
#   from models.generator import UNetGenerator
#   gen = UNetGenerator(
#       in_channels=4, out_channels=3,
#       ngf=args.ngf, n_downsample=7,
#       use_dropout=False
#   ).to(device)
#
# To:
#
#   from models.ladlenet import LadleNet
#   gen = LadleNet(
#       in_channels=4, out_channels=3,
#       ngf=args.ngf, use_dropout=False
#   ).to(device)
#
# Everything else (discriminator, losses, semantic constraint,
# dataset, checkpointing, evaluation metrics) stays UNCHANGED.
# ──────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    # Quick sanity check — matches your Colab setup
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = LadleNet(in_channels=4, out_channels=3, ngf=64, use_dropout=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"LadleNet params: {total_params:,}")

    # Simulate a batch from ColorizationDataset
    dummy = torch.randn(8, 4, 256, 256).to(device)   # batch_size=8, 256×256 tiles
    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]  (should be in [-1, 1])")
    print("✓ LadleNet sanity check passed")
