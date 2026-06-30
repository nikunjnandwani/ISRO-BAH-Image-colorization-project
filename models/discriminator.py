"""
PatchGAN Discriminator for Pix2Pix Colorization
=================================================
Classifies every 70×70 overlapping patch of the input as real or fake.

Input : concatenation of condition (4-ch IR) and image (3-ch RGB) = 7 channels
Output: 2-D map of patch predictions
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator (Markovian discriminator).

    Parameters
    ----------
    in_channels : condition channels + image channels (default 4+3 = 7)
    ndf         : base number of filters
    n_layers    : number of intermediate conv layers
    """

    def __init__(self, in_channels: int = 7, ndf: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Second-to-last layer with stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final 1-channel prediction map
        layers += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, condition, target):
        """
        Parameters
        ----------
        condition : (B, 4, H, W) IR input
        target    : (B, 3, H, W) real or generated RGB

        Returns
        -------
        patch_pred : (B, 1, H', W') patch-level real/fake predictions
        """
        x = torch.cat([condition, target], dim=1)  # (B, 7, H, W)
        return self.model(x)


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    disc = PatchGANDiscriminator(in_channels=7, ndf=64, n_layers=3)
    cond = torch.randn(2, 4, 128, 128)
    tgt  = torch.randn(2, 3, 128, 128)
    pred = disc(cond, tgt)
    print(f"Condition : {cond.shape}")
    print(f"Target    : {tgt.shape}")
    print(f"Prediction: {pred.shape}")
    print(f"Params    : {sum(p.numel() for p in disc.parameters()):,}")
