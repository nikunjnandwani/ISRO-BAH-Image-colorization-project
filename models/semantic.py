"""
Semantic Branch — Spectral Index Constraints
=============================================
Computes NDVI and NDWI from predicted and ground-truth bands to
enforce semantic consistency during training.

These indices ensure that:
  • Vegetation regions (high NDVI) get mapped to green
  • Water bodies  (high NDWI) get mapped to blue
  • The colorization does not "hallucinate" incorrect land-cover colours

During training:
  Ground-truth indices are computed from the real RGB + IR bands.
  Predicted indices use the generator's output + known IR bands.

During inference:
  The semantic branch is NOT needed — the generator has already
  learned to produce semantically consistent outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7   # avoid division by zero in index computation


def compute_ndvi(nir: torch.Tensor, red: torch.Tensor) -> torch.Tensor:
    """
    Normalized Difference Vegetation Index.

    NDVI = (NIR − Red) / (NIR + Red)

    Parameters
    ----------
    nir : (B, 1, H, W)  —  Band 5 (from IR input, unnormalised reflectance)
    red : (B, 1, H, W)  —  Band 4 (from RGB, unnormalised reflectance)

    Returns
    -------
    ndvi : (B, 1, H, W)  ∈ [-1, 1]
    """
    return (nir - red) / (nir + red + EPS)


def compute_ndwi(green: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
    """
    Normalized Difference Water Index.

    NDWI = (Green − NIR) / (Green + NIR)

    Parameters
    ----------
    green : (B, 1, H, W) — Band 3 (from RGB, unnormalised reflectance)
    nir   : (B, 1, H, W) — Band 5 (from IR input, unnormalised reflectance)

    Returns
    -------
    ndwi : (B, 1, H, W)  ∈ [-1, 1]
    """
    return (green - nir) / (green + nir + EPS)


def compute_ndbi(swir1: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
    """
    Normalized Difference Built-up Index.

    NDBI = (SWIR1 − NIR) / (SWIR1 + NIR)

    Parameters
    ----------
    swir1 : (B, 1, H, W) — Band 6 (from IR input)
    nir   : (B, 1, H, W) — Band 5 (from IR input)

    Returns
    -------
    ndbi : (B, 1, H, W)  ∈ [-1, 1]
    """
    return (swir1 - nir) / (swir1 + nir + EPS)


class SemanticConstraint(nn.Module):
    """
    Computes semantic consistency loss between predicted and ground-truth
    spectral indices (NDVI, NDWI).

    Expects **unnormalised** (raw reflectance) bands as input.

    The raw_bands tensor from ColorizationDataset has layout:
        [R(B4), G(B3), B(B2), NIR(B5), SWIR1(B6), SWIR2(B7), Thermal(B10)]
        idx: 0      1      2      3        4          5           6
    """

    def __init__(self):
        super().__init__()

    def _extract_bands(self, raw_bands: torch.Tensor):
        """Extract individual bands from the raw 7-channel tensor."""
        red   = raw_bands[:, 0:1]   # B4
        green = raw_bands[:, 1:2]   # B3
        nir   = raw_bands[:, 3:4]   # B5
        swir1 = raw_bands[:, 4:5]   # B6
        return red, green, nir, swir1

    def compute_gt_indices(self, raw_bands: torch.Tensor):
        """Compute ground-truth NDVI and NDWI from real bands."""
        red, green, nir, swir1 = self._extract_bands(raw_bands)
        ndvi = compute_ndvi(nir, red)
        ndwi = compute_ndwi(green, nir)
        return ndvi, ndwi

    def compute_pred_indices(self, pred_rgb_raw: torch.Tensor,
                              raw_bands: torch.Tensor):
        """
        Compute predicted NDVI and NDWI using the generator's output
        (denormalised to reflectance) and known IR bands.

        Parameters
        ----------
        pred_rgb_raw : (B, 3, H, W) predicted [R, G, B] in reflectance scale
        raw_bands    : (B, 7, H, W) all bands, raw
        """
        pred_red   = pred_rgb_raw[:, 0:1]   # predicted B4
        pred_green = pred_rgb_raw[:, 1:2]   # predicted B3
        nir = raw_bands[:, 3:4]             # known B5

        ndvi = compute_ndvi(nir, pred_red)
        ndwi = compute_ndwi(pred_green, nir)
        return ndvi, ndwi

    def forward(self, pred_rgb_raw: torch.Tensor,
                raw_bands: torch.Tensor) -> torch.Tensor:
        """
        Semantic consistency loss = MSE(pred_indices, gt_indices).

        Parameters
        ----------
        pred_rgb_raw : (B, 3, H, W) predicted RGB in raw reflectance scale
        raw_bands    : (B, 7, H, W) all raw bands from dataset

        Returns
        -------
        loss : scalar tensor
        """
        ndvi_gt, ndwi_gt = self.compute_gt_indices(raw_bands)
        ndvi_pred, ndwi_pred = self.compute_pred_indices(pred_rgb_raw, raw_bands)

        loss_ndvi = F.mse_loss(ndvi_pred, ndvi_gt)
        loss_ndwi = F.mse_loss(ndwi_pred, ndwi_gt)

        return loss_ndvi + loss_ndwi


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    sem = SemanticConstraint()
    raw = torch.rand(2, 7, 128, 128) * 0.5       # fake raw bands
    pred_rgb = torch.rand(2, 3, 128, 128) * 0.5  # fake prediction

    loss = sem(pred_rgb, raw)
    print(f"Semantic loss: {loss.item():.6f}")

    ndvi_gt, ndwi_gt = sem.compute_gt_indices(raw)
    print(f"NDVI range: [{ndvi_gt.min():.3f}, {ndvi_gt.max():.3f}]")
    print(f"NDWI range: [{ndwi_gt.min():.3f}, {ndwi_gt.max():.3f}]")
