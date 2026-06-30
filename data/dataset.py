"""
PyTorch Dataset Classes for IR Colorization Pipeline
=====================================================
Provides two datasets:
  • SRDataset         – (LR_B10, HR_B10) pairs for ESRGAN training
  • ColorizationDataset – (IR_4ch, RGB_3ch) pairs for Pix2Pix training

Both load .npy tiles produced by prepare_dataset.py.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2


# ── Helpers ────────────────────────────────────────────────────────

def _random_augment(arrays: list[np.ndarray], flip: bool = True,
                    rotate: bool = True) -> list[np.ndarray]:
    """Apply identical random flip + 90° rotation to a list of (C,H,W) arrays."""
    if flip and np.random.rand() > 0.5:
        arrays = [a[:, :, ::-1].copy() for a in arrays]   # horizontal flip
    if flip and np.random.rand() > 0.5:
        arrays = [a[:, ::-1, :].copy() for a in arrays]   # vertical flip
    if rotate:
        k = np.random.randint(0, 4)
        if k > 0:
            arrays = [np.rot90(a, k, axes=(1, 2)).copy() for a in arrays]
    return arrays


def _normalize_reflectance(x: np.ndarray, clip_lo: float = 0.0,
                           clip_hi: float = 0.5) -> np.ndarray:
    """Clip surface-reflectance values and map to [-1, 1]."""
    x = np.clip(x, clip_lo, clip_hi)
    x = (x - clip_lo) / (clip_hi - clip_lo)   # → [0, 1]
    x = x * 2.0 - 1.0                          # → [-1, 1]
    return x


def _normalize_thermal(x: np.ndarray, mean: float = 300.0,
                        std: float = 10.0) -> np.ndarray:
    """Z-score thermal band then tanh-squash to ~[-1, 1]."""
    x = (x - mean) / (std + 1e-8)
    x = np.clip(x, -3.0, 3.0)   # clip outliers
    x = x / 3.0                  # → roughly [-1, 1]
    return x


def _denormalize_reflectance(x: torch.Tensor, clip_lo: float = 0.0,
                              clip_hi: float = 0.5) -> torch.Tensor:
    """Inverse of _normalize_reflectance.  [-1,1] → [clip_lo, clip_hi]."""
    x = (x + 1.0) / 2.0                         # → [0, 1]
    x = x * (clip_hi - clip_lo) + clip_lo        # → original scale
    return x


def _denormalize_thermal(x: torch.Tensor, mean: float = 300.0,
                          std: float = 10.0) -> torch.Tensor:
    """Inverse of _normalize_thermal."""
    x = x * 3.0 * (std + 1e-8) + mean
    return x


# ── SR Dataset ─────────────────────────────────────────────────────

class SRDataset(Dataset):
    """
    Super-Resolution dataset for the thermal band (B10).

    Each sample returns:
        lr  – float32 tensor, shape (1, H//sf, W//sf)
        hr  – float32 tensor, shape (1, H, W)

    LR is created on-the-fly by down-sampling the HR patch.
    """

    def __init__(self, data_dir: str, scale_factor: int = 4,
                 augment: bool = True,
                 thermal_mean: float = 300.0, thermal_std: float = 10.0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.scale_factor = scale_factor
        self.augment = augment
        self.thermal_mean = thermal_mean
        self.thermal_std = thermal_std

        self.files = sorted(self.data_dir.glob('*.npy'))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tile = np.load(self.files[idx])          # (7, 128, 128)
        b10 = tile[6:7, :, :]                     # (1, 128, 128)

        if self.augment:
            [b10] = _random_augment([b10])

        h, w = b10.shape[1], b10.shape[2]
        sf = self.scale_factor

        # Create LR by down-sampling with bicubic interpolation
        b10_2d = b10[0]                            # (H, W)
        lr_2d = cv2.resize(b10_2d, (w // sf, h // sf),
                           interpolation=cv2.INTER_CUBIC)
        lr = lr_2d[np.newaxis, :, :]               # (1, H//sf, W//sf)

        # Normalise
        hr = _normalize_thermal(b10, self.thermal_mean, self.thermal_std)
        lr = _normalize_thermal(lr,  self.thermal_mean, self.thermal_std)

        return (torch.from_numpy(lr.astype(np.float32)),
                torch.from_numpy(hr.astype(np.float32)))


# ── Colorization Dataset ───────────────────────────────────────────

class ColorizationDataset(Dataset):
    """
    Pix2Pix colorization dataset.

    Each sample returns:
        ir_input   – float32 tensor (4, H, W) : [B5, B6, B7, B10]
        rgb_target – float32 tensor (3, H, W) : [B4(R), B3(G), B2(B)]
        raw_bands  – float32 tensor (7, H, W) : all bands, unnormalised
                     (used for semantic-index computation in loss)

    If ``sr_model`` is provided, B10 is super-resolved before stacking.
    """

    # Indices into the 7-band tile
    IR_INDICES  = [3, 4, 5, 6]   # B5, B6, B7, B10
    RGB_INDICES = [2, 1, 0]      # B4(Red), B3(Green), B2(Blue)

    def __init__(self, data_dir: str, augment: bool = True,
                 thermal_mean: float = 300.0, thermal_std: float = 10.0,
                 stats_path: str | None = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.thermal_mean = thermal_mean
        self.thermal_std = thermal_std

        self.files = sorted(self.data_dir.glob('*.npy'))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")

        # Load computed stats if available
        if stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                self.stats = json.load(f)
        else:
            self.stats = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tile = np.load(self.files[idx])          # (7, 128, 128)

        # Separate bands
        ir_raw  = tile[self.IR_INDICES]           # (4, H, W) — B5, B6, B7, B10
        rgb_raw = tile[self.RGB_INDICES]           # (3, H, W) — R, G, B

        if self.augment:
            [ir_raw, rgb_raw] = _random_augment([ir_raw, rgb_raw])

        # Keep an unnormalised copy for semantic index computation
        raw_bands = np.concatenate([rgb_raw, ir_raw], axis=0)  # (7, H, W)
        # raw_bands layout: [R, G, B, NIR, SWIR1, SWIR2, Thermal]

        # Normalise reflectance bands (first 3 of IR, all of RGB)
        ir_norm = np.empty_like(ir_raw)
        ir_norm[:3] = _normalize_reflectance(ir_raw[:3])       # B5, B6, B7
        ir_norm[3:] = _normalize_thermal(ir_raw[3:],
                                         self.thermal_mean,
                                         self.thermal_std)     # B10

        rgb_norm = _normalize_reflectance(rgb_raw)             # R, G, B

        return (torch.from_numpy(ir_norm.astype(np.float32)),
                torch.from_numpy(rgb_norm.astype(np.float32)),
                torch.from_numpy(raw_bands.astype(np.float32)))


# ── DataLoader factory ─────────────────────────────────────────────

def get_sr_loaders(prepared_dir: str, batch_size: int = 16,
                   scale_factor: int = 4, num_workers: int = 4,
                   thermal_mean: float = 300.0,
                   thermal_std: float = 10.0):
    """Return train / val DataLoaders for the SR task."""
    train_ds = SRDataset(f"{prepared_dir}/train", scale_factor,
                         augment=True,
                         thermal_mean=thermal_mean,
                         thermal_std=thermal_std)
    val_ds   = SRDataset(f"{prepared_dir}/val", scale_factor,
                         augment=False,
                         thermal_mean=thermal_mean,
                         thermal_std=thermal_std)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


def get_colorize_loaders(prepared_dir: str, batch_size: int = 8,
                         num_workers: int = 4,
                         thermal_mean: float = 300.0,
                         thermal_std: float = 10.0):
    """Return train / val DataLoaders for the colorization task."""
    stats_path = f"{prepared_dir}/stats.json"
    train_ds = ColorizationDataset(f"{prepared_dir}/train", augment=True,
                                   thermal_mean=thermal_mean,
                                   thermal_std=thermal_std,
                                   stats_path=stats_path)
    val_ds   = ColorizationDataset(f"{prepared_dir}/val", augment=False,
                                   thermal_mean=thermal_mean,
                                   thermal_std=thermal_std,
                                   stats_path=stats_path)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl
