"""
Data Preparation Script — Tile Extraction from Landsat GeoTIFFs
================================================================
Reads multi-band Landsat 8/9 GeoTIFFs and extracts overlapping
128×128 patches, filtering out tiles with excessive NaN coverage.

Output structure:
    prepared_data/
        train/ val/ test/
            {region}_{index:05d}.npy   → float32, shape (7, 128, 128)
        stats.json                     → per-band normalisation statistics

Band order in each .npy (matches GeoTIFF):
    0: SR_B2 (Blue)   1: SR_B3 (Green)  2: SR_B4 (Red)
    3: SR_B5 (NIR)    4: SR_B6 (SWIR1)  5: SR_B7 (SWIR2)
    6: ST_B10 (Thermal)

Usage:
    python data/prepare_dataset.py --archive_dir archive --output_dir prepared_data
"""

import os
import sys
import json
import argparse
import numpy as np
import rasterio
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {desc}: {i+1}/{total}", end='', flush=True)
            yield item
        print()

# ── Constants ──────────────────────────────────────────────────────
BAND_NAMES = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10']

# Region filename → train / val / test
SPLIT_MAP = {
    'landsat_delhi_ncr.tif': 'train',
    'landsat_rajasthan.tif': 'train',
    'landsat_gangetic_plain-0000000000-0000000000.tif': 'train',
    'landsat_gangetic_plain-0000000000-0000008960.tif': 'train',
    'landsat_himalayan_foot.tif': 'val',
    'landsat_sundarbans.tif': 'test',
    'landsat_western_ghats.tif': 'test',
}


def extract_tiles(filepath: str, patch_size: int, stride: int,
                  max_nan_ratio: float) -> list:
    """
    Slide a window over a GeoTIFF and keep patches where the fraction
    of NaN pixels (across all bands) is below *max_nan_ratio*.

    NaN pixels inside kept patches are filled with the per-band local
    mean so the downstream network never sees NaN.
    """
    tiles = []

    with rasterio.open(filepath) as ds:
        data = ds.read()                     # (7, H, W), float64
        num_bands, H, W = data.shape

        # Validity mask — True where every band is finite
        valid_mask = np.all(np.isfinite(data), axis=0)  # (H, W)

        # Replace NaN / Inf with 0 for safe indexing
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        n_rows = max(1, (H - patch_size) // stride + 1)
        n_cols = max(1, (W - patch_size) // stride + 1)
        total_patches = n_rows * n_cols

        checked = 0
        for i in range(n_rows):
            for j in range(n_cols):
                y = i * stride
                x = j * stride
                y_end = y + patch_size
                x_end = x + patch_size

                # Safety clamp
                if y_end > H or x_end > W:
                    continue

                patch_mask = valid_mask[y:y_end, x:x_end]
                nan_ratio = 1.0 - patch_mask.mean()

                if nan_ratio <= max_nan_ratio:
                    patch = data[:, y:y_end, x:x_end].copy()

                    # Fill remaining invalid pixels with per-band local mean
                    if not patch_mask.all():
                        for b in range(num_bands):
                            bp = patch[b]
                            valid_vals = bp[patch_mask]
                            fill = valid_vals.mean() if valid_vals.size > 0 else 0.0
                            bp[~patch_mask] = fill
                            patch[b] = bp

                    tiles.append(patch.astype(np.float32))

                checked += 1
                if checked % 5000 == 0:
                    print(f"    Checked {checked}/{total_patches} windows, "
                          f"kept {len(tiles)} tiles so far")

    return tiles


def compute_stats(tiles: list) -> dict:
    """Per-band mean / std / min / max computed from a list of tiles."""
    stacked = np.stack(tiles, axis=0)  # (N, 7, H, W)
    stats = {}
    for b in range(stacked.shape[1]):
        bd = stacked[:, b]
        stats[BAND_NAMES[b]] = {
            'mean': float(bd.mean()),
            'std':  float(bd.std()),
            'min':  float(bd.min()),
            'max':  float(bd.max()),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training tiles from Landsat GeoTIFFs')
    parser.add_argument('--archive_dir', type=str, default='archive')
    parser.add_argument('--output_dir', type=str, default='prepared_data')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=96)
    parser.add_argument('--max_nan_ratio', type=float, default=0.15)
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    output_dir  = Path(args.output_dir)

    for split in ('train', 'val', 'test'):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    all_train_tiles = []
    split_counts = {'train': 0, 'val': 0, 'test': 0}

    for filename, split in SPLIT_MAP.items():
        filepath = archive_dir / filename
        if not filepath.exists():
            print(f"⚠  {filepath} not found — skipping")
            continue

        region = filename.replace('.tif', '').replace('landsat_', '')
        print(f"\n{'─'*60}")
        print(f"Region : {region}")
        print(f"File   : {filename}")
        print(f"Split  : {split}")

        tiles = extract_tiles(
            str(filepath), args.patch_size, args.stride, args.max_nan_ratio)
        print(f"  → Extracted {len(tiles)} valid tiles")

        for idx, tile in enumerate(tiles):
            np.save(output_dir / split / f"{region}_{idx:05d}.npy", tile)

        split_counts[split] += len(tiles)
        if split == 'train':
            all_train_tiles.extend(tiles)

    # ── Normalisation statistics (training set only) ──
    print(f"\n{'─'*60}")
    print("Computing per-band normalisation statistics (train set) …")
    if all_train_tiles:
        stats = compute_stats(all_train_tiles)
        for name, s in stats.items():
            print(f"  {name:8s}  mean={s['mean']:.5f}  std={s['std']:.5f}  "
                  f"range=[{s['min']:.5f}, {s['max']:.5f}]")
    else:
        stats = {}
        print("  WARNING: no training tiles — stats empty")

    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"  Train : {split_counts['train']:>6,} tiles")
    print(f"  Val   : {split_counts['val']:>6,} tiles")
    print(f"  Test  : {split_counts['test']:>6,} tiles")
    print(f"  Total : {sum(split_counts.values()):>6,} tiles")
    print(f"  Stats : {stats_path}")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
