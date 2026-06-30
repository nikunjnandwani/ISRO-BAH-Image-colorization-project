"""
End-to-End Inference Pipeline
===============================
Takes a full Landsat GeoTIFF and produces a colorized RGB output.

Pipeline:
    1. Read GeoTIFF → extract IR bands (B5, B6, B7, B10)
    2. Apply ESRGAN to sharpen B10
    3. Stack [B5, B6, B7, B10_SR], normalize
    4. Tile-based Pix2Pix inference with overlap blending
    5. Write output GeoTIFF with original CRS / transform

Usage:
    python inference.py --input archive/landsat_sundarbans.tif \\
                        --sr_checkpoint checkpoints/sr/sr_final.pth \\
                        --colorize_checkpoint checkpoints/colorize/colorize_best.pth \\
                        --output outputs/sundarbans_colorized.tif
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.esrgan import ESRGANGenerator
from models.generator import UNetGenerator


def normalize_reflectance(x, lo=0.0, hi=0.5):
    """Clip and scale to [-1, 1]."""
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x * 2.0 - 1.0


def denormalize_reflectance(x, lo=0.0, hi=0.5):
    """Convert [-1, 1] back to reflectance."""
    x = (x + 1.0) / 2.0
    return x * (hi - lo) + lo


def normalize_thermal(x, mean=300.0, std=10.0):
    x = (x - mean) / (std + 1e-8)
    x = np.clip(x, -3.0, 3.0) / 3.0
    return x


def load_models(sr_ckpt_path, colorize_ckpt_path, device,
                scale_factor=4, num_rrdb=4, nf=64, ngf=64):
    """Load trained ESRGAN and Pix2Pix generators."""

    # Super-resolution model
    sr_model = None
    if sr_ckpt_path and os.path.exists(sr_ckpt_path):
        sr_model = ESRGANGenerator(
            in_channels=1, out_channels=1,
            nf=nf, nb=num_rrdb, scale_factor=scale_factor
        ).to(device)
        ckpt = torch.load(sr_ckpt_path, map_location=device, weights_only=False)
        sr_model.load_state_dict(ckpt['generator'])
        sr_model.eval()
        print(f"✓ Loaded SR model from {sr_ckpt_path}")
    else:
        print("⚠ No SR model provided — using raw B10")

    # Colorization model
    color_model = UNetGenerator(
        in_channels=4, out_channels=3,
        ngf=ngf, n_downsample=7,
        use_dropout=False   # no dropout at inference
    ).to(device)
    ckpt = torch.load(colorize_ckpt_path, map_location=device, weights_only=False)
    color_model.load_state_dict(ckpt['generator'])
    color_model.eval()
    print(f"✓ Loaded colorization model from {colorize_ckpt_path}")

    return sr_model, color_model


def tile_inference(ir_stack, color_model, device,
                   tile_size=128, overlap=32):
    """
    Run Pix2Pix on a large image using overlapping tiles with
    linear blending to avoid seam artifacts.

    Parameters
    ----------
    ir_stack   : (4, H, W) normalised IR input (numpy)
    color_model: trained UNetGenerator
    tile_size  : inference tile size
    overlap    : overlap between adjacent tiles

    Returns
    -------
    rgb_output : (3, H, W) predicted RGB in [-1, 1] (numpy)
    """
    C, H, W = ir_stack.shape
    stride = tile_size - overlap

    # Pad to make dimensions divisible by stride
    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride
    ir_padded = np.pad(ir_stack,
                       ((0, 0), (0, pad_h), (0, pad_w)),
                       mode='reflect')
    _, Hp, Wp = ir_padded.shape

    output = np.zeros((3, Hp, Wp), dtype=np.float32)
    weight = np.zeros((1, Hp, Wp), dtype=np.float32)

    # Create blending weight (linear ramp at edges)
    blend = np.ones((tile_size, tile_size), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap)
    blend[:overlap, :] *= ramp[:, None]
    blend[-overlap:, :] *= ramp[::-1, None]
    blend[:, :overlap] *= ramp[None, :]
    blend[:, -overlap:] *= ramp[None, ::-1]

    n_tiles = 0
    for y in range(0, Hp - tile_size + 1, stride):
        for x in range(0, Wp - tile_size + 1, stride):
            tile = ir_padded[:, y:y+tile_size, x:x+tile_size]
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = color_model(tile_tensor)
            pred_np = pred.squeeze(0).cpu().numpy()

            output[:, y:y+tile_size, x:x+tile_size] += pred_np * blend[None]
            weight[:, y:y+tile_size, x:x+tile_size] += blend[None]
            n_tiles += 1

    # Normalize by weights
    weight = np.maximum(weight, 1e-8)
    output /= weight

    # Remove padding
    output = output[:, :H, :W]
    return output


@torch.no_grad()
def run_sr_on_thermal(b10, sr_model, device, thermal_mean, thermal_std,
                      tile_size=128, overlap=32, scale_factor=4):
    """
    Apply ESRGAN to the thermal band.
    For simplicity, we downsample then upsample (since B10 is already at 30m grid).
    """
    if sr_model is None:
        return b10

    H, W = b10.shape
    # Normalize thermal
    b10_norm = normalize_thermal(b10, thermal_mean, thermal_std)

    # Process in tiles
    sf = scale_factor
    lr_h, lr_w = H // sf, W // sf

    # Downsample to simulate LR (100m-like)
    import cv2
    b10_lr = cv2.resize(b10_norm, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

    # Run SR on the LR version
    stride = tile_size // sf - overlap // sf
    if stride <= 0:
        stride = tile_size // sf

    # Pad LR image
    pad_h = (stride - (lr_h % stride)) % stride
    pad_w = (stride - (lr_w % stride)) % stride
    b10_lr_pad = np.pad(b10_lr, ((pad_h, 0), (pad_w, 0)), mode='reflect')

    lr_hp, lr_wp = b10_lr_pad.shape
    hr_hp, hr_wp = lr_hp * sf, lr_wp * sf

    output = np.zeros((hr_hp, hr_wp), dtype=np.float32)
    weight = np.zeros((hr_hp, hr_wp), dtype=np.float32)

    lr_tile = tile_size // sf
    for y in range(0, lr_hp - lr_tile + 1, stride):
        for x in range(0, lr_wp - lr_tile + 1, stride):
            tile = b10_lr_pad[y:y+lr_tile, x:x+lr_tile]
            tile_t = torch.from_numpy(
                tile[np.newaxis, np.newaxis].astype(np.float32)).to(device)
            sr_out = sr_model(tile_t).squeeze().cpu().numpy()

            oy, ox = y * sf, x * sf
            oh, ow = sr_out.shape
            output[oy:oy+oh, ox:ox+ow] += sr_out
            weight[oy:oy+oh, ox:ox+ow] += 1.0

    weight = np.maximum(weight, 1e-8)
    output /= weight

    # Crop to original size
    output = output[pad_h*sf:pad_h*sf+H, pad_w*sf:pad_w*sf+W]
    return output


def process_geotiff(input_path, sr_model, color_model, device,
                    thermal_mean, thermal_std, tile_size=128, overlap=32):
    """Process a full GeoTIFF through the pipeline."""

    with rasterio.open(input_path) as ds:
        data = ds.read()         # (7, H, W)
        profile = ds.profile.copy()
        crs = ds.crs
        transform = ds.transform
        H, W = ds.height, ds.width

    print(f"  Input: {H}×{W}, {data.shape[0]} bands")

    # Replace NaN with 0
    valid_mask = np.all(np.isfinite(data), axis=0)
    data = np.nan_to_num(data, nan=0.0).astype(np.float32)

    # Extract bands
    b5  = data[3]   # NIR
    b6  = data[4]   # SWIR1
    b7  = data[5]   # SWIR2
    b10 = data[6]   # Thermal

    # Step 1: SR on B10
    print("  Step 1: Super-resolving B10...")
    t0 = time.time()
    if sr_model is not None:
        b10_sr = run_sr_on_thermal(b10, sr_model, device,
                                    thermal_mean, thermal_std,
                                    tile_size, overlap)
    else:
        b10_sr = normalize_thermal(b10, thermal_mean, thermal_std)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Step 2: Normalize and stack
    print("  Step 2: Stacking IR bands...")
    b5_norm  = normalize_reflectance(b5)
    b6_norm  = normalize_reflectance(b6)
    b7_norm  = normalize_reflectance(b7)
    if sr_model is not None:
        b10_norm = b10_sr   # already normalised inside run_sr_on_thermal
    else:
        b10_norm = b10_sr

    ir_stack = np.stack([b5_norm, b6_norm, b7_norm, b10_norm], axis=0)

    # Step 3: Colorize
    print("  Step 3: Colorizing...")
    t0 = time.time()
    rgb_pred = tile_inference(ir_stack, color_model, device,
                              tile_size=tile_size, overlap=overlap)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Step 4: Denormalize to reflectance
    rgb_reflectance = denormalize_reflectance(rgb_pred)

    # Mask out originally invalid pixels
    for c in range(3):
        rgb_reflectance[c][~valid_mask] = 0.0

    # Convert to uint8 for visualization
    rgb_uint8 = np.clip(rgb_reflectance * 255 / 0.3, 0, 255).astype(np.uint8)

    return rgb_reflectance, rgb_uint8, profile, crs, transform


def main():
    parser = argparse.ArgumentParser(description='IR to RGB Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input Landsat GeoTIFF')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: outputs/<name>_colorized.tif)')
    parser.add_argument('--sr_checkpoint', type=str, default=None)
    parser.add_argument('--colorize_checkpoint', type=str, required=True)
    parser.add_argument('--tile_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--thermal_mean', type=float, default=300.0)
    parser.add_argument('--thermal_std', type=float, default=10.0)
    parser.add_argument('--stats_file', type=str, default=None,
                        help='Path to stats.json from data preparation')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load stats
    t_mean, t_std = args.thermal_mean, args.thermal_std
    if args.stats_file and os.path.exists(args.stats_file):
        with open(args.stats_file) as f:
            stats = json.load(f)
        t_mean = stats.get('ST_B10', {}).get('mean', t_mean)
        t_std  = stats.get('ST_B10', {}).get('std', t_std)
        print(f"Thermal stats from file: mean={t_mean:.2f}, std={t_std:.2f}")

    # Load models
    sr_model, color_model = load_models(
        args.sr_checkpoint, args.colorize_checkpoint, device)

    # Output path
    if args.output is None:
        name = Path(args.input).stem
        output_path = f"outputs/{name}_colorized.tif"
    else:
        output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Process
    print(f"\nProcessing: {args.input}")
    t_start = time.time()

    rgb_refl, rgb_uint8, profile, crs, transform = process_geotiff(
        args.input, sr_model, color_model, device,
        t_mean, t_std, args.tile_size, args.overlap)

    total_time = time.time() - t_start

    # Save GeoTIFF (RGB reflectance)
    profile.update({
        'count': 3,
        'dtype': 'float32',
        'driver': 'GTiff',
        'compress': 'lzw',
    })
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rgb_refl)
    print(f"\n✓ Reflectance GeoTIFF saved: {output_path}")

    # Save PNG preview
    png_path = output_path.replace('.tif', '_preview.png')
    import cv2
    # RGB → BGR for cv2
    preview = np.transpose(rgb_uint8, (1, 2, 0))[:, :, ::-1]
    # Downscale if too large
    max_dim = 2048
    h, w = preview.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        preview = cv2.resize(preview, None, fx=scale, fy=scale)
    cv2.imwrite(png_path, preview)
    print(f"✓ Preview PNG saved: {png_path}")

    print(f"\nTotal inference time: {total_time:.1f}s")
    print(f"Image size: {rgb_refl.shape[1]}×{rgb_refl.shape[2]}")
    h, w = rgb_refl.shape[1], rgb_refl.shape[2]
    n_tiles = ((h // (args.tile_size - args.overlap)) *
               (w // (args.tile_size - args.overlap)))
    if n_tiles > 0:
        print(f"Per-tile inference time: {total_time/n_tiles*1000:.1f}ms")


if __name__ == '__main__':
    main()
