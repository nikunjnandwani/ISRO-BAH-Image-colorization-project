"""
Evaluation Script — PSNR / SSIM / FID / Inference Time
=======================================================
Compares generated RGB images against ground truth on the test set.

Metrics:
    • PSNR  — Peak Signal-to-Noise Ratio (higher = better)
    • SSIM  — Structural Similarity Index (higher = better)
    • FID   — Fréchet Inception Distance (lower = better)
    • Time  — Per-tile inference latency

Usage:
    python evaluate.py --data_dir prepared_data/test \\
                       --checkpoint checkpoints/colorize/colorize_best.pth
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import ColorizationDataset
from models.generator import UNetGenerator

try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── PSNR ───────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 max_val: float = 1.0) -> float:
    """
    Compute PSNR between predicted and target images.
    Inputs should be in [0, 1] range.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


# ── SSIM ───────────────────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5,
                     channels: int = 3, device='cpu'):
    """Create a Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, channels: int = 3) -> float:
    """
    Compute SSIM between predicted and target images.
    Inputs: (B, C, H, W) in [0, 1].
    """
    device = pred.device
    kernel = _gaussian_kernel(window_size, 1.5, channels, device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12   = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# ── FID ────────────────────────────────────────────────────────────

class InceptionFeatureExtractor:
    """Extract features from InceptionV3 for FID computation."""

    def __init__(self, device='cpu'):
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.device = device
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()  # Remove classification head
        self.model.eval().to(device)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract InceptionV3 features.
        images: (B, 3, H, W) in [0, 1]
        """
        # Resize to 299×299 (Inception input size)
        x = F.interpolate(images, size=(299, 299), mode='bilinear',
                          align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.model(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        return feat.cpu().numpy()


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Compute FID from precomputed InceptionV3 features.

    FID = ||mu_r - mu_f||^2 + Tr(Σ_r + Σ_f - 2*(Σ_r·Σ_f)^0.5)
    """
    from scipy import linalg

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    diff_sq = diff @ diff

    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_sq + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


# ── Main Evaluation ────────────────────────────────────────────────

def load_stats(prepared_dir):
    stats_path = Path(prepared_dir).parent / 'stats.json'
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        return (stats.get('ST_B10', {}).get('mean', 300.0),
                stats.get('ST_B10', {}).get('std', 10.0))
    return 300.0, 10.0


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # Load thermal stats
    t_mean, t_std = load_stats(args.data_dir)

    # Dataset
    stats_path = str(Path(args.data_dir).parent / 'stats.json')
    test_ds = ColorizationDataset(
        args.data_dir, augment=False,
        thermal_mean=t_mean, thermal_std=t_std,
        stats_path=stats_path)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)
    print(f"Test set: {len(test_ds)} tiles")

    # Model
    gen = UNetGenerator(
        in_channels=4, out_channels=3,
        ngf=args.ngf, n_downsample=7,
        use_dropout=False
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['generator'])
    gen.eval()
    print(f"✓ Loaded model from {args.checkpoint}")

    # ── Evaluate ──
    all_psnr = []
    all_ssim = []
    all_times = []
    real_features = []
    fake_features = []

    # FID extractor (optional, requires more memory)
    inception = None
    if args.compute_fid:
        try:
            inception = InceptionFeatureExtractor(device)
            print("✓ InceptionV3 loaded for FID computation")
        except Exception as e:
            print(f"⚠ Could not load InceptionV3: {e}")
            print("  FID computation skipped")

    for batch_idx, (ir_input, rgb_target, _) in enumerate(
            tqdm(test_dl, desc='Evaluating')):
        ir_input = ir_input.to(device)
        rgb_target = rgb_target.to(device)

        # Inference with timing
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        pred_rgb = gen(ir_input)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        dt = time.time() - t0
        all_times.append(dt / ir_input.shape[0])

        # Convert to [0, 1] for metrics
        pred_01 = (pred_rgb + 1) / 2
        tgt_01  = (rgb_target + 1) / 2

        # Per-sample PSNR
        for i in range(ir_input.shape[0]):
            psnr = compute_psnr(pred_01[i:i+1], tgt_01[i:i+1])
            all_psnr.append(psnr)

        # Batch SSIM
        ssim = compute_ssim(pred_01, tgt_01)
        all_ssim.append(ssim)

        # FID features
        if inception:
            real_features.append(inception.extract(tgt_01))
            fake_features.append(inception.extract(pred_01))

    # ── Aggregate results ──
    results = {
        'psnr_mean': float(np.mean(all_psnr)),
        'psnr_std': float(np.std(all_psnr)),
        'ssim_mean': float(np.mean(all_ssim)),
        'ssim_std': float(np.std(all_ssim)),
        'inference_time_ms': float(np.mean(all_times) * 1000),
        'num_samples': len(all_psnr),
    }

    if inception and real_features:
        real_feat = np.concatenate(real_features, axis=0)
        fake_feat = np.concatenate(fake_features, axis=0)
        fid = compute_fid(real_feat, fake_feat)
        results['fid'] = fid

    # ── Print Results ──
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({results['num_samples']} test tiles)")
    print(f"{'='*60}")
    print(f"  PSNR  : {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  SSIM  : {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    if 'fid' in results:
        print(f"  FID   : {results['fid']:.2f}")
    print(f"  Time  : {results['inference_time_ms']:.1f} ms/tile")
    print(f"{'='*60}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_dir / 'evaluation_results.json'}")

    # ── Visualizations ──
    if args.save_vis:
        save_comparison_grid(gen, test_dl, device, output_dir, n_samples=args.n_vis)

    return results


def save_comparison_grid(gen, test_dl, device, output_dir, n_samples=20):
    """Save visual comparison grid."""
    gen.eval()
    output_dir = Path(output_dir) / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    count = 0
    for ir_input, rgb_target, raw_bands in test_dl:
        ir_input = ir_input.to(device)
        rgb_target = rgb_target.to(device)
        pred_rgb = gen(ir_input)

        for i in range(ir_input.shape[0]):
            if count >= n_samples:
                return

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # IR false-colour
            ir_vis = ir_input[i, :3].cpu().numpy()
            ir_vis = (ir_vis - ir_vis.min()) / (ir_vis.max() - ir_vis.min() + 1e-8)
            axes[0].imshow(np.transpose(ir_vis, (1, 2, 0)))
            axes[0].set_title('IR False Color')

            # Thermal
            thermal = ir_input[i, 3].cpu().numpy()
            thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)
            axes[1].imshow(thermal, cmap='inferno')
            axes[1].set_title('Thermal B10')

            # Ground truth
            gt = (rgb_target[i].cpu().numpy() + 1) / 2
            gt = np.clip(np.transpose(gt, (1, 2, 0)), 0, 1)
            axes[2].imshow(gt)
            axes[2].set_title('Ground Truth RGB')

            # Predicted
            pr = (pred_rgb[i].cpu().numpy() + 1) / 2
            pr = np.clip(np.transpose(pr, (1, 2, 0)), 0, 1)
            axes[3].imshow(pr)
            axes[3].set_title('Predicted RGB')

            for ax in axes:
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(output_dir / f'comparison_{count:03d}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            count += 1

    print(f"  ✓ Saved {count} comparison images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate colorization model')
    parser.add_argument('--data_dir', type=str, default='prepared_data/test')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--compute_fid', action='store_true', default=True)
    parser.add_argument('--save_vis', action='store_true', default=True)
    parser.add_argument('--n_vis', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
