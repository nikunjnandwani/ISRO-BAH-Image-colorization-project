"""
Pix2Pix Colorization Training Script
======================================
Trains an enhanced Pix2Pix model (U-Net generator + PatchGAN discriminator)
to translate 4-channel IR imagery to 3-channel RGB.

Loss = λ_L1·L1 + λ_adv·L_adv + λ_perc·L_perceptual + λ_sem·L_semantic

Usage:
    python train_colorize.py --data_dir prepared_data --epochs 200

For Colab:
    !python train_colorize.py --data_dir /content/drive/MyDrive/isro_data/prepared_data \\
                              --checkpoint_dir /content/drive/MyDrive/isro_checkpoints/colorize
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import ColorizationDataset
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.semantic import SemanticConstraint
from models.losses import CombinedColorizeLoss

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── Utilities ──────────────────────────────────────────────────────

def denormalize_reflectance(x, lo=0.0, hi=0.5):
    """Convert [-1,1] back to [lo, hi] reflectance scale."""
    return (x + 1.0) / 2.0 * (hi - lo) + lo


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_stats(prepared_dir):
    """Load thermal normalization statistics."""
    stats_path = Path(prepared_dir) / 'stats.json'
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        t_mean = stats.get('ST_B10', {}).get('mean', 300.0)
        t_std  = stats.get('ST_B10', {}).get('std', 10.0)
        return t_mean, t_std
    return 300.0, 10.0


def save_visualizations(gen, val_dl, device, epoch, output_dir, n_samples=4):
    """Save side-by-side comparison images."""
    gen.eval()
    output_dir = Path(output_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (ir_input, rgb_target, raw_bands) in enumerate(val_dl):
            if batch_idx >= 1:
                break
            ir_input = ir_input.to(device)
            rgb_target = rgb_target.to(device)
            pred_rgb = gen(ir_input)

            n = min(n_samples, ir_input.shape[0])

            fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
            if n == 1:
                axes = axes[np.newaxis, :]

            for i in range(n):
                # IR false-colour (B5, B6, B7 as RGB)
                ir_vis = ir_input[i, :3].cpu().numpy()
                ir_vis = (ir_vis - ir_vis.min()) / (ir_vis.max() - ir_vis.min() + 1e-8)
                ir_vis = np.transpose(ir_vis, (1, 2, 0))

                # Ground truth RGB
                gt = rgb_target[i].cpu().numpy()
                gt = (gt + 1) / 2  # [-1,1] → [0,1]
                gt = np.clip(np.transpose(gt, (1, 2, 0)), 0, 1)

                # Predicted RGB
                pr = pred_rgb[i].cpu().numpy()
                pr = (pr + 1) / 2
                pr = np.clip(np.transpose(pr, (1, 2, 0)), 0, 1)

                # Thermal band (B10)
                thermal = ir_input[i, 3].cpu().numpy()
                thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)

                axes[i, 0].imshow(ir_vis)
                axes[i, 0].set_title('IR False Color (B5,B6,B7)')
                axes[i, 1].imshow(thermal, cmap='inferno')
                axes[i, 1].set_title('Thermal B10')
                axes[i, 2].imshow(gt)
                axes[i, 2].set_title('Ground Truth RGB')
                axes[i, 3].imshow(pr)
                axes[i, 3].set_title('Predicted RGB')

                for ax in axes[i]:
                    ax.axis('off')

            plt.suptitle(f'Epoch {epoch+1}', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f'epoch_{epoch+1:04d}.png', dpi=150,
                        bbox_inches='tight')
            plt.close()


# ── Training Loop ──────────────────────────────────────────────────

def train_one_epoch(gen, disc, semantic, train_dl, opt_g, opt_d,
                    criterion, device, grad_clip=1.0):
    """Train one epoch of Pix2Pix."""
    gen.train()
    disc.train()

    epoch_losses = {
        'l1': 0, 'perceptual': 0, 'adversarial_g': 0,
        'semantic': 0, 'total_g': 0, 'total_d': 0
    }
    n_batches = 0

    for ir_input, rgb_target, raw_bands in tqdm(train_dl, desc='  Train',
                                                 leave=False):
        ir_input   = ir_input.to(device)
        rgb_target = rgb_target.to(device)
        raw_bands  = raw_bands.to(device)

        # ── Generator forward ──
        pred_rgb = gen(ir_input)

        # ── Discriminator update ──
        opt_d.zero_grad()
        disc_real = disc(ir_input, rgb_target)
        disc_fake = disc(ir_input, pred_rgb.detach())
        d_loss, d_dict = criterion.discriminator_loss(disc_real, disc_fake)
        d_loss.backward()
        nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        opt_d.step()

        # ── Generator update ──
        opt_g.zero_grad()
        disc_pred = disc(ir_input, pred_rgb)

        # Semantic loss: denormalize predicted RGB to reflectance scale
        pred_rgb_raw = denormalize_reflectance(pred_rgb)
        sem_loss = semantic(pred_rgb_raw, raw_bands)

        g_loss, g_dict = criterion.generator_loss(
            pred_rgb, rgb_target, disc_pred, semantic_loss=sem_loss)
        g_loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
        opt_g.step()

        # Accumulate losses
        for k, v in g_dict.items():
            if k in epoch_losses:
                epoch_losses[k] += v
        for k, v in d_dict.items():
            if k in epoch_losses:
                epoch_losses[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(gen, val_dl, device):
    """Compute validation L1 loss and PSNR."""
    gen.eval()
    total_l1 = 0
    total_psnr = 0
    n = 0

    for ir_input, rgb_target, _ in val_dl:
        ir_input   = ir_input.to(device)
        rgb_target = rgb_target.to(device)
        pred_rgb = gen(ir_input)

        # L1
        total_l1 += nn.functional.l1_loss(pred_rgb, rgb_target).item() * ir_input.shape[0]

        # PSNR
        pred_01 = (pred_rgb + 1) / 2
        tgt_01  = (rgb_target + 1) / 2
        mse = torch.mean((pred_01 - tgt_01) ** 2, dim=[1, 2, 3])
        psnr = -10 * torch.log10(mse + 1e-8)
        total_psnr += psnr.sum().item()
        n += ir_input.shape[0]

    return total_l1 / max(n, 1), total_psnr / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix Colorization')
    parser.add_argument('--data_dir', type=str, default='prepared_data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/colorize')
    parser.add_argument('--log_dir', type=str, default='logs/colorize')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--lambda_perceptual', type=float, default=10.0)
    parser.add_argument('--lambda_semantic', type=float, default=5.0)
    parser.add_argument('--lambda_adv', type=float, default=1.0)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--lr_decay_start', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--vis_interval', type=int, default=5,
                        help='Save visualizations every N epochs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Thermal stats
    t_mean, t_std = load_stats(args.data_dir)

    # Data
    stats_path = f"{args.data_dir}/stats.json"
    train_ds = ColorizationDataset(f"{args.data_dir}/train", augment=True,
                                    thermal_mean=t_mean, thermal_std=t_std,
                                    stats_path=stats_path)
    val_ds   = ColorizationDataset(f"{args.data_dir}/val", augment=False,
                                    thermal_mean=t_mean, thermal_std=t_std,
                                    stats_path=stats_path)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} tiles ({len(train_dl)} batches)")
    print(f"Val  : {len(val_ds)} tiles ({len(val_dl)} batches)")

    # Models
    gen = UNetGenerator(
        in_channels=4, out_channels=3,
        ngf=args.ngf, n_downsample=7,
        use_dropout=True, dropout_rate=0.5
    ).to(device)

    disc = PatchGANDiscriminator(
        in_channels=7,   # 4 IR + 3 RGB
        ndf=args.ndf,
        n_layers=3
    ).to(device)

    semantic = SemanticConstraint().to(device)

    print(f"Generator params     : {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params : {sum(p.numel() for p in disc.parameters()):,}")

    # Losses
    criterion = CombinedColorizeLoss(
        lambda_l1=args.lambda_l1,
        lambda_perceptual=args.lambda_perceptual,
        lambda_semantic=args.lambda_semantic,
        lambda_adv=args.lambda_adv,
    ).to(device)

    # Optimisers
    opt_g = optim.Adam(gen.parameters(), lr=args.lr_g,
                       betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=args.lr_d,
                       betas=(0.5, 0.999))

    # Linear LR decay after decay_start epoch
    def lr_lambda(epoch):
        if epoch < args.lr_decay_start:
            return 1.0
        return 1.0 - (epoch - args.lr_decay_start) / (args.epochs - args.lr_decay_start + 1e-8)

    sched_g = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)

    # Resume
    start_epoch = 0
    best_psnr = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt['generator'])
        disc.load_state_dict(ckpt['discriminator'])
        if 'optimizer_g' in ckpt:
            opt_g.load_state_dict(ckpt['optimizer_g'])
        if 'optimizer_d' in ckpt:
            opt_d.load_state_dict(ckpt['optimizer_d'])
        start_epoch = ckpt.get('epoch', 0)
        best_psnr = ckpt.get('best_psnr', 0)
        print(f"Resumed from epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    # TensorBoard
    writer = SummaryWriter(args.log_dir) if HAS_TB else None

    # ── Training Loop ──
    print(f"\n{'='*60}")
    print(f"Training Pix2Pix Colorization ({args.epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        losses = train_one_epoch(
            gen, disc, semantic, train_dl, opt_g, opt_d,
            criterion, device)

        val_l1, val_psnr = validate(gen, val_dl, device)

        sched_g.step()
        sched_d.step()

        dt = time.time() - t0
        lr_current = opt_g.param_groups[0]['lr']

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"G={losses['total_g']:.3f} D={losses['total_d']:.3f} | "
              f"L1={losses['l1']:.4f} Sem={losses['semantic']:.4f} | "
              f"Val PSNR={val_psnr:.2f} | LR={lr_current:.6f} | "
              f"{dt:.1f}s")

        if writer:
            for k, v in losses.items():
                writer.add_scalar(f'train/{k}', v, epoch)
            writer.add_scalar('val/l1', val_l1, epoch)
            writer.add_scalar('val/psnr', val_psnr, epoch)
            writer.add_scalar('lr', lr_current, epoch)

        # Visualizations
        if (epoch + 1) % args.vis_interval == 0:
            save_visualizations(gen, val_dl, device, epoch, args.output_dir)

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint({
                'epoch': epoch + 1,
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'optimizer_g': opt_g.state_dict(),
                'optimizer_d': opt_d.state_dict(),
                'best_psnr': best_psnr,
            }, f"{args.checkpoint_dir}/colorize_best.pth")
            print(f"    ✓ New best PSNR: {best_psnr:.2f}")

        # Periodic checkpoints
        if (epoch + 1) % 20 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'optimizer_g': opt_g.state_dict(),
                'optimizer_d': opt_d.state_dict(),
                'best_psnr': best_psnr,
            }, f"{args.checkpoint_dir}/colorize_epoch{epoch+1}.pth")

    # Final save
    save_checkpoint({
        'generator': gen.state_dict(),
        'discriminator': disc.state_dict(),
    }, f"{args.checkpoint_dir}/colorize_final.pth")

    if writer:
        writer.close()

    print(f"\n{'='*60}")
    print(f"Colorization training complete!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Models saved to: {args.checkpoint_dir}/")


if __name__ == '__main__':
    main()
