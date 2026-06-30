"""
ESRGAN Training Script — Super-Resolution for Thermal Band B10
================================================================
Two-phase training:
  Phase 1: Pre-train generator with L1 pixel loss (PSNR-oriented)
  Phase 2: Fine-tune with GAN (L1 + perceptual + adversarial)

Usage:
    python train_sr.py --data_dir prepared_data --epochs_pretrain 50 --epochs_gan 50

For Colab:
    !python train_sr.py --data_dir /content/drive/MyDrive/isro_data/prepared_data \\
                        --checkpoint_dir /content/drive/MyDrive/isro_checkpoints/sr
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SRDataset
from models.esrgan import ESRGANGenerator, ESRGANDiscriminator
from models.losses import CombinedSRLoss, GANLoss

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


def save_checkpoint(state, path):
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved: {path}")


def load_stats(prepared_dir):
    """Load dataset statistics for thermal normalization."""
    stats_path = Path(prepared_dir) / 'stats.json'
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        t_mean = stats.get('ST_B10', {}).get('mean', 300.0)
        t_std  = stats.get('ST_B10', {}).get('std', 10.0)
        print(f"  Thermal stats: mean={t_mean:.2f}, std={t_std:.2f}")
        return t_mean, t_std
    return 300.0, 10.0


def train_one_epoch(gen, disc, train_dl, opt_g, opt_d, criterion, gan_loss,
                    device, use_gan=False, grad_clip=1.0):
    """Train for one epoch, return average losses."""
    gen.train()
    if disc is not None:
        disc.train()

    epoch_losses = {'pixel': 0, 'perceptual': 0, 'adversarial': 0,
                    'total': 0, 'disc': 0}
    n_batches = 0

    for lr, hr in tqdm(train_dl, desc='  Train', leave=False):
        lr = lr.to(device)
        hr = hr.to(device)
        B = lr.shape[0]

        # ── Generator forward ──
        sr = gen(lr)

        if use_gan and disc is not None:
            # ── Discriminator update ──
            opt_d.zero_grad()
            disc_real = disc(hr)
            disc_fake = disc(sr.detach())
            d_loss_real = gan_loss(disc_real, is_real=True)
            d_loss_fake = gan_loss(disc_fake, is_real=False)
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
            opt_d.step()
            epoch_losses['disc'] += d_loss.item()

            # ── Generator update (with GAN) ──
            opt_g.zero_grad()
            disc_pred = disc(sr)
            g_loss, loss_dict = criterion(sr, hr, disc_pred, use_gan=True)
            g_loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
            opt_g.step()
        else:
            # ── Generator update (pretrain, no GAN) ──
            opt_g.zero_grad()
            g_loss, loss_dict = criterion(sr, hr, use_gan=False)
            g_loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), grad_clip)
            opt_g.step()

        for k in epoch_losses:
            if k in loss_dict:
                epoch_losses[k] += loss_dict[k]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(gen, val_dl, criterion, device):
    """Validate and return average pixel loss + PSNR."""
    gen.eval()
    total_psnr = 0
    total_loss = 0
    n = 0

    for lr, hr in val_dl:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = gen(lr)

        loss, _ = criterion(sr, hr, use_gan=False)
        total_loss += loss.item()

        # Compute PSNR (on [-1,1] data, rescale to [0,1])
        sr_01 = (sr + 1) / 2
        hr_01 = (hr + 1) / 2
        mse = torch.mean((sr_01 - hr_01) ** 2, dim=[1, 2, 3])
        psnr = -10 * torch.log10(mse + 1e-8)
        total_psnr += psnr.sum().item()
        n += lr.shape[0]

    return total_loss / max(n, 1), total_psnr / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description='Train ESRGAN for thermal SR')
    parser.add_argument('--data_dir', type=str, default='prepared_data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/sr')
    parser.add_argument('--log_dir', type=str, default='logs/sr')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--num_rrdb', type=int, default=4)
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--epochs_pretrain', type=int, default=50)
    parser.add_argument('--epochs_gan', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_pretrain', type=float, default=2e-4)
    parser.add_argument('--lr_gan', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load thermal normalization stats
    t_mean, t_std = load_stats(args.data_dir)

    # Data
    train_ds = SRDataset(f"{args.data_dir}/train", args.scale_factor,
                         augment=True, thermal_mean=t_mean, thermal_std=t_std)
    val_ds   = SRDataset(f"{args.data_dir}/val", args.scale_factor,
                         augment=False, thermal_mean=t_mean, thermal_std=t_std)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} tiles, Val: {len(val_ds)} tiles")
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # Models
    gen = ESRGANGenerator(
        in_channels=1, out_channels=1,
        nf=args.nf, nb=args.num_rrdb,
        scale_factor=args.scale_factor
    ).to(device)

    disc = ESRGANDiscriminator(in_channels=1, nf=args.nf).to(device)

    print(f"Generator params : {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    # Losses
    criterion = CombinedSRLoss(
        lambda_pixel=1.0, lambda_perceptual=1.0, lambda_adv=0.005
    ).to(device)
    gan_loss_fn = GANLoss(mode='vanilla').to(device)

    # TensorBoard
    writer = SummaryWriter(args.log_dir) if HAS_TB else None

    # ── PHASE 1: Pre-training with L1 ──
    print(f"\n{'='*60}")
    print(f"PHASE 1: Pre-training Generator ({args.epochs_pretrain} epochs)")
    print(f"{'='*60}")

    opt_g = optim.Adam(gen.parameters(), lr=args.lr_pretrain, betas=(0.9, 0.999))
    sched_g = optim.lr_scheduler.StepLR(opt_g, step_size=20, gamma=0.5)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        gen.load_state_dict(ckpt['generator'])
        if 'optimizer_g' in ckpt:
            opt_g.load_state_dict(ckpt['optimizer_g'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    best_psnr = 0
    for epoch in range(start_epoch, args.epochs_pretrain):
        t0 = time.time()
        losses = train_one_epoch(
            gen, None, train_dl, opt_g, None, criterion, None,
            device, use_gan=False)
        val_loss, val_psnr = validate(gen, val_dl, criterion, device)
        sched_g.step()
        dt = time.time() - t0

        print(f"  Epoch {epoch+1:3d}/{args.epochs_pretrain} | "
              f"L1={losses['pixel']:.5f} | "
              f"Val PSNR={val_psnr:.2f} dB | "
              f"Time={dt:.1f}s")

        if writer:
            writer.add_scalar('pretrain/pixel_loss', losses['pixel'], epoch)
            writer.add_scalar('pretrain/val_psnr', val_psnr, epoch)

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint({
                'epoch': epoch + 1,
                'generator': gen.state_dict(),
                'optimizer_g': opt_g.state_dict(),
                'best_psnr': best_psnr,
                'phase': 'pretrain',
            }, f"{args.checkpoint_dir}/sr_pretrain_best.pth")

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'generator': gen.state_dict(),
                'optimizer_g': opt_g.state_dict(),
                'phase': 'pretrain',
            }, f"{args.checkpoint_dir}/sr_pretrain_epoch{epoch+1}.pth")

    # ── PHASE 2: GAN Fine-tuning ──
    print(f"\n{'='*60}")
    print(f"PHASE 2: GAN Fine-tuning ({args.epochs_gan} epochs)")
    print(f"{'='*60}")

    opt_g = optim.Adam(gen.parameters(), lr=args.lr_gan, betas=(0.9, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=args.lr_gan, betas=(0.9, 0.999))

    for epoch in range(args.epochs_gan):
        t0 = time.time()
        losses = train_one_epoch(
            gen, disc, train_dl, opt_g, opt_d, criterion, gan_loss_fn,
            device, use_gan=True)
        val_loss, val_psnr = validate(gen, val_dl, criterion, device)
        dt = time.time() - t0

        print(f"  Epoch {epoch+1:3d}/{args.epochs_gan} | "
              f"G={losses['total']:.4f} D={losses['disc']:.4f} | "
              f"Val PSNR={val_psnr:.2f} dB | Time={dt:.1f}s")

        if writer:
            writer.add_scalar('gan/g_total', losses['total'], epoch)
            writer.add_scalar('gan/d_loss', losses['disc'], epoch)
            writer.add_scalar('gan/val_psnr', val_psnr, epoch)

        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'optimizer_g': opt_g.state_dict(),
                'optimizer_d': opt_d.state_dict(),
                'phase': 'gan',
            }, f"{args.checkpoint_dir}/sr_gan_epoch{epoch+1}.pth")

    # Save final model
    save_checkpoint({
        'generator': gen.state_dict(),
        'phase': 'final',
    }, f"{args.checkpoint_dir}/sr_final.pth")

    if writer:
        writer.close()

    print(f"\n{'='*60}")
    print("ESRGAN training complete!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Final model saved to: {args.checkpoint_dir}/sr_final.pth")


if __name__ == '__main__':
    main()
