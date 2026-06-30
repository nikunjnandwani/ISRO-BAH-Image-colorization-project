"""
Loss Functions for IR Colorization Pipeline
=============================================
Implements all loss components for both ESRGAN and Pix2Pix training:

  • PixelLoss         — L1 reconstruction
  • PerceptualLoss    — VGG-16 feature matching
  • GANLoss           — Adversarial loss (vanilla / LSGAN)
  • CombinedSRLoss    — For ESRGAN (pixel + perceptual + adversarial)
  • CombinedColorLoss — For Pix2Pix (L1 + perceptual + adversarial + semantic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Pixel Loss ─────────────────────────────────────────────────────

class PixelLoss(nn.Module):
    """L1 pixel-wise loss."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


# ── Perceptual (VGG) Loss ──────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """
    Extract features from VGG-16 for perceptual loss computation.
    Uses relu3_3 layer by default (layer index 16 in VGG-16 features).

    The network expects 3-channel input; for 1-channel SR data
    we repeat the channel 3 times.
    """

    def __init__(self, layer_idx: int = 16, use_input_norm: bool = True):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features[:layer_idx + 1]))

        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False

        self.use_input_norm = use_input_norm
        if use_input_norm:
            # ImageNet normalisation
            self.register_buffer(
                'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer(
                'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Handle single-channel input (for ESRGAN)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Rescale from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0

        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        return self.features(x)


class PerceptualLoss(nn.Module):
    """Perceptual loss via VGG feature matching."""

    def __init__(self, layer_idx: int = 16):
        super().__init__()
        self.vgg = VGGFeatureExtractor(layer_idx=layer_idx)

    def forward(self, pred, target):
        pred_feat   = self.vgg(pred)
        target_feat = self.vgg(target)
        return F.l1_loss(pred_feat, target_feat)


# ── GAN Loss ───────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    GAN loss supporting vanilla BCE and LSGAN (MSE) modes.

    Parameters
    ----------
    mode : 'vanilla' (BCE with logits) or 'lsgan' (MSE)
    """

    def __init__(self, mode: str = 'vanilla'):
        super().__init__()
        self.mode = mode
        if mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}")

    def forward(self, pred, is_real: bool):
        if is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return self.loss(pred, target)


# ── Combined SR Loss ───────────────────────────────────────────────

class CombinedSRLoss(nn.Module):
    """
    Combined loss for ESRGAN training.

    Phase 1 (pre-train):  pixel only
    Phase 2 (GAN):        pixel + perceptual + adversarial
    """

    def __init__(self, lambda_pixel: float = 1.0,
                 lambda_perceptual: float = 1.0,
                 lambda_adv: float = 0.005,
                 gan_mode: str = 'vanilla'):
        super().__init__()
        self.pixel_loss = PixelLoss()
        self.perceptual_loss = PerceptualLoss()
        self.gan_loss = GANLoss(gan_mode)

        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv

    def forward(self, pred, target, disc_pred=None, use_gan: bool = False):
        """
        Returns
        -------
        total_loss : scalar
        loss_dict  : dict of individual losses for logging
        """
        l_pixel = self.pixel_loss(pred, target)
        loss_dict = {'pixel': l_pixel.item()}
        total = self.lambda_pixel * l_pixel

        if use_gan:
            l_percep = self.perceptual_loss(pred, target)
            loss_dict['perceptual'] = l_percep.item()
            total = total + self.lambda_perceptual * l_percep

            if disc_pred is not None:
                l_adv = self.gan_loss(disc_pred, is_real=True)
                loss_dict['adversarial'] = l_adv.item()
                total = total + self.lambda_adv * l_adv

        loss_dict['total'] = total.item()
        return total, loss_dict


# ── Combined Colorization Loss ─────────────────────────────────────

class CombinedColorizeLoss(nn.Module):
    """
    Combined loss for Pix2Pix colorization.

    Total = λ_L1 · L1 + λ_adv · L_adv + λ_perc · L_perceptual + λ_sem · L_semantic
    """

    def __init__(self, lambda_l1: float = 100.0,
                 lambda_perceptual: float = 10.0,
                 lambda_semantic: float = 5.0,
                 lambda_adv: float = 1.0,
                 gan_mode: str = 'vanilla'):
        super().__init__()
        self.pixel_loss = PixelLoss()
        self.perceptual_loss = PerceptualLoss()
        self.gan_loss = GANLoss(gan_mode)

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_semantic = lambda_semantic
        self.lambda_adv = lambda_adv

    def generator_loss(self, pred_rgb, target_rgb, disc_pred,
                       semantic_loss: torch.Tensor | None = None):
        """
        Compute total generator loss.

        Parameters
        ----------
        pred_rgb    : (B, 3, H, W) generated RGB in [-1, 1]
        target_rgb  : (B, 3, H, W) real RGB in [-1, 1]
        disc_pred   : discriminator output on fake image
        semantic_loss : precomputed semantic consistency loss (scalar)

        Returns
        -------
        total_loss  : scalar
        loss_dict   : dict for logging
        """
        l_l1     = self.pixel_loss(pred_rgb, target_rgb)
        l_percep = self.perceptual_loss(pred_rgb, target_rgb)
        l_adv    = self.gan_loss(disc_pred, is_real=True)

        total = (self.lambda_l1 * l_l1
                 + self.lambda_perceptual * l_percep
                 + self.lambda_adv * l_adv)

        loss_dict = {
            'l1': l_l1.item(),
            'perceptual': l_percep.item(),
            'adversarial_g': l_adv.item(),
        }

        if semantic_loss is not None:
            total = total + self.lambda_semantic * semantic_loss
            loss_dict['semantic'] = semantic_loss.item()

        loss_dict['total_g'] = total.item()
        return total, loss_dict

    def discriminator_loss(self, disc_real, disc_fake):
        """
        Compute discriminator loss.

        Parameters
        ----------
        disc_real : discriminator output on real images
        disc_fake : discriminator output on fake images (detached)

        Returns
        -------
        total_loss : scalar
        loss_dict  : dict for logging
        """
        l_real = self.gan_loss(disc_real, is_real=True)
        l_fake = self.gan_loss(disc_fake, is_real=False)
        total = (l_real + l_fake) * 0.5

        loss_dict = {
            'disc_real': l_real.item(),
            'disc_fake': l_fake.item(),
            'total_d': total.item(),
        }
        return total, loss_dict


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    # Test SR loss
    sr_loss = CombinedSRLoss()
    pred = torch.randn(2, 1, 128, 128)
    tgt  = torch.randn(2, 1, 128, 128)
    loss, d = sr_loss(pred, tgt, use_gan=False)
    print(f"SR Loss (pretrain): {d}")

    # Test colorization loss
    col_loss = CombinedColorizeLoss()
    pred_rgb = torch.randn(2, 3, 128, 128)
    tgt_rgb  = torch.randn(2, 3, 128, 128)
    disc_out = torch.randn(2, 1, 14, 14)
    sem_loss = torch.tensor(0.5)
    loss_g, dg = col_loss.generator_loss(pred_rgb, tgt_rgb, disc_out, sem_loss)
    print(f"Generator Loss: {dg}")

    disc_real = torch.randn(2, 1, 14, 14)
    disc_fake = torch.randn(2, 1, 14, 14)
    loss_d, dd = col_loss.discriminator_loss(disc_real, disc_fake)
    print(f"Discriminator Loss: {dd}")
