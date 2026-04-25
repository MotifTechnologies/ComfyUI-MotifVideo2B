"""MotifVideo latent format definition.

Uses the same VAE statistics as Wan2.1 (AutoencoderKLWan):
  - 16-channel latents
  - spatial downscale 8x, temporal downscale 4x
  - latents_mean / latents_std from
    /lustrefs/team-multimodal/checkpoints/base_checkpoint/model/vae/config.json
"""

import torch
import comfy.latent_formats


class MotifVideoLatent(comfy.latent_formats.LatentFormat):
    """16-channel video latent format for MotifVideo 1.9B.

    Inherits process_in / process_out from comfy.latent_formats.Wan21 logic
    but is defined independently so future VAE changes are localised here.
    """

    latent_channels = 16
    latent_dimensions = 3  # temporal video (B, C, T, H, W)
    spacial_downscale_ratio = 8

    # Visualisation factors (same as Wan21 — same VAE architecture)
    latent_rgb_factors = [
        [-0.1299, -0.1692,  0.2932],
        [ 0.0671,  0.0406,  0.0442],
        [ 0.3568,  0.2548,  0.1747],
        [ 0.0372,  0.2344,  0.1420],
        [ 0.0313,  0.0189, -0.0328],
        [ 0.0296, -0.0956, -0.0665],
        [-0.3477, -0.4059, -0.2925],
        [ 0.0166,  0.1902,  0.1975],
        [-0.0412,  0.0267, -0.1364],
        [-0.1293,  0.0740,  0.1636],
        [ 0.0680,  0.3019,  0.1128],
        [ 0.0032,  0.0581,  0.0639],
        [-0.1251,  0.0927,  0.1699],
        [ 0.0060, -0.0633,  0.0005],
        [ 0.3477,  0.2275,  0.2950],
        [ 0.1984,  0.0913,  0.1861],
    ]
    latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

    def __init__(self):
        self.scale_factor = 1.0

        # Values from vae/config.json (mean_of_means / std_of_means)
        self.latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
             0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
        ]).view(1, self.latent_channels, 1, 1, 1)

        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]).view(1, self.latent_channels, 1, 1, 1)

    def process_in(self, latent):
        """Normalise VAE latent into model input space."""
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - mean) * self.scale_factor / std

    def process_out(self, latent):
        """Denormalise model output back to VAE latent space."""
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return latent * std / self.scale_factor + mean
