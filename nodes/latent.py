"""Empty MotifVideo latent node.

Creates an empty latent tensor for MotifVideo video generation.
VAE specs: 16 latent channels, 8x spatial downscale, 4x temporal downscale.

Latent shape: (batch, 16, T//4+1, H//8, W//8)
The +1 on temporal dim accounts for the initial frame.
"""

import torch
import comfy.model_management


class EmptyMotifLatent:
    """Create empty latent for MotifVideo video generation.

    Output is compatible with KSampler latent_image input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 736, "min": 64, "max": 8192, "step": 16}),
                "num_frames": ("INT", {"default": 121, "min": 1, "max": 1024}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "motifvideo"

    # VAE downscale factors
    SPATIAL_FACTOR = 8
    TEMPORAL_FACTOR = 4
    LATENT_CHANNELS = 16

    def generate(self, width, height, num_frames, batch_size):
        # WAN VAE requires pixel H/W to be a multiple of 16 — floor-snap here
        # so non-conformant values from saved workflows still encode cleanly.
        width -= width % 16
        height -= height % 16
        # Latent spatial dimensions
        latent_h = height // self.SPATIAL_FACTOR
        latent_w = width // self.SPATIAL_FACTOR

        # Latent temporal dimension: T//4 + 1 (accounts for initial frame)
        latent_t = num_frames // self.TEMPORAL_FACTOR + 1

        latent = torch.zeros(
            [batch_size, self.LATENT_CHANNELS, latent_t, latent_h, latent_w],
            device=comfy.model_management.intermediate_device(),
        )

        return ({"samples": latent},)
