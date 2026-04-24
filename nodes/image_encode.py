"""MotifVideo image encode node.

Encodes a single image via VAE and injects it as concat_latent_image
into positive and negative conditioning dicts.

The image is resized (aspect-ratio preserving, center-cropped) to match
the target latent's pixel resolution so that concat_cond downstream
receives shapes that align with the noise latent. Target pixel H/W is
derived as latent H/W * 8 (WAN VAE spatial downscale).

Temporal padding and first-frame mask are handled inside concat_cond()
in models/__init__.py — no extra processing is needed here.
"""

import comfy.utils
import node_helpers


class MotifImageEncode:
    """Encode an image with MotifVideo VAE and inject into conditioning.

    Outputs positive and negative conditioning dicts with
    ``concat_latent_image`` set, ready for KSampler.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "motifvideo"

    def encode(self, positive, negative, vae, image, latent):
        # Match the image to the latent's pixel footprint (VAE 8x downscale).
        latent_h = latent["samples"].shape[-2]
        latent_w = latent["samples"].shape[-1]
        target_h = latent_h * 8
        target_w = latent_w * 8

        # ComfyUI IMAGE is BHWC; common_upscale expects BCHW.
        image = image.movedim(-1, 1)
        image = comfy.utils.common_upscale(
            image, target_w, target_h, "bilinear", "center"
        )
        image = image.movedim(1, -1)

        # Encode only RGB channels; shape: [B, H, W, 3] -> latent
        concat_latent_image = vae.encode(image[:, :, :, :3])

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image}
        )

        return (positive, negative)
