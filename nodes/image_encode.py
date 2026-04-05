"""MotifVideo image encode node.

Encodes a single image via VAE and injects it as concat_latent_image
into positive and negative conditioning dicts.

Temporal padding and first-frame mask are handled inside concat_cond()
in models/__init__.py — no extra processing is needed here.
"""

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
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "motifvideo"

    def encode(self, positive, negative, vae, image):
        # Encode only RGB channels; shape: [B, H, W, 3] -> latent
        concat_latent_image = vae.encode(image[:, :, :, :3])

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image}
        )

        return (positive, negative)
