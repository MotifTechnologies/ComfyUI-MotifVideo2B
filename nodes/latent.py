# TODO: Full implementation in checklist item 7.
# Input:  width, height, num_frames, batch_size
# Output: LATENT with shape (B, 16, T//4+1, H//8, W//8)


class EmptyMotifLatent:
    """Placeholder — full implementation in checklist item 7."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "num_frames": ("INT", {"default": 16, "min": 1, "max": 256}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "motifvideo"

    def generate(self, width, height, num_frames, batch_size):
        raise NotImplementedError("EmptyMotifLatent: not yet implemented (checklist item 7)")
