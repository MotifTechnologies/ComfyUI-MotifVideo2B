# TODO: Full implementation in checklist item 5.
# Input:  model_path (transformer), text_encoder_path, vae_path, weight_dtype (fp8 option)
# Output: MODEL, CLIP, VAE
# Notes:
#   - Text encoder is 16.4 GB — confirm ComfyUI offload compatibility
#   - Check load_diffusion_model() compatibility; implement custom loader if needed


class MotifVideoModelLoader:
    """Placeholder — full implementation in checklist item 5."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "text_encoder_path": ("STRING", {"default": ""}),
                "vae_path": ("STRING", {"default": ""}),
                "weight_dtype": (["fp16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_fast"],),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_model"
    CATEGORY = "motifvideo"

    def load_model(self, model_path, text_encoder_path, vae_path, weight_dtype):
        raise NotImplementedError("MotifVideoModelLoader: not yet implemented (checklist item 5)")
