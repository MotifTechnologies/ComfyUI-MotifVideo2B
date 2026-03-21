"""MotifVideo loader nodes."""

import logging


class MotifTextEncoderLoader:
    """Load MotifVideo 1.9B text encoder (T5Gemma2) into ComfyUI CLIP system.

    Loads the encoder-half of T5Gemma2Model from a safetensors checkpoint
    and wraps it in comfy.sd.CLIP so ComfyUI memory management and offloading
    work automatically.

    Output CLIP is compatible with the standard CLIPTextEncode node.

    Args:
        text_encoder_path: Directory that contains model.safetensors and
            config.json (the HuggingFace checkpoint layout).
        tokenizer_path: Directory that contains tokenizer.json and
            tokenizer_config.json.
        dtype: Compute dtype.  bfloat16 recommended for 16.4 GB model.
    """

    DTYPE_MAP = {
        "bfloat16": "bfloat16",
        "float16": "float16",
        "float32": "float32",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder_path": (
                    "STRING",
                    {
                        "default": (
                            "/lustrefs/team-multimodal/checkpoints/"
                            "base_checkpoint/model/text_encoder"
                        )
                    },
                ),
                "tokenizer_path": (
                    "STRING",
                    {
                        "default": (
                            "/lustrefs/team-multimodal/checkpoints/"
                            "base_checkpoint/model/tokenizer"
                        )
                    },
                ),
                "dtype": (["bfloat16", "float16", "float32"],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "motifvideo"

    def load_text_encoder(self, text_encoder_path, tokenizer_path, dtype):
        import os
        import torch
        import safetensors.torch
        import comfy.sd
        import comfy.supported_models_base as supported_models_base

        from ..text_encoders.t5_gemma2 import MotifVideoSD1Tokenizer, te

        # --- dtype ---
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        # --- state dict ---
        ckpt_file = os.path.join(text_encoder_path, "model.safetensors")
        logging.info("[MotifVideo] Loading text encoder from %s", ckpt_file)
        state_dict = safetensors.torch.load_file(ckpt_file, device="cpu")

        # --- config path → pass through model_options ---
        config_path = os.path.join(text_encoder_path, "config.json")
        model_options = {"motifvideo_config_path": config_path}

        # --- tokenizer_data carries tokenizer path ---
        tokenizer_data = {"motifvideo_tokenizer_path": tokenizer_path}

        # --- build ClipTarget ---
        target = supported_models_base.ClipTarget(MotifVideoSD1Tokenizer, te(dtype_t5gemma2=torch_dtype))
        target.params = {}  # no extra constructor params

        # --- count parameters for memory estimation ---
        # Only encoder half (prefix 'encoder.') matters
        param_count = sum(
            v.numel() for k, v in state_dict.items() if k.startswith("encoder.")
        )
        logging.info("[MotifVideo] text encoder param count: %d", param_count)

        clip = comfy.sd.CLIP(
            target=target,
            embedding_directory=None,
            tokenizer_data=tokenizer_data,
            parameters=param_count,
            state_dict=[state_dict],
            model_options=model_options,
        )

        logging.info("[MotifVideo] Text encoder loaded.")
        return (clip,)
