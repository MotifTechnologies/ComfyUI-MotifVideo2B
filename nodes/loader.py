"""MotifVideo loader nodes.

MotifTextEncoderLoader: CLIPLoader-style node for T5Gemma2 text encoder.
Loads from ComfyUI's text_encoders folder, returns CLIP compatible with
CLIPTextEncode and KSampler.
"""

import os
import logging
import torch

import folder_paths
import comfy.sd
import comfy.supported_models_base as supported_models_base


class MotifTextEncoderLoader:
    """Load MotifVideo T5Gemma2 text encoder.

    Expects a directory (symlinked in models/text_encoders/) containing:
      - model.safetensors (16.4GB)
      - config.json

    Tokenizer directory must also be in models/text_encoders/ containing:
      - tokenizer.json
      - tokenizer_config.json

    Output CLIP is compatible with CLIPTextEncode node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # List directories in text_encoders folder
        te_files = folder_paths.get_filename_list("text_encoders")
        # Also allow directory names (for our symlinked dirs)
        te_dir = folder_paths.get_folder_paths("text_encoders")[0]
        te_dirs = []
        if os.path.exists(te_dir):
            for name in sorted(os.listdir(te_dir)):
                full = os.path.join(te_dir, name)
                if os.path.isdir(full) and os.path.exists(os.path.join(full, "model.safetensors")):
                    te_dirs.append(name)

        return {
            "required": {
                "text_encoder": (te_dirs if te_dirs else ["(no model directories found)"],),
                "tokenizer": (
                    [d for d in sorted(os.listdir(te_dir))
                     if os.path.isdir(os.path.join(te_dir, d)) and
                     os.path.exists(os.path.join(te_dir, d, "tokenizer.json"))]
                    if os.path.exists(te_dir) else ["(no tokenizer directories found)"],
                ),
                "dtype": (["bfloat16", "float16", "float32"],),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "motifvideo"

    DESCRIPTION = "Load MotifVideo T5Gemma2 text encoder.\nOutput CLIP is compatible with CLIPTextEncode."

    def load_text_encoder(self, text_encoder, tokenizer, dtype, device="default"):
        import safetensors.torch
        from ..text_encoders.t5_gemma2 import MotifVideoSD1Tokenizer, te

        te_dir = folder_paths.get_folder_paths("text_encoders")[0]
        text_encoder_path = os.path.join(te_dir, text_encoder)
        tokenizer_path = os.path.join(te_dir, tokenizer)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        # Load state dict
        ckpt_file = os.path.join(text_encoder_path, "model.safetensors")
        logging.info("[MotifVideo] Loading text encoder from %s", ckpt_file)
        state_dict = safetensors.torch.load_file(ckpt_file, device="cpu")

        # Config and tokenizer paths
        config_path = os.path.join(text_encoder_path, "config.json")
        model_options = {"motifvideo_config_path": config_path}
        tokenizer_data = {"motifvideo_tokenizer_path": tokenizer_path}

        if device == "cpu":
            model_options["load_device"] = torch.device("cpu")
            model_options["offload_device"] = torch.device("cpu")

        # Build ClipTarget
        target = supported_models_base.ClipTarget(
            MotifVideoSD1Tokenizer, te(dtype_t5gemma2=torch_dtype)
        )
        target.params = {}

        # Param count for memory estimation
        param_count = sum(
            v.numel() for k, v in state_dict.items() if k.startswith("encoder.")
        )

        clip = comfy.sd.CLIP(
            target=target,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            tokenizer_data=tokenizer_data,
            parameters=param_count,
            state_dict=[state_dict],
            model_options=model_options,
        )

        logging.info("[MotifVideo] Text encoder loaded (%d params)", param_count)
        return (clip,)
