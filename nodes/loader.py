"""MotifVideo loader nodes."""

import os
import logging
import torch

import folder_paths
import comfy.sd
import comfy.supported_models_base as supported_models_base


class MotifTextEncoderLoader:
    """Load MotifVideo T5Gemma2 text encoder (CLIPLoader style).

    Select model.safetensors from models/text_encoders/ dropdown.
    config.json is auto-detected in the same directory.
    Tokenizer is auto-detected as sibling directory (same parent + /tokenizer/).

    Output CLIP is compatible with CLIPTextEncode and KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "dtype": (["bfloat16", "float16", "float32"],),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "motifvideo"

    DESCRIPTION = (
        "Load MotifVideo T5Gemma2 text encoder.\n"
        "Select model.safetensors from text_encoders folder.\n"
        "config.json and tokenizer are auto-detected.\n"
        "Output is compatible with CLIPTextEncode."
    )

    def load_clip(self, clip_name, dtype, device="default"):
        import safetensors.torch
        from ..text_encoders.t5_gemma2 import MotifVideoSD1Tokenizer, te

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)

        # Auto-detect: if clip_name is inside a directory, look for config.json there
        clip_dir = os.path.dirname(clip_path)
        config_path = os.path.join(clip_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.json not found in {clip_dir}. "
                "Put model.safetensors and config.json in the same directory."
            )

        # Auto-detect tokenizer: look for sibling 'tokenizer' dir or parent/tokenizer
        tokenizer_path = None
        parent = os.path.dirname(clip_dir)
        for candidate in [
            os.path.join(parent, "tokenizer"),          # sibling: ../tokenizer/
            os.path.join(clip_dir, "tokenizer"),         # child: ./tokenizer/
            os.path.join(clip_dir, "..", "tokenizer"),    # parent sibling
        ]:
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "tokenizer.json")):
                tokenizer_path = os.path.realpath(candidate)
                break

        if tokenizer_path is None:
            raise FileNotFoundError(
                f"tokenizer directory not found near {clip_dir}. "
                "Expected a 'tokenizer/' directory with tokenizer.json next to the text_encoder."
            )

        logging.info("[MotifVideo] Loading text encoder: %s", clip_path)
        logging.info("[MotifVideo] Config: %s", config_path)
        logging.info("[MotifVideo] Tokenizer: %s", tokenizer_path)

        state_dict = safetensors.torch.load_file(clip_path, device="cpu")

        model_options = {}
        tokenizer_data = {"motifvideo_tokenizer_path": tokenizer_path}

        if device == "cpu":
            model_options["load_device"] = torch.device("cpu")
            model_options["offload_device"] = torch.device("cpu")

        target = supported_models_base.ClipTarget(
            MotifVideoSD1Tokenizer, te(dtype_t5gemma2=torch_dtype)
        )
        target.params = {}

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
