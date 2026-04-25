"""MotifVideo loader nodes."""

import logging
import torch

import folder_paths
import comfy.sd
import comfy.supported_models_base as supported_models_base


class MotifTextEncoderLoader:
    """Load MotifVideo T5Gemma2 text encoder (CLIPLoader style).

    Select the `motifvideo_t5gemma2.safetensors` file from the `text_encoders` dropdown.
    Config values and tokenizer are bundled with the node.

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
        "Select the `motifvideo_t5gemma2.safetensors` file from the `text_encoders` dropdown.\n"
        "Config values and tokenizer are bundled with the node.\n"
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

        logging.info("[MotifVideo] Loading text encoder: %s", clip_path)

        state_dict = safetensors.torch.load_file(clip_path, device="cpu")

        model_options = {}
        tokenizer_data = {}

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
