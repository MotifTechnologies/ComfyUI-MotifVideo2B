"""MotifVideo 1.9B ComfyUI model config.

Extends supported_models_base.BASE so ComfyUI can:
  - detect the model via unet_config["image_model"] == "motif_video"
  - apply the correct latent format (16-channel, 8x spatial, 4x temporal)
  - instantiate the diffusion model wrapper (MotifVideoModel)
  - apply flow-matching sampling with shift = 2.5

unet_config values are validated against:
  /lustrefs/team-multimodal/checkpoints/base_checkpoint/model/transformer/config.json

sampling_settings["shift"] is validated against:
  /lustrefs/team-multimodal/checkpoints/base_checkpoint/model/scheduler/scheduler_config.json
  ("shift": 2.5, "use_dynamic_shifting": false)
"""

import torch
import comfy.supported_models_base as supported_models_base
import comfy.model_base

from .models import MotifVideoModel
from .models.latent_format import MotifVideoLatent


class MotifVideo19B(supported_models_base.BASE):
    unet_config = {
        # Marker key — used by our loader and matches() for explicit detection.
        "image_model": "motif_video",
        # Values from transformer/config.json:
        "in_channels": 33,
        "out_channels": 16,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "num_layers": 12,
        "num_single_layers": 24,
        "num_decoder_layers": 8,
        "text_embed_dim": 2560,
        "image_embed_dim": 1152,
        "patch_size": 2,
        "patch_size_t": 1,
        "rope_axes_dim": (16, 56, 56),
    }

    unet_extra_config = {}

    sampling_settings = {
        # FlowMatchEulerDiscreteScheduler: "shift": 2.5, "use_dynamic_shifting": false
        "shift": 2.5,
    }

    latent_format = MotifVideoLatent
    memory_usage_factor = 1.0

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    # FP8 quantisation is supported — UNETLoader weight_dtype selector applies.
    optimizations = {"fp8": True}

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.FLOW

    def get_model(self, state_dict, prefix="", device=None):
        return MotifVideoModel(self, device=device)

    def clip_target(self, state_dict={}):
        # TODO: checklist item 4 — T5Gemma2 text encoder integration.
        # The loader node handles text encoder loading independently for now.
        return None
