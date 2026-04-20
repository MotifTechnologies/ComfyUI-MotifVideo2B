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
    # Only the marker key is used for matches(). Architecture params are detected
    # dynamically from the checkpoint and stored in unet_config at load time.
    # This avoids match failures when the model architecture evolves.
    unet_config = {
        "image_model": "motif_video",
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

    # FP8 quantisation is intentionally disabled. MotifVideoTransformer3DModel
    # uses vanilla torch.nn.Conv3d / nn.Linear, so ComfyUI's fp8-aware
    # manual_cast ops never wrap this model and fp8-stored weights end up
    # mismatched with fp32 biases at Conv3d forward
    # (RuntimeError: Input type (BFloat16) and bias type (float) should be the same).
    # Proper fp8 support would require rewriting the transformer layers on top
    # of comfy.ops.manual_cast / fp8_ops — tracked as a follow-up issue.
    optimizations = {"fp8": False}

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.FLOW

    def get_model(self, state_dict, prefix="", device=None):
        return MotifVideoModel(self, device=device)

    def clip_target(self, state_dict={}):
        from .text_encoders.t5_gemma2 import MotifVideoSD1Tokenizer, te
        return supported_models_base.ClipTarget(MotifVideoSD1Tokenizer, te())
