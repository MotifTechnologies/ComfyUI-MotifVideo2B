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
    # ComfyUI BaseModel.memory_required = area * dtype_size * 0.01 * factor * MB.
    # area = B * T * H * W = 1 * 121 * 92 * 160 = 1,781,120 for 720p/121f.
    # factor=4.0 would predict activation peak ~139 GB (97% of an H200's 143 GB),
    # which pushes ComfyUI into "OOM risk → force staged" territory and ends up
    # being counterproductive. The factor values used by HunyuanVideo/CosmosT2V
    # are tuned for much smaller T (e.g. 33 frames); applying them here without
    # accounting for the 3.7x frame-count difference over-estimates activation.
    # Measured: with --highvram full-load, activation peak ~18 GB, which gives
    # factor = 18 GB * 1024 / (area * 2 * 0.01 * MB) ~= 0.52. The original 1.0
    # is closer to measurement than 4.0.
    # (Independent of this factor, NORMAL_VRAM + DynamicVRAM tends to keep the
    # model on the staged path anyway; avoiding --highvram would need a
    # separate approach — future work.)
    memory_usage_factor = 1.0

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    # FP8 quantisation enabled. Phase 1 (#17) wrapped all directly-created
    # layers with comfy.ops (manual_cast / fp8_ops); Phase 2 (#18) replaced
    # the diffusers `Attention` inside each block with a comfy.ops-based
    # MotifVideoAttention (to_q / to_k / to_v / to_out / QKNorm all ops-
    # managed). With both phases shipped, `pick_operations(..., fp8_
    # optimizations=True, ...)` routes every weight through fp8_ops at load
    # time and the forward path stays numerically correct via manual_cast.
    optimizations = {"fp8": True}

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.FLOW

    def get_model(self, state_dict, prefix="", device=None):
        return MotifVideoModel(self, device=device)

    def clip_target(self, state_dict={}):
        from .text_encoders.t5_gemma2 import MotifVideoSD1Tokenizer, te
        return supported_models_base.ClipTarget(MotifVideoSD1Tokenizer, te())
