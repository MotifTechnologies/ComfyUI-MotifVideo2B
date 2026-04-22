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
    # Video transformer — activation peak 이 weight 보다 훨씬 크므로 image-scale 1.0 은
    # 과소 평가. ComfyUI 가 이 값으로 BaseModel.memory_required 를 계산해 full-load vs
    # staged 결정. 1.0 으로 두면 NORMAL_VRAM 환경에서 "작은 모델" 로 오판하여 Staged
    # partial load 로 빠지고, DynamicVRAM async offload 왕복이 매 step 추가돼 10배 이상
    # 느려진다 (222s/step 관찰). 4.0 = CosmosT2V 수준, 121 frame × 720p + cross-attn 구조
    # 기준 충분한 여유. HunyuanVideo 5.5 는 좀 더 큰 모델이라 5.5, 우리는 2B 로 더 작으므로 4.0.
    memory_usage_factor = 4.0

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
