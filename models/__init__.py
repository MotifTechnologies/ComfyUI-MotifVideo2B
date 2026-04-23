"""ComfyUI model_base wrapper for MotifVideo 1.9B.

MotifVideoModel subclasses comfy.model_base.BaseModel and overrides:
  - concat_cond()  — builds the 17-channel concat condition
                     (16 latent_condition + 1 mask) that is prepended to the
                     16-channel noise to form the 33-channel input.
  - extra_conds()  — passes text encoder outputs and attention mask to the
                     diffusion model via the ComfyUI cond dict.

The actual MotifVideoTransformer3DModel is embedded in models/transformer/
(copied from motif_core with import paths localised).

The transformer is set directly as self.diffusion_model (NOT wrapped in an
adapter nn.Module) so that state_dict keys match the checkpoint without a
spurious 'transformer.' prefix. The ComfyUI calling convention is handled
by monkey-patching the transformer's forward method.
"""

import torch
import comfy.model_base
import comfy.conds
import comfy.ops

from .transformer import MotifVideoTransformer3DModel
from .latent_format import MotifVideoLatent


def _make_comfyui_forward(original_forward):
    """Create a ComfyUI-compatible forward that delegates to the transformer.

    ComfyUI calls: diffusion_model(x, timestep, context=..., control=...,
                                    transformer_options=..., **extra_conds)
    Transformer expects: forward(hidden_states, timestep,
                                  encoder_hidden_states, ...)
    """
    def comfyui_forward(
        x,
        timestep,
        context=None,
        control=None,
        transformer_options=None,
        encoder_attention_mask=None,
        pooled_projections=None,
        image_embeds=None,
        **kwargs,
    ):
        output = original_forward(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            image_embeds=image_embeds,
            return_dict=False,
        )
        return output[0]
    return comfyui_forward


class MotifVideoModel(comfy.model_base.BaseModel):
    """ComfyUI BaseModel wrapper for MotifVideo 1.9B.

    The diffusion_model is the MotifVideoTransformer3DModel directly (no
    adapter wrapper) so that checkpoint state_dict keys match without a
    'transformer.' prefix. The forward method is monkey-patched to translate
    ComfyUI's calling convention.

    in_channels breakdown (33 total):
      16  — noised latent  (xc, managed by BaseModel._apply_model)
      16  — latent_condition  (first-frame latent for I2V, zeros for T2V)
       1  — mask              (1.0 on first frame for I2V, zeros for T2V)
    """

    def __init__(
        self,
        model_config,
        model_type=comfy.model_base.ModelType.FLOW,
        device=None,
    ):
        unet_config = model_config.unet_config

        # fp8 안전장치: BaseModel.__init__ 표준 pick_operations 호출 전에 적용.
        # config.py 의 optimizations["fp8"] = True 는 weight_dtype 이 실제 fp8 계열일
        # 때만 fp8_ops 경로로 내려보낸다. bf16/fp16 weight 에 fp8_ops 가 선택되면
        # 매 Linear 마다 on-the-fly fp8 dispatch 가 붙어 10배 이상 느려진다
        # (bf16 smoke 에서 sage-off 222s/step 의 근본 원인).
        # model_config.optimizations 는 인스턴스 생성 시 .copy() 된 mutable dict.
        if model_config.custom_operations is None:
            _weight_dtype = unet_config.get("dtype", None)
            _fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
            _is_fp8_weight = _weight_dtype in _fp8_types if _weight_dtype is not None else False
            if not _is_fp8_weight:
                # weight 가 fp8 아닌 경우 fp8 최적화 비활성 — BaseModel 이 pick_operations
                # 호출 시 이 값을 읽어 fp8_ops 대신 disable_weight_init 선택.
                model_config.optimizations["fp8"] = False

        # BaseModel 표준 경로: unet_model=MotifVideoTransformer3DModel 을 전달하면
        # BaseModel.__init__ 이 pick_operations + unet_model(**unet_config) + eval() +
        # archive_model_dtypes() 를 순서대로 호출. vbar 시스템과 정상 협력하는 표준 흐름.
        # MotifVideoTransformer3DModel.__init__ 에 **kwargs 를 추가하여 unet_config 의
        # 모르는 키(image_model 등)를 무시.
        super().__init__(
            model_config,
            model_type,
            device=device,
            unet_model=MotifVideoTransformer3DModel,
        )

        # DEBUG: ops class 정보 출력 (BaseModel 이 선택한 operations 는 직접 조회 불가 —
        # pick_operations 결과는 diffusion_model 내부에 bound 됨. dtype 으로 추론).
        _weight_dtype = unet_config.get("dtype", None)
        _fp8_opt = model_config.optimizations.get("fp8", False)
        print(
            f"[MotifVideo DEBUG ops] weight_dtype={_weight_dtype} "
            f"manual_cast_dtype={self.manual_cast_dtype} "
            f"fp8_opt_effective={_fp8_opt} "
            f"diffusion_model={type(self.diffusion_model).__name__}",
            flush=True,
        )
        print(
            f"[MotifVideo] unet_config keys forwarded to transformer: "
            f"{ {k: v for k, v in unet_config.items() if k in ('rope_theta', 'num_decoder_layers', 'num_layers', 'num_single_layers', 'num_attention_heads', 'enable_text_cross_attention_dual', 'enable_text_cross_attention_single')} }",
            flush=True,
        )

        # Monkey-patch forward to translate ComfyUI calling convention.
        # The transformer is set directly as diffusion_model (not wrapped in
        # an adapter nn.Module) so state_dict keys match the checkpoint.
        # super().__init__ 이 self.diffusion_model 을 이미 생성하고 eval() 을 완료한 뒤에 적용.
        original_forward = self.diffusion_model.forward
        self.diffusion_model.forward = _make_comfyui_forward(original_forward)

    # ------------------------------------------------------------------
    # concat_cond: build the 17-channel prepend condition
    # ------------------------------------------------------------------

    def concat_cond(self, **kwargs):
        """Return [latent_condition (16ch), mask (1ch)] concatenated.

        For T2V  : both tensors are zeros.
        For I2V  : latent_condition = process_latent_in(first_frame_latent),
                   mask = 1.0 on the first temporal slice, 0.0 elsewhere.
        """
        noise = kwargs.get("noise", None)
        if noise is None:
            return None

        device = kwargs["device"]
        dtype = noise.dtype

        # noise shape: [B, 16, T, H, W]
        B, _, T, H, W = noise.shape

        # --- latent_condition ---
        latent_condition = torch.zeros(B, 16, T, H, W, dtype=dtype, device=device)

        # --- mask ---
        latent_mask = torch.zeros(B, 1, T, H, W, dtype=dtype, device=device)

        # I2V: use concat_latent_image for first-frame conditioning
        image = kwargs.get("concat_latent_image", None)
        if image is not None:
            # image may be [B, 16, 1, H, W] or [B, 16, H, W] — normalise shape
            image = self.process_latent_in(image)
            if image.ndim == 4:
                # [B, 16, H, W] → [B, 16, 1, H, W]
                image = image.unsqueeze(2)
            # Pad / crop temporal dim to match noise
            if image.shape[2] < T:
                pad = torch.zeros(B, 16, T - image.shape[2], H, W, dtype=dtype, device=device)
                image = torch.cat([image, pad], dim=2)
            else:
                image = image[:, :, :T]

            latent_condition = image
            # First temporal slice → mask = 1.0
            latent_mask[:, :, 0] = 1.0

        return torch.cat([latent_condition, latent_mask], dim=1)  # [B, 17, T, H, W]

    # ------------------------------------------------------------------
    # extra_conds: pass text encoder outputs into the diffusion model
    # ------------------------------------------------------------------

    def extra_conds(self, **kwargs):
        """Extend the base extra_conds with MotifVideo-specific conditioning.

        Keys added to the cond dict:
          c_crossattn            — text encoder hidden states [B, E, D] (padding trimmed in
                                   MotifVideoT5Gemma2Model.encode — no mask needed downstream)
          pooled_projections     — pooled text embedding [B, D_pool] (optional)
          image_embeds           — vision encoder output [B, N, D] (I2V, optional)

        attention_mask 는 의도적으로 cond dict 에 넣지 않는다. text_encoder 가 padded
        뒷부분을 미리 잘라 valid text token 만 넘기므로 transformer 가 mask 없이
        올바르게 cross-attention 수행. mask=None 경로로 PyTorch SDPA 가 cuDNN/Flash
        백엔드를 자동 선택해 최고 속도. (Confluence H200 벤치 19.5s/step 경로)
        """
        out = super().extra_conds(**kwargs)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        pooled_projections = kwargs.get("pooled_projections", None)
        if pooled_projections is not None:
            out["pooled_projections"] = comfy.conds.CONDRegular(pooled_projections)

        image_embeds = kwargs.get("image_embeds", None)
        if image_embeds is not None:
            out["image_embeds"] = comfy.conds.CONDRegular(image_embeds)

        return out


__all__ = ["MotifVideoModel", "MotifVideoLatent", "_make_comfyui_forward"]
