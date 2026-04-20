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

from .transformer import MotifVideoTransformer3DModel

from .adapter import MotifVideoModelAdapter
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
        # Disable ComfyUI's default UNetModel instantiation — we create the
        # transformer ourselves below.
        unet_config_override = dict(model_config.unet_config)
        unet_config_override["disable_unet_model_creation"] = True

        # Temporarily patch the config so BaseModel.__init__ does not try to
        # instantiate a UNetModel with our custom unet_config.
        original_unet_config = model_config.unet_config
        model_config.unet_config = unet_config_override
        super().__init__(model_config, model_type, device=device)
        model_config.unet_config = original_unet_config  # restore

        # Filter unet_config to only valid MotifVideoTransformer3DModel params.
        _TRANSFORMER_PARAMS = {
            "in_channels", "out_channels", "num_attention_heads", "attention_head_dim",
            "num_layers", "num_single_layers", "num_decoder_layers", "mlp_ratio",
            "patch_size", "patch_size_t", "qk_norm", "norm_type",
            "text_embed_dim", "image_embed_dim", "pooled_projection_dim",
            "rope_theta", "rope_axes_dim", "base_latent_size",
            "enable_text_cross_attention_dual", "enable_text_cross_attention_single",
        }
        transformer_kwargs = {
            k: v for k, v in original_unet_config.items()
            if k in _TRANSFORMER_PARAMS
        }
        print(f"[MotifVideo] transformer_kwargs: { {k: v for k, v in transformer_kwargs.items() if k in ('rope_theta', 'num_decoder_layers', 'num_layers', 'num_single_layers', 'num_attention_heads', 'enable_text_cross_attention_dual', 'enable_text_cross_attention_single')} }")
        transformer = MotifVideoTransformer3DModel(**transformer_kwargs)
        # Cast to bfloat16 to match checkpoint weights — avoids dtype mismatch
        # when ComfyUI force-loads bfloat16 weights but biases stay float32.
        transformer = transformer.to(dtype=torch.bfloat16)

        # Monkey-patch forward to translate ComfyUI calling convention.
        # The transformer is set directly as diffusion_model (not wrapped in
        # an adapter nn.Module) so state_dict keys match the checkpoint.
        original_forward = transformer.forward
        transformer.forward = _make_comfyui_forward(original_forward)

        self.diffusion_model = transformer
        self.diffusion_model.eval()

        # NOTE: channels_last_3d / torch.compile은 __init__ 시점이 아니라
        # load_model_weights() 이후에 적용해야 한다. apply_compile 이 만드는
        # OptimizedModule 은 state_dict key prefix 가 "_orig_mod." 로 바뀌어
        # ComfyUI 가 전달하는 `transformer_blocks.X` 형태 키와 전부 mismatch
        # 되어 load_state_dict 가 unet missing 으로 처리한다. 결과적으로
        # 파라미터가 로드되지 않고 random-init transformer 로 샘플링 →
        # 출력이 노이즈. 아래 load_model_weights 오버라이드에서 처리한다.

    # ------------------------------------------------------------------
    # load_model_weights: apply channels_last_3d + torch.compile
    # AFTER checkpoint state_dict has been loaded
    # ------------------------------------------------------------------

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        # 1) 원본 BaseModel 로직으로 state_dict 로드
        super().load_model_weights(sd, unet_prefix=unet_prefix, assign=assign)

        # 2) weight 로드 완료 후 메모리 레이아웃 + compile 적용 (1회만)
        try:
            import torch._dynamo.eval_frame as _dynamo_eval
            _OptimizedModule = _dynamo_eval.OptimizedModule
        except Exception:
            _OptimizedModule = tuple()  # isinstance check no-op

        if isinstance(self.diffusion_model, _OptimizedModule):
            # 이미 wrap 된 상태면 skip (재호출 방지 + revert 후 stale 경로 방어)
            return self

        # NOTE: apply_compile 호출은 reverted. torch.compile(OptimizedModule) 은
        # ComfyUI ModelPatcher 의 dynamic VRAM offload 가 모델 파라미터를 탐색하는
        # 경로와 구조적으로 충돌하여 "Force pre-loaded 838 weights" 현상과 VRAM
        # 120GB 폭증을 유발했다. autotune/freezing 완화로도 offload 실패 자체는
        # 해결 불가. 속도 회복은 SageAttention 이식(이슈 #16) 에서 처리한다.
        # 04_log.md '2026-04-20 P3.1 최종 revert' 참조.
        from .compile_config import apply_channels_last_3d
        self.diffusion_model = apply_channels_last_3d(self.diffusion_model)
        return self

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
          c_crossattn            — text encoder hidden states [B, E, D]
          encoder_attention_mask — text token boolean mask [B, E]
          pooled_projections     — pooled text embedding [B, D_pool] (optional)
          image_embeds           — vision encoder output [B, N, D] (I2V, optional)
        """
        out = super().extra_conds(**kwargs)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out["encoder_attention_mask"] = comfy.conds.CONDRegular(attention_mask)
        elif cross_attn is not None:
            # Auto-generate all-ones mask if text encoder didn't provide one.
            # Shape: [B, seq_len] matching cross_attn [B, seq_len, hidden_dim].
            out["encoder_attention_mask"] = comfy.conds.CONDRegular(
                torch.ones(cross_attn.shape[:2], dtype=torch.bool, device=cross_attn.device)
            )

        pooled_projections = kwargs.get("pooled_projections", None)
        if pooled_projections is not None:
            out["pooled_projections"] = comfy.conds.CONDRegular(pooled_projections)

        image_embeds = kwargs.get("image_embeds", None)
        if image_embeds is not None:
            out["image_embeds"] = comfy.conds.CONDRegular(image_embeds)

        return out


__all__ = ["MotifVideoModel", "MotifVideoModelAdapter", "MotifVideoLatent",
           "_make_comfyui_forward"]
