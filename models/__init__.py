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

import os

import torch
import comfy.model_base
import comfy.conds
import comfy.ops
import comfy.model_management

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

        # Select operations class based on weight/compute dtype.
        # Mirrors comfy.model_base.BaseModel.__init__ pattern (model_base.py:143-147).
        # custom_operations override takes precedence; otherwise pick_operations chooses
        # manual_cast / fp8_ops / disable_weight_init depending on model_config and dtype.
        if model_config.custom_operations is None:
            fp8 = model_config.optimizations.get("fp8", False)
            operations = comfy.ops.pick_operations(
                original_unet_config.get("dtype", None),
                self.manual_cast_dtype,
                fp8_optimizations=fp8,
                model_config=model_config,
            )
        else:
            operations = model_config.custom_operations

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
        # Use unet_config.dtype (weight storage dtype) rather than self.get_dtype()
        # — the latter reads self.diffusion_model.dtype which doesn't exist until
        # transformer is constructed below. Matches comfy flux/sd3 conventions.
        weight_dtype = original_unet_config.get("dtype", None)
        transformer = MotifVideoTransformer3DModel(
            **transformer_kwargs,
            operations=operations,
            dtype=weight_dtype,
            device=device,
        )
        # NOTE: .to(dtype=bfloat16) 강제 cast 제거 — comfy.ops 가 weight load 시점에
        # weight_dtype 과 compute_dtype 매핑을 담당. 기존 강제 cast 는 manual_cast 회로를
        # 우회하여 fp8/quantized weight 를 망가뜨릴 수 있음.

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
    # apply_model: sequential offload (HF model_cpu_offload_seq equivalent)
    # ------------------------------------------------------------------

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                    transformer_options={}, **kwargs):
        """Override to perform targeted unload of T5Gemma2 text encoder before denoising.

        HF pipeline 의 model_cpu_offload_seq="text_encoder->transformer->vae" 와
        동등 의미론으로, 우리 T5Gemma2 text encoder 만 targeted unload.
        control / LoRA / hooks / patches 등은 절대 건드리지 않음 (Codex HIGH 리뷰 결과 반영).

        HIGH_VRAM 환경(H200)에서 ComfyUI 의 smart memory 가 skip 하는 경우에 대비해
        직접 unload. sequential_offload / unload_all_models 방식을 사용하지 않으므로
        denoise time 에 참여하는 control, LoRA, hooks, patches, TREAD, TeaCache 등
        보조 모델은 GPU 상태가 유지된다.

        상태 기반 idempotent 구현: current_loaded_models 에서 MotifVideoT5Gemma2Model
        인스턴스를 찾아 있으면 unload, 없으면 skip (이미 unload 된 상태면 no-op 자동).
        """
        # local import: circular import 방지
        # (text_encoders 는 models 를 import 하지 않으므로 실제 순환은 없으나
        #  top-level import 를 피해 의존 방향을 명시적으로 단방향으로 유지)
        from text_encoders.t5_gemma2 import MotifVideoT5Gemma2Model

        device = comfy.model_management.get_torch_device()

        # current_loaded_models 순회하여 T5Gemma2 text encoder LoadedModel 식별.
        # LoadedModel.model 은 ModelPatcher, .model.model 이 nn.Module(BaseModel 또는
        # 직접 nn.Module 서브클래스). 두 경로 모두 체크.
        text_encoder_lms = []
        for lm in comfy.model_management.current_loaded_models:
            if lm.model is None:
                continue
            # ModelPatcher 경로: lm.model.model → nn.Module
            mp_model = getattr(lm.model, "model", None)
            if isinstance(mp_model, MotifVideoT5Gemma2Model):
                text_encoder_lms.append(lm)
                continue
            # 직접 nn.Module 경로: lm.model 자체가 MotifVideoT5Gemma2Model
            if isinstance(lm.model, MotifVideoT5Gemma2Model):
                text_encoder_lms.append(lm)

        if text_encoder_lms:
            # T5Gemma2 text encoder 만 targeted unload.
            # keep_loaded = "T5Gemma2 이외의 모든 LoadedModel" 흑리스트 방식.
            # control, LoRA, hooks, patches, 기타 모든 보조 모델은 keep 에 포함.
            non_text_encoder_lms = [
                lm for lm in comfy.model_management.current_loaded_models
                if lm not in text_encoder_lms
            ]
            comfy.model_management.free_memory(1e30, device, keep_loaded=non_text_encoder_lms)

        return super().apply_model(x, t, c_concat, c_crossattn, control,
                                   transformer_options, **kwargs)

    # ------------------------------------------------------------------
    # memory_required: add activation peak margin for smart memory hint
    # ------------------------------------------------------------------

    def memory_required(self, input_shape, cond_shapes={}):
        """Return memory estimate with activation peak margin.

        HF sequential_offload 정렬 보조: normal_vram 환경에서 ComfyUI smart
        memory 가 올바른 unload 결정을 내리도록 activation peak 여유를 1.3배
        margin 으로 반영. HIGH_VRAM 환경에서는 apply_model 의 명시적 free_memory
        호출이 주 역할이므로 본 메서드는 보조 역할만 함.
        """
        base = super().memory_required(input_shape, cond_shapes)
        return base * 1.3

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

        # compile / channels_last_3d 호출은 직전 플랜(20260419-perf-sampling) 에서 revert.
        # - apply_compile: OptimizedModule wrapping 이 ComfyUI ModelPatcher offload 와 구조
        #   충돌 (VRAM 폭증). 04_log '2026-04-20 P3.1 최종 revert' 참조.
        # - apply_channels_last_3d: sage/compile 과 coupled 도입 전제라 단독은 이득 불투명.
        #   sage 활성화 후 별도 gate 로 재도입 검토 (현재 gate: MOTIFVIDEO_ENABLE_CHANNELS_LAST=1,
        #   아직 도입 안 됨).
        #
        # 속도 회복 경로는 SageAttention 이식 (이슈 #16). attention processor 교체만 수행
        # 하므로 OptimizedModule wrapping 없음 → ModelPatcher offload 와 호환.
        # sageattention 미설치 환경에선 helper 가 자동 no-op.
        if os.environ.get("MOTIFVIDEO_ENABLE_SAGE") == "1":
            from .compile_config import apply_sage_attention
            self.diffusion_model = apply_sage_attention(self.diffusion_model)
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
            # Fallback: auto-generate all-ones mask when the text encoder did not
            # supply a padding mask.  This path is intentionally kept for backward
            # compatibility with the standard CLIPTextEncode node, but it is a
            # quality-divergence path: padded tokens participate in cross-attention,
            # causing a training/inference distribution mismatch.  When
            # MotifTextEncode is used, attention_mask is populated via
            # MotifVideoT5Gemma2Model.encode_token_weights() and this branch is
            # never reached.
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
