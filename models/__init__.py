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

        # fp8 safety: applied before BaseModel.__init__ runs its standard
        # pick_operations. The `optimizations["fp8"] = True` in config.py is
        # only safe to route through fp8_ops when weight_dtype is actually an
        # fp8 variant. If fp8_ops is selected for bf16/fp16 weights, every
        # Linear call gains an on-the-fly fp8 dispatch that runs 10x+ slower
        # (this was the root cause of the bf16 smoke "sage-off 222 s/step"
        # regression). model_config.optimizations is a mutable dict copied
        # per instance construction, so overwriting it here is local.
        if model_config.custom_operations is None:
            _weight_dtype = unet_config.get("dtype", None)
            _fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
            _is_fp8_weight = _weight_dtype in _fp8_types if _weight_dtype is not None else False
            if not _is_fp8_weight:
                # Non-fp8 weights: disable the fp8 optimization flag so that
                # BaseModel's pick_operations picks disable_weight_init instead
                # of fp8_ops.
                model_config.optimizations["fp8"] = False

        # Standard BaseModel construction path: passing
        # unet_model=MotifVideoTransformer3DModel lets BaseModel.__init__ run
        # pick_operations + unet_model(**unet_config) + eval() +
        # archive_model_dtypes() in order, cooperating cleanly with the vbar
        # system. MotifVideoTransformer3DModel.__init__ accepts **kwargs so
        # that unknown keys in unet_config (e.g. image_model) are ignored.
        super().__init__(
            model_config,
            model_type,
            device=device,
            unet_model=MotifVideoTransformer3DModel,
        )

        # DEBUG: report ops-class info. The operations object BaseModel
        # selected is not directly queryable — pick_operations' result is
        # bound inside diffusion_model, so we infer it from dtypes.
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
        # Applied after super().__init__ has already built self.diffusion_model
        # and called eval() on it.
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

        attention_mask is intentionally not put into the cond dict. The text
        encoder trims the padded tail ahead of time and only forwards valid
        text tokens, so the transformer performs cross-attention correctly
        without a mask. Going through the mask=None path lets PyTorch SDPA
        pick cuDNN/Flash automatically for best throughput (this is the
        19.5 s/step path measured on the H200 Confluence benchmark).
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
