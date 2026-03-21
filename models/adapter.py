"""ComfyUI diffusion_model interface adapter for MotifVideoTransformer3DModel.

ComfyUI's BaseModel._apply_model() calls:
    self.diffusion_model(xc, t, context=context, control=control,
                         transformer_options=transformer_options, **extra_conds)

MotifVideoTransformer3DModel.forward() expects:
    forward(hidden_states, timestep, encoder_hidden_states,
            encoder_attention_mask, ...)

This adapter bridges the two calling conventions so neither the ComfyUI
core nor the motif_core transformer need to be modified.
"""

import torch
import torch.nn as nn


class MotifVideoModelAdapter(nn.Module):
    """Wrap MotifVideoTransformer3DModel to satisfy the ComfyUI diffusion_model
    calling convention.

    ComfyUI passes keyword arguments from extra_conds() as **extra_conds.
    We forward ``encoder_attention_mask`` and ``pooled_projections`` through
    that mechanism; see MotifVideoModel.extra_conds().
    """

    def __init__(self, transformer):
        """
        Args:
            transformer: A MotifVideoTransformer3DModel instance.
        """
        super().__init__()
        self.transformer = transformer

    # ------------------------------------------------------------------
    # Proxy common attribute access so ComfyUI can inspect the wrapped
    # model (e.g. dtype, patch_size) without knowing about the adapter.
    # ------------------------------------------------------------------

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def forward(
        self,
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
        """Translate ComfyUI calling convention to MotifVideoTransformer3DModel.

        Args:
            x: Noisy latent tensor [B, C_in, T, H, W].
               C_in = 33 for MotifVideo (16 noise + 16 latent_condition + 1 mask).
            timestep: Diffusion timestep tensor [B] (already scalar-converted by
                      model_sampling.timestep() in BaseModel._apply_model).
            context: Text encoder hidden states [B, E, D]  (c_crossattn).
            control: ControlNet output dict (unused — passed through for future
                     compatibility).
            transformer_options: ComfyUI transformer options dict.
            encoder_attention_mask: Boolean/float mask [B, E] for text tokens.
            pooled_projections: Pooled text embedding [B, D_pool] (optional).
            image_embeds: Vision encoder embeddings [B, N, D] (I2V, optional).
            **kwargs: Any remaining extra_conds (ignored but accepted for
                      forward-compatibility).

        Returns:
            Denoised latent tensor [B, 16, T, H, W].
        """
        if transformer_options is None:
            transformer_options = {}

        output = self.transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            image_embeds=image_embeds,
            return_dict=False,
        )
        # forward() returns (sample,) when return_dict=False
        return output[0]
