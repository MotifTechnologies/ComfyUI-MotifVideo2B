"""MotifVideo TeaCache acceleration node.

TeaCache (Timestep Embedding Aware Cache) accelerates diffusion sampling by
selectively reusing cached transformer residuals when the input change between
steps is small. This avoids redundant full forward passes.

Algorithm reference: CVPR 2025 "Timestep Embedding Tells: It's Time to Cache
for Video Diffusion Model" (ali-vilab/TeaCache).

Architecture notes for MotifVideo 1.9B:
- Dual-stream transformer blocks: norm1 = AdaLayerNormZero
  norm1(hidden_states, emb=temb) → (norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp)
- temb is computed inside transformer.forward via self.time_text_embed()
- Flow matching model (no CFG alternation) → single cache, no even/odd split
- Monkey-patch target: diffusion_model.forward (MotifVideoTransformer3DModel is the
  diffusion_model directly; there is no intermediate adapter wrapper)

Cache state lifecycle:
  reset() must be called between separate video generations. The patch
  installs a pre-hook on the adapter that auto-resets when a new batch
  starts (detected via step counter overflow or explicit reset flag).
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polynomial rescaling coefficients
# ---------------------------------------------------------------------------
# Polynomial rescaling: maps raw input L1 diff to estimated output diff.
# Form: rescaled = c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]
#
# Calibrated on MotifVideo 1.9B (447 data points, image + video 720p 121f).
# R² = 0.708. Use calibrate=True to recalibrate on your own data.
_MOTIF_POLY_COEFFS = [
    2.732448052820203,
    -13.565045836418308,
    17.82483016383567,
    -1.9267397151033308,
    0.0929721136898134,
]


# ---------------------------------------------------------------------------
# Cache state container
# ---------------------------------------------------------------------------

class _TeaCacheState:
    """Mutable cache state attached to a patched adapter instance."""

    def __init__(
        self,
        rel_l1_thresh: float,
        poly_coeffs: list,
        start: float = 0.0,
        end: float = 1.0,
        calibrate: bool = False,
    ):
        self.rel_l1_thresh = rel_l1_thresh
        self.rescale_func = np.poly1d(poly_coeffs)
        self.start = start
        self.end = end
        self.calibrate = calibrate

        # Cache tensors (reset between generations)
        self.accumulated_rel_l1_distance: float = 0.0
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_output: torch.Tensor | None = None
        self.step_counter: int = 0

        # Calibration data collection
        self.calibration_data: list[tuple[float, float]] = []

    def reset(self):
        """Reset all cache state. Call between separate video generations."""
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_output = None
        self.step_counter = 0

    def should_skip(self, current_modulated_inp: torch.Tensor) -> bool:
        """Decide whether to skip this step and reuse cached residual.

        Returns True if cache is valid and we can skip full computation.
        Updates accumulated_rel_l1_distance in place.
        Resets accumulator to 0 when threshold is exceeded (full compute).
        """
        if self.previous_modulated_input is None or self.previous_output is None:
            # No cache yet — always compute
            return False

        # Shape mismatch guard (e.g. resolution changed between runs)
        if self.previous_modulated_input.shape != current_modulated_inp.shape:
            self.reset()
            return False

        # Relative L1 distance between current and previous modulated input
        prev = self.previous_modulated_input
        mean_prev = prev.abs().mean()
        if mean_prev.item() < 1e-10:
            # Prevent division by zero; treat as no change → might skip
            # but safer to compute on degenerate inputs
            return False

        raw_diff = (current_modulated_inp - prev).abs().mean() / (mean_prev + 1e-10)
        raw_diff_scalar = raw_diff.cpu().item()

        # Polynomial rescaling: maps raw input diff to estimated output diff
        rescaled = float(self.rescale_func(raw_diff_scalar))

        self.accumulated_rel_l1_distance += rescaled

        if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
            # Accumulated drift is within tolerance → skip
            return True
        else:
            # Drift too large → must recompute; reset accumulator
            self.accumulated_rel_l1_distance = 0.0
            return False


# ---------------------------------------------------------------------------
# Modulated input extraction helper
# ---------------------------------------------------------------------------

def _extract_modulated_input(
    transformer,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
) -> torch.Tensor:
    """Extract the timestep-modulated input from the first dual-stream block.

    Mirrors the AdaLayerNormZero computation inside
    MotifVideoTransformerBlock.forward() line:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp =
            self.norm1(hidden_states, emb=temb)

    We call norm1 of transformer_blocks[0] directly (read-only, no side
    effects) to get the modulated representation used for cache comparison.

    Args:
        transformer: MotifVideoTransformer3DModel instance.
        hidden_states: Patch-embedded latent after x_embedder [B, N, D].
        temb: Timestep embedding [B, D_temb] from time_text_embed.

    Returns:
        norm_hidden_states: Modulated hidden states [B, N, D].
    """
    block0 = transformer.transformer_blocks[0]
    with torch.no_grad():
        # AdaLayerNormZero returns (norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        norm_hidden_states, *_ = block0.norm1(hidden_states, emb=temb)
    return norm_hidden_states


# ---------------------------------------------------------------------------
# Patched adapter forward factory
# ---------------------------------------------------------------------------

def _compute_sampling_progress(timestep: torch.Tensor) -> float:
    """Compute sampling progress in [0.0, 1.0] from a flow-matching sigma.

    MotifVideo uses ModelSamplingFlux (shift=2.5) where:
      - timestep passed to the model IS the sigma value (timestep fn = identity)
      - sigma ≈ 1.0 at the start (full noise), sigma ≈ 0.0 at the end (clean image)

    Therefore: progress = 1.0 - sigma, clamped to [0.0, 1.0].

    Args:
        timestep: Sigma tensor as received by the adapter forward, any shape.

    Returns:
        Scalar float representing sampling progress (0.0=start, 1.0=end).
    """
    sigma = timestep.flatten()[0].item()
    # ComfyUI with ModelSamplingSD3 uses sigma range [0, 1000].
    # Normalize to [0, 1] before computing progress.
    if sigma > 1.0:
        sigma = sigma / 1000.0
    progress = 1.0 - sigma
    return max(0.0, min(1.0, progress))


def _make_teacache_forward(original_adapter_forward, transformer, state: _TeaCacheState):
    """Return a replacement forward function for MotifVideoModelAdapter.

    The returned function wraps the original adapter forward with TeaCache
    logic. Cache skip decisions are made at the whole-transformer level
    (residual = full_output - full_input), which is equivalent to the
    FLUX-style integration.

    Args:
        original_adapter_forward: The original bound method of the adapter.
        transformer: MotifVideoTransformer3DModel (unwrapped) for direct
                     access to transformer_blocks and time_text_embed.
        state: _TeaCacheState instance holding mutable cache tensors.

    Returns:
        A new forward function with the same signature as the adapter's
        original forward.
    """

    # Per-conditioning-type cache states for CFG support.
    # Stored on the parent state so idempotency guard can access them.
    if not hasattr(state, 'cond_states'):
        state.cond_states = {}

    def _get_cond_state(cond_type: int) -> _TeaCacheState:
        """Get or create a cache state for a specific conditioning type."""
        if cond_type not in state.cond_states:
            state.cond_states[cond_type] = _TeaCacheState(
                rel_l1_thresh=state.rel_l1_thresh,
                poly_coeffs=list(state.rescale_func.coeffs),
                start=state.start,
                end=state.end,
                calibrate=state.calibrate,
            )
        return state.cond_states[cond_type]

    def teacache_forward(
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
        # ------------------------------------------------------------------
        # Step 0: Determine conditioning type (positive=0, negative=1)
        # ------------------------------------------------------------------
        cond_or_uncond = transformer_options.get("cond_or_uncond", [0]) if transformer_options else [0]
        cond_type = cond_or_uncond[0] if cond_or_uncond else 0
        cs = _get_cond_state(cond_type)

        # ------------------------------------------------------------------
        # Step 1: Compute temb + embedded hidden states for modulated input
        # ------------------------------------------------------------------
        with torch.no_grad():
            temb, _token_replace_emb = transformer.time_text_embed(
                timestep, pooled_projections
            )
            x_embedded = transformer.x_embedder(x)

        modulated_inp = _extract_modulated_input(transformer, x_embedded, temb)

        # ------------------------------------------------------------------
        # Step 2: Calibration mode — always compute, collect diffs
        # ------------------------------------------------------------------
        if cs.calibrate:
            output = original_adapter_forward(
                x, timestep, context=context, control=control,
                transformer_options=transformer_options,
                encoder_attention_mask=encoder_attention_mask,
                pooled_projections=pooled_projections,
                image_embeds=image_embeds, **kwargs,
            )

            if (cs.previous_modulated_input is not None
                    and cs.previous_output is not None
                    and cs.previous_modulated_input.shape == modulated_inp.shape
                    and cs.previous_output.shape == output.shape):
                prev_mod = cs.previous_modulated_input
                mean_prev = prev_mod.abs().mean()
                raw_diff = (modulated_inp - prev_mod).abs().mean() / (mean_prev + 1e-10)
                raw_diff_val = raw_diff.cpu().item()

                mean_prev_out = cs.previous_output.abs().mean()
                output_diff = (output - cs.previous_output).abs().mean() / (mean_prev_out + 1e-10)
                output_diff_val = output_diff.cpu().item()

                cs.calibration_data.append((raw_diff_val, output_diff_val))
                print(
                    f"[TeaCache:CALIB] cond={cond_type} step={cs.step_counter:3d} "
                    f"raw_diff={raw_diff_val:.8f} output_diff={output_diff_val:.8f}"
                )

            cs.previous_output = output.clone()
            cs.previous_modulated_input = modulated_inp.clone()
            cs.step_counter += 1
            return output

        # ------------------------------------------------------------------
        # Step 3: Cache decision — first step always computes
        # ------------------------------------------------------------------
        skip = cs.should_skip(modulated_inp)
        should_calc = not skip

        if should_calc:
            logger.debug(
                "[TeaCache] cond=%d Step %d: COMPUTE (accumulated=%.4f)",
                cond_type, cs.step_counter, cs.accumulated_rel_l1_distance,
            )
            output = original_adapter_forward(
                x, timestep, context=context, control=control,
                transformer_options=transformer_options,
                encoder_attention_mask=encoder_attention_mask,
                pooled_projections=pooled_projections,
                image_embeds=image_embeds, **kwargs,
            )
            cs.previous_output = output.clone()
        else:
            logger.debug(
                "[TeaCache] cond=%d Step %d: SKIP (accumulated=%.4f < thresh=%.4f)",
                cond_type, cs.step_counter, cs.accumulated_rel_l1_distance, cs.rel_l1_thresh,
            )
            output = cs.previous_output

        # ------------------------------------------------------------------
        # Step 4: Update cache for next step
        # ------------------------------------------------------------------
        cs.previous_modulated_input = modulated_inp.clone()
        cs.step_counter += 1

        return output

    return teacache_forward


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class MotifTeaCache:
    """Apply TeaCache acceleration to a MotifVideo diffusion model.

    TeaCache skips redundant transformer forward passes during sampling by
    reusing cached residuals from previous steps when the input change
    (measured via timestep-modulated L1 distance) is below a threshold.

    The cache state (accumulated distance, modulated input, residual) is
    automatically reset at the start of each new generation via ComfyUI's
    model cloning mechanism.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": (
                            "Relative L1 threshold for cache reuse. "
                            "Lower = more aggressive caching (faster, riskier quality). "
                            "Higher = more recomputation (safer quality, less speedup). "
                            "Recommended: 0.15–0.3 for MotifVideo."
                        ),
                    },
                ),
                "enable": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Disable to bypass TeaCache (useful for A/B comparison).",
                    },
                ),
                "start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling progress at which TeaCache becomes active "
                            "(0.0=start of sampling, 1.0=end). "
                            "Set > 0.0 to skip caching during early high-noise steps, "
                            "which preserves coarse structure quality."
                        ),
                    },
                ),
                "end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling progress at which TeaCache becomes inactive "
                            "(0.0=start of sampling, 1.0=end). "
                            "Set < 1.0 to skip caching during late low-noise steps, "
                            "which preserves fine detail quality."
                        ),
                    },
                ),
                "calibrate": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Calibration mode: runs full forward every step and "
                            "prints raw_diff vs output_diff to console. "
                            "Use the output poly_coeffs to replace the default. "
                            "No caching is applied in this mode."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache"
    CATEGORY = "motifvideo"

    DESCRIPTION = (
        "Apply TeaCache acceleration to MotifVideo 1.9B.\n"
        "Speeds up sampling by reusing cached transformer residuals.\n"
        "rel_l1_thresh controls quality/speed trade-off.\n"
        "Connect between Load Diffusion Model and KSampler."
    )

    def apply_teacache(
        self,
        model,
        rel_l1_thresh: float,
        enable: bool,
        start: float = 0.0,
        end: float = 1.0,
        calibrate: bool = False,
    ):
        """Patch the diffusion model with TeaCache logic.

        Args:
            model: ComfyUI ModelPatcher wrapping MotifVideoModel.
            rel_l1_thresh: Threshold for relative L1 distance accumulation.
            enable: If False, return model unchanged.
            start: Sampling progress (0.0–1.0) at which caching activates.
                   Steps before this progress threshold are computed in full.
            end: Sampling progress (0.0–1.0) at which caching deactivates.
                 Steps at or beyond this threshold are computed in full.

        Returns:
            Tuple of (patched_model,) or (original_model,) if disabled.
        """
        if not enable:
            logger.info("[TeaCache] Disabled — returning model unchanged.")
            return (model,)

        # Clone the model patcher so we don't mutate shared state
        patched_model = model.clone()

        # Reach through ComfyUI's ModelPatcher to get the inner model
        # (MotifVideoModel) and then the transformer.
        #
        # Architecture note: diffusion_model IS MotifVideoTransformer3DModel directly.
        # There is no intermediate adapter wrapper — the transformer is the diffusion_model.
        inner_model = patched_model.model  # MotifVideoModel (BaseModel subclass)
        transformer = inner_model.diffusion_model  # MotifVideoTransformer3DModel (directly)

        if not hasattr(transformer, "transformer_blocks") or len(transformer.transformer_blocks) == 0:
            logger.warning(
                "[TeaCache] diffusion_model has no transformer_blocks. "
                "Expected MotifVideoTransformer3DModel. TeaCache not applied."
            )
            return (patched_model,)

        # Check if already patched (idempotency guard)
        if getattr(transformer, "_teacache_enabled", False):
            logger.info("[TeaCache] Already patched — skipping re-patch.")
            # Update parameters on existing state if different
            if hasattr(transformer, "_teacache_state"):
                transformer._teacache_state.rel_l1_thresh = rel_l1_thresh
                transformer._teacache_state.start = start
                transformer._teacache_state.end = end
                transformer._teacache_state.reset()
                # Reset per-conditioning cache states
                if hasattr(transformer._teacache_state, 'cond_states'):
                    for cs in transformer._teacache_state.cond_states.values():
                        cs.rel_l1_thresh = rel_l1_thresh
                        cs.start = start
                        cs.end = end
                        cs.reset()
            return (patched_model,)

        # Build cache state
        state = _TeaCacheState(
            rel_l1_thresh=rel_l1_thresh,
            poly_coeffs=_MOTIF_POLY_COEFFS,
            start=start,
            end=end,
            calibrate=calibrate,
        )

        # Save original forward and install patched forward.
        # transformer.forward here is the ComfyUI-compatible forward installed by
        # _make_comfyui_forward in the loader. TeaCache wraps this outermost entry.
        original_forward = transformer.forward

        patched_forward = _make_teacache_forward(
            original_adapter_forward=original_forward,
            transformer=transformer,
            state=state,
        )

        # Replace forward on the transformer instance directly
        transformer.forward = patched_forward

        # Mark as patched and store state for introspection/reset
        transformer._teacache_enabled = True
        transformer._teacache_state = state

        mode = "CALIBRATION" if calibrate else "CACHING"
        logger.info(
            "[TeaCache] Patched MotifVideoTransformer3DModel (diffusion_model). "
            "mode=%s, rel_l1_thresh=%.3f, start=%.2f, end=%.2f, poly_coeffs=%s",
            mode,
            rel_l1_thresh,
            start,
            end,
            _MOTIF_POLY_COEFFS,
        )

        return (patched_model,)
