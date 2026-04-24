# attention.py
#
# MotifVideoAttention — ops-aware Attention replacement for MotifVideo transformer.
#
# Subset principle (inherited from ops_primitives.py):
#   Only branches exercised by MotifVideo transformer are implemented.
#   Attribute names mirror diffusers Attention exactly for state_dict key parity.
#
# Supported configurations:
#   single-block (pre_only=True,  added_kv=False) — SingleTransformerBlock
#   dual-block   (pre_only=False, added_kv=True)  — TransformerBlock
#
# Design note:
#   Designed against the P1.1 snapshot `tests/transformer/expected_attn_keys.json`
#   for block index 0. All other block indexes are assumed to share the same
#   structure (recommender proposal 2).

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from comfy.ldm.modules.attention import optimized_attention

# DEBUG: on the very first forward call, print optimized_attention's actual
# backend + mask state once. This is how we confirm which kernel ComfyUI
# auto-selected (attention_pytorch / attention_flash / attention_xformers /
# ...). Without the MOTIFVIDEO_DEBUG_ATTN env var set this is effectively a
# no-op.
_DEBUG_ATTN_PRINTED = False


# Local copy of `apply_rotary_emb` (same impl as transformer_motif_video.py:64-116).
# Duplicated intentionally so attention.py has zero diffusers dependency; the copy
# in transformer_motif_video.py will be removed in P4.2 alongside the legacy
# MotifVideoAttnProcessor2_0, at which point transformer_motif_video.py will
# import this one.
def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    if use_real:
        cos, sin = freqs_cis
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        if cos.dim() != 4 or sin.dim() != 4:
            raise RuntimeError(f"RoPE must be 2D or 4D, got cos={cos.dim()}D, sin={sin.dim()}D")
        cos, sin = cos.to(x.device), sin.to(x.device)
        if cos.size(-2) != x.size(-2) or cos.size(-1) != x.size(-1):
            raise RuntimeError(
                f"RoPE shape mismatch: rope[-2:]=({cos.size(-2)},{cos.size(-1)}) vs x[-2:]=({x.size(-2)},{x.size(-1)})"
            )
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    x_rot = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(2)
    x_out = torch.view_as_real(x_rot * freqs).flatten(3)
    return x_out.type_as(x)

# ---------------------------------------------------------------------------
# Default ops fallback (same pattern as ops_primitives._get_default_ops)
# ---------------------------------------------------------------------------
_DEFAULT_OPS = None


def _get_default_ops():
    global _DEFAULT_OPS
    if _DEFAULT_OPS is None:
        import comfy.ops
        _DEFAULT_OPS = comfy.ops.disable_weight_init
    return _DEFAULT_OPS


# ---------------------------------------------------------------------------
# MotifVideoAttention
# ---------------------------------------------------------------------------

class MotifVideoAttention(nn.Module):
    """Ops-aware attention module for MotifVideo transformer.

    Replaces diffusers ``Attention`` + ``MotifVideoAttnProcessor2_0``.
    ``forward`` is a placeholder until P2.2/P2.3.

    Args:
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension per head.
        qk_norm: ``"rms_norm"`` → per-head RMSNorm on Q and K; ``None`` → no norm.
        pre_only: ``True`` for SingleTransformerBlock (no ``to_out`` projection).
        added_kv: ``True`` for TransformerBlock (has ``add_*_proj`` / ``to_add_out`` /
            ``norm_added_*``).
        eps: Epsilon for RMSNorm layers.
        dropout: Dropout probability for ``to_out[1]``.
        bias: Whether Linear layers include a bias term.
        dtype: Tensor dtype passed through to ops constructors.
        device: Device passed through to ops constructors.
        operations: ComfyUI ops namespace (e.g. ``comfy.ops.disable_weight_init``).
            Defaults to ``comfy.ops.disable_weight_init`` via lazy import.

    State-dict key parity:
        single-block: to_q.{weight,bias}, to_k.{weight,bias}, to_v.{weight,bias},
                      norm_q.weight, norm_k.weight  (8 keys when qk_norm="rms_norm")
        dual-block:   above + to_out.0.{weight,bias},
                      add_q_proj.{weight,bias}, add_k_proj.{weight,bias},
                      add_v_proj.{weight,bias}, to_add_out.{weight,bias},
                      norm_added_q.weight, norm_added_k.weight  (20 keys)
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        pre_only: bool,
        added_kv: bool,
        qk_norm: Optional[str] = "rms_norm",
        eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        # Only two combinations are supported:
        #   Single : pre_only=True  & added_kv=False
        #   Dual   : pre_only=False & added_kv=True
        # Anything else (True/True, False/False) would mismatch the checkpoint
        # state_dict, so we reject it up front.
        if pre_only == added_kv:
            raise ValueError(
                f"MotifVideoAttention: unsupported combo pre_only={pre_only}, added_kv={added_kv}. "
                "Use (pre_only=True, added_kv=False) for Single blocks or "
                "(pre_only=False, added_kv=True) for Dual blocks."
            )

        ops = operations or _get_default_ops()
        hidden_size = num_attention_heads * attention_head_dim

        # Exposed as external attribute contract (R5)
        self.heads = num_attention_heads

        # ------------------------------------------------------------------
        # Core QKV projections — always present
        # ------------------------------------------------------------------
        self.to_q = ops.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype, device=device)
        self.to_k = ops.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype, device=device)
        self.to_v = ops.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype, device=device)

        # ------------------------------------------------------------------
        # QK norm — optional
        # ------------------------------------------------------------------
        if qk_norm == "rms_norm":
            self.norm_q = ops.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
            self.norm_k = ops.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        elif qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        else:
            raise ValueError(
                f"MotifVideoAttention: qk_norm={qk_norm!r} is not supported. "
                "Use 'rms_norm' or None."
            )

        # ------------------------------------------------------------------
        # Output projection
        #   pre_only=True  (Single block) → None   (diffusers Attention contract)
        #   pre_only=False (Dual block)   → ModuleList([Linear, Dropout])
        # ------------------------------------------------------------------
        if pre_only:
            self.to_out = None
        else:
            self.to_out = nn.ModuleList([
                ops.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype, device=device),
                nn.Dropout(dropout),
            ])

        # ------------------------------------------------------------------
        # Added-KV branch — Dual block only
        # ------------------------------------------------------------------
        if added_kv:
            self.add_q_proj = ops.Linear(
                hidden_size, hidden_size, bias=bias, dtype=dtype, device=device
            )
            self.add_k_proj = ops.Linear(
                hidden_size, hidden_size, bias=bias, dtype=dtype, device=device
            )
            self.add_v_proj = ops.Linear(
                hidden_size, hidden_size, bias=bias, dtype=dtype, device=device
            )
            self.to_add_out = ops.Linear(
                hidden_size, hidden_size, bias=bias, dtype=dtype, device=device
            )
            if qk_norm == "rms_norm":
                self.norm_added_q = ops.RMSNorm(
                    attention_head_dim, eps=eps, dtype=dtype, device=device
                )
                self.norm_added_k = ops.RMSNorm(
                    attention_head_dim, eps=eps, dtype=dtype, device=device
                )
            else:
                self.norm_added_q = None
                self.norm_added_k = None
        else:
            # Single block: None attributes (plain assignment, not register_module)
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None
            self.norm_added_q = None
            self.norm_added_k = None


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        query_input: Optional[torch.Tensor] = None,
        key_input: Optional[torch.Tensor] = None,
        value_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # DEBUG: declare the global flag once at the top of forward. Re-
        # declaring `global` on the same name later in the same function
        # raises SyntaxError in Python.
        global _DEBUG_ATTN_PRINTED

        # Cross-attention mode: query already projected externally (cross_attn_query_proj + norm),
        # skip to_q and only apply reshape + norm_q + RoPE. K/V use to_k/to_v as normal.
        if query_input is not None:
            query = query_input.unflatten(2, (self.heads, -1)).transpose(1, 2)
            key = self.to_k(key_input)
            value = self.to_v(value_input)

            key = key.unflatten(2, (self.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (self.heads, -1)).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)

            if not _DEBUG_ATTN_PRINTED:
                _DEBUG_ATTN_PRINTED = True
                _m = attention_mask
                print(
                    f"[MotifVideo DEBUG attn(cross)] backend={optimized_attention.__name__} "
                    f"mask_present={_m is not None} "
                    f"mask_shape={tuple(_m.shape) if _m is not None else None} "
                    f"mask_dtype={_m.dtype if _m is not None else None} "
                    f"q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)}",
                    flush=True,
                )

            hidden_states = optimized_attention(query, key, value, self.heads, skip_reshape=True, mask=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
            return hidden_states, None

        if self.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            if self.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if self.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (self.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (self.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (self.heads, -1)).transpose(1, 2)

            if self.norm_added_q is not None:
                encoder_query = self.norm_added_q(encoder_query)
            if self.norm_added_k is not None:
                encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention (ComfyUI optimized_attention — auto-selects Flash / cuDNN / xFormers)
        if not _DEBUG_ATTN_PRINTED:
            _DEBUG_ATTN_PRINTED = True
            _m = attention_mask
            print(
                f"[MotifVideo DEBUG attn(joint)] backend={optimized_attention.__name__} "
                f"mask_present={_m is not None} "
                f"mask_shape={tuple(_m.shape) if _m is not None else None} "
                f"mask_dtype={_m.dtype if _m is not None else None} "
                f"q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)}",
                flush=True,
            )
        hidden_states = optimized_attention(query, key, value, self.heads, skip_reshape=True, mask=attention_mask)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if self.to_out is not None:
                hidden_states = self.to_out[0](hidden_states)
                hidden_states = self.to_out[1](hidden_states)

            if self.to_add_out is not None:
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
