"""T5Gemma2 native implementation scaffold for ComfyUI partial offload.

This module hosts the native T5Gemma2 encoder. RMSNorm uses raw `nn.Parameter`
(small weight, no offload benefit). The Linear/Embedding layers added in
later items will receive `comfy.ops` injection for `comfy_cast_weights`
support — the `comfy.ops` import is wired here so subsequent additions
land without further bootstrap edits.

Derived from transformers.models.t5gemma2.modeling_t5gemma2 (Apache 2.0).
"""

import torch
import torch.nn as nn

import comfy.ops  # noqa: F401  # used by Linear/Embedding wrappers in later items


# ---------------------------------------------------------------------------
# RoPE helpers (HF modeling_t5gemma2.py:175-205, Apache 2.0)
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to q, k.

    cos/sin shape: (B, S, head_dim). After unsqueeze_dim=1: (B, 1, S, head_dim),
    broadcasts to q/k of shape (B, heads, S, head_dim).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# GQA helper (HF modeling_t5gemma2.py:208-217, Apache 2.0)
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """KV head repeat for GQA: (B, n_kv, S, D) → (B, n_q, S, D) where n_q = n_kv * n_rep."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class T5Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # T5Gemma2 specific: (x * w).to(dtype) NOT x.to(dtype) * w
        # See: https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Self-Attention (HF modeling_t5gemma2.py:255-330, Apache 2.0)
# ---------------------------------------------------------------------------

class T5Gemma2SelfAttention(nn.Module):
    """Self-attention block — encoder, GQA, q_norm/k_norm + RoPE.

    Mask + RoPE responsibility split:
      - `attention_mask` and `position_embeddings` are built externally by the
        caller (T5Gemma2TextEncoder, item 5). The caller dispatches per
        `layer_types[layer_idx]` and supplies the matching mask/RoPE table
        for "full_attention" or "sliding_attention".
      - `self.sliding_window` and `self.is_sliding` are stored only as a hint
        for the caller; this forward does not derive a mask from them. SDPA
        receives the prebuilt `attn_mask` as-is.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        operations = operations if operations is not None else comfy.ops.disable_weight_init

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # HF uses query_pre_attn_scalar (=256) rather than head_dim for the scale denominator;
        # numerically identical for this model but kept for interface parity.
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # encoder

        self.q_proj = operations.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias, dtype=dtype, device=device,
        )
        self.k_proj = operations.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, dtype=dtype, device=device,
        )
        self.v_proj = operations.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, dtype=dtype, device=device,
        )
        self.o_proj = operations.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias, dtype=dtype, device=device,
        )

        # attn_logit_softcapping: null in our model — stored for interface match.
        # If non-null, SDPA path cannot be used as-is; requires pre-softmax cap.
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.is_sliding = self.layer_type == "sliding_attention"

        # q_norm/k_norm: per-head RMSNorm over head_dim — Gemma2 specific.
        # Raw nn.Parameter inside; no comfy.ops injection needed (small weight).
        self.q_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (B, n_q, S, D)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # (B, n_kv, S, D)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (B, n_kv, S, D)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA: expand KV heads to match Q heads (explicit expand for portability over torch 2.5+).
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output
