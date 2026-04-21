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
#   P1.1 snapshot `tests/transformer/expected_attn_keys.json` block index 0 기준으로 설계됨.
#   다른 block index 도 동일 구조 가정 (recommender 제안 2).

from __future__ import annotations

from typing import Optional

import torch.nn as nn

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

        # 지원 조합은 2가지: (Single) pre_only=True & added_kv=False, (Dual) pre_only=False & added_kv=True.
        # 그 외 조합 (True/True, False/False) 은 checkpoint state_dict 과 mismatch 되므로 즉시 차단.
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

        # P2.3/P4.1 에서 apply_sage_attention 이 True 로 설정
        self.use_sage = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MotifVideoAttention.forward is not implemented yet. "
            "Implementation is scheduled for P2.2/P2.3."
        )
