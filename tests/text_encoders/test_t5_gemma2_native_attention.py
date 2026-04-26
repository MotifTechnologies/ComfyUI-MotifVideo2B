"""
Tests for attention helpers and T5Gemma2SelfAttention in
text_encoders/t5_gemma2_native.py.

Scope: rotate_half, apply_rotary_pos_emb, repeat_kv, T5Gemma2SelfAttention
(checklist item 3). RMSNorm is covered in a separate file.

Blind-test principle: tests are written from the spec only — implementation
code was NOT read before writing these tests.

Environment: must be run with ComfyUI venv python:
  /lustrefs/team-multimodal/minsu/ComfyUI/.venv/bin/python \\
      -m pytest tests/text_encoders/test_t5_gemma2_native_attention.py -v

The bootstrap below derives <comfy_root> from this file's location (assumes
layout <comfy_root>/custom_nodes/<repo>/tests/...).
"""

import math
import os
import sys
from pathlib import Path

# --- ComfyUI environment bootstrap (must precede any comfy import) ----------
# Layout: <comfy_root>/custom_nodes/<repo>/tests/text_encoders/<this_file>
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_COMFYUI_ROOT_GUESS = _THIS_FILE.parents[4]
_COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", _COMFYUI_ROOT_GUESS)).resolve()

sys.argv = ["x", "--cpu"]
for _p in (str(_COMFYUI_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import pytest  # noqa: E402
import torch  # noqa: E402
import comfy.ops  # noqa: E402

from text_encoders.t5_gemma2_native import (  # noqa: E402
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    T5Gemma2SelfAttention,
)


# ---------------------------------------------------------------------------
# Config fixture shared across attention tests
# ---------------------------------------------------------------------------

def _make_config():
    """Return T5Gemma2TextConfig built from the project's bundled defaults."""
    from text_encoders.t5_gemma2_config import T5_GEMMA2_CONFIG
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2EncoderConfig,
    )
    return T5Gemma2EncoderConfig(**T5_GEMMA2_CONFIG).text_config


# ---------------------------------------------------------------------------
# 1. rotate_half
# ---------------------------------------------------------------------------

class TestRotateHalf:

    def test_rotate_half_shape(self):
        """Output shape must equal input shape — last-dim split-and-cat must not change size."""
        for shape in [(2, 4, 8), (1, 1, 256), (3, 7, 16)]:
            x = torch.randn(*shape)
            out = rotate_half(x)
            assert out.shape == x.shape, (
                f"rotate_half shape mismatch for input {shape}: got {out.shape}"
            )

    def test_rotate_half_double_apply_negates(self):
        """Applying rotate_half twice must equal -x (180-degree rotation)."""
        x = torch.randn(2, 8, 256)
        result = rotate_half(rotate_half(x))
        assert torch.allclose(result, -x, atol=1e-6), (
            "rotate_half(rotate_half(x)) must equal -x"
        )

    def test_rotate_half_values(self):
        """
        Concrete check: for x = [x1 | x2] (halved along last dim),
        output must be [-x2 | x1].
        """
        # Use even last-dim so halving is exact
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
        out = rotate_half(x)
        # x1=[1,2], x2=[3,4] → [-x2, x1] = [-3,-4, 1,2]
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(out, expected, atol=1e-6), (
            f"rotate_half([1,2,3,4]) expected [-3,-4,1,2], got {out.tolist()}"
        )


# ---------------------------------------------------------------------------
# 2. apply_rotary_pos_emb
# ---------------------------------------------------------------------------

class TestApplyRotaryPosEmb:

    # Typical shapes: q/k (B, heads, S, D), cos/sin (B, S, D)
    _B, _H, _S, _D = 2, 8, 16, 256

    def _make_cos_sin(self, val_cos, val_sin):
        cos = torch.full((self._B, self._S, self._D), val_cos)
        sin = torch.full((self._B, self._S, self._D), val_sin)
        return cos, sin

    def test_apply_rope_shape_preserved(self):
        """Output q_rot and k_rot shapes must match input q and k shapes."""
        q = torch.randn(self._B, self._H, self._S, self._D)
        k = torch.randn(self._B, self._H, self._S, self._D)
        cos, sin = self._make_cos_sin(1.0, 0.0)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape, (
            f"q_rot shape {q_rot.shape} != q shape {q.shape}"
        )
        assert k_rot.shape == k.shape, (
            f"k_rot shape {k_rot.shape} != k shape {k.shape}"
        )

    def test_apply_rope_identity_when_cos1_sin0(self):
        """cos=1, sin=0 → rotation by 0° → output equals input."""
        q = torch.randn(self._B, self._H, self._S, self._D)
        k = torch.randn(self._B, self._H, self._S, self._D)
        cos, sin = self._make_cos_sin(1.0, 0.0)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert torch.allclose(q_rot, q, atol=1e-6), (
            "apply_rope with cos=1,sin=0 must be identity for q"
        )
        assert torch.allclose(k_rot, k, atol=1e-6), (
            "apply_rope with cos=1,sin=0 must be identity for k"
        )

    def test_apply_rope_pure_rotate_when_cos0_sin1(self):
        """cos=0, sin=1 → output = (rotate_half(q), rotate_half(k))."""
        q = torch.randn(self._B, self._H, self._S, self._D)
        k = torch.randn(self._B, self._H, self._S, self._D)
        cos, sin = self._make_cos_sin(0.0, 1.0)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert torch.allclose(q_rot, rotate_half(q), atol=1e-6), (
            "apply_rope with cos=0,sin=1 must give rotate_half(q)"
        )
        assert torch.allclose(k_rot, rotate_half(k), atol=1e-6), (
            "apply_rope with cos=0,sin=1 must give rotate_half(k)"
        )

    def test_apply_rope_formula(self):
        """Verify the full formula: q*cos + rotate_half(q)*sin."""
        q = torch.randn(self._B, self._H, self._S, self._D)
        k = torch.randn(self._B, self._H, self._S, self._D)

        cos = torch.rand(self._B, self._S, self._D)
        sin = torch.rand(self._B, self._S, self._D)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # Compute expected manually with broadcasting
        cos_b = cos.unsqueeze(1)  # (B, 1, S, D)
        sin_b = sin.unsqueeze(1)  # (B, 1, S, D)
        q_expected = q * cos_b + rotate_half(q) * sin_b
        k_expected = k * cos_b + rotate_half(k) * sin_b

        assert torch.allclose(q_rot, q_expected, atol=1e-5), "q formula mismatch"
        assert torch.allclose(k_rot, k_expected, atol=1e-5), "k formula mismatch"

    def test_apply_rope_unsqueeze_dim_default(self):
        """Default unsqueeze_dim=1 must work without explicit kwarg."""
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        cos = torch.ones(1, 8, 16)
        sin = torch.zeros(1, 8, 16)
        # Should not raise
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape


# ---------------------------------------------------------------------------
# 3. repeat_kv
# ---------------------------------------------------------------------------

class TestRepeatKv:

    def test_repeat_kv_n_rep_1_identity(self):
        """n_rep=1 → output is equivalent to input (no duplication)."""
        x = torch.randn(2, 4, 16, 64)
        out = repeat_kv(x, 1)
        assert out.shape == x.shape, f"n_rep=1 shape mismatch: {out.shape}"
        assert torch.allclose(out, x, atol=0.0), "n_rep=1 must return equal tensor"

    def test_repeat_kv_n_rep_2_shape(self):
        """n_rep=2: (B, n_kv, S, D) → (B, 2*n_kv, S, D)."""
        B, n_kv, S, D = 2, 4, 16, 64
        x = torch.randn(B, n_kv, S, D)
        out = repeat_kv(x, 2)
        assert out.shape == (B, 2 * n_kv, S, D), (
            f"Expected {(B, 2*n_kv, S, D)}, got {out.shape}"
        )

    def test_repeat_kv_n_rep_3_shape(self):
        """n_rep=3: (B, n_kv, S, D) → (B, 3*n_kv, S, D)."""
        B, n_kv, S, D = 1, 4, 8, 32
        x = torch.randn(B, n_kv, S, D)
        out = repeat_kv(x, 3)
        assert out.shape == (B, 3 * n_kv, S, D)

    def test_repeat_kv_interleave_pattern(self):
        """
        Each KV head must be replicated n_rep times consecutively (NOT block repeat).
        n_rep=2, n_kv=2:
          output head 0 == output head 1 == input head 0
          output head 2 == output head 3 == input head 1
        """
        B, S, D = 1, 4, 8
        n_kv = 2
        n_rep = 2
        # Distinct values per kv-head so identity is unambiguous
        x = torch.zeros(B, n_kv, S, D)
        x[0, 0] = 1.0  # kv-head 0
        x[0, 1] = 2.0  # kv-head 1

        out = repeat_kv(x, n_rep)
        # heads 0,1 should both be kv-head 0 (value=1)
        assert torch.allclose(out[0, 0], out[0, 1]), (
            "Interleave: output head 0 must == output head 1 (both from kv-head 0)"
        )
        assert torch.all(out[0, 0] == 1.0), "output head 0 must carry kv-head 0 value"
        # heads 2,3 should both be kv-head 1 (value=2)
        assert torch.allclose(out[0, 2], out[0, 3]), (
            "Interleave: output head 2 must == output head 3 (both from kv-head 1)"
        )
        assert torch.all(out[0, 2] == 2.0), "output head 2 must carry kv-head 1 value"

    def test_repeat_kv_large_n_rep(self):
        """n_rep larger than n_kv must still produce correct shape."""
        B, n_kv, S, D = 1, 2, 4, 16
        x = torch.randn(B, n_kv, S, D)
        out = repeat_kv(x, 8)
        assert out.shape == (B, 2 * 8, S, D)


# ---------------------------------------------------------------------------
# 4. T5Gemma2SelfAttention — initialisation
# ---------------------------------------------------------------------------

class TestT5Gemma2SelfAttentionInit:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_config()

    def test_init_attribute_shapes(self, cfg):
        """q/k/v/o_proj weight shapes must match HF spec."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        # hidden_size=2560, num_q_heads=8, num_kv_heads=4, head_dim=256
        assert attn.q_proj.weight.shape == (2048, 2560), (
            f"q_proj.weight shape: expected (2048,2560), got {attn.q_proj.weight.shape}"
        )
        assert attn.k_proj.weight.shape == (1024, 2560), (
            f"k_proj.weight shape: expected (1024,2560), got {attn.k_proj.weight.shape}"
        )
        assert attn.v_proj.weight.shape == (1024, 2560), (
            f"v_proj.weight shape: expected (1024,2560), got {attn.v_proj.weight.shape}"
        )
        assert attn.o_proj.weight.shape == (2560, 2048), (
            f"o_proj.weight shape: expected (2560,2048), got {attn.o_proj.weight.shape}"
        )

    def test_init_no_bias(self, cfg):
        """All 4 projection layers must have bias=None (attention_bias=False)."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer = getattr(attn, name)
            assert layer.bias is None, (
                f"{name}.bias should be None (attention_bias=False), got {layer.bias}"
            )

    def test_init_layer_type_sliding(self, cfg):
        """layer_idx=0 → sliding_attention: is_sliding=True, sliding_window=1024."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        assert attn.is_sliding is True, (
            f"layer_idx=0: expected is_sliding=True, got {attn.is_sliding}"
        )
        assert attn.sliding_window == 1024, (
            f"layer_idx=0: expected sliding_window=1024, got {attn.sliding_window}"
        )

    def test_init_layer_type_full(self, cfg):
        """layer_idx=5 → full_attention: is_sliding=False, sliding_window=None."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=5)
        assert attn.is_sliding is False, (
            f"layer_idx=5: expected is_sliding=False, got {attn.is_sliding}"
        )
        assert attn.sliding_window is None, (
            f"layer_idx=5: expected sliding_window=None, got {attn.sliding_window}"
        )

    def test_init_q_k_norm_shapes(self, cfg):
        """q_norm and k_norm weight shapes must both be (head_dim,) = (256,)."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        assert attn.q_norm.weight.shape == (256,), (
            f"q_norm.weight.shape expected (256,), got {attn.q_norm.weight.shape}"
        )
        assert attn.k_norm.weight.shape == (256,), (
            f"k_norm.weight.shape expected (256,), got {attn.k_norm.weight.shape}"
        )

    def test_init_scaling_value(self, cfg):
        """scaling must equal head_dim**-0.5 = 256**-0.5."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        expected = 256 ** -0.5
        assert abs(attn.scaling - expected) < 1e-9, (
            f"scaling expected {expected}, got {attn.scaling}"
        )

    def test_init_num_kv_groups(self, cfg):
        """num_key_value_groups must equal num_q_heads / num_kv_heads = 8/4 = 2."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        assert attn.num_key_value_groups == 2, (
            f"num_key_value_groups expected 2, got {attn.num_key_value_groups}"
        )

    def test_init_is_causal_false_and_no_dropout(self, cfg):
        """is_causal must be False and attention_dropout must be 0 (encoder)."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        assert attn.is_causal is False, (
            f"is_causal expected False, got {attn.is_causal}"
        )
        assert attn.attention_dropout == 0, (
            f"attention_dropout expected 0, got {attn.attention_dropout}"
        )

    def test_init_head_dim(self, cfg):
        """head_dim attribute must be 256."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        assert attn.head_dim == 256, (
            f"head_dim expected 256, got {attn.head_dim}"
        )


# ---------------------------------------------------------------------------
# 5. T5Gemma2SelfAttention — operations injection
# ---------------------------------------------------------------------------

class TestT5Gemma2SelfAttentionOps:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_config()

    def test_operations_manual_cast_comfy_cast_weights(self, cfg):
        """
        operations=comfy.ops.manual_cast → all 4 proj layers must have
        comfy_cast_weights == True.
        """
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0, operations=comfy.ops.manual_cast)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer = getattr(attn, name)
            val = getattr(layer, "comfy_cast_weights", False)
            assert val is True, (
                f"{name}.comfy_cast_weights expected True with manual_cast, got {val}"
            )

    def test_operations_none_uses_disable_weight_init(self, cfg):
        """
        operations=None (default) should use comfy.ops.disable_weight_init,
        which wraps Linear but sets comfy_cast_weights=False.
        """
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0, operations=None)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer = getattr(attn, name)
            # disable_weight_init Linear is NOT raw nn.Linear
            import torch.nn as nn
            # It should be a comfy wrapped type (subclass), not the base nn.Linear class
            assert type(layer) is not nn.Linear, (
                f"{name} should be comfy ops wrapper, not raw nn.Linear"
            )
            # comfy_cast_weights should be False (not True) for disable_weight_init
            val = getattr(layer, "comfy_cast_weights", None)
            assert val is not True, (
                f"{name}.comfy_cast_weights should not be True with disable_weight_init, got {val}"
            )


# ---------------------------------------------------------------------------
# 6. T5Gemma2SelfAttention — forward
# ---------------------------------------------------------------------------

def _make_dummy_rope(B: int, S: int, D: int = 256):
    """Return (cos, sin) that act as identity rotation (cos=1, sin=0)."""
    cos = torch.ones(B, S, D)
    sin = torch.zeros(B, S, D)
    return cos, sin


class TestT5Gemma2SelfAttentionForward:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_config()

    def test_forward_smoke(self, cfg):
        """Basic smoke: (B=1, S=16, D=2560) input → same shape output."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        attn.eval()
        B, S, D = 1, 16, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _make_dummy_rope(B, S)
        with torch.no_grad():
            out = attn(hidden, position_embeddings=(cos, sin))
        assert out.shape == (B, S, D), (
            f"forward output shape expected {(B, S, D)}, got {out.shape}"
        )

    def test_forward_with_dummy_rope(self, cfg):
        """Forward with identity RoPE (cos=ones, sin=zeros) must complete without error."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        attn.eval()
        B, S, D = 2, 8, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _make_dummy_rope(B, S)
        with torch.no_grad():
            out = attn(hidden, position_embeddings=(cos, sin))
        assert out is not None
        assert not torch.isnan(out).any(), "NaN in forward output with identity RoPE"
        assert not torch.isinf(out).any(), "Inf in forward output with identity RoPE"

    def test_forward_dtype_preserved_bfloat16(self, cfg):
        """bfloat16 input → bfloat16 output."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        attn = attn.to(torch.bfloat16)
        attn.eval()
        B, S, D = 1, 16, 2560
        hidden = torch.randn(B, S, D, dtype=torch.bfloat16)
        cos = torch.ones(B, S, 256, dtype=torch.bfloat16)
        sin = torch.zeros(B, S, 256, dtype=torch.bfloat16)
        with torch.no_grad():
            out = attn(hidden, position_embeddings=(cos, sin))
        assert out.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got {out.dtype}"
        )

    def test_attention_mask_passthrough(self, cfg):
        """Additive attention mask (B, 1, S, S) must be accepted without error."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        attn.eval()
        B, S, D = 1, 8, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _make_dummy_rope(B, S)
        # Additive mask: zeros (no masking) with shape (B, 1, S, S)
        mask = torch.zeros(B, 1, S, S)
        with torch.no_grad():
            out = attn(hidden, position_embeddings=(cos, sin), attention_mask=mask)
        assert out.shape == (B, S, D), (
            f"With attention mask: output shape expected {(B, S, D)}, got {out.shape}"
        )

    def test_forward_full_attention_layer(self, cfg):
        """Full-attention layer (layer_idx=5) forward must also produce correct shape."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=5)
        attn.eval()
        B, S, D = 1, 16, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _make_dummy_rope(B, S)
        with torch.no_grad():
            out = attn(hidden, position_embeddings=(cos, sin))
        assert out.shape == (B, S, D)

    def test_forward_batch_2(self, cfg):
        """Batch size > 1 must work correctly."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        attn.eval()
        B, S, D = 3, 12, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _make_dummy_rope(B, S)
        with torch.no_grad():
            out = attn(hidden, position_embeddings=(cos, sin))
        assert out.shape == (B, S, D)

    def test_forward_no_nan_for_random_input(self, cfg):
        """Random float32 input must not produce NaN/Inf."""
        attn = T5Gemma2SelfAttention(cfg, layer_idx=0)
        attn.eval()
        B, S, D = 2, 16, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _make_dummy_rope(B, S)
        with torch.no_grad():
            out = attn(hidden, position_embeddings=(cos, sin))
        assert not torch.isnan(out).any(), "NaN detected in forward output"
        assert not torch.isinf(out).any(), "Inf detected in forward output"
