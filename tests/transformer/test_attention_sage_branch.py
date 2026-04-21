"""P2.3: MotifVideoAttention sage branch — spy tests.

Patches dispatch_optimized_attention and F.scaled_dot_product_attention to verify
which path is taken under 5 scenarios.

The attention module is loaded directly via spec_from_file_location (same pattern
as test_attention_forward_sdpa.py). In that path the relative import
``from ..sage_ops import dispatch_optimized_attention`` falls back to None
(try/except in attention.py). Tests inject a real callable via module-level
attribute patching before each test.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import pathlib
import sys
import types
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BASE = _REPO_ROOT / "models" / "transformer"
_COMFYUI_ROOT = _REPO_ROOT.parent.parent

if str(_COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMFYUI_ROOT))

# ---------------------------------------------------------------------------
# comfy.ops mock (identical to test_attention_forward_sdpa.py)
# ---------------------------------------------------------------------------

def _make_comfy_ops_mock():
    class _MockOps:
        class Linear(nn.Linear):
            def __init__(self, *args, dtype=None, device=None, **kwargs):
                super().__init__(*args, **kwargs)

        class RMSNorm(nn.RMSNorm):
            def __init__(self, normalized_shape, eps=None, dtype=None, device=None, **kwargs):
                super().__init__(normalized_shape, eps=eps or 1e-6)

    mock_comfy = types.ModuleType("comfy")
    mock_ops = types.ModuleType("comfy.ops")
    mock_ops.disable_weight_init = _MockOps
    mock_comfy.ops = mock_ops
    return mock_comfy, mock_ops


def _should_use_real_comfy() -> bool:
    if os.environ.get("MOTIF_FORCE_MOCK_COMFY") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import comfy.ops  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def comfy_mock_fixture():
    if _should_use_real_comfy():
        yield
        return

    _prev_comfy = sys.modules.get("comfy", None)
    _prev_comfy_ops = sys.modules.get("comfy.ops", None)

    mock_comfy, mock_ops = _make_comfy_ops_mock()
    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.ops"] = mock_ops

    yield

    if _prev_comfy is None:
        sys.modules.pop("comfy", None)
    else:
        sys.modules["comfy"] = _prev_comfy

    if _prev_comfy_ops is None:
        sys.modules.pop("comfy.ops", None)
    else:
        sys.modules["comfy.ops"] = _prev_comfy_ops


# ---------------------------------------------------------------------------
# Load attention module directly
# ---------------------------------------------------------------------------
_ATTENTION_PATH = _BASE / "attention.py"
_spec = _ilu.spec_from_file_location("attention_sage_test", _ATTENTION_PATH)
_attn_mod = _ilu.module_from_spec(_spec)
_attn_mod._DEFAULT_OPS = None
_spec.loader.exec_module(_attn_mod)

MotifVideoAttention = _attn_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Dimensions from fixture
# ---------------------------------------------------------------------------
EXPECTED = json.loads(
    (_REPO_ROOT / "tests" / "transformer" / "expected_attn_keys.json").read_text()
)


def _dims():
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]
    return hidden // head_dim, head_dim, hidden


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dual(num_heads, head_dim):
    torch.manual_seed(0)
    return MotifVideoAttention(
        num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm"
    )


def _single(num_heads, head_dim):
    torch.manual_seed(0)
    return MotifVideoAttention(
        num_heads, head_dim, pre_only=True, added_kv=False, qk_norm="rms_norm"
    )


def _sdpa_spy_return(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Minimal SDPA stub: returns zero tensor of correct shape [B, H, S, D]."""
    return torch.zeros(
        query.shape[0], query.shape[1], query.shape[2], query.shape[3],
        dtype=query.dtype,
    )


def _dispatch_spy_return(query, key, value, attention_mask):
    """Minimal dispatch stub: returns zero tensor of correct shape [B, H, S, D]."""
    return torch.zeros(
        query.shape[0], query.shape[1], query.shape[2], query.shape[3],
        dtype=query.dtype,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sage_positive_joint_path():
    """use_sage=True + dual block + bool mask [B,1,1,L+E] → dispatch called once, SDPA 0."""
    num_heads, head_dim, hidden = _dims()
    attn = _dual(num_heads, head_dim)
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    dispatch_mock = mock.MagicMock(side_effect=_dispatch_spy_return)
    sdpa_mock = mock.MagicMock(side_effect=_sdpa_spy_return)

    _original_dispatch = _attn_mod.dispatch_optimized_attention
    _attn_mod.dispatch_optimized_attention = dispatch_mock
    try:
        with mock.patch.object(_attn_mod.F, "scaled_dot_product_attention", sdpa_mock):
            with torch.no_grad():
                attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)
    finally:
        _attn_mod.dispatch_optimized_attention = _original_dispatch

    assert dispatch_mock.call_count == 1, f"expected dispatch called 1 time, got {dispatch_mock.call_count}"
    assert sdpa_mock.call_count == 0, f"expected SDPA called 0 times, got {sdpa_mock.call_count}"


def test_sage_negative_use_sage_false():
    """use_sage=False → SDPA called once, dispatch 0 (even with joint path + bool mask)."""
    num_heads, head_dim, hidden = _dims()
    attn = _dual(num_heads, head_dim)
    attn.use_sage = False  # default, explicit here for clarity

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    dispatch_mock = mock.MagicMock(side_effect=_dispatch_spy_return)
    sdpa_mock = mock.MagicMock(side_effect=_sdpa_spy_return)

    _original_dispatch = _attn_mod.dispatch_optimized_attention
    _attn_mod.dispatch_optimized_attention = dispatch_mock
    try:
        with mock.patch.object(_attn_mod.F, "scaled_dot_product_attention", sdpa_mock):
            with torch.no_grad():
                attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)
    finally:
        _attn_mod.dispatch_optimized_attention = _original_dispatch

    assert sdpa_mock.call_count == 1, f"expected SDPA called 1 time, got {sdpa_mock.call_count}"
    assert dispatch_mock.call_count == 0, f"expected dispatch called 0 times, got {dispatch_mock.call_count}"


def test_sage_negative_cross_attn():
    """use_sage=True + query_input is not None → cross-attn always uses SDPA."""
    num_heads, head_dim, hidden = _dims()
    attn = _dual(num_heads, head_dim)
    attn.use_sage = True

    B, L, L_kv = 2, 16, 10
    q = torch.randn(B, L, hidden)
    kv = torch.randn(B, L_kv, hidden)

    dispatch_mock = mock.MagicMock(side_effect=_dispatch_spy_return)
    sdpa_mock = mock.MagicMock(side_effect=_sdpa_spy_return)

    _original_dispatch = _attn_mod.dispatch_optimized_attention
    _attn_mod.dispatch_optimized_attention = dispatch_mock
    try:
        with mock.patch.object(_attn_mod.F, "scaled_dot_product_attention", sdpa_mock):
            with torch.no_grad():
                attn(hidden_states=q, query_input=q, key_input=kv, value_input=kv)
    finally:
        _attn_mod.dispatch_optimized_attention = _original_dispatch

    assert sdpa_mock.call_count == 1, f"expected SDPA called 1 time, got {sdpa_mock.call_count}"
    assert dispatch_mock.call_count == 0, f"expected dispatch called 0 times, got {dispatch_mock.call_count}"


def test_sage_positive_single_block_joint_path():
    """P2.3 fix: single block (pre_only=True, add_q_proj=None) + use_sage=True + encoder + bool mask → dispatch 1회."""
    num_heads, head_dim, hidden = _dims()
    attn = _single(num_heads, head_dim)
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    dispatch_mock = mock.MagicMock(side_effect=_dispatch_spy_return)
    sdpa_mock = mock.MagicMock(side_effect=_sdpa_spy_return)

    _original_dispatch = _attn_mod.dispatch_optimized_attention
    _attn_mod.dispatch_optimized_attention = dispatch_mock
    try:
        with mock.patch.object(_attn_mod.F, "scaled_dot_product_attention", sdpa_mock):
            with torch.no_grad():
                attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)
    finally:
        _attn_mod.dispatch_optimized_attention = _original_dispatch

    assert dispatch_mock.call_count == 1, f"single block joint path: dispatch 기대 1, 실제 {dispatch_mock.call_count}"
    assert sdpa_mock.call_count == 0, f"single block joint path: SDPA 기대 0, 실제 {sdpa_mock.call_count}"


def test_sage_negative_self_attn_no_encoder():
    """use_sage=True + encoder_hidden_states=None + single block → SDPA (joint concat never executed)."""
    num_heads, head_dim, hidden = _dims()
    attn = _single(num_heads, head_dim)
    attn.use_sage = True

    B, L = 2, 16
    hs = torch.randn(B, L, hidden)

    dispatch_mock = mock.MagicMock(side_effect=_dispatch_spy_return)
    sdpa_mock = mock.MagicMock(side_effect=_sdpa_spy_return)

    _original_dispatch = _attn_mod.dispatch_optimized_attention
    _attn_mod.dispatch_optimized_attention = dispatch_mock
    try:
        with mock.patch.object(_attn_mod.F, "scaled_dot_product_attention", sdpa_mock):
            with torch.no_grad():
                attn(hidden_states=hs, encoder_hidden_states=None, attention_mask=None)
    finally:
        _attn_mod.dispatch_optimized_attention = _original_dispatch

    assert sdpa_mock.call_count == 1, f"expected SDPA called 1 time, got {sdpa_mock.call_count}"
    assert dispatch_mock.call_count == 0, f"expected dispatch called 0 times, got {dispatch_mock.call_count}"


def test_sage_negative_mask_shape_violation():
    """use_sage=True + joint path + bad mask ([B, S] or float additive) → SDPA forced."""
    num_heads, head_dim, hidden = _dims()

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)

    for bad_mask in [
        # [B, S] length-only mask
        torch.ones(B, L + E, dtype=torch.bool),
        # float additive mask (wrong dtype)
        torch.zeros(B, 1, 1, L + E, dtype=torch.float32),
    ]:
        attn = _dual(num_heads, head_dim)
        attn.use_sage = True

        dispatch_mock = mock.MagicMock(side_effect=_dispatch_spy_return)
        sdpa_mock = mock.MagicMock(side_effect=_sdpa_spy_return)

        _original_dispatch = _attn_mod.dispatch_optimized_attention
        _attn_mod.dispatch_optimized_attention = dispatch_mock
        try:
            with mock.patch.object(_attn_mod.F, "scaled_dot_product_attention", sdpa_mock):
                with torch.no_grad():
                    attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=bad_mask)
        finally:
            _attn_mod.dispatch_optimized_attention = _original_dispatch

        assert sdpa_mock.call_count == 1, (
            f"bad_mask shape={bad_mask.shape} dtype={bad_mask.dtype}: "
            f"expected SDPA 1, got {sdpa_mock.call_count}"
        )
        assert dispatch_mock.call_count == 0, (
            f"bad_mask shape={bad_mask.shape} dtype={bad_mask.dtype}: "
            f"expected dispatch 0, got {dispatch_mock.call_count}"
        )
