"""P2.2: MotifVideoAttention.forward (SDPA 고정) — golden fixture 비교.

CUDA-free design: comfy.ops is replaced with a lightweight mock so the test
suite can run without a GPU. The production load path always has CUDA available
and uses real comfy.ops — only the test environment needs this workaround.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import pathlib
import sys
import types

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BASE = _REPO_ROOT / "models" / "transformer"
_COMFYUI_ROOT = _REPO_ROOT.parent.parent

if str(_COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMFYUI_ROOT))

# ---------------------------------------------------------------------------
# comfy.ops mock
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
# Load attention module directly (avoids models/__init__.py CUDA init)
# ---------------------------------------------------------------------------
_ATTENTION_PATH = _BASE / "attention.py"
_spec = _ilu.spec_from_file_location("attention", _ATTENTION_PATH)
_attention_mod = _ilu.module_from_spec(_spec)
_attention_mod._DEFAULT_OPS = None
_spec.loader.exec_module(_attention_mod)

MotifVideoAttention = _attention_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Expected keys and fixture paths
# ---------------------------------------------------------------------------
EXPECTED = json.loads(
    (_REPO_ROOT / "tests" / "transformer" / "expected_attn_keys.json").read_text()
)
FIX = _REPO_ROOT / "tests" / "transformer" / "fixtures"


def _dims():
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]
    return hidden // head_dim, head_dim, hidden


def _inputs(hidden, seed=42):
    g = torch.Generator().manual_seed(seed)
    B, L, E = 2, 16, 8
    return torch.randn(B, L, hidden, generator=g), torch.randn(B, E, hidden, generator=g)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["single", "dual"])
def test_forward_matches_golden(kind):
    num_heads, head_dim, hidden = _dims()
    torch.manual_seed(0)  # must match generate_attention_golden.py seeding
    if kind == "single":
        attn = MotifVideoAttention(num_heads, head_dim, pre_only=True, added_kv=False, qk_norm="rms_norm")
    else:
        attn = MotifVideoAttention(num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm")
    # no load_state_dict — deterministic init reproduces the fixture parameters

    hs, eh = _inputs(hidden)
    with torch.no_grad():
        out, ctx = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)

    ref_out = torch.load(FIX / f"attention_golden_{kind}_out.pt", weights_only=True)
    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-4)

    # ctx:
    #   single (pre_only=True): processor returns (hidden_states, sliced_encoder).
    #   dual: (hidden_states, encoder_out_via_to_add_out).
    ctx_path = FIX / f"attention_golden_{kind}_ctx.pt"
    if ctx_path.exists():
        ref_ctx = torch.load(ctx_path, weights_only=True)
        torch.testing.assert_close(ctx, ref_ctx, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kind", ["single", "dual"])
def test_forward_no_nan(kind):
    num_heads, head_dim, hidden = _dims()
    if kind == "single":
        attn = MotifVideoAttention(num_heads, head_dim, pre_only=True, added_kv=False, qk_norm="rms_norm")
    else:
        attn = MotifVideoAttention(num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm")
    hs, eh = _inputs(hidden)
    with torch.no_grad():
        out, ctx = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    if ctx is not None:
        assert not torch.isnan(ctx).any()
        assert not torch.isinf(ctx).any()
