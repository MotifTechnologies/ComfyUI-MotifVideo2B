"""P2.3 cross-attn branch (query_input not None) — basic contract.

P2.2 already implements the forward; this file satisfies the checklist P2.3
verify requirement: output shape [B, L, hidden] and second return is None.
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
# Load attention module directly
# ---------------------------------------------------------------------------
_ATTENTION_PATH = _BASE / "attention.py"
_spec = _ilu.spec_from_file_location("attention_cross_test", _ATTENTION_PATH)
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
# Tests
# ---------------------------------------------------------------------------

def test_cross_attn_returns_shape_and_none_second():
    num_heads, head_dim, hidden = _dims()
    torch.manual_seed(0)
    attn = MotifVideoAttention(
        num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm"
    )
    B, L, L_kv = 2, 16, 10
    q = torch.randn(B, L, hidden)
    kv = torch.randn(B, L_kv, hidden)
    with torch.no_grad():
        out, second = attn(hidden_states=q, query_input=q, key_input=kv, value_input=kv)
    assert out.shape == (B, L, hidden), f"got {out.shape}"
    assert second is None, f"cross-attn second return must be None, got {type(second)}"
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
