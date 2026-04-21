"""P2.1: 외부 attribute contract (R5 대응).

CUDA-free design: comfy.ops is replaced with a lightweight mock.
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
_ATTENTION_PATH = _REPO_ROOT / "models" / "transformer" / "attention.py"
_spec = _ilu.spec_from_file_location("attention", _ATTENTION_PATH)
_attention_mod = _ilu.module_from_spec(_spec)
_attention_mod._DEFAULT_OPS = None
_spec.loader.exec_module(_attention_mod)

MotifVideoAttention = _attention_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Expected keys
# ---------------------------------------------------------------------------
EXPECTED = json.loads(
    (_REPO_ROOT / "tests" / "transformer" / "expected_attn_keys.json").read_text()
)


def _dims():
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]      # 1536
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]  # 128
    return hidden // head_dim, head_dim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_heads_exposed():
    num_heads, head_dim = _dims()
    s = MotifVideoAttention(num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False)
    d = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    assert s.heads == num_heads
    assert d.heads == num_heads


def test_single_contract():
    num_heads, head_dim = _dims()
    s = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False
    )
    assert s.add_q_proj is None and s.add_k_proj is None and s.add_v_proj is None
    assert s.to_add_out is None
    assert s.norm_added_q is None and s.norm_added_k is None
    assert s.to_out is None, "pre_only=True 일 때 to_out 은 None (diffusers 원본 계약)"


def test_dual_contract():
    num_heads, head_dim = _dims()
    d = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    assert d.add_q_proj is not None and d.to_add_out is not None
    assert isinstance(d.to_out, nn.ModuleList)
    # to_out[0] 은 ops.Linear (nn.Linear 서브클래스), to_out[1] 은 Dropout
    assert d.to_out[0] is not None
    assert isinstance(d.to_out[1], nn.Dropout)


def test_qk_norm_rms():
    num_heads, head_dim = _dims()
    a = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    assert a.norm_q is not None and a.norm_k is not None
    assert a.norm_added_q is not None and a.norm_added_k is not None


def test_qk_norm_none():
    num_heads, head_dim = _dims()
    a = MotifVideoAttention(
        num_heads, head_dim, qk_norm=None, pre_only=False, added_kv=True
    )
    assert a.norm_q is None and a.norm_k is None
    assert a.norm_added_q is None and a.norm_added_k is None
