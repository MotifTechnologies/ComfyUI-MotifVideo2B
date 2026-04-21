"""P2.1: MotifVideoAttention parameter shape 이 expected_attn_keys.json 과 일치.

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

@pytest.mark.parametrize("block_kind, pre_only, added_kv", [
    ("single_block", True, False),
    ("dual_block", False, True),
])
def test_shape_matches(block_kind, pre_only, added_kv):
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=pre_only, added_kv=added_kv
    )
    sd = attn.state_dict()
    for key, meta in EXPECTED[block_kind].items():
        assert list(sd[key].shape) == meta["shape"], (
            f"{block_kind}.{key}: got {list(sd[key].shape)}, want {meta['shape']}"
        )
