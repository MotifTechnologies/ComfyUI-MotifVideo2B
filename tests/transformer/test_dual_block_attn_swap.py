"""P3.2: MotifVideoTransformerBlock 의 self.attn 교체 + ops 주입 검증.

CUDA-free design: comfy.ops is replaced with a lightweight mock so the test
suite can run without a GPU. The production load path always has CUDA available
and uses real comfy.ops — only the test environment needs this workaround.

diffusers stub 및 models namespace 사전 주입은 conftest.py 의 session-scoped
autouse fixture (_diffusers_stub_session, _models_namespace_session) 가 담당한다.
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
# comfy.ops mock — mirrors test_single_block_attn_swap.py pattern
# ---------------------------------------------------------------------------

def _make_comfy_ops_mock():
    class _MockOps:
        class Linear(nn.Linear):
            def __init__(self, *args, dtype=None, device=None, **kwargs):
                super().__init__(*args, **kwargs)

        class RMSNorm(nn.RMSNorm):
            def __init__(self, normalized_shape, eps=None, dtype=None, device=None, **kwargs):
                super().__init__(normalized_shape, eps=eps or 1e-6)

        class LayerNorm(nn.LayerNorm):
            def __init__(self, *args, dtype=None, device=None, **kwargs):
                super().__init__(*args, **kwargs)

        class Conv3d(nn.Conv3d):
            def __init__(self, *args, dtype=None, device=None, **kwargs):
                super().__init__(*args, **kwargs)

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
# Load transformer_motif_video directly (avoids models/__init__.py CUDA init)
# ---------------------------------------------------------------------------

def _load_module(name: str, path: pathlib.Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# models / models.transformer namespace 는 conftest._models_namespace_session 이 주입한다.

_ops_mod = _load_module("models.transformer.ops_primitives", _BASE / "ops_primitives.py")
_attn_mod = _load_module("models.transformer.attention", _BASE / "attention.py")
_load_module("models.transformer.tread_mixin", _BASE / "tread_mixin.py")
_load_module("models.transformer.accelerate_patch", _BASE / "accelerate_patch.py")
_tmv_mod = _load_module(
    "models.transformer.transformer_motif_video",
    _BASE / "transformer_motif_video.py",
)

MotifVideoTransformerBlock = _tmv_mod.MotifVideoTransformerBlock
MotifVideoAttention = _attn_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EXPECTED = json.loads((_REPO_ROOT / "tests/transformer/expected_attn_keys.json").read_text())


def _dims():
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]
    return hidden // head_dim, head_dim, hidden


def _make_block():
    num_heads, head_dim, _ = _dims()
    return MotifVideoTransformerBlock(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attn_is_motifvideoattention():
    block = _make_block()
    assert type(block.attn).__name__ == "MotifVideoAttention"
    assert isinstance(block.attn, MotifVideoAttention)


def test_added_kv_contract():
    block = _make_block()
    assert block.attn.add_q_proj is not None
    assert block.attn.to_add_out is not None
    assert block.attn.to_out is not None
    assert isinstance(block.attn.to_out, nn.ModuleList)


def test_self_attn_call_signature():
    """block.attn(hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
    호출이 crash 없이 (out, ctx) 튜플 반환, 두 텐서 shape 일치."""
    block = _make_block()
    num_heads, head_dim, hidden = _dims()
    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    with torch.no_grad():
        out, ctx = block.attn(
            hidden_states=hs,
            encoder_hidden_states=eh,
            attention_mask=None,
            image_rotary_emb=None,
        )
    assert out.shape == (B, L, hidden)
    assert ctx.shape == (B, E, hidden)


def test_ops_injection_consistency():
    """recommender 제안 1: block 내부 ops 주입 동일성.
    Dual block 의 feedforward 내부 첫 proj (ff.net[0].proj) 와 attn.to_q 타입 동일."""
    block = _make_block()
    attn_lin = block.attn.to_q
    # Try common FeedForward paths
    ff_lin = None
    if hasattr(block, "ff") and hasattr(block.ff, "net"):
        # diffusers FeedForward.net = ModuleList([GEGLU(Linear...), Dropout, Linear])
        first = block.ff.net[0]
        if hasattr(first, "proj"):
            ff_lin = first.proj
        elif isinstance(first, nn.Linear):
            ff_lin = first
    if ff_lin is None:
        # fallback: norm2 가 ops.LayerNorm — 같은 ops 네임스페이스 소속 확인
        ff_lin = block.norm2
    assert type(attn_lin) is type(ff_lin), (
        f"attn.to_q type={type(attn_lin).__name__} != ff first layer type={type(ff_lin).__name__}"
    )


def test_state_dict_keys_match_expected():
    """recommender P3.2 제안 2: attn.* state_dict key 가 expected dual_block 와 정확히 일치."""
    block = _make_block()
    attn_state = {
        k[len("attn."):]: v for k, v in block.state_dict().items() if k.startswith("attn.")
    }
    actual = set(attn_state.keys())
    expected = set(EXPECTED["dual_block"].keys())
    assert actual == expected, f"mismatch: missing={expected - actual}, unexpected={actual - expected}"


def test_sage_flag_based_dispatch_in_compile_config():
    """P4.1: apply_sage_attention 이 use_sage=True 방식으로 재작성됨."""
    src = (_REPO_ROOT / "models" / "compile_config.py").read_text()
    # 새 방식
    assert "block.attn.use_sage = True" in src, "P4.1: use_sage 플래그 세팅 누락"
    # 레거시 방식 제거 확인
    assert 'block.attn.processor = xDiTMotifVideoAttnProcessor()' not in src, (
        "P4.1: legacy processor 대입 잔류. 제거 필요"
    )
    # P3.2 hasattr 가드 제거 확인
    assert 'hasattr(block.attn, "processor")' not in src, (
        "P4.1: P3.2 임시 hasattr 가드 잔류. 제거 필요"
    )
