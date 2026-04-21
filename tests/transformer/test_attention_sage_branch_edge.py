"""P2.3 sage 분기 엣지 케이스 — 기존 5개 시나리오 gap 보완.

기존 test_attention_sage_branch.py 가 커버하지 않는 갭:
  1.  단일 블록 + encoder=None + use_sage=True → SDPA (add_q_proj is None 조건 위반)
  2.  joint path + float dtype mask ([B,1,1,L+E] 형태지만 bool 아님) → SDPA (dtype 위반)
  3.  joint path + bool mask 이지만 shape [B,1,L,L+E] (3번째 차원 확장) → SDPA (shape 위반)
  4.  encoder 존재 + dual block + use_sage=True 이지만 mask.shape[-1] != L+E → SDPA
  5.  sage fallback 수치 동등성: dispatch_optimized_attention(_SAGE_AVAILABLE=False) vs 직접 SDPA
  6.  use_sage 기본값 False 시 dispatch 미호출
  7.  sage 분기 후 to_out/to_add_out 공통 로직 적용됨 (output shape 검증)
  8.  cross-attn 결정성: seed 고정 두 번 실행 → 동일 output
  9.  use_sage 런타임 동적 토글: False→True→False
 10.  sage fallback dtype 보존: float16 입력 시 dispatch 반환 dtype 유지
 11.  encoder 있지만 add_q_proj is None (single block) + use_sage=True → SDPA
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
# comfy.ops mock (CUDA-free, 기존 파일과 동일 패턴)
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
# Load attention module (별도 네임스페이스로 로드 — 기존 테스트와 독립)
# ---------------------------------------------------------------------------
_ATTENTION_PATH = _BASE / "attention.py"
_spec = _ilu.spec_from_file_location("attention_sage_edge_test", _ATTENTION_PATH)
_attn_mod = _ilu.module_from_spec(_spec)
_attn_mod._DEFAULT_OPS = None
_spec.loader.exec_module(_attn_mod)

MotifVideoAttention = _attn_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Dimensions
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

def _dual(seed=0):
    num_heads, head_dim, _ = _dims()
    torch.manual_seed(seed)
    return MotifVideoAttention(
        num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm"
    )


def _single(seed=0):
    num_heads, head_dim, _ = _dims()
    torch.manual_seed(seed)
    return MotifVideoAttention(
        num_heads, head_dim, pre_only=True, added_kv=False, qk_norm="rms_norm"
    )


def _sdpa_stub(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Shape-correct zero stub for SDPA spy."""
    return torch.zeros(
        query.shape[0], query.shape[1], query.shape[2], query.shape[3],
        dtype=query.dtype,
    )


def _dispatch_stub(query, key, value, attention_mask):
    """Shape-correct zero stub for dispatch spy."""
    return torch.zeros(
        query.shape[0], query.shape[1], query.shape[2], query.shape[3],
        dtype=query.dtype,
    )


def _run_with_spies(attn, *, hidden_states, encoder_hidden_states=None,
                   attention_mask=None, query_input=None,
                   key_input=None, value_input=None):
    """Run forward with dispatch + SDPA spies. Returns (dispatch_count, sdpa_count)."""
    dispatch_mock = mock.MagicMock(side_effect=_dispatch_stub)
    sdpa_mock = mock.MagicMock(side_effect=_sdpa_stub)
    _original_dispatch = _attn_mod.dispatch_optimized_attention
    _attn_mod.dispatch_optimized_attention = dispatch_mock
    try:
        with mock.patch.object(_attn_mod.F, "scaled_dot_product_attention", sdpa_mock):
            with torch.no_grad():
                attn(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    query_input=query_input,
                    key_input=key_input,
                    value_input=value_input,
                )
    finally:
        _attn_mod.dispatch_optimized_attention = _original_dispatch
    return dispatch_mock.call_count, sdpa_mock.call_count


# ---------------------------------------------------------------------------
# 포인트 1: single block + use_sage=True + encoder=None → SDPA
# (add_q_proj is None 조건 미충족 → sage 진입 금지)
# ---------------------------------------------------------------------------

def test_sage_single_block_no_encoder_uses_sdpa():
    """use_sage=True + single block (add_q_proj is None) + encoder=None → SDPA, dispatch=0."""
    _, _, hidden = _dims()
    attn = _single()
    attn.use_sage = True

    B, L = 2, 16
    hs = torch.randn(B, L, hidden)
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)  # shape 맞아도 add_q_proj 없으면 안 됨

    d_count, s_count = _run_with_spies(attn, hidden_states=hs, attention_mask=mask)

    assert d_count == 0, f"single block: dispatch must not be called, got {d_count}"
    assert s_count == 1, f"single block: SDPA must be called once, got {s_count}"


# ---------------------------------------------------------------------------
# 포인트 2: joint path + float mask (dtype 위반) → SDPA
# 기존 테스트가 [B,S] 와 float32 [B,1,1,L+E] 를 함께 묶었으나,
# 이 테스트는 float64 와 bfloat16 dtype 도 각각 검증.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_sage_joint_path_float_mask_dtype_violation_uses_sdpa(dtype):
    """use_sage=True + joint path + float mask (dtype 위반) → SDPA 강제."""
    _, _, hidden = _dims()
    attn = _dual()
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    # [B,1,1,L+E] 형태지만 bool이 아닌 float dtype
    mask = torch.zeros(B, 1, 1, L + E, dtype=dtype)

    d_count, s_count = _run_with_spies(
        attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask
    )

    assert d_count == 0, (
        f"float {dtype} mask: dispatch must be 0, got {d_count}"
    )
    assert s_count == 1, (
        f"float {dtype} mask: SDPA must be 1, got {s_count}"
    )


# ---------------------------------------------------------------------------
# 포인트 3: joint path + bool mask 이지만 shape [B,1,L,L+E] → SDPA
# 3번째 차원이 1이 아니라 L로 확장된 케이스
# ---------------------------------------------------------------------------

def test_sage_joint_path_bool_mask_wrong_shape_expanded_dim_uses_sdpa():
    """use_sage=True + joint path + bool mask [B,1,L,L+E] (shape 위반) → SDPA."""
    _, _, hidden = _dims()
    attn = _dual()
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    # 올바른 shape는 [B,1,1,L+E], 이 케이스는 dim[2]=L (확장됨)
    mask = torch.ones(B, 1, L, L + E, dtype=torch.bool)

    d_count, s_count = _run_with_spies(
        attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask
    )

    assert d_count == 0, (
        f"mask shape [B,1,L,L+E]: dispatch must be 0, got {d_count}"
    )
    assert s_count == 1, (
        f"mask shape [B,1,L,L+E]: SDPA must be 1, got {s_count}"
    )


# ---------------------------------------------------------------------------
# 포인트 4: joint path + bool mask 이지만 shape[-1] != L+E → SDPA
# ---------------------------------------------------------------------------

def test_sage_joint_path_bool_mask_wrong_last_dim_uses_sdpa():
    """use_sage=True + joint path + bool mask shape[-1] != L+E → SDPA."""
    _, _, hidden = _dims()
    attn = _dual()
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    # 마지막 dim 이 L+E 가 아님 (L 만)
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)

    d_count, s_count = _run_with_spies(
        attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask
    )

    assert d_count == 0, (
        f"mask last dim={L} (not L+E={L+E}): dispatch must be 0, got {d_count}"
    )
    assert s_count == 1, (
        f"mask last dim={L} (not L+E={L+E}): SDPA must be 1, got {s_count}"
    )


# ---------------------------------------------------------------------------
# 포인트 5: sage fallback 수치 동등성
# _SAGE_AVAILABLE=False 환경에서 dispatch_optimized_attention 이 반환하는 값이
# 직접 SDPA 호출과 수치적으로 동등한지 검증.
# (sage_ops.py fallback: F.scaled_dot_product_attention(...).contiguous())
# ---------------------------------------------------------------------------

def test_sage_fallback_numeric_equivalence_with_direct_sdpa(monkeypatch):
    """sage 미설치 환경에서 dispatch fallback == 직접 SDPA (atol=1e-5).

    이 테스트는 명시적으로 SDPA fallback 경로만 검증한다.
    _SAGE_AVAILABLE=False 를 강제하여 sageattention 설치 여부와 무관하게
    fallback 코드 경로가 올바른 수치를 반환하는지 검증한다.
    """
    # models/__init__.py 가 comfy 패키지를 요구하므로 직접 파일 로드로 회피
    _sage_spec = _ilu.spec_from_file_location(
        "sage_ops_numeric", _REPO_ROOT / "models" / "sage_ops.py"
    )
    _sage_mod = _ilu.module_from_spec(_sage_spec)
    _sage_spec.loader.exec_module(_sage_mod)
    # _SAGE_AVAILABLE=False 를 강제하여 sage 설치 여부와 무관하게 fallback 경로만 실행
    monkeypatch.setattr(_sage_mod, "_SAGE_AVAILABLE", False)
    dispatch_optimized_attention = _sage_mod.dispatch_optimized_attention

    num_heads, head_dim, _ = _dims()
    B, L, E = 2, 8, 4

    torch.manual_seed(42)
    # [B, H, S, D] 형태 — dispatch 는 이 형태를 받음
    query = torch.randn(B, num_heads, L + E, head_dim)
    key = torch.randn(B, num_heads, L + E, head_dim)
    value = torch.randn(B, num_heads, L + E, head_dim)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    with torch.no_grad():
        dispatch_out = dispatch_optimized_attention(query, key, value, mask)
        sdpa_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False
        ).contiguous()

    torch.testing.assert_close(dispatch_out, sdpa_out, atol=1e-5, rtol=1e-4), (
        "sage fallback (dispatch) output differs from direct SDPA by more than atol=1e-5"
    )


# ---------------------------------------------------------------------------
# 포인트 6: use_sage 기본값 False → dispatch 미호출
# 명시적 설정 없이 새 인스턴스 생성 시
# ---------------------------------------------------------------------------

def test_sage_default_false_dispatch_not_called():
    """새 인스턴스 기본값 use_sage=False → dispatch 0회."""
    _, _, hidden = _dims()
    attn = _dual()
    # use_sage 를 명시적으로 설정하지 않음

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    d_count, s_count = _run_with_spies(
        attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask
    )

    assert d_count == 0, (
        f"default use_sage: dispatch should not be called (got {d_count}). "
        "use_sage 기본값이 False 여야 함."
    )
    assert s_count == 1, f"default use_sage: SDPA should be called once (got {s_count})"


# ---------------------------------------------------------------------------
# 포인트 7: sage 분기 후 to_out/to_add_out 공통 로직 적용 검증
# dispatch 가 진입되어도 (B, L_total, hidden) shape 반환이 보장돼야 함
# ---------------------------------------------------------------------------

def test_sage_branch_output_shape_after_to_out():
    """sage 분기 진입 후에도 to_out/to_add_out 을 거친 최종 shape 가 올바름."""
    _, _, hidden = _dims()
    attn = _dual()
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    # spy 없이 실제 dispatch (sage fallback) 경로로 실행
    with torch.no_grad():
        out, ctx = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)

    assert out.shape == (B, L, hidden), (
        f"sage 분기 후 out.shape 기대 ({B},{L},{hidden}), 실제 {tuple(out.shape)}"
    )
    assert ctx is not None, "dual block sage 분기 후 ctx (두번째 반환) 가 None 이면 안 됨"
    assert ctx.shape == (B, E, hidden), (
        f"sage 분기 후 ctx.shape 기대 ({B},{E},{hidden}), 실제 {tuple(ctx.shape)}"
    )


def test_sage_branch_output_no_nan_inf():
    """sage 분기 진입 후 output 에 NaN/Inf 없음."""
    _, _, hidden = _dims()
    attn = _dual()
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    with torch.no_grad():
        out, ctx = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)

    assert not torch.isnan(out).any(), "sage 분기 출력에 NaN 발생"
    assert not torch.isinf(out).any(), "sage 분기 출력에 Inf 발생"
    if ctx is not None:
        assert not torch.isnan(ctx).any(), "sage 분기 ctx 에 NaN 발생"
        assert not torch.isinf(ctx).any(), "sage 분기 ctx 에 Inf 발생"


# ---------------------------------------------------------------------------
# 포인트 8: cross-attn 결정성 — seed 고정 두 번 실행 → 동일 output
# ---------------------------------------------------------------------------

def test_cross_attn_determinism_same_seed():
    """torch.manual_seed(0) 두 번 → cross-attn output 완전 일치."""
    num_heads, head_dim, hidden = _dims()
    B, L, L_kv = 2, 16, 10

    def _run():
        torch.manual_seed(0)
        attn = MotifVideoAttention(
            num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm"
        )
        g = torch.Generator().manual_seed(0)
        q = torch.randn(B, L, hidden, generator=g)
        kv = torch.randn(B, L_kv, hidden, generator=g)
        with torch.no_grad():
            out, second = attn(hidden_states=q, query_input=q, key_input=kv, value_input=kv)
        return out

    out1 = _run()
    out2 = _run()

    torch.testing.assert_close(out1, out2, atol=0, rtol=0), (
        "cross-attn: seed 고정 두 번 실행 결과가 다름 (비결정적)"
    )


# ---------------------------------------------------------------------------
# 포인트 9: use_sage 런타임 동적 토글
# False(기본) → True 설정 후 dispatch 진입 확인 → 다시 False 로 set 후 미진입 확인
# ---------------------------------------------------------------------------

def test_sage_runtime_toggle_false_then_true_then_false():
    """런타임 use_sage 토글: False→True (dispatch 호출) → False (dispatch 미호출)."""
    _, _, hidden = _dims()
    attn = _dual()

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    # 1단계: 기본 False → dispatch=0
    d1, s1 = _run_with_spies(attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)
    assert d1 == 0, f"toggle 1단계(False): dispatch 기대 0, 실제 {d1}"
    assert s1 == 1, f"toggle 1단계(False): SDPA 기대 1, 실제 {s1}"

    # 2단계: True 로 토글 → dispatch=1
    attn.use_sage = True
    d2, s2 = _run_with_spies(attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)
    assert d2 == 1, f"toggle 2단계(True): dispatch 기대 1, 실제 {d2}"
    assert s2 == 0, f"toggle 2단계(True): SDPA 기대 0, 실제 {s2}"

    # 3단계: False 로 되돌림 → dispatch=0 복귀
    attn.use_sage = False
    d3, s3 = _run_with_spies(attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)
    assert d3 == 0, f"toggle 3단계(False 복귀): dispatch 기대 0, 실제 {d3}"
    assert s3 == 1, f"toggle 3단계(False 복귀): SDPA 기대 1, 실제 {s3}"


# ---------------------------------------------------------------------------
# 포인트 10: sage fallback dtype 보존 — float16 입력 시
# dispatch 반환 tensor 의 dtype 이 query.dtype 과 일치해야 함
# ---------------------------------------------------------------------------

def test_sage_branch_dtype_preserved_fp16():
    """sage 분기 진입 후 output dtype 이 fp16 입력과 일치 (또는 후처리로 강제됨)."""
    _, _, hidden = _dims()
    attn = _dual().to(torch.float16)
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden, dtype=torch.float16)
    eh = torch.randn(B, E, hidden, dtype=torch.float16)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    with torch.no_grad():
        out, ctx = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask)

    assert out.dtype == torch.float16, (
        f"fp16 입력 sage 분기: 기대 dtype=float16, 실제 {out.dtype}"
    )
    if ctx is not None:
        assert ctx.dtype == torch.float16, (
            f"fp16 입력 sage 분기: ctx dtype 기대 float16, 실제 {ctx.dtype}"
        )


# ---------------------------------------------------------------------------
# 포인트 11: encoder 있지만 single block (add_q_proj is None) + use_sage=True → SDPA
# docstring 3조건: joint concat 이후 조건 위반
# ---------------------------------------------------------------------------

def test_sage_encoder_present_but_single_block_uses_sdpa():
    """encoder_hidden_states 있어도 single block (add_q_proj None) + use_sage=True → SDPA."""
    _, _, hidden = _dims()
    # single block 은 added_kv=False → add_q_proj 없음
    attn = _single()
    attn.use_sage = True

    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden)
    eh = torch.randn(B, E, hidden)
    mask = torch.ones(B, 1, 1, L + E, dtype=torch.bool)

    d_count, s_count = _run_with_spies(
        attn, hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask
    )

    assert d_count == 0, (
        f"single block + encoder: dispatch 금지인데 {d_count}회 호출됨. "
        "add_q_proj is None 인 경로에서 sage 분기 진입 → silent correctness bug"
    )
    assert s_count == 1, (
        f"single block + encoder: SDPA 기대 1, 실제 {s_count}"
    )
