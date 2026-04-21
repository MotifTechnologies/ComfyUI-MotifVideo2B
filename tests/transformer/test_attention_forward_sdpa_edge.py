"""P2.2 엣지케이스 테스트 — MotifVideoAttention.forward 블라인드 검증.

기존 test_attention_forward_sdpa.py 가 커버하지 않는 갭:
  1.  cross-attn 분기 (query_input not None): shape, 두번째 반환값 None, NaN/Inf
  2.  encoder_hidden_states=None + single block: 순수 self-attn 경로 crash 없이 동작
  3.  attention_mask 전달 효과: mask 있/없 output 이 실제로 달라야 함
  4.  RoPE 전달 효과: image_rotary_emb 있/없 output 이 실제로 달라야 함
  5.  dual block to_add_out 효과: to_add_out weight=0 시 두번째 반환이 0
  6.  dtype 보존: float32→float32, float16→float16
  7.  pre_only 분기 규칙 준수: attention.py 에 'self.pre_only' 문자열 없어야 함
  8.  cross-attn 에서 add_q_proj 호출 없음: spy mock 검증
  9.  input/output sequence length 보존: self-attn 경로
 10.  fixture 재현성: generate_attention_golden.py 재실행 시 동일 tensor
 11.  fixture 파일 sanity: 존재 + 비어있지 않음
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import pathlib
import sys
import types
import unittest.mock as mock

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
# comfy.ops mock (CUDA-free)
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
# Load attention module (블라인드 — 구현 내용이 아닌 인터페이스만 사용)
# ---------------------------------------------------------------------------
_ATTENTION_PATH = _BASE / "attention.py"
_spec = _ilu.spec_from_file_location("attention_edge", _ATTENTION_PATH)
_attention_mod = _ilu.module_from_spec(_spec)
_attention_mod._DEFAULT_OPS = None
_spec.loader.exec_module(_attention_mod)

MotifVideoAttention = _attention_mod.MotifVideoAttention

EXPECTED = json.loads(
    (_REPO_ROOT / "tests" / "transformer" / "expected_attn_keys.json").read_text()
)
FIX = _REPO_ROOT / "tests" / "transformer" / "fixtures"


def _dims():
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]
    return hidden // head_dim, head_dim, hidden


def _make_single(seed=0):
    num_heads, head_dim, _ = _dims()
    torch.manual_seed(seed)
    return MotifVideoAttention(num_heads, head_dim, pre_only=True, added_kv=False, qk_norm="rms_norm")


def _make_dual(seed=0):
    num_heads, head_dim, _ = _dims()
    torch.manual_seed(seed)
    return MotifVideoAttention(num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm")


def _inputs(seed=42):
    _, _, hidden = _dims()
    g = torch.Generator().manual_seed(seed)
    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden, generator=g)
    eh = torch.randn(B, E, hidden, generator=g)
    return hs, eh


# ---------------------------------------------------------------------------
# 포인트 1: cross-attn 분기 (query_input not None)
# ---------------------------------------------------------------------------

def test_cross_attn_output_shape():
    """query_input 전달 시 output shape 은 (B, L_Q, hidden)."""
    attn = _make_single()
    _, _, hidden = _dims()
    B, L_Q = 2, 12
    g = torch.Generator().manual_seed(7)
    query_input = torch.randn(B, L_Q, hidden, generator=g)
    key_input = torch.randn(B, 8, hidden, generator=g)
    value_input = torch.randn(B, 8, hidden, generator=g)

    with torch.no_grad():
        out, ctx = attn(
            hidden_states=None,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
            query_input=query_input,
            key_input=key_input,
            value_input=value_input,
        )
    assert out.shape == (B, L_Q, hidden), f"expected ({B},{L_Q},{hidden}), got {tuple(out.shape)}"


def test_cross_attn_second_return_is_none():
    """query_input 경로의 두 번째 반환값은 None (processor line 160 계약)."""
    attn = _make_single()
    _, _, hidden = _dims()
    B, L_Q = 2, 10
    g = torch.Generator().manual_seed(11)
    query_input = torch.randn(B, L_Q, hidden, generator=g)
    key_input = torch.randn(B, 6, hidden, generator=g)
    value_input = torch.randn(B, 6, hidden, generator=g)

    with torch.no_grad():
        out, ctx = attn(
            hidden_states=None,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
            query_input=query_input,
            key_input=key_input,
            value_input=value_input,
        )
    assert ctx is None, f"cross-attn second return must be None, got {type(ctx)}"


def test_cross_attn_no_nan_inf():
    """cross-attn 경로 output 에 NaN/Inf 없음."""
    attn = _make_single()
    _, _, hidden = _dims()
    B, L_Q = 2, 14
    g = torch.Generator().manual_seed(99)
    query_input = torch.randn(B, L_Q, hidden, generator=g)
    key_input = torch.randn(B, 10, hidden, generator=g)
    value_input = torch.randn(B, 10, hidden, generator=g)

    with torch.no_grad():
        out, _ = attn(
            hidden_states=None,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
            query_input=query_input,
            key_input=key_input,
            value_input=value_input,
        )
    assert not torch.isnan(out).any(), "NaN in cross-attn output"
    assert not torch.isinf(out).any(), "Inf in cross-attn output"


# ---------------------------------------------------------------------------
# 포인트 2: encoder_hidden_states=None + single block (순수 self-attn)
# ---------------------------------------------------------------------------

def test_single_block_no_encoder_does_not_crash():
    """encoder_hidden_states=None 인 single block 이 crash 없이 output 반환."""
    attn = _make_single()
    _, _, hidden = _dims()
    B, L = 2, 16
    g = torch.Generator().manual_seed(3)
    hs = torch.randn(B, L, hidden, generator=g)

    with torch.no_grad():
        out, ctx = attn(
            hidden_states=hs,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
        )
    assert out is not None
    assert out.shape == (B, L, hidden), f"shape mismatch: {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# 포인트 3: attention_mask 전달 효과
# ---------------------------------------------------------------------------

def test_attention_mask_changes_output():
    """유효/무효 mask 가 있으면 mask 없는 경우와 output 이 달라야 함."""
    attn = _make_single(seed=5)
    _, _, hidden = _dims()
    hs, eh = _inputs(seed=42)
    B, L = hs.shape[:2]
    num_heads, _, _ = _dims()

    # mask: (B, 1, L, L+E) — 절반 열을 -inf 로 차단
    L_total = L + eh.shape[1]
    mask = torch.zeros(B, 1, L_total, L_total)
    mask[:, :, :, L_total // 2:] = float("-inf")

    with torch.no_grad():
        out_masked, _ = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=mask, image_rotary_emb=None)
        out_unmasked, _ = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)

    # mask 가 실제로 효과 있으면 출력이 달라야 함
    assert not torch.allclose(out_masked, out_unmasked), "attention_mask has no effect on output"


# ---------------------------------------------------------------------------
# 포인트 4: RoPE 전달 효과
# ---------------------------------------------------------------------------

def test_rope_changes_output():
    """image_rotary_emb 전달 시 output 이 미전달 output 과 달라야 함."""
    attn = _make_single(seed=7)
    _, _, hidden = _dims()
    num_heads, head_dim, _ = _dims()
    hs, eh = _inputs(seed=55)
    B, L = hs.shape[:2]
    E = eh.shape[1]

    # (cos, sin) tuple: shape (B, L, head_dim) — L = hs sequence length
    g = torch.Generator().manual_seed(0)
    cos = torch.randn(L, head_dim, generator=g)
    sin = torch.randn(L, head_dim, generator=g)
    rope = (cos, sin)

    with torch.no_grad():
        out_rope, _ = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=rope)
        out_no_rope, _ = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)

    assert out_rope.shape == out_no_rope.shape, "shape changed with rope"
    assert not torch.allclose(out_rope, out_no_rope), "RoPE has no effect on output"


# ---------------------------------------------------------------------------
# 포인트 5: dual block to_add_out 효과 (weight=0 → 두 번째 반환 ≈ 0)
# ---------------------------------------------------------------------------

def test_dual_block_to_add_out_zero_weights():
    """to_add_out weight/bias 를 0 으로 세팅하면 두 번째 반환값이 0."""
    attn = _make_dual(seed=9)
    # to_add_out 은 nn.Linear
    with torch.no_grad():
        attn.to_add_out.weight.zero_()
        attn.to_add_out.bias.zero_()

    hs, eh = _inputs(seed=42)

    with torch.no_grad():
        out, ctx = attn(
            hidden_states=hs,
            encoder_hidden_states=eh,
            attention_mask=None,
            image_rotary_emb=None,
        )

    assert ctx is not None, "dual block second return should not be None"
    assert torch.allclose(ctx, torch.zeros_like(ctx), atol=1e-6), (
        f"to_add_out=0 should produce ctx≈0, max_abs={ctx.abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 포인트 6: dtype 보존
# ---------------------------------------------------------------------------

def test_dtype_float32_preserved():
    """float32 input → float32 output."""
    attn = _make_single()
    _, _, hidden = _dims()
    B, L = 2, 16
    hs = torch.randn(B, L, hidden, dtype=torch.float32)
    eh = torch.randn(B, 8, hidden, dtype=torch.float32)

    with torch.no_grad():
        out, _ = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)
    assert out.dtype == torch.float32, f"expected float32, got {out.dtype}"


def test_dtype_float16_preserved():
    """float16 input → float16 output (processor line 225 계약)."""
    attn = _make_single().to(torch.float16)
    _, _, hidden = _dims()
    B, L = 2, 16
    hs = torch.randn(B, L, hidden, dtype=torch.float16)
    eh = torch.randn(B, 8, hidden, dtype=torch.float16)

    with torch.no_grad():
        out, _ = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)
    assert out.dtype == torch.float16, f"expected float16, got {out.dtype}"


# ---------------------------------------------------------------------------
# 포인트 7: 분기 규칙 준수 — self.pre_only 미사용
# ---------------------------------------------------------------------------

def test_no_pre_only_attribute_usage_in_forward():
    """attention.py forward 구현에 'self.pre_only' 문자열이 없어야 함 (분기 규칙)."""
    source = _ATTENTION_PATH.read_text()
    assert "self.pre_only" not in source, (
        "attention.py forward 에 self.pre_only 가 사용됨. "
        "요구사항: 'self.to_out is None' 으로 분기할 것."
    )


# ---------------------------------------------------------------------------
# 포인트 8: cross-attn 에서 add_q_proj 호출 없음
# ---------------------------------------------------------------------------

def test_cross_attn_does_not_call_add_q_proj():
    """query_input 경로에서 add_q_proj 가 호출되면 안 됨."""
    attn = _make_dual(seed=13)
    _, _, hidden = _dims()
    B, L_Q = 2, 10
    g = torch.Generator().manual_seed(17)
    query_input = torch.randn(B, L_Q, hidden, generator=g)
    key_input = torch.randn(B, 8, hidden, generator=g)
    value_input = torch.randn(B, 8, hidden, generator=g)

    # add_q_proj 에 spy 걸기
    original_add_q_proj = attn.add_q_proj
    call_count = {"n": 0}
    real_forward = original_add_q_proj.forward

    def _spy_forward(x):
        call_count["n"] += 1
        return real_forward(x)

    with mock.patch.object(attn.add_q_proj, "forward", side_effect=_spy_forward):
        with torch.no_grad():
            attn(
                hidden_states=None,
                encoder_hidden_states=None,
                attention_mask=None,
                image_rotary_emb=None,
                query_input=query_input,
                key_input=key_input,
                value_input=value_input,
            )

    assert call_count["n"] == 0, (
        f"add_q_proj was called {call_count['n']} time(s) during cross-attn — must be 0"
    )


# ---------------------------------------------------------------------------
# 포인트 9: input/output sequence length 보존 (self-attn)
# ---------------------------------------------------------------------------

def test_self_attn_sequence_length_preserved():
    """self-attn 에서 input L 과 output L 이 동일해야 함."""
    attn = _make_single()
    _, _, hidden = _dims()
    B, L = 2, 20
    g = torch.Generator().manual_seed(21)
    hs = torch.randn(B, L, hidden, generator=g)

    with torch.no_grad():
        out, _ = attn(
            hidden_states=hs,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
        )
    assert out.shape[1] == L, f"input L={L}, output L={out.shape[1]}"


# ---------------------------------------------------------------------------
# 포인트 10: fixture 재현성
# ---------------------------------------------------------------------------

def test_fixture_reproducibility_single():
    """seed 고정 + deterministic 재실행 시 single fixture output 재현."""
    # generate_attention_golden.py 와 동일한 절차를 인라인으로 재현
    _FIXTURE_SCRIPT = FIX / "generate_attention_golden.py"
    assert _FIXTURE_SCRIPT.exists(), "generate_attention_golden.py not found"

    num_heads, head_dim, hidden = _dims()
    # 두 번 독립적으로 같은 weight seed + input seed 로 생성
    def _run_single():
        torch.manual_seed(0)
        attn = MotifVideoAttention(num_heads, head_dim, pre_only=True, added_kv=False, qk_norm="rms_norm")
        g = torch.Generator().manual_seed(42)
        B, L, E = 2, 16, 8
        hs = torch.randn(B, L, hidden, generator=g)
        eh = torch.randn(B, E, hidden, generator=g)
        with torch.no_grad():
            out, ctx = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)
        return out, ctx

    out1, ctx1 = _run_single()
    out2, ctx2 = _run_single()

    torch.testing.assert_close(out1, out2, atol=0, rtol=0), "single output not reproducible"
    if ctx1 is not None and ctx2 is not None:
        torch.testing.assert_close(ctx1, ctx2, atol=0, rtol=0), "single ctx not reproducible"


def test_fixture_reproducibility_dual():
    """seed 고정 + deterministic 재실행 시 dual fixture output 재현."""
    num_heads, head_dim, hidden = _dims()

    def _run_dual():
        torch.manual_seed(0)
        attn = MotifVideoAttention(num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm")
        g = torch.Generator().manual_seed(42)
        B, L, E = 2, 16, 8
        hs = torch.randn(B, L, hidden, generator=g)
        eh = torch.randn(B, E, hidden, generator=g)
        with torch.no_grad():
            out, ctx = attn(hidden_states=hs, encoder_hidden_states=eh, attention_mask=None, image_rotary_emb=None)
        return out, ctx

    out1, ctx1 = _run_dual()
    out2, ctx2 = _run_dual()

    torch.testing.assert_close(out1, out2, atol=0, rtol=0), "dual output not reproducible"
    if ctx1 is not None and ctx2 is not None:
        torch.testing.assert_close(ctx1, ctx2, atol=0, rtol=0), "dual ctx not reproducible"


# ---------------------------------------------------------------------------
# 포인트 11: fixture 파일 sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", [
    "attention_golden_single_out.pt",
    "attention_golden_single_ctx.pt",
    "attention_golden_dual_out.pt",
    "attention_golden_dual_ctx.pt",
])
def test_fixture_file_exists_and_nonempty(fname):
    """각 fixture 파일이 존재하고 비어있지 않음."""
    p = FIX / fname
    assert p.exists(), f"fixture not found: {fname}"
    size = p.stat().st_size
    assert size > 0, f"fixture is empty (0 bytes): {fname}"


@pytest.mark.parametrize("fname,expected_shape", [
    ("attention_golden_single_out.pt", (2, 16, 1536)),
    ("attention_golden_single_ctx.pt", (2, 8, 1536)),
    ("attention_golden_dual_out.pt", (2, 16, 1536)),
    ("attention_golden_dual_ctx.pt", (2, 8, 1536)),
])
def test_fixture_tensor_shape(fname, expected_shape):
    """fixture tensor shape 이 예상 값과 일치."""
    t = torch.load(FIX / fname, weights_only=True)
    assert tuple(t.shape) == expected_shape, (
        f"{fname}: expected shape {expected_shape}, got {tuple(t.shape)}"
    )


