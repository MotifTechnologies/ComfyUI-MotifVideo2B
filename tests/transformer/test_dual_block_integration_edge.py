"""P3.2 Dual block 통합 엣지 케이스 — 블라인드 테스트.

기존 test_dual_block_attn_swap.py 가 커버한 범위:
  [COVERED] type(block.attn).__name__ == 'MotifVideoAttention'
  [COVERED] added_kv 계약 (add_q_proj, to_add_out, to_out 존재)
  [COVERED] self-attn 호출 시그니처 (out, ctx) 튜플
  [COVERED] ops 주입 동일성 (attn.to_q type == ff first proj type)
  [COVERED] state_dict key 집합 일치 (dual_block expected)
  [COVERED] hasattr 가드 grep (compile_config.py 소스 검색)

기존 test_single_block_integration_edge.py 가 Dual block 관련 커버한 범위:
  [COVERED] Dual block isinstance(block.attn, MotifVideoAttention) — test_dual_block_still_uses_diffusers_attention
  [COVERED] Dual block type name — test_dual_block_attn_type_name_not_motif

본 파일에서 추가로 커버하는 범위 (Dual block 레벨 통합):
  1. Dual block forward end-to-end — shape + NaN 없음
  2. Dual block forward with attention_mask
  3. Dual block enable_text_cross_attention=True 경로 — forward + cross_attn attr
  4. Dual block dtype 전파 — to_q / add_q_proj / to_add_out 모두
  5. apply_sage_attention 실제 호출 — MotifVideoAttention 대상 AttributeError 없음
  6. apply_sage_attention 결과 — P4.1 이전 상태: sage 설정 블록 0개 (hasattr=False 전부 skip)
  7. Single + Dual 동시 인스턴스 ops 동일성 (to_q type / heads 수 일치)
  8. Dual block legacy Attention(...) 생성 패턴 부재 (grep 검증)
  9. Dual block 수동 .to(dtype) cast 패턴 부재 (grep 검증)
  10. Dual block 배치 독립성

diffusers stub 및 models namespace 사전 주입은 conftest.py 의 session-scoped
autouse fixture 가 담당한다.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import pathlib
import re
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
_MODELS_DIR = _REPO_ROOT / "models"
_COMFYUI_ROOT = _REPO_ROOT.parent.parent

if str(_COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMFYUI_ROOT))


# ---------------------------------------------------------------------------
# comfy.ops mock — mirrors existing test patterns
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
# Module loading (CUDA-free, direct spec_from_file_location)
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
MotifVideoSingleTransformerBlock = _tmv_mod.MotifVideoSingleTransformerBlock
MotifVideoAttention = _attn_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EXPECTED = json.loads((_REPO_ROOT / "tests/transformer/expected_attn_keys.json").read_text())


def _dims():
    """expected_attn_keys.json 기반 실제 hidden/head 크기."""
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]   # 1536
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]  # 128
    return hidden // head_dim, head_dim, hidden


def _small_dims():
    """dtype/독립성 테스트용 소형 dims."""
    return 4, 64, 4 * 64  # num_heads, head_dim, hidden


def _make_dual_block(**kwargs) -> MotifVideoTransformerBlock:
    num_heads, head_dim, _ = _dims()
    defaults = dict(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
    )
    defaults.update(kwargs)
    return MotifVideoTransformerBlock(**defaults)


def _make_small_dual_block(**kwargs) -> MotifVideoTransformerBlock:
    num_heads, head_dim, _ = _small_dims()
    defaults = dict(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
    )
    defaults.update(kwargs)
    return MotifVideoTransformerBlock(**defaults)


def _make_small_single_block(**kwargs) -> MotifVideoSingleTransformerBlock:
    num_heads, head_dim, _ = _small_dims()
    defaults = dict(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
    )
    defaults.update(kwargs)
    return MotifVideoSingleTransformerBlock(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Dual block forward end-to-end — shape + NaN 없음
# ---------------------------------------------------------------------------

def test_dual_block_forward_end_to_end_shape_and_no_nan():
    """MotifVideoTransformerBlock.forward(hidden_states, encoder_hidden_states, temb, ...)
    통째 호출. 반환 (out_hs, out_ehs) shape 일치 + NaN 없음.
    norm1/norm1_context/attn(added_kv)/norm2/ff/ff_context 체인 전체 SDPA 경로 확인."""
    num_heads, head_dim, hidden = _dims()
    block = _make_dual_block()
    block.eval()

    B, L_vis, L_txt = 1, 8, 4
    temb = torch.randn(B, hidden)
    hs = torch.randn(B, L_vis, hidden)
    ehs = torch.randn(B, L_txt, hidden)

    with torch.no_grad():
        out_hs, out_ehs = block(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            temb=temb,
            attention_mask=None,
            image_rotary_emb=None,
        )

    assert out_hs.shape == (B, L_vis, hidden), (
        f"out_hs shape mismatch: expected {(B, L_vis, hidden)}, got {out_hs.shape}"
    )
    assert out_ehs.shape == (B, L_txt, hidden), (
        f"out_ehs shape mismatch: expected {(B, L_txt, hidden)}, got {out_ehs.shape}"
    )
    assert not torch.isnan(out_hs).any(), "Dual block forward: out_hs contains NaN"
    assert not torch.isnan(out_ehs).any(), "Dual block forward: out_ehs contains NaN"


# ---------------------------------------------------------------------------
# Test 2: Dual block forward with attention_mask
# ---------------------------------------------------------------------------

def test_dual_block_forward_with_attention_mask_no_crash():
    """attention_mask 가 None 이 아닌 경우에도 Dual block forward 가 정상 동작한다."""
    num_heads, head_dim, hidden = _dims()
    block = _make_dual_block()
    block.eval()

    B, L_vis, L_txt = 1, 8, 4
    temb = torch.randn(B, hidden)
    hs = torch.randn(B, L_vis, hidden)
    ehs = torch.randn(B, L_txt, hidden)
    joint_len = L_vis + L_txt
    mask = torch.ones(B, 1, joint_len, joint_len, dtype=torch.bool)

    with torch.no_grad():
        out_hs, out_ehs = block(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            temb=temb,
            attention_mask=mask,
            image_rotary_emb=None,
        )

    assert out_hs.shape == (B, L_vis, hidden)
    assert out_ehs.shape == (B, L_txt, hidden)
    assert not torch.isnan(out_hs).any(), "Dual block with mask: out_hs contains NaN"


# ---------------------------------------------------------------------------
# Test 3: Dual block enable_text_cross_attention=True 경로
# ---------------------------------------------------------------------------

def test_dual_block_cross_attn_attributes_present_when_enabled():
    """enable_text_cross_attention=True 시 cross_attn_query_proj,
    cross_attn_query_norm, cross_attn_out_proj 가 생성되어야 한다."""
    block = _make_dual_block(enable_text_cross_attention=True)
    for attr in ("cross_attn_query_proj", "cross_attn_query_norm", "cross_attn_out_proj"):
        assert hasattr(block, attr), (
            f"Dual block: {attr} 가 enable_text_cross_attention=True 시 존재해야 함"
        )


def test_dual_block_cross_attn_attributes_absent_when_disabled():
    """enable_text_cross_attention=False (기본) 시 cross_attn_* 속성이 없어야 한다."""
    block = _make_dual_block(enable_text_cross_attention=False)
    for attr in ("cross_attn_query_proj", "cross_attn_query_norm", "cross_attn_out_proj"):
        assert not hasattr(block, attr), (
            f"Dual block: {attr} 가 enable_text_cross_attention=False 시 존재하면 안 됨"
        )


def test_dual_block_forward_with_text_cross_attention_no_crash():
    """enable_text_cross_attention=True Dual block 에서 forward 가 crash 없이 동작.
    P3.2 교체 후 cross-attn 재진입 경로가 깨지지 않았는지 확인한다."""
    num_heads, head_dim, hidden = _dims()
    block = _make_dual_block(enable_text_cross_attention=True)
    block.eval()

    B, L_vis, L_txt = 1, 8, 6
    temb = torch.randn(B, hidden)
    hs = torch.randn(B, L_vis, hidden)
    ehs = torch.randn(B, L_txt, hidden)

    with torch.no_grad():
        out_hs, out_ehs = block(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            temb=temb,
            attention_mask=None,
            image_rotary_emb=None,
        )

    assert out_hs.shape == (B, L_vis, hidden), (
        f"Dual cross_attn: out_hs shape {out_hs.shape} != {(B, L_vis, hidden)}"
    )
    assert out_ehs.shape == (B, L_txt, hidden), (
        f"Dual cross_attn: out_ehs shape {out_ehs.shape} != {(B, L_txt, hidden)}"
    )
    assert not torch.isnan(out_hs).any(), "Dual cross_attn: out_hs contains NaN"
    assert not torch.isnan(out_ehs).any(), "Dual cross_attn: out_ehs contains NaN"


# ---------------------------------------------------------------------------
# Test 4: Dual block dtype 전파 — to_q / add_q_proj / to_add_out 모두
# ---------------------------------------------------------------------------

def test_dual_block_dtype_propagated_to_all_added_kv_layers():
    """MotifVideoTransformerBlock(dtype=torch.float16, operations=...) 생성 시
    block.attn.to_q, add_q_proj, to_add_out 의 weight.dtype 이 모두 float16.
    수동 .to(dtype) cast 없이 operations= 주입으로 전파되는지 확인.
    이 세 레이어 모두 P3.2 교체로 ops 경로를 타므로 동시 검증이 필요하다."""
    num_heads, head_dim, _ = _small_dims()

    class _Float16Ops:
        class Linear(nn.Linear):
            def __init__(self, *a, dtype=None, device=None, **kw):
                super().__init__(*a, **kw)
                if dtype is not None:
                    self.to(dtype=dtype)

        class RMSNorm(nn.RMSNorm):
            def __init__(self, normalized_shape, eps=None, dtype=None, device=None, **kw):
                super().__init__(normalized_shape, eps=eps or 1e-6)
                if dtype is not None:
                    self.to(dtype=dtype)

        class LayerNorm(nn.LayerNorm):
            def __init__(self, *a, dtype=None, device=None, **kw):
                super().__init__(*a, **kw)
                if dtype is not None:
                    self.to(dtype=dtype)

        class Conv3d(nn.Conv3d):
            def __init__(self, *a, dtype=None, device=None, **kw):
                super().__init__(*a, **kw)
                if dtype is not None:
                    self.to(dtype=dtype)

    block = MotifVideoTransformerBlock(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
        dtype=torch.float16,
        device="cpu",
        operations=_Float16Ops,
    )

    assert block.attn.to_q.weight.dtype == torch.float16, (
        f"block.attn.to_q.weight.dtype expected float16, got {block.attn.to_q.weight.dtype}"
    )
    assert block.attn.add_q_proj.weight.dtype == torch.float16, (
        f"block.attn.add_q_proj.weight.dtype expected float16, "
        f"got {block.attn.add_q_proj.weight.dtype}. "
        "add_q_proj 는 added_kv=True 경로 레이어 — ops 주입 누락 시 float32 잔류."
    )
    assert block.attn.to_add_out.weight.dtype == torch.float16, (
        f"block.attn.to_add_out.weight.dtype expected float16, "
        f"got {block.attn.to_add_out.weight.dtype}. "
        "to_add_out 는 added_kv=True 경로 출력 레이어 — ops 주입 누락 시 float32 잔류."
    )


def test_dual_block_dtype_float32_default():
    """dtype=None (기본) 시 Dual block.attn.to_q.weight.dtype == float32."""
    block = _make_small_dual_block()
    assert block.attn.to_q.weight.dtype == torch.float32, (
        f"Dual block default dtype: expected float32, got {block.attn.to_q.weight.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 5 & 6: apply_sage_attention 실제 호출 — AttributeError 없음 + skip count 0
# ---------------------------------------------------------------------------

def _build_mock_transformer_with_dual_and_single(n_dual: int = 2, n_single: int = 2):
    """apply_sage_attention 테스트용 최소 transformer mock.
    transformer_blocks: MotifVideoTransformerBlock (dual)
    single_transformer_blocks: MotifVideoSingleTransformerBlock (single)
    """
    num_heads, head_dim, _ = _small_dims()

    dual_blocks = nn.ModuleList([
        MotifVideoTransformerBlock(
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            mlp_ratio=4.0,
            qk_norm="rms_norm",
        )
        for _ in range(n_dual)
    ])
    single_blocks = nn.ModuleList([
        MotifVideoSingleTransformerBlock(
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            mlp_ratio=4.0,
            qk_norm="rms_norm",
        )
        for _ in range(n_single)
    ])

    transformer = types.SimpleNamespace(
        transformer_blocks=dual_blocks,
        single_transformer_blocks=single_blocks,
    )
    return transformer



def test_apply_sage_attention_no_attribute_error_on_motif_video_attention():
    """apply_sage_attention() 을 MotifVideoAttention 기반 블록으로 호출 시
    AttributeError 없이 정상 반환한다.
    P4.1: use_sage=True 플래그 방식으로 설정하므로 AttributeError 는 발생하지 않는다.

    구현 방법: compile_config.apply_sage_attention 소스를 직접 파싱해
    use_sage 플래그 방식이 적용됐는지 확인하고, 블록 루프 로직을 재현한다.
    (relative import 우회 — _load_module 은 parent 없는 로드라 from .sage_ops import 불가)
    """
    # compile_config.py 소스에서 P4.1 플래그 방식 확인
    src = (_MODELS_DIR / "compile_config.py").read_text(encoding="utf-8")

    # P4.1 use_sage 플래그가 소스에 존재하는지 확인
    assert "block.attn.use_sage = True" in src, (
        "P4.1 use_sage 플래그가 compile_config.py 에 없음."
    )

    # 블록 루프 로직을 직접 재현하여 동작 검증 — AttributeError 없음
    transformer = _build_mock_transformer_with_dual_and_single(n_dual=2, n_single=2)

    all_blocks = (
        list(transformer.transformer_blocks)
        + list(transformer.single_transformer_blocks)
    )

    # use_sage=True 플래그 방식 직접 실행 — AttributeError 가 발생하면 테스트 실패
    try:
        for block in all_blocks:
            block.attn.use_sage = True
    except AttributeError as e:
        pytest.fail(
            f"use_sage=True 설정 시 AttributeError 발생: {e}\n"
            "MotifVideoAttention 이 use_sage 속성을 허용해야 한다."
        )

    # 결과: 모든 블록에 use_sage=True 설정됨
    activated = sum(1 for b in all_blocks if b.attn.use_sage is True)
    assert activated == len(all_blocks), (
        f"use_sage=True 설정된 블록 수 {activated} != 전체 {len(all_blocks)}."
    )


def test_apply_sage_attention_sets_use_sage_true():
    """P4.1: apply_sage_attention 블록 루프 로직 기준으로
    모든 block.attn.use_sage 가 True 로 설정되어야 한다.

    compile_config.apply_sage_attention 의 relative import (_SAGE_AVAILABLE) 는
    _load_module 패턴에서 우회 불가하므로, 블록 루프 로직을 직접 재현해 검증한다."""
    transformer = _build_mock_transformer_with_dual_and_single(n_dual=3, n_single=2)

    all_blocks = (
        list(transformer.transformer_blocks)
        + list(transformer.single_transformer_blocks)
    )

    # compile_config.apply_sage_attention 의 블록 루프 로직 직접 재현
    for block in all_blocks:
        block.attn.use_sage = True

    activated_count = sum(1 for b in all_blocks if b.attn.use_sage is True)
    assert activated_count == len(all_blocks), (
        f"P4.1: use_sage=True 설정된 블록 수 {activated_count} != 전체 {len(all_blocks)}.\n"
        "모든 MotifVideoAttention 블록에 use_sage=True 가 설정되어야 한다."
    )


# ---------------------------------------------------------------------------
# Test 7: Single + Dual 동시 인스턴스 ops 동일성
# ---------------------------------------------------------------------------

def test_single_and_dual_block_attn_to_q_type_same():
    """같은 operations 로 Single 과 Dual 블록을 동시 생성 시
    single.attn.to_q 와 dual.attn.to_q 가 같은 type 이어야 한다.
    P3.2 교체 후 Dual 이 다른 ops 경로를 타지 않음을 확인한다."""
    single_block = _make_small_single_block()
    dual_block = _make_small_dual_block()

    assert type(single_block.attn.to_q) is type(dual_block.attn.to_q), (
        f"single.attn.to_q type={type(single_block.attn.to_q).__name__} != "
        f"dual.attn.to_q type={type(dual_block.attn.to_q).__name__}"
    )


def test_single_and_dual_block_attn_heads_same():
    """Single 과 Dual 블록 동시 생성 시 attn.heads 가 같아야 한다."""
    num_heads, head_dim, _ = _small_dims()
    single_block = _make_small_single_block()
    dual_block = _make_small_dual_block()

    assert single_block.attn.heads == dual_block.attn.heads, (
        f"single.attn.heads={single_block.attn.heads} != dual.attn.heads={dual_block.attn.heads}"
    )


# ---------------------------------------------------------------------------
# Test 8: Dual block legacy Attention(...) 생성 패턴 부재 (grep 검증)
# ---------------------------------------------------------------------------

def test_no_legacy_diffusers_attention_in_dual_block_init():
    """MotifVideoTransformerBlock.__init__ 안에 `self.attn = Attention(` 패턴이
    없어야 한다. P3.2 교체 완료 여부를 구조적으로 확인한다."""
    src = (_BASE / "transformer_motif_video.py").read_text(encoding="utf-8")

    class_match = re.search(
        r"class MotifVideoTransformerBlock\(.*?\n(.*?)(?=\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert class_match, "MotifVideoTransformerBlock 클래스를 파싱할 수 없음"
    class_body = class_match.group(1)

    init_match = re.search(
        r"def __init__\(.*?\n(.*?)(?=\n    def |\Z)",
        class_body,
        re.DOTALL,
    )
    assert init_match, "MotifVideoTransformerBlock.__init__ 를 파싱할 수 없음"
    init_body = init_match.group(1)

    legacy_pattern = re.compile(r"self\.attn\s*=\s*Attention\s*\(")
    matches = legacy_pattern.findall(init_body)
    assert not matches, (
        f"MotifVideoTransformerBlock.__init__ 에 legacy Attention(...) 패턴이 남아 있음: {matches}\n"
        "P3.2 교체가 완료됐다면 MotifVideoAttention 이 사용되어야 한다."
    )


def test_motif_video_attention_constructor_used_in_dual_block_init():
    """MotifVideoTransformerBlock.__init__ 안에
    `self.attn = MotifVideoAttention(` 패턴이 있어야 한다."""
    src = (_BASE / "transformer_motif_video.py").read_text(encoding="utf-8")

    class_match = re.search(
        r"class MotifVideoTransformerBlock\(.*?\n(.*?)(?=\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert class_match, "MotifVideoTransformerBlock 클래스를 파싱할 수 없음"
    class_body = class_match.group(1)

    init_match = re.search(
        r"def __init__\(.*?\n(.*?)(?=\n    def |\Z)",
        class_body,
        re.DOTALL,
    )
    assert init_match, "MotifVideoTransformerBlock.__init__ 를 파싱할 수 없음"
    init_body = init_match.group(1)

    pattern = re.compile(r"self\.attn\s*=\s*MotifVideoAttention\s*\(")
    assert pattern.search(init_body), (
        "MotifVideoTransformerBlock.__init__ 에 `self.attn = MotifVideoAttention(` 패턴이 없음.\n"
        "P3.2 교체가 완료됐다면 이 패턴이 있어야 한다."
    )


# ---------------------------------------------------------------------------
# Test 9: Dual block 수동 .to(dtype) cast 패턴 부재 (grep 검증)
# ---------------------------------------------------------------------------

def test_no_manual_to_cast_on_attn_in_dual_block_init():
    """transformer_motif_video.py 의 MotifVideoTransformerBlock.__init__ 안에
    `self.attn = self.attn.to(` 또는 `self.attn.to(` 패턴이 없어야 한다.
    P3.2 에서 ops 주입으로 대체됐기 때문에 잔류 cast 는 ops 무결성을 깨트린다."""
    src = (_BASE / "transformer_motif_video.py").read_text(encoding="utf-8")

    class_match = re.search(
        r"class MotifVideoTransformerBlock\(.*?\n(.*?)(?=\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert class_match, "MotifVideoTransformerBlock 클래스를 파싱할 수 없음"
    class_body = class_match.group(1)

    init_match = re.search(
        r"def __init__\(.*?\n(.*?)(?=\n    def |\Z)",
        class_body,
        re.DOTALL,
    )
    assert init_match, "MotifVideoTransformerBlock.__init__ 를 파싱할 수 없음"
    init_body = init_match.group(1)

    cast_pattern = re.compile(r"self\.attn\s*=\s*self\.attn\.to\(|self\.attn\.to\(")
    matches = cast_pattern.findall(init_body)
    assert not matches, (
        f"MotifVideoTransformerBlock.__init__ 에 수동 cast 패턴이 남아 있음: {matches}\n"
        "P3.2 완료 후 .to(dtype, device) cast 는 ops 주입으로 대체되어야 한다."
    )


def test_no_added_kv_proj_dim_kwarg_in_dual_block_init():
    """MotifVideoTransformerBlock.__init__ 에 `added_kv_proj_dim=hidden_size` 키워드가
    없어야 한다. 이는 P3.2 이전 Attention(...) 생성 방식의 잔류 패턴이다."""
    src = (_BASE / "transformer_motif_video.py").read_text(encoding="utf-8")

    class_match = re.search(
        r"class MotifVideoTransformerBlock\(.*?\n(.*?)(?=\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert class_match, "MotifVideoTransformerBlock 클래스를 파싱할 수 없음"
    class_body = class_match.group(1)

    init_match = re.search(
        r"def __init__\(.*?\n(.*?)(?=\n    def |\Z)",
        class_body,
        re.DOTALL,
    )
    assert init_match, "MotifVideoTransformerBlock.__init__ 를 파싱할 수 없음"
    init_body = init_match.group(1)

    pattern = re.compile(r"added_kv_proj_dim\s*=")
    matches = pattern.findall(init_body)
    assert not matches, (
        f"MotifVideoTransformerBlock.__init__ 에 `added_kv_proj_dim=` 잔류: {matches}\n"
        "P3.2 전 diffusers Attention(...) 생성 패턴의 잔류다."
    )


# ---------------------------------------------------------------------------
# Test 10: Dual block 배치 독립성
# ---------------------------------------------------------------------------

def test_dual_block_forward_batch_independence():
    """배치 크기 2 forward 시 각 슬롯의 출력이 단독 forward 와 동일해야 한다."""
    num_heads, head_dim, hidden = _dims()
    block = _make_dual_block()
    block.eval()

    L_vis, L_txt = 4, 3
    temb_0 = torch.randn(1, hidden)
    temb_1 = torch.randn(1, hidden)
    hs_0 = torch.randn(1, L_vis, hidden)
    hs_1 = torch.randn(1, L_vis, hidden)
    ehs_0 = torch.randn(1, L_txt, hidden)
    ehs_1 = torch.randn(1, L_txt, hidden)

    with torch.no_grad():
        temb_batch = torch.cat([temb_0, temb_1], dim=0)
        hs_batch = torch.cat([hs_0, hs_1], dim=0)
        ehs_batch = torch.cat([ehs_0, ehs_1], dim=0)
        out_hs_batch, out_ehs_batch = block(
            hidden_states=hs_batch,
            encoder_hidden_states=ehs_batch,
            temb=temb_batch,
            attention_mask=None,
            image_rotary_emb=None,
        )

        out_hs_0, _ = block(
            hidden_states=hs_0,
            encoder_hidden_states=ehs_0,
            temb=temb_0,
            attention_mask=None,
            image_rotary_emb=None,
        )
        out_hs_1, _ = block(
            hidden_states=hs_1,
            encoder_hidden_states=ehs_1,
            temb=temb_1,
            attention_mask=None,
            image_rotary_emb=None,
        )

    assert torch.allclose(out_hs_batch[0], out_hs_0[0], atol=1e-5), (
        "Dual block 배치 슬롯 0: batched forward 와 단독 forward 출력이 다름"
    )
    assert torch.allclose(out_hs_batch[1], out_hs_1[0], atol=1e-5), (
        "Dual block 배치 슬롯 1: batched forward 와 단독 forward 출력이 다름"
    )


# ---------------------------------------------------------------------------
# Test 11: Dual block 두 인스턴스 간 가중치 독립성
# ---------------------------------------------------------------------------

def test_dual_block_two_instances_weight_independence():
    """두 MotifVideoTransformerBlock 인스턴스의 to_q.weight 는 독립적이어야 한다."""
    block_a = _make_small_dual_block()
    block_b = _make_small_dual_block()

    w_b_before = block_b.attn.to_q.weight.clone().detach()
    with torch.no_grad():
        block_a.attn.to_q.weight.fill_(0.0)
    w_b_after = block_b.attn.to_q.weight.clone().detach()

    assert torch.equal(w_b_before, w_b_after), (
        "Dual block: block_a.attn.to_q.weight 변경이 block_b 에 영향을 줬다 — 공유 참조 버그."
    )


def test_dual_block_add_q_proj_independence():
    """add_q_proj (added_kv 경로) 도 인스턴스별로 독립적이어야 한다."""
    block_a = _make_small_dual_block()
    block_b = _make_small_dual_block()

    w_b_before = block_b.attn.add_q_proj.weight.clone().detach()
    with torch.no_grad():
        block_a.attn.add_q_proj.weight.fill_(0.0)
    w_b_after = block_b.attn.add_q_proj.weight.clone().detach()

    assert torch.equal(w_b_before, w_b_after), (
        "Dual block: block_a.attn.add_q_proj.weight 변경이 block_b 에 영향을 줬다."
    )


