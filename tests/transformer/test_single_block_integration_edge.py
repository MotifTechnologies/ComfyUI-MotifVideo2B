"""P3.1 블록 레벨 통합 엣지 케이스 — 블라인드 테스트.

기존 test_single_block_attn_swap.py 가 커버한 범위:
  [COVERED] type(block.attn).__name__ == 'MotifVideoAttention'
  [COVERED] pre_only 계약 (add_q_proj is None, to_out is None)
  [COVERED] self-attn 호출 시그니처 (attn 직접 호출)
  [COVERED] cross-attn 호출 시그니처 (attn 직접 호출)
  [COVERED] ops 주입 동일성 (type(attn.to_q) is type(proj_mlp))

본 파일에서 추가로 커버하는 범위 (블록 레벨 통합):
  1. block.forward(...) end-to-end — shape + NaN 없음
  2. state_dict key 집합 — attn.* prefix 하위 key 가 expected_attn_keys.json 과 정확히 일치
  3. enable_text_cross_attention=True 경로 — forward 통과 + shape 확인
  4. dtype 전파 — block.attn.to_q.weight.dtype == 지정 dtype
  5. 여러 블록 인스턴스 간 가중치 독립성
  6. .to(dtype/device) 수동 cast 패턴 부재 (grep 검증)
  7. legacy Attention(...) 생성 패턴 부재 (grep 검증)
  8. Dual block P3.2 완료 — MotifVideoAttention 사용 확인

diffusers stub 및 models namespace 사전 주입은 conftest.py 의 session-scoped
autouse fixture (_diffusers_stub_session, _models_namespace_session) 가 담당한다.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
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
# comfy.ops mock (동일 패턴: test_single_block_attn_swap.py 준수)
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
    import os
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

MotifVideoSingleTransformerBlock = _tmv_mod.MotifVideoSingleTransformerBlock
MotifVideoTransformerBlock = _tmv_mod.MotifVideoTransformerBlock
MotifVideoAttention = _attn_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
EXPECTED = json.loads((_REPO_ROOT / "tests/transformer/expected_attn_keys.json").read_text())


def _dims():
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]   # 1536
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]  # 128
    return hidden // head_dim, head_dim, hidden


def _make_block(**kwargs) -> MotifVideoSingleTransformerBlock:
    num_heads, head_dim, _ = _dims()
    defaults = dict(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
    )
    defaults.update(kwargs)
    return MotifVideoSingleTransformerBlock(**defaults)


def _small_dims():
    """소형 dims — dtype 전파 / 독립성 테스트용 (크기 상관없음)."""
    return 4, 64, 4 * 64  # num_heads, head_dim, hidden


def _make_small_block(**kwargs) -> MotifVideoSingleTransformerBlock:
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
# Test 1: block.forward() end-to-end — output shape + NaN 없음
# ---------------------------------------------------------------------------

def test_block_forward_end_to_end_shape_and_no_nan():
    """block.forward(hidden_states, encoder_hidden_states, temb, ...) 통째 호출.
    반환 shape == 입력 shape, NaN 없음을 확인한다.
    이는 attn→norm→proj_mlp→proj_out 체인이 정상 동작하는지 블록 레벨에서 검증한다."""
    num_heads, head_dim, hidden = _dims()
    block = _make_block()
    block.eval()

    B, L_vis, L_txt = 1, 8, 4
    # AdaLayerNormZeroSingle 은 temb 로부터 gate 생성, 크기: [B, hidden_size * 2 + ?] 필요.
    # 실제 AdaLayerNormZeroSingle.linear 출력 크기를 확인한 뒤 temb 크기를 맞춘다.
    # ops_primitives.AdaLayerNormZeroSingle 의 linear: hidden_size → 3 * hidden_size
    # → temb 는 hidden_size 크기여야 함
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
    assert not torch.isnan(out_hs).any(), "out_hs contains NaN"
    assert not torch.isnan(out_ehs).any(), "out_ehs contains NaN"


def test_block_forward_with_attention_mask():
    """attention_mask 가 None 이 아닌 경우에도 forward 가 정상 동작한다."""
    num_heads, head_dim, hidden = _dims()
    block = _make_block()
    block.eval()

    B, L_vis, L_txt = 1, 8, 4
    temb = torch.randn(B, hidden)
    hs = torch.randn(B, L_vis, hidden)
    ehs = torch.randn(B, L_txt, hidden)
    # joint seq_len = L_vis + L_txt
    joint_len = L_vis + L_txt
    # attention_mask: [B, 1, joint_len, joint_len] bool
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
    assert not torch.isnan(out_hs).any(), "out_hs contains NaN with attention_mask"


# ---------------------------------------------------------------------------
# Test 2: state_dict key 집합 — attn.* prefix 가 expected_attn_keys.json 과 일치
# ---------------------------------------------------------------------------

def test_single_block_attn_state_dict_keys_match_expected():
    """MotifVideoSingleTransformerBlock.state_dict() 에서 attn.* prefix 를 strip 한
    key 집합이 expected_attn_keys.json["single_block"] 와 정확히 일치해야 한다.
    P3.1 교체로 key 이름이 바뀌면 checkpoint 로드가 깨진다."""
    block = _make_block()
    full_sd = block.state_dict()

    # attn.* prefix 키만 추출하고 prefix 제거
    attn_keys = {
        k[len("attn."):] for k in full_sd if k.startswith("attn.")
    }
    expected = set(EXPECTED["single_block"].keys())

    extra = attn_keys - expected
    missing = expected - attn_keys

    assert attn_keys == expected, (
        f"attn.* state_dict key mismatch.\n"
        f"  extra  (구현에만 있음): {sorted(extra)}\n"
        f"  missing (expected에만 있음): {sorted(missing)}"
    )


def test_single_block_state_dict_no_unexpected_attn_subkeys():
    """state_dict 에 attn.to_out.*, attn.add_q_proj.* 같은
    pre_only=True 위반 키가 없어야 한다."""
    block = _make_block()
    sd = block.state_dict()

    forbidden_prefixes = ["attn.to_out.", "attn.add_q_proj.", "attn.add_k_proj.", "attn.add_v_proj."]
    violations = [k for k in sd for pfx in forbidden_prefixes if k.startswith(pfx)]
    assert not violations, (
        f"pre_only=True 위반 key 가 state_dict 에 존재: {violations}"
    )


# ---------------------------------------------------------------------------
# Test 3: enable_text_cross_attention=True 경로 — forward 통과 + shape
# ---------------------------------------------------------------------------

def test_block_forward_with_text_cross_attention():
    """enable_text_cross_attention=True 시 forward 에서 cross-attn 재진입 경로가
    정상 동작한다. P3.1 교체로 self.attn 두 번째 호출(cross-attn 분기)이 깨지지
    않았는지 확인한다."""
    num_heads, head_dim, hidden = _dims()
    block = _make_block(enable_text_cross_attention=True)
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
        f"cross_attn path: out_hs shape {out_hs.shape} != expected {(B, L_vis, hidden)}"
    )
    assert out_ehs.shape == (B, L_txt, hidden), (
        f"cross_attn path: out_ehs shape {out_ehs.shape} != expected {(B, L_txt, hidden)}"
    )
    assert not torch.isnan(out_hs).any(), "cross_attn path: out_hs contains NaN"
    assert not torch.isnan(out_ehs).any(), "cross_attn path: out_ehs contains NaN"


def test_block_cross_attn_attributes_present():
    """enable_text_cross_attention=True 시 cross_attn_query_proj,
    cross_attn_query_norm, cross_attn_out_proj 가 생성되어야 한다."""
    block = _make_block(enable_text_cross_attention=True)
    for attr in ("cross_attn_query_proj", "cross_attn_query_norm", "cross_attn_out_proj"):
        assert hasattr(block, attr), f"{attr} 가 enable_text_cross_attention=True 시 존재해야 함"


def test_block_cross_attn_attributes_absent_when_disabled():
    """enable_text_cross_attention=False (기본) 시 cross_attn_* 속성이 없어야 한다."""
    block = _make_block(enable_text_cross_attention=False)
    for attr in ("cross_attn_query_proj", "cross_attn_query_norm", "cross_attn_out_proj"):
        assert not hasattr(block, attr), f"{attr} 가 enable_text_cross_attention=False 시 존재하면 안 됨"


# ---------------------------------------------------------------------------
# Test 4: dtype 전파 — block.attn.to_q.weight.dtype == 지정 dtype
# ---------------------------------------------------------------------------

def test_dtype_propagated_to_attn_to_q():
    """MotifVideoSingleTransformerBlock(dtype=torch.float16) 생성 시
    block.attn.to_q.weight.dtype == float16 이어야 한다.
    수동 .to(dtype) cast 없이 operations= 주입으로 전파되는지 확인한다."""
    num_heads, head_dim, _ = _small_dims()

    # float16 mock ops
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

    block = MotifVideoSingleTransformerBlock(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
        dtype=torch.float16,
        device="cpu",
        operations=_Float16Ops,
    )

    assert block.attn.to_q.weight.dtype == torch.float16, (
        f"block.attn.to_q.weight.dtype expected float16, got {block.attn.to_q.weight.dtype}. "
        "operations= 주입 경로를 통한 dtype 전파 누락 의심."
    )


def test_dtype_float32_default():
    """dtype=None (기본) 시 block.attn.to_q.weight.dtype == float32."""
    block = _make_small_block()
    assert block.attn.to_q.weight.dtype == torch.float32, (
        f"block.attn.to_q.weight.dtype expected float32 by default, "
        f"got {block.attn.to_q.weight.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 5: 여러 블록 인스턴스 간 가중치 독립성
# ---------------------------------------------------------------------------

def test_two_block_instances_weight_independence():
    """두 MotifVideoSingleTransformerBlock 인스턴스를 생성 후
    한 쪽 to_q.weight 를 변경해도 다른 쪽은 영향받지 않아야 한다."""
    block_a = _make_small_block()
    block_b = _make_small_block()

    w_b_before = block_b.attn.to_q.weight.clone().detach()

    # block_a 가중치를 0 으로 만들기
    with torch.no_grad():
        block_a.attn.to_q.weight.fill_(0.0)

    w_b_after = block_b.attn.to_q.weight.clone().detach()

    assert torch.equal(w_b_before, w_b_after), (
        "block_a 의 to_q.weight 변경이 block_b 에 영향을 줬다 — 공유 참조 버그 의심."
    )


def test_two_block_instances_proj_mlp_independence():
    """proj_mlp 도 인스턴스별로 독립적이어야 한다."""
    block_a = _make_small_block()
    block_b = _make_small_block()

    w_b_before = block_b.proj_mlp.weight.clone().detach()

    with torch.no_grad():
        block_a.proj_mlp.weight.zero_()

    w_b_after = block_b.proj_mlp.weight.clone().detach()
    assert torch.equal(w_b_before, w_b_after), (
        "block_a.proj_mlp.weight 변경이 block_b 에 영향을 줬다."
    )


# ---------------------------------------------------------------------------
# Test 6: .to(dtype/device) 수동 cast 패턴 부재 (grep 검증)
# ---------------------------------------------------------------------------

def test_no_manual_to_cast_on_attn_in_single_block_init():
    """transformer_motif_video.py 의 MotifVideoSingleTransformerBlock.__init__ 안에
    `self.attn = self.attn.to(` 또는 `self.attn.to(` 패턴이 없어야 한다.
    P3.1 에서 ops 주입으로 대체됐기 때문에 잔류 cast 는 ops 무결성을 깨트린다."""
    import re

    src = (_BASE / "transformer_motif_video.py").read_text(encoding="utf-8")

    # MotifVideoSingleTransformerBlock 클래스 범위만 추출
    # 클래스 시작 ~ 다음 최상위 클래스 시작 전까지
    class_match = re.search(
        r"class MotifVideoSingleTransformerBlock\(.*?\n(.*?)(?=\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert class_match, "MotifVideoSingleTransformerBlock 클래스를 파싱할 수 없음"
    class_body = class_match.group(1)

    # __init__ 메서드 범위만 추출 (첫 def __init__ ~ 다음 def 까지)
    init_match = re.search(
        r"def __init__\(.*?\n(.*?)(?=\n    def |\Z)",
        class_body,
        re.DOTALL,
    )
    assert init_match, "MotifVideoSingleTransformerBlock.__init__ 를 파싱할 수 없음"
    init_body = init_match.group(1)

    # 수동 cast 패턴 검색
    cast_pattern = re.compile(r"self\.attn\s*=\s*self\.attn\.to\(|self\.attn\.to\(")
    matches = cast_pattern.findall(init_body)
    assert not matches, (
        f"MotifVideoSingleTransformerBlock.__init__ 에 수동 cast 패턴이 남아 있음: {matches}"
    )


# ---------------------------------------------------------------------------
# Test 7: legacy Attention(...) 생성 패턴 부재 (grep 검증)
# ---------------------------------------------------------------------------

def test_no_legacy_diffusers_attention_in_single_block_init():
    """MotifVideoSingleTransformerBlock.__init__ 안에 `self.attn = Attention(` 패턴이
    없어야 한다. P3.1 교체 완료 여부를 구조적으로 확인한다."""
    import re

    src = (_BASE / "transformer_motif_video.py").read_text(encoding="utf-8")

    class_match = re.search(
        r"class MotifVideoSingleTransformerBlock\(.*?\n(.*?)(?=\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert class_match, "MotifVideoSingleTransformerBlock 클래스를 파싱할 수 없음"
    class_body = class_match.group(1)

    init_match = re.search(
        r"def __init__\(.*?\n(.*?)(?=\n    def |\Z)",
        class_body,
        re.DOTALL,
    )
    assert init_match, "MotifVideoSingleTransformerBlock.__init__ 를 파싱할 수 없음"
    init_body = init_match.group(1)

    # diffusers Attention 직접 생성 패턴 — MotifVideoAttention 과 구별
    legacy_pattern = re.compile(r"self\.attn\s*=\s*Attention\s*\(")
    matches = legacy_pattern.findall(init_body)
    assert not matches, (
        f"MotifVideoSingleTransformerBlock.__init__ 에 legacy Attention(...) 패턴이 남아 있음: {matches}"
    )


def test_motif_video_attention_constructor_used_in_single_block_init():
    """MotifVideoSingleTransformerBlock.__init__ 안에
    `self.attn = MotifVideoAttention(` 패턴이 있어야 한다."""
    import re

    src = (_BASE / "transformer_motif_video.py").read_text(encoding="utf-8")

    class_match = re.search(
        r"class MotifVideoSingleTransformerBlock\(.*?\n(.*?)(?=\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert class_match
    class_body = class_match.group(1)

    init_match = re.search(
        r"def __init__\(.*?\n(.*?)(?=\n    def |\Z)",
        class_body,
        re.DOTALL,
    )
    assert init_match
    init_body = init_match.group(1)

    pattern = re.compile(r"self\.attn\s*=\s*MotifVideoAttention\s*\(")
    assert pattern.search(init_body), (
        "MotifVideoSingleTransformerBlock.__init__ 에 `self.attn = MotifVideoAttention(` 패턴이 없음"
    )


# ---------------------------------------------------------------------------
# Test 8: Dual block P3.2 완료 — MotifVideoAttention 사용 확인
# ---------------------------------------------------------------------------

def test_dual_block_still_uses_diffusers_attention():
    """P3.2 완료: MotifVideoTransformerBlock(=Dual block) 의 self.attn 은
    MotifVideoAttention 이어야 한다."""
    num_heads, head_dim, _ = _small_dims()
    dual_block = MotifVideoTransformerBlock(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
    )
    assert isinstance(dual_block.attn, MotifVideoAttention), (
        f"Dual block self.attn 이 MotifVideoAttention 이어야 하는데 "
        f"{type(dual_block.attn).__name__} 임 — P3.2 교체 누락."
    )


def test_dual_block_attn_type_name_not_motif():
    """P3.2 완료: MotifVideoTransformerBlock.attn 의 __name__ 이 'MotifVideoAttention' 이어야 함."""
    num_heads, head_dim, _ = _small_dims()
    dual_block = MotifVideoTransformerBlock(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
    )
    assert type(dual_block.attn).__name__ == "MotifVideoAttention", (
        "Dual block attn 이 MotifVideoAttention 이어야 하는데 "
        f"{type(dual_block.attn).__name__} 임 — P3.2 교체 누락."
    )


# ---------------------------------------------------------------------------
# Test: forward 배치 크기 2 — 배치 독립성
# ---------------------------------------------------------------------------

def test_block_forward_batch_independence():
    """배치 크기 2 로 forward 시, 각 배치 슬롯의 출력이 단독 forward 와 동일해야 한다."""
    num_heads, head_dim, hidden = _dims()
    block = _make_block()
    block.eval()

    L_vis, L_txt = 4, 3
    temb_0 = torch.randn(1, hidden)
    temb_1 = torch.randn(1, hidden)
    hs_0 = torch.randn(1, L_vis, hidden)
    hs_1 = torch.randn(1, L_vis, hidden)
    ehs_0 = torch.randn(1, L_txt, hidden)
    ehs_1 = torch.randn(1, L_txt, hidden)

    with torch.no_grad():
        # 배치 묶어서 forward
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

        # 각각 따로 forward
        out_hs_0, out_ehs_0 = block(
            hidden_states=hs_0,
            encoder_hidden_states=ehs_0,
            temb=temb_0,
            attention_mask=None,
            image_rotary_emb=None,
        )
        out_hs_1, out_ehs_1 = block(
            hidden_states=hs_1,
            encoder_hidden_states=ehs_1,
            temb=temb_1,
            attention_mask=None,
            image_rotary_emb=None,
        )

    # SDPA 는 배치 독립적이어야 함
    assert torch.allclose(out_hs_batch[0], out_hs_0[0], atol=1e-5), (
        "배치 슬롯 0: batched forward 와 단독 forward 출력이 다름"
    )
    assert torch.allclose(out_hs_batch[1], out_hs_1[0], atol=1e-5), (
        "배치 슬롯 1: batched forward 와 단독 forward 출력이 다름"
    )
