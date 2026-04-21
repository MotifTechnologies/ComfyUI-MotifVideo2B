"""P2.1 엣지 케이스: MotifVideoAttention 스켈레톤 심층 검증.

기존 테스트가 커버하지 않는 영역을 독립 시각으로 검증:
- 포인트 1: ops.Linear 서브클래스 여부
- 포인트 2: dtype/device 전파
- 포인트 3: 커스텀 operations 주입
- 포인트 4 (심화): norm_q.weight.shape 및 타입 확인
- 포인트 5: use_sage 기본값 False
- 포인트 6: forward() → NotImplementedError (다른 에러 타입 구분)
- 포인트 7: registered submodule 개수
- 포인트 8: None attribute 가 state_dict 에 누출 안 됨
- 포인트 9: eps 전파 (custom eps 가 RMSNorm 에 실제 적용되는지)
- 포인트 10: bias=False 경로
- 포인트 11: added_kv=True + pre_only=True 조합 (설계 경계)

CUDA-free: comfy.ops 를 mock 으로 교체하여 CPU 만으로 실행 가능.
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
# comfy.ops mock — dtype/device 전파를 실제로 검증하기 위해 파라미터를 보존
# ---------------------------------------------------------------------------

def _make_comfy_ops_mock(track_instances: bool = False):
    """
    track_instances=True 일 때 생성된 Linear/RMSNorm 인스턴스를 내부 리스트에 기록.
    이를 통해 '어떤 ops 클래스가 실제 사용됐는지' 확인 가능.
    """
    class _MockOps:
        _linear_instances: list = []
        _rmsnorm_instances: list = []

        class Linear(nn.Linear):
            def __init__(self, in_features, out_features, bias=True,
                         dtype=None, device=None, **kwargs):
                # dtype/device 를 실제로 PyTorch 에 전달
                factory_kwargs = {}
                if dtype is not None:
                    factory_kwargs["dtype"] = dtype
                if device is not None:
                    factory_kwargs["device"] = device
                super().__init__(in_features, out_features, bias=bias, **factory_kwargs)
                if track_instances:
                    _MockOps._linear_instances.append(self)

        class RMSNorm(nn.RMSNorm):
            def __init__(self, normalized_shape, eps=None, dtype=None, device=None, **kwargs):
                actual_eps = eps if eps is not None else 1e-6
                factory_kwargs = {}
                if dtype is not None:
                    factory_kwargs["dtype"] = dtype
                if device is not None:
                    factory_kwargs["device"] = device
                super().__init__(normalized_shape, eps=actual_eps, **factory_kwargs)
                # eps 를 직접 저장하여 검증 가능하게 함
                self.eps = actual_eps
                if track_instances:
                    _MockOps._rmsnorm_instances.append(self)

    mock_comfy = types.ModuleType("comfy")
    mock_ops = types.ModuleType("comfy.ops")
    mock_ops.disable_weight_init = _MockOps
    mock_comfy.ops = mock_ops
    return mock_comfy, mock_ops, _MockOps


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

    mock_comfy, mock_ops, _ = _make_comfy_ops_mock()
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
_spec = _ilu.spec_from_file_location("attention_edge", _ATTENTION_PATH)
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
    return hidden // head_dim, head_dim   # 12, 128


# ---------------------------------------------------------------------------
# 포인트 1: ops.Linear 서브클래스 여부
# to_q/to_k/to_v 가 ops.Linear (nn.Linear 서브클래스) 인지 확인
# ---------------------------------------------------------------------------

def test_point1_to_q_is_nn_linear_subclass():
    """to_q 는 ops.Linear 이어야 하며, 최소한 nn.Linear 서브클래스여야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False)
    assert isinstance(attn.to_q, nn.Linear), (
        f"to_q 타입 {type(attn.to_q)} 은 nn.Linear 서브클래스가 아님"
    )
    assert isinstance(attn.to_k, nn.Linear), (
        f"to_k 타입 {type(attn.to_k)} 은 nn.Linear 서브클래스가 아님"
    )
    assert isinstance(attn.to_v, nn.Linear), (
        f"to_v 타입 {type(attn.to_v)} 은 nn.Linear 서브클래스가 아님"
    )


def test_point1_to_out_linear_is_nn_linear_subclass():
    """dual block 의 to_out[0] 도 nn.Linear 서브클래스여야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    assert isinstance(attn.to_out[0], nn.Linear), (
        f"to_out[0] 타입 {type(attn.to_out[0])} 은 nn.Linear 서브클래스가 아님"
    )


# ---------------------------------------------------------------------------
# 포인트 2: dtype/device 전파
# float16 + cpu 로 생성 시 Linear 와 RMSNorm 전체에 전파되어야 한다
# ---------------------------------------------------------------------------

def _load_attn_with_mock():
    """dtype/device 전파 검증용 mock ops 로 attention 모듈을 재로드하여 반환."""
    mock_comfy, mock_ops, MockOps = _make_comfy_ops_mock(track_instances=False)
    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.ops"] = mock_ops
    spec = _ilu.spec_from_file_location("attention_dtype_test", _ATTENTION_PATH)
    mod = _ilu.module_from_spec(spec)
    mod._DEFAULT_OPS = None
    spec.loader.exec_module(mod)
    return mod.MotifVideoAttention


def _restore_comfy(prev_comfy, prev_comfy_ops):
    if prev_comfy is None:
        sys.modules.pop("comfy", None)
    else:
        sys.modules["comfy"] = prev_comfy
    if prev_comfy_ops is None:
        sys.modules.pop("comfy.ops", None)
    else:
        sys.modules["comfy.ops"] = prev_comfy_ops


def test_point2_dtype_propagation_float16():
    """dtype=torch.float16 전달 시 to_q/k/v.weight 가 float16 이어야 한다 (single block)."""
    if _should_use_real_comfy():
        pytest.skip("real comfy 환경: dtype mock 검증 스킵")

    prev_comfy = sys.modules.get("comfy", None)
    prev_comfy_ops = sys.modules.get("comfy.ops", None)
    try:
        Attn = _load_attn_with_mock()
        num_heads, head_dim = _dims()
        attn = Attn(
            num_heads, head_dim,
            qk_norm="rms_norm", pre_only=True, added_kv=False,
            dtype=torch.float16, device="cpu",
        )
        assert attn.to_q.weight.dtype == torch.float16, (
            f"dtype=float16 전달했으나 to_q.weight.dtype={attn.to_q.weight.dtype}"
        )
        assert attn.to_k.weight.dtype == torch.float16
        assert attn.to_v.weight.dtype == torch.float16
    finally:
        _restore_comfy(prev_comfy, prev_comfy_ops)


def test_point2_dtype_device_propagation():
    """dtype + device 전파를 Linear 와 RMSNorm 전체에 대해 검증 (dual block)."""
    if _should_use_real_comfy():
        pytest.skip("real comfy 환경: dtype mock 검증 스킵")

    prev_comfy = sys.modules.get("comfy", None)
    prev_comfy_ops = sys.modules.get("comfy.ops", None)
    try:
        Attn = _load_attn_with_mock()
        num_heads, head_dim = _dims()
        cpu = torch.device("cpu")
        d = Attn(
            num_heads, head_dim, qk_norm="rms_norm",
            pre_only=False, added_kv=True,
            dtype=torch.float16, device=cpu,
        )
        # Linear weights — add_* 포함
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
            lin = getattr(d, name)
            assert lin.weight.dtype == torch.float16, f"{name}.weight dtype"
            assert lin.weight.device.type == cpu.type, f"{name}.weight device"
        # to_out[0] (Linear)
        assert d.to_out[0].weight.dtype == torch.float16, "to_out[0].weight dtype"
        assert d.to_out[0].weight.device.type == cpu.type, "to_out[0].weight device"
        # RMSNorm weights (norm_q/k + norm_added_q/k)
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            rms = getattr(d, name)
            assert rms.weight.dtype == torch.float16, f"{name}.weight dtype"
            assert rms.weight.device.type == cpu.type, f"{name}.weight device"
    finally:
        _restore_comfy(prev_comfy, prev_comfy_ops)


# ---------------------------------------------------------------------------
# 포인트 3: 커스텀 operations 주입
# operations=my_ops 전달 시 내부 Linear 가 my_ops.Linear 인스턴스이어야 한다
# ---------------------------------------------------------------------------

def test_point3_custom_operations_injection():
    """operations 인자로 전달된 커스텀 ops 클래스가 내부에 실제 사용되어야 한다."""
    created_by_custom: list = []

    class _CustomOps:
        class Linear(nn.Linear):
            def __init__(self, in_f, out_f, bias=True, dtype=None, device=None, **kw):
                super().__init__(in_f, out_f, bias=bias)
                created_by_custom.append("Linear")

        class RMSNorm(nn.RMSNorm):
            def __init__(self, normalized_shape, eps=None, dtype=None, device=None, **kw):
                super().__init__(normalized_shape, eps=eps or 1e-6)
                created_by_custom.append("RMSNorm")

    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim,
        qk_norm="rms_norm", pre_only=True, added_kv=False,
        operations=_CustomOps
    )

    assert len(created_by_custom) > 0, (
        "operations=_CustomOps 전달했으나 _CustomOps.Linear/RMSNorm 이 한 번도 호출되지 않음 "
        "— operations 인자가 무시되고 default fallback 이 사용된 것으로 추정"
    )
    assert isinstance(attn.to_q, _CustomOps.Linear), (
        f"to_q 타입 {type(attn.to_q)} 이 _CustomOps.Linear 가 아님 "
        "— operations 주입이 to_q 에 적용되지 않음"
    )


# ---------------------------------------------------------------------------
# 포인트 4 (심화): norm_q.weight.shape 및 타입
# ---------------------------------------------------------------------------

def test_point4_norm_q_weight_shape():
    """norm_q.weight.shape == [head_dim] 이어야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False
    )
    assert attn.norm_q is not None
    assert list(attn.norm_q.weight.shape) == [head_dim], (
        f"norm_q.weight.shape={list(attn.norm_q.weight.shape)}, expected=[{head_dim}]"
    )
    assert list(attn.norm_k.weight.shape) == [head_dim], (
        f"norm_k.weight.shape={list(attn.norm_k.weight.shape)}, expected=[{head_dim}]"
    )


def test_point4_norm_added_weight_shape():
    """dual block 의 norm_added_q/k.weight.shape == [head_dim]."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    assert list(attn.norm_added_q.weight.shape) == [head_dim], (
        f"norm_added_q.weight.shape={list(attn.norm_added_q.weight.shape)}"
    )
    assert list(attn.norm_added_k.weight.shape) == [head_dim], (
        f"norm_added_k.weight.shape={list(attn.norm_added_k.weight.shape)}"
    )


# ---------------------------------------------------------------------------
# 포인트 5: use_sage 기본값 False
# ---------------------------------------------------------------------------

def test_point5_use_sage_default_false():
    """use_sage 는 attribute 로 존재해야 하며 기본값이 False 여야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False
    )
    assert hasattr(attn, "use_sage"), "use_sage attribute 가 존재하지 않음"
    assert attn.use_sage is False, (
        f"use_sage 기본값이 {attn.use_sage!r} — False 가 아니면 P4.1 이전에 sage 경로가 실행되어 crash 발생 가능"
    )


def test_point5_use_sage_default_false_dual():
    """dual block 에서도 use_sage 기본값 False."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    assert hasattr(attn, "use_sage"), "dual block: use_sage attribute 없음"
    assert attn.use_sage is False


# ---------------------------------------------------------------------------
# 포인트 6: forward() → NotImplementedError (다른 에러 타입은 실패)
# ---------------------------------------------------------------------------

def test_point6_forward_raises_not_implemented_error():
    """attn(...) 호출 시 NotImplementedError 가 raise 되어야 한다.
    TypeError / AttributeError 등 다른 예외는 구현 오류.
    """
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False
    )
    with pytest.raises(NotImplementedError):
        # 임의 텐서로 호출 — 어떤 인자가 와도 NotImplementedError 여야 한다
        dummy = torch.zeros(1, 4, num_heads * head_dim)
        attn(dummy)


def test_point6_forward_not_other_exception():
    """forward 가 TypeError/AttributeError 등 다른 예외를 raise 하면 실패."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    try:
        dummy = torch.zeros(1, 4, num_heads * head_dim)
        attn(dummy)
        pytest.fail("forward 가 예외를 전혀 raise 하지 않음 — 구현 필요")
    except NotImplementedError:
        pass  # 정상
    except Exception as exc:
        pytest.fail(
            f"forward 가 NotImplementedError 가 아닌 {type(exc).__name__} 를 raise 함: {exc}"
        )


# ---------------------------------------------------------------------------
# 포인트 7: registered submodule 개수
# ---------------------------------------------------------------------------

def _count_direct_submodules(attn: nn.Module) -> int:
    """직접 자식 모듈(depth=1)만 센다."""
    return sum(1 for _ in attn.children())


def test_point7_single_block_submodule_count():
    """single block (pre_only=True, added_kv=False, qk_norm=rms_norm) 의 직접 서브모듈 수.

    expected_attn_keys.json single_block 의 파라미터 보유 모듈:
    - to_q, to_k, to_v (Linear x3)
    - norm_q, norm_k (RMSNorm x2)
    합계 = 5

    to_out=None, add_* 없으므로 5개.
    """
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False
    )
    count = _count_direct_submodules(attn)
    assert count == 5, (
        f"single block 직접 서브모듈 수 = {count}, expected = 5. "
        f"모듈 목록: {[name for name, _ in attn.named_children()]}"
    )


def test_point7_dual_block_submodule_count():
    """dual block (pre_only=False, added_kv=True, qk_norm=rms_norm) 의 직접 서브모듈 수.

    expected_attn_keys.json dual_block 의 파라미터 보유 직접 자식 모듈:
    - to_q, to_k, to_v (x3)
    - norm_q, norm_k (x2)
    - add_q_proj, add_k_proj, add_v_proj (x3)
    - norm_added_q, norm_added_k (x2)
    - to_add_out (x1)
    - to_out (ModuleList x1 — 내부 Linear/Dropout 은 ModuleList 의 자식이므로 1로 집계)
    합계 = 12  (ModuleList 자체가 1개 child)
    """
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=True
    )
    count = _count_direct_submodules(attn)
    assert count == 12, (
        f"dual block 직접 서브모듈 수 = {count}, expected = 12. "
        f"모듈 목록: {[name for name, _ in attn.named_children()]}"
    )


# ---------------------------------------------------------------------------
# 포인트 8: None attribute 가 state_dict 에 누출 안 됨
# ---------------------------------------------------------------------------

def test_point8_single_block_no_add_keys_in_state_dict():
    """single block (added_kv=False) 의 state_dict 에 add_* 키가 없어야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False
    )
    sd = attn.state_dict()
    leaked = [k for k in sd if "add_" in k]
    assert leaked == [], (
        f"added_kv=False 임에도 state_dict 에 add_* 키 노출: {leaked}"
    )


def test_point8_single_block_no_to_out_keys_in_state_dict():
    """single block (pre_only=True) 의 state_dict 에 to_out 키가 없어야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=False
    )
    sd = attn.state_dict()
    leaked = [k for k in sd if k.startswith("to_out")]
    assert leaked == [], (
        f"pre_only=True 임에도 state_dict 에 to_out 키 노출: {leaked}"
    )


# ---------------------------------------------------------------------------
# 포인트 9: eps 전파
# ---------------------------------------------------------------------------

def test_point9_eps_propagation():
    """eps=1e-5 전달 시 norm_q.eps == 1e-5 이어야 한다."""
    num_heads, head_dim = _dims()
    custom_eps = 1e-5
    attn = MotifVideoAttention(
        num_heads, head_dim,
        qk_norm="rms_norm", pre_only=True, added_kv=False,
        eps=custom_eps
    )
    assert attn.norm_q is not None, "norm_q 가 None — qk_norm='rms_norm' 이어야 함"
    assert attn.norm_q.eps == pytest.approx(custom_eps), (
        f"norm_q.eps={attn.norm_q.eps}, expected={custom_eps} — "
        "eps 인자가 RMSNorm 에 전파되지 않음"
    )
    assert attn.norm_k.eps == pytest.approx(custom_eps), (
        f"norm_k.eps={attn.norm_k.eps}, expected={custom_eps}"
    )


def test_point9_eps_propagation_dual():
    """dual block 에서 norm_added_q/k.eps 도 전파되어야 한다."""
    num_heads, head_dim = _dims()
    custom_eps = 1e-8
    attn = MotifVideoAttention(
        num_heads, head_dim,
        qk_norm="rms_norm", pre_only=False, added_kv=True,
        eps=custom_eps
    )
    assert attn.norm_added_q.eps == pytest.approx(custom_eps), (
        f"norm_added_q.eps={attn.norm_added_q.eps}, expected={custom_eps}"
    )
    assert attn.norm_added_k.eps == pytest.approx(custom_eps), (
        f"norm_added_k.eps={attn.norm_added_k.eps}, expected={custom_eps}"
    )


# ---------------------------------------------------------------------------
# 포인트 10: bias=False 경로
# ---------------------------------------------------------------------------

def test_point10_bias_false_to_q():
    """bias=False 전달 시 to_q.bias is None 이어야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim,
        qk_norm="rms_norm", pre_only=True, added_kv=False,
        bias=False
    )
    assert attn.to_q.bias is None, (
        f"bias=False 전달했으나 to_q.bias={attn.to_q.bias} — bias 인자가 무시됨"
    )
    assert attn.to_k.bias is None, "to_k.bias 가 None 이 아님"
    assert attn.to_v.bias is None, "to_v.bias 가 None 이 아님"


def test_point10_bias_false_no_bias_keys_in_state_dict():
    """bias=False 시 state_dict 에 to_q.bias 키가 없어야 한다."""
    num_heads, head_dim = _dims()
    attn = MotifVideoAttention(
        num_heads, head_dim,
        qk_norm="rms_norm", pre_only=True, added_kv=False,
        bias=False
    )
    sd = attn.state_dict()
    bias_keys = [k for k in sd if k.endswith(".bias")]
    assert bias_keys == [], (
        f"bias=False 임에도 state_dict 에 bias 키 발견: {bias_keys}"
    )


# ---------------------------------------------------------------------------
# 포인트 11: 무효 조합 (pre_only == added_kv) 은 ValueError 로 즉시 거부되어야 한다
# ---------------------------------------------------------------------------

def test_point11_invalid_combo_raises():
    """무효 조합 2가지가 모두 ValueError("unsupported combo") 를 raise 해야 한다."""
    num_heads, head_dim = _dims()
    # 조합 1: pre_only=True, added_kv=True (두 역할 섞임 — 지원 안 됨)
    with pytest.raises(ValueError, match="unsupported combo"):
        MotifVideoAttention(num_heads, head_dim, qk_norm="rms_norm", pre_only=True, added_kv=True)
    # 조합 2: pre_only=False, added_kv=False (Dual 인데 add branch 없음 — 지원 안 됨)
    with pytest.raises(ValueError, match="unsupported combo"):
        MotifVideoAttention(num_heads, head_dim, qk_norm="rms_norm", pre_only=False, added_kv=False)
