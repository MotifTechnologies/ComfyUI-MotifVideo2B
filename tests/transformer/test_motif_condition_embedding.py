# tests/transformer/test_motif_condition_embedding.py
#
# P1.1 — MotifVideoConditionEmbedding.forward dtype 계약 블라인드 검증
#
# 검증 대상:
#   1. fp8 weight storage (float8_e4m3fn, float8_e5m2) → conditioning.dtype == bfloat16
#   2. bf16 weight storage → conditioning.dtype == bfloat16 (기존 경로 유지)
#   3. 반환 tuple (conditioning, token_replace_emb) 형태 + token_replace_emb is None
#   4. pooled_projection 제공 시 dtype 일관성
#   5. timestep scalar dtype 변화 (int, float32, float64) → forward 동작 유지
#   6. float8_e5m2 variant 도 fp8 분기로 진입하는지
#
# CUDA-free 설계: comfy.ops / comfy.ldm 전부 mock. CPU 에서 실행 가능.
# fp8 dtype 은 torch.float8_e4m3fn/e5m2 이 torch 2.1+ 에서 지원됨.
# scaled_mm 실행 없이 dtype 계약만 검증 (GPU 불필요).

from __future__ import annotations

import sys
import types
import importlib.util as ilu
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# comfy stub builder (teardown-safe)
# ---------------------------------------------------------------------------
# conftest.py 가 이미 diffusers stub + models namespace 를 설치한다.
# 추가로 comfy.ldm 경로와 comfy.ops mock 이 필요하다.
#
# sys.modules pollution 방지:
#   _COMFY_STUB_KEYS 목록의 기존 값을 모듈 로드 전에 snapshot 하고,
#   module-scoped autouse fixture 가 teardown 시 복원한다.
#   conftest._stub_teardown_session 과 동일한 패턴.

_COMFY_STUB_KEYS = (
    "comfy",
    "comfy.ops",
    "comfy.ldm",
    "comfy.ldm.modules",
    "comfy.ldm.modules.attention",
)

# collection 시점에 snapshot (conftest import 직후, test 파일 import 시)
_comfy_pre_snapshot: dict = {k: sys.modules.get(k) for k in _COMFY_STUB_KEYS}


def _install_comfy_stubs() -> None:
    """comfy.ops + comfy.ldm.modules.attention stub 설치."""
    class _MockOps:
        class Linear(nn.Linear):
            def __init__(self, *a, dtype=None, device=None, **kw):
                super().__init__(*a, **kw)

        class LayerNorm(nn.LayerNorm):
            def __init__(self, *a, dtype=None, device=None, **kw):
                super().__init__(*a, **kw)

        class Conv3d(nn.Conv3d):
            def __init__(self, *a, dtype=None, device=None, **kw):
                super().__init__(*a, **kw)

    mock_comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    mock_ops = types.ModuleType("comfy.ops")
    mock_ops.disable_weight_init = _MockOps
    mock_comfy.ops = mock_ops
    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.ops"] = mock_ops

    for name in ["comfy.ldm", "comfy.ldm.modules", "comfy.ldm.modules.attention"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["comfy.ldm.modules.attention"].optimized_attention = lambda *a, **kw: None


def _restore_comfy_stubs(snapshot: dict) -> None:
    """snapshot 기준으로 sys.modules 를 복원한다."""
    for name, mod in snapshot.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


# module-level 에서 즉시 설치 (transformer 모듈 로드에 필요)
_install_comfy_stubs()


@pytest.fixture(autouse=True, scope="module")
def _comfy_stubs_teardown():
    """module 종료 시 comfy stub 을 snapshot 으로 복원한다.

    conftest._stub_teardown_session 과 동일한 패턴.
    setup 은 이미 module-level 에서 완료됐으므로 yield 전 작업 없음.
    """
    yield
    _restore_comfy_stubs(_comfy_pre_snapshot)


def _load_transformer_module():
    """models.transformer.transformer_motif_video 를 importlib 로 로드."""
    if "models.transformer.transformer_motif_video" in sys.modules:
        return sys.modules["models.transformer.transformer_motif_video"]

    # models namespace 가 conftest 에 의해 이미 주입돼 있어야 함
    t_ns = sys.modules.get("models.transformer") or types.ModuleType("models.transformer")
    sys.modules["models.transformer"] = t_ns

    transformer_dir = _REPO_ROOT / "models" / "transformer"
    for mod_name in ["ops_primitives", "tread_mixin", "accelerate_patch", "attention"]:
        full_name = f"models.transformer.{mod_name}"
        if full_name not in sys.modules:
            spec = ilu.spec_from_file_location(full_name, transformer_dir / f"{mod_name}.py")
            m = ilu.module_from_spec(spec)
            sys.modules[full_name] = m
            setattr(t_ns, mod_name, m)
            spec.loader.exec_module(m)

    full_name = "models.transformer.transformer_motif_video"
    spec = ilu.spec_from_file_location(full_name, transformer_dir / "transformer_motif_video.py")
    m = ilu.module_from_spec(spec)
    sys.modules[full_name] = m
    setattr(t_ns, "transformer_motif_video", m)
    spec.loader.exec_module(m)
    return m


_tmv = _load_transformer_module()
MotifVideoConditionEmbedding = _tmv.MotifVideoConditionEmbedding

# ---------------------------------------------------------------------------
# fp8 dtype 가용성 체크
# ---------------------------------------------------------------------------

_FP8_DTYPES_AVAILABLE = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 256  # time_embed_dim 과 일치

def _make_time_proj_mock(out_dtype: torch.dtype = torch.float32):
    """diffusers.Timesteps stub 은 identity forward 를 반환해 shape mismatch 가 발생한다.
    sinusoidal embedding (batch, 256) 을 반환하는 float32 mock 으로 대체.

    time_proj 는 nn.Module 이므로 patch.object(emb, 'time_proj', ...) 는
    MagicMock 치환이 불가 (TypeError: cannot assign MagicMock as child module).
    대신 time_proj.forward 를 직접 patch 한다.
    """
    def _mock_time_proj_forward(timestep: torch.Tensor) -> torch.Tensor:
        batch = timestep.shape[0] if timestep.ndim >= 1 else 1
        return torch.randn(batch, EMBEDDING_DIM, dtype=out_dtype)
    return _mock_time_proj_forward


def _make_te_mock(out_dtype: torch.dtype = torch.bfloat16):
    """timestep_embedder forward mock: 받은 input dtype 기록 후 out_dtype 으로 반환."""
    received: list[torch.dtype] = []

    def _mock_te(x: torch.Tensor) -> torch.Tensor:
        received.append(x.dtype)
        batch = x.shape[0] if x.ndim >= 1 else 1
        return torch.zeros(batch, EMBEDDING_DIM, dtype=out_dtype)

    return _mock_te, received


def _make_text_embedder_mock(embedding_dim: int = EMBEDDING_DIM):
    """text_embedder forward mock: bf16 출력 반환."""
    def _mock_text_emb(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0] if x.ndim >= 1 else 1
        return torch.zeros(batch, embedding_dim, dtype=torch.bfloat16)
    return _mock_text_emb


def _make_embedding(
    embedding_dim: int = EMBEDDING_DIM,
    pooled_projection_dim: int | None = None,
) -> MotifVideoConditionEmbedding:
    """기본 weight storage dtype = bfloat16 로 임베딩 생성."""
    emb = MotifVideoConditionEmbedding(
        embedding_dim=embedding_dim,
        pooled_projection_dim=pooled_projection_dim,
    )
    return emb.to(torch.bfloat16)


def _cast_params_to_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """모듈 파라미터를 직접 dtype 으로 교체 (mock weight storage)."""
    for name, param in list(module.named_parameters(recurse=False)):
        setattr(module, name, nn.Parameter(param.data.to(dtype), requires_grad=False))
    for child in module.children():
        _cast_params_to_dtype(child, dtype)


# ---------------------------------------------------------------------------
# 1. 기본 경로: bf16 weight storage → conditioning.dtype == bfloat16
#
# 설계 노트:
#   conftest 의 diffusers.Timesteps stub 은 identity forward 를 반환하므로
#   shape mismatch (1x1 vs 256x256) 가 발생한다.
#   time_proj 를 sinusoidal embedding 형태(float32, shape (batch, 256)) 로 mock.
#   timestep_embedder.forward 도 mock 하여 dtype 계약만 검증한다.
# ---------------------------------------------------------------------------

class TestBf16Path:
    def test_bf16_conditioning_dtype(self):
        """bf16 weight storage → conditioning.dtype is bfloat16."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, received = _make_te_mock(out_dtype=torch.bfloat16)
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            conditioning, token_replace_emb = emb(torch.tensor([0.5]))
        assert conditioning.dtype == torch.bfloat16, (
            f"bf16 path: expected conditioning.dtype=bfloat16, got {conditioning.dtype}"
        )

    def test_bf16_token_replace_emb_is_none(self):
        """bf16 path 에서 token_replace_emb 는 None 이어야 한다."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            _, token_replace_emb = emb(torch.tensor([0.5]))
        assert token_replace_emb is None, (
            f"token_replace_emb must be None, got {token_replace_emb}"
        )

    def test_bf16_return_is_tuple_length_2(self):
        """반환 타입은 tuple, 길이 2."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            result = emb(torch.tensor([0.5]))
        assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
        assert len(result) == 2, f"expected len=2, got {len(result)}"

    def test_bf16_conditioning_is_tensor(self):
        """conditioning 은 torch.Tensor 여야 한다."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            conditioning, _ = emb(torch.tensor([0.5]))
        assert isinstance(conditioning, torch.Tensor), (
            f"conditioning must be Tensor, got {type(conditioning)}"
        )


# ---------------------------------------------------------------------------
# 2. fp8 분기: float8_e4m3fn weight storage → conditioning.dtype == bfloat16
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FP8_DTYPES_AVAILABLE, reason="torch.float8_e4m3fn not available")
class TestFp8E4m3Path:
    """float8_e4m3fn weight storage → forward dtype 계약 검증.

    CPU pod 에서는 `torch._scaled_mm` fp8 kernel 실행 불가. 본 유닛은
    `forward` 의 dtype 계약 (fp8 분기 → bf16 input, Issue #25 회귀 가드) 만 검증.

    실제 fp8 kernel 경로 (scaled_mm `out_dtype=bf16 + bias` 정상 진입,
    fallback 0회) 는 GPU pod 에서 검증 — 플랜
    `.plans/20260423-fp8-bias-fallback-fix/03_checklist.md` P3.4 (F2
    `fallback_log_count = 0`).
    """

    def test_fp8_e4m3fn_conditioning_dtype_is_bfloat16(self):
        """float8_e4m3fn weight storage → conditioning.dtype must be bfloat16."""
        import unittest.mock as mock
        emb = _make_embedding()
        _cast_params_to_dtype(emb.timestep_embedder, torch.float8_e4m3fn)

        def _mock_forward(x: torch.Tensor) -> torch.Tensor:
            assert x.dtype == torch.bfloat16, (
                f"fp8 path: timestep_embedder 는 bfloat16 input 을 받아야 함, got {x.dtype}"
            )
            return torch.zeros(x.shape[0], EMBEDDING_DIM, dtype=torch.bfloat16)

        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=_mock_forward):
            conditioning, token_replace_emb = emb(torch.tensor([0.5]))

        assert conditioning.dtype == torch.bfloat16, (
            f"fp8_e4m3fn path: conditioning.dtype must be bfloat16, got {conditioning.dtype}"
        )
        assert token_replace_emb is None

    def test_fp8_e4m3fn_timestep_embedder_receives_bfloat16_input(self):
        """fp8 경로에서 timestep_embedder 가 bfloat16 input 을 받는지 검증."""
        import unittest.mock as mock
        emb = _make_embedding()
        _cast_params_to_dtype(emb.timestep_embedder, torch.float8_e4m3fn)

        te_mock, received = _make_te_mock(out_dtype=torch.bfloat16)
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            emb(torch.tensor([1.0]))

        assert len(received) == 1
        assert received[0] == torch.bfloat16, (
            f"fp8_e4m3fn path: input to timestep_embedder must be bfloat16, "
            f"got {received[0]}"
        )


@pytest.mark.skipif(not _FP8_DTYPES_AVAILABLE, reason="torch.float8_e5m2 not available")
class TestFp8E5m2Path:
    """float8_e5m2 weight storage → forward dtype 계약 검증.

    CPU pod 에서는 `torch._scaled_mm` fp8 kernel 실행 불가. 본 유닛은
    `forward` 의 dtype 계약 (fp8 분기 → bf16 input, Issue #25 회귀 가드) 만 검증.

    실제 fp8 kernel 경로 (scaled_mm `out_dtype=bf16 + bias` 정상 진입,
    fallback 0회) 는 GPU pod 에서 검증 — 플랜
    `.plans/20260423-fp8-bias-fallback-fix/03_checklist.md` P3.4 (F2
    `fallback_log_count = 0`).
    """

    def test_fp8_e5m2_variant_triggers_fp8_branch(self):
        """float8_e5m2 variant 도 fp8 분기로 진입해야 한다.

        fp8_e5m2 weight 를 가진 timestep_embedder 에서 호출 시
        bfloat16 input 이 전달되는지 확인.
        """
        import unittest.mock as mock
        emb = _make_embedding()
        _cast_params_to_dtype(emb.timestep_embedder, torch.float8_e5m2)

        te_mock, received = _make_te_mock(out_dtype=torch.bfloat16)
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            emb(torch.tensor([1.0]))

        assert len(received) == 1
        assert received[0] == torch.bfloat16, (
            f"fp8_e5m2 path: input to timestep_embedder must be bfloat16, "
            f"got {received[0]}. float8_e5m2 must trigger fp8 branch."
        )

    def test_fp8_e5m2_token_replace_emb_is_none(self):
        """fp8_e5m2 path 에서도 token_replace_emb 는 None."""
        import unittest.mock as mock
        emb = _make_embedding()
        _cast_params_to_dtype(emb.timestep_embedder, torch.float8_e5m2)

        te_mock, _ = _make_te_mock(out_dtype=torch.bfloat16)
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            _, token_replace_emb = emb(torch.tensor([0.5]))

        assert token_replace_emb is None


# ---------------------------------------------------------------------------
# 3. pooled_projection 제공 시 dtype 일관성
# ---------------------------------------------------------------------------

class TestPooledProjection:
    def test_pooled_projection_dtype_consistency_bf16(self):
        """pooled_projection 제공 시 conditioning + text_embedder 합산 후 dtype = bfloat16."""
        import unittest.mock as mock
        pooled_dim = 128
        emb = _make_embedding(embedding_dim=EMBEDDING_DIM, pooled_projection_dim=pooled_dim)
        ts = torch.tensor([0.5])
        pooled = torch.randn(1, pooled_dim).to(torch.bfloat16)

        te_mock, _ = _make_te_mock(out_dtype=torch.bfloat16)
        txt_mock = _make_text_embedder_mock(embedding_dim=EMBEDDING_DIM)
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock), \
             mock.patch.object(emb.text_embedder, "forward", side_effect=txt_mock):
            conditioning, _ = emb(ts, pooled_projection=pooled)
        assert conditioning.dtype == torch.bfloat16, (
            f"pooled_projection path: conditioning.dtype must be bfloat16, got {conditioning.dtype}"
        )

    def test_pooled_projection_none_does_not_invoke_text_embedder(self):
        """pooled_projection=None 시 text_embedder.forward 가 호출되지 않아야 한다.

        text_embedder 는 nn.Module 이므로 patch.object(emb, 'text_embedder', ...) 은
        MagicMock 치환 불가. text_embedder.forward 를 spy 로 패치하여 호출 여부 확인.
        """
        import unittest.mock as mock
        pooled_dim = 128
        emb = _make_embedding(embedding_dim=EMBEDDING_DIM, pooled_projection_dim=pooled_dim)

        te_mock, _ = _make_te_mock()
        txt_call_count = []

        def _spy_text_embedder_forward(x):
            txt_call_count.append(1)
            batch = x.shape[0] if x.ndim >= 1 else 1
            return torch.zeros(batch, EMBEDDING_DIM, dtype=torch.bfloat16)

        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock), \
             mock.patch.object(emb.text_embedder, "forward", side_effect=_spy_text_embedder_forward):
            emb(torch.tensor([0.5]), pooled_projection=None)

        assert len(txt_call_count) == 0, (
            f"pooled_projection=None 인데 text_embedder.forward 가 {len(txt_call_count)}회 호출됨"
        )

    def test_pooled_projection_none_when_no_text_embedder(self):
        """pooled_projection_dim=None 이면 text_embedder 속성이 없어야 한다."""
        emb = _make_embedding(embedding_dim=EMBEDDING_DIM, pooled_projection_dim=None)
        assert not hasattr(emb, "text_embedder"), (
            "pooled_projection_dim=None 인데 text_embedder 가 존재함"
        )


# ---------------------------------------------------------------------------
# 4. 경계값: timestep scalar dtype 변화 → forward 안전성
#
# 설계 노트:
#   time_proj stub + timestep_embedder mock 을 사용해 shape/dtype 계약만 검증.
#   int64 timestep 은 일부 구현에서 time_proj 에서 거부할 수 있다.
#   여기서는 time_proj 도 mock 하므로 input dtype 의 전파 경로를 검증한다.
# ---------------------------------------------------------------------------

class TestTimestepEdgeCases:
    def test_timestep_int_tensor(self):
        """timestep 이 int dtype tensor 일 때도 forward 가 crash 없이 동작해야 한다."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        ts = torch.tensor([100], dtype=torch.int64)
        try:
            with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
                 mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
                conditioning, _ = emb(ts)
            assert isinstance(conditioning, torch.Tensor)
        except Exception as e:
            pytest.fail(f"timestep int64 tensor → forward raised: {e}")

    def test_timestep_float32_tensor(self):
        """timestep 이 float32 tensor 일 때 conditioning.dtype = bfloat16."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock(out_dtype=torch.bfloat16)
        ts = torch.tensor([0.5], dtype=torch.float32)
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            conditioning, _ = emb(ts)
        assert conditioning.dtype == torch.bfloat16, (
            f"float32 timestep → conditioning.dtype must be bfloat16, got {conditioning.dtype}"
        )

    def test_timestep_float64_tensor(self):
        """timestep 이 float64 tensor 일 때 forward 가 crash 없이 동작해야 한다."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        ts = torch.tensor([0.5], dtype=torch.float64)
        try:
            with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
                 mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
                conditioning, _ = emb(ts)
            assert isinstance(conditioning, torch.Tensor)
        except Exception as e:
            pytest.fail(f"timestep float64 tensor → forward raised: {e}")

    def test_timestep_zero(self):
        """timestep = 0 경계값 — conditioning 이 유효한 tensor 를 반환해야 한다."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        ts = torch.tensor([0.0])
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            conditioning, _ = emb(ts)
        assert isinstance(conditioning, torch.Tensor)
        assert not torch.isnan(conditioning).any(), "timestep=0 → NaN in conditioning"

    def test_timestep_batch_size_1(self):
        """배치 크기 1 — 정상 동작."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            conditioning, token_replace_emb = emb(torch.tensor([0.5]))
        assert conditioning.ndim >= 1
        assert token_replace_emb is None

    def test_timestep_large_value(self):
        """timestep 이 매우 큰 값일 때 NaN/Inf 없이 반환돼야 한다."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        ts = torch.tensor([1000.0])
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            conditioning, _ = emb(ts)
        assert not torch.isnan(conditioning).any(), "large timestep → NaN in conditioning"
        assert not torch.isinf(conditioning).any(), "large timestep → Inf in conditioning"


# ---------------------------------------------------------------------------
# 5. 회귀 risk: bf16 분기가 fp8 변경으로 영향 받지 않는지
# ---------------------------------------------------------------------------

class TestBf16RegressionGuard:
    def test_bf16_path_not_affected_by_fp8_dtype_check(self):
        """bf16 weight 인 경우 fp8 분기가 발동되지 않아야 한다.

        bf16 경로에서 timestep_embedder 에 전달되는 input dtype 이
        bf16 이어야 한다 (float32 가 그대로 전달되면 regression).
        """
        import unittest.mock as mock
        emb = _make_embedding()
        param_dtype = next(emb.timestep_embedder.parameters()).dtype
        assert param_dtype == torch.bfloat16, f"test setup: expected bf16, got {param_dtype}"

        te_mock, received = _make_te_mock(out_dtype=torch.bfloat16)
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            emb(torch.tensor([0.5]))

        assert len(received) == 1
        assert received[0] == torch.bfloat16, (
            f"bf16 path: timestep_embedder input should be bfloat16, "
            f"got {received[0]} — float32 bleed regression"
        )

    def test_bf16_path_param_dtype_unchanged_after_forward(self):
        """forward 호출 후 timestep_embedder 파라미터 dtype 이 변경되지 않아야 한다."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            emb(torch.tensor([0.5]))
        for p in emb.timestep_embedder.parameters():
            assert p.dtype == torch.bfloat16, (
                f"bf16 param dtype changed after forward: {p.dtype}"
            )


# ---------------------------------------------------------------------------
# 6. 타입 불일치 엣지케이스
# ---------------------------------------------------------------------------

class TestTypeEdgeCases:
    def test_timestep_wrong_shape_raises_or_handles(self):
        """timestep 이 2D tensor (N, 1) 일 때 forward 가 crash 하지 않거나 명확한 에러를 냄."""
        import unittest.mock as mock
        emb = _make_embedding()
        te_mock, _ = _make_te_mock()
        ts_2d = torch.tensor([[0.5]])  # shape (1, 1)
        try:
            with mock.patch.object(emb.time_proj, "forward", side_effect=_make_time_proj_mock()), \
                 mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
                conditioning, _ = emb(ts_2d)
            assert isinstance(conditioning, torch.Tensor)
        except (RuntimeError, ValueError, TypeError):
            pass  # shape 불일치로 crash → 명확한 에러, acceptable

    def test_float32_does_not_bleed_without_fp8(self):
        """bf16 weight 환경에서 diffusers.Timesteps 의 float32 output 이
        conditioning 까지 번지지 않아야 한다 (Issue #25 regression guard).

        time_proj 가 float32 를 반환해도 conditioning 은 bfloat16 이어야 한다.
        핵심 검증: forward 내에서 float32 → bf16 cast 가 수행되는지.
        """
        import unittest.mock as mock
        emb = _make_embedding()

        # time_proj.forward 가 float32 를 반환 (diffusers Timesteps 실제 동작)
        def _float32_time_proj_forward(x):
            batch = x.shape[0] if x.ndim >= 1 else 1
            return torch.randn(batch, EMBEDDING_DIM, dtype=torch.float32)

        # timestep_embedder 가 받는 input dtype 을 기록하면서 bf16 출력 반환
        te_mock, received = _make_te_mock(out_dtype=torch.bfloat16)

        with mock.patch.object(emb.time_proj, "forward", side_effect=_float32_time_proj_forward), \
             mock.patch.object(emb.timestep_embedder, "forward", side_effect=te_mock):
            conditioning, _ = emb(torch.tensor([0.5]))

        # bf16 경로: timestep_embedder 가 bf16 input 을 받아야 함
        assert len(received) == 1
        assert received[0] == torch.bfloat16, (
            f"float32 time_proj output must be cast to bfloat16 before timestep_embedder. "
            f"got {received[0]} (Issue #25 regression guard)"
        )
        assert conditioning.dtype == torch.bfloat16, (
            f"float32 time_proj output must not bleed to conditioning. "
            f"conditioning.dtype={conditioning.dtype} (Issue #25 regression)"
        )
