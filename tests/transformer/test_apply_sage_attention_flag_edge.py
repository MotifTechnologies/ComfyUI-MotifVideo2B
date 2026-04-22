"""P4.1 apply_sage_attention edge-case — 블라인드 독립 검증.

기존 테스트 커버리지 분석:
  [COVERED] apply_sage_attention → 모든 block.attn.use_sage=True (flag 방식)
  [COVERED] MOTIFVIDEO_ENABLE_SAGE 미설정 → use_sage 건드리지 않음 (기본 OFF)
  [COVERED] _SAGE_AVAILABLE=False → no-op
  [COVERED] hasattr(block.attn, "processor") 소스 grep 0건
  [COVERED] block.attn.processor = xDiTMotifVideoAttnProcessor() 소스 grep 0건
  [COVERED] block.attn.use_sage = True 소스 확인 (dual integration test)

  [GAP — 본 파일에서 추가 커버]
  G1. 두 번 호출 idempotent — 이미 use_sage=True 인 상태에서 재호출 OK
  G2. use_sage=False 수동 복원 가능 — 사용자가 False 로 되돌릴 수 있음
  G3. log 출력 검증 — caplog 기반 (sage 활성화 수 포함 메시지)
  G4. early-return 경로의 use_sage 보존 — opt-in 없을 시 기존 False 유지 + True 도 유지
  G5. "Patched … xDiTMotifVideoAttnProcessor" 구문 소스 부재 확인
  G6. "skipped" warning 메시지(P3.2 hasattr 가드) 소스 부재 확인
  G7. sage 미설치 경로 log 문구 존재 확인 (skip 메시지)
  G8. 빈 transformer (block 수 0) — crash 없이 정상 반환
  G9. MOTIFVIDEO_ENABLE_SAGE 미설정 시 이미 True인 블록도 건드리지 않음 (기존 True 보존)
  G10. single_transformer_blocks 만 있는 transformer — use_sage=True 설정됨

conftest.py 의 diffusers stub + models namespace 자동 주입 사용.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO_ROOT / "models"


# ---------------------------------------------------------------------------
# Module loading helpers (기존 test_apply_sage_attention_flag.py 패턴 재사용)
# ---------------------------------------------------------------------------

def _ensure_sage_ops_loaded():
    if "models.sage_ops" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "models.sage_ops", _MODELS_DIR / "sage_ops.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["models.sage_ops"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["models.sage_ops"]


def _load_compile_config():
    """compile_config 를 매 호출마다 fresh 로드 (monkeypatch 독립성 보장).
    __package__ = "models" 로 설정하여 from .sage_ops import 가 동작하도록 한다."""
    _ensure_sage_ops_loaded()
    mod_name = "models.compile_config"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        _MODELS_DIR / "compile_config.py",
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "models"
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Stub transformer helpers
# ---------------------------------------------------------------------------

class _StubAttn:
    """MotifVideoAttention 대리 — use_sage 속성만 필요."""
    def __init__(self, initial_use_sage: bool = False):
        self.use_sage = initial_use_sage


class _StubBlock:
    def __init__(self, initial_use_sage: bool = False):
        self.attn = _StubAttn(initial_use_sage)


class _StubTransformer:
    def __init__(self, n_dual: int = 2, n_single: int = 3):
        self.transformer_blocks = [_StubBlock() for _ in range(n_dual)]
        self.single_transformer_blocks = [_StubBlock() for _ in range(n_single)]


# ---------------------------------------------------------------------------
# G1: 두 번 호출 idempotent
# ---------------------------------------------------------------------------

def test_apply_sage_attention_idempotent_double_call(monkeypatch):
    """apply_sage_attention 을 동일 transformer 에 두 번 호출해도
    결과가 동일 (use_sage=True) 하고 예외가 발생하지 않는다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=2, n_single=2)

    # 첫 번째 호출
    cc.apply_sage_attention(t)
    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    assert all(b.attn.use_sage is True for b in all_blocks), "1차 호출 후 use_sage=True 아님"

    # 두 번째 호출 — 예외 없어야 하고 여전히 True
    cc.apply_sage_attention(t)
    assert all(b.attn.use_sage is True for b in all_blocks), "2차 호출 후 use_sage=True 아님"


# ---------------------------------------------------------------------------
# G2: use_sage=False 수동 복원 가능
# ---------------------------------------------------------------------------

def test_apply_sage_attention_use_sage_manually_reversible(monkeypatch):
    """apply_sage_attention 호출 후 사용자가 수동으로 use_sage=False 로 되돌릴 수 있다.
    즉, use_sage 가 일반 쓰기 가능 속성임을 검증한다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=2, n_single=1)
    cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    assert all(b.attn.use_sage is True for b in all_blocks), "apply 후 use_sage=True 아님"

    # 사용자가 수동으로 False 로 되돌리기
    for b in all_blocks:
        b.attn.use_sage = False

    assert all(b.attn.use_sage is False for b in all_blocks), (
        "수동 use_sage=False 복원 실패 — use_sage 가 읽기 전용이거나 재귀 setter 문제"
    )


# ---------------------------------------------------------------------------
# G3: log 출력 검증 — sage 활성화 메시지 포함
# ---------------------------------------------------------------------------

def test_apply_sage_attention_logs_activation_count(monkeypatch, caplog):
    """apply_sage_attention 성공 경로에서 로그 출력이 있어야 한다.
    최소 조건: 숫자(활성화된 블록 수)가 포함된 메시지 또는 'sage'/'Sage' 키워드."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=2, n_single=3)

    with caplog.at_level(logging.DEBUG, logger="models.compile_config"):
        cc.apply_sage_attention(t)

    all_log = " ".join(r.getMessage() for r in caplog.records).lower()
    assert ("sage" in all_log or "use_sage" in all_log), (
        f"apply_sage_attention 성공 경로에서 'sage' 관련 로그 메시지가 없음. "
        f"기록된 로그: {[r.getMessage() for r in caplog.records]}"
    )


def test_apply_sage_attention_log_not_patched_xdit_phrase(monkeypatch, caplog):
    """apply_sage_attention 성공 경로에서 'Patched … xDiTMotifVideoAttnProcessor' 구문이
    로그에 없어야 한다 — P4.1 이후 processor 대입 방식이 제거되었으므로."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=1, n_single=1)

    with caplog.at_level(logging.DEBUG, logger="models.compile_config"):
        cc.apply_sage_attention(t)

    all_log = " ".join(r.getMessage() for r in caplog.records)
    assert "xDiTMotifVideoAttnProcessor" not in all_log, (
        "apply_sage_attention 로그에 'xDiTMotifVideoAttnProcessor' 구문 잔류. "
        "P4.1 이후 processor 대입 로그가 제거되어야 한다."
    )


# ---------------------------------------------------------------------------
# G4: early-return 경로의 use_sage 보존
# G4a: ENABLE 미설정 + 기존 False → False 유지
# G4b: ENABLE 미설정 + 기존 True  → True 유지 (건드리지 않음)
# ---------------------------------------------------------------------------

def test_apply_sage_attention_disable_env_preserves_existing_false(monkeypatch):
    """MOTIFVIDEO_ENABLE_SAGE 미설정 시 기존 use_sage=False 블록을 건드리지 않는다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.delenv("MOTIFVIDEO_ENABLE_SAGE", raising=False)

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=2, n_single=2)
    # 기본값 False
    cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    assert all(b.attn.use_sage is False for b in all_blocks), (
        "MOTIFVIDEO_ENABLE_SAGE 미설정 경로에서 use_sage=False 가 변경됐다"
    )


def test_apply_sage_attention_disable_env_preserves_existing_true(monkeypatch):
    """MOTIFVIDEO_ENABLE_SAGE 미설정 시 이미 True 인 블록도 건드리지 않는다.
    early-return 이 순수 no-op 임을 검증한다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.delenv("MOTIFVIDEO_ENABLE_SAGE", raising=False)

    cc = _load_compile_config()
    # 미리 use_sage=True 로 설정된 블록
    t = _StubTransformer.__new__(_StubTransformer)
    t.transformer_blocks = [_StubBlock(initial_use_sage=True) for _ in range(2)]
    t.single_transformer_blocks = [_StubBlock(initial_use_sage=True) for _ in range(2)]

    cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    assert all(b.attn.use_sage is True for b in all_blocks), (
        "MOTIFVIDEO_ENABLE_SAGE 미설정에서 기존 True 블록이 False 로 변경됐다 — "
        "early-return 이 use_sage 를 건드리면 안 된다"
    )


# ---------------------------------------------------------------------------
# G5: 소스 grep — "Patched" + "xDiTMotifVideoAttnProcessor" 조합 부재
# ---------------------------------------------------------------------------

def test_source_no_patched_xdit_processor_phrase():
    """compile_config.py 소스에 'Patched … xDiTMotifVideoAttnProcessor' 구문이 없어야 한다.
    P4.1 이후 processor 대입 기반 로그 문구가 제거됐는지 확인한다."""
    src = (_MODELS_DIR / "compile_config.py").read_text(encoding="utf-8")
    assert "Patched" not in src or "xDiTMotifVideoAttnProcessor" not in src, (
        "compile_config.py 에 'Patched … xDiTMotifVideoAttnProcessor' 로그 구문 잔류.\n"
        "P4.1 이후 이 문구는 use_sage 기반 로그로 대체되어야 한다."
    )
    # 보다 직접적 조합 검증
    assert not ("Patched" in src and "xDiTMotifVideoAttnProcessor" in src), (
        "compile_config.py 에 'Patched'와 'xDiTMotifVideoAttnProcessor' 가 동시에 존재.\n"
        "P4.1 에서 이 조합을 제거했어야 한다."
    )


# ---------------------------------------------------------------------------
# G6: 소스 grep — P3.2 hasattr 가드 warning 문구 부재
# ---------------------------------------------------------------------------

def test_source_no_p3_2_skip_warning_message():
    """compile_config.py 소스에 P3.2 hasattr 가드에서 발생하던
    'skipped' + 'apply_sage_attention' 조합 warning 문구가 없어야 한다."""
    src = (_MODELS_DIR / "compile_config.py").read_text(encoding="utf-8")
    # P3.2 임시 가드: logger.warning("[MotifVideo] apply_sage_attention: skipped ...")
    assert "apply_sage_attention: skipped" not in src, (
        "compile_config.py 에 P3.2 임시 hasattr 가드의 'skipped' warning 문구가 잔류.\n"
        "P4.1 에서 hasattr 가드 전체가 제거되어야 한다."
    )


def test_source_no_hasattr_processor_guard():
    """compile_config.py 에 hasattr(block.attn, 'processor') 패턴이 없어야 한다."""
    src = (_MODELS_DIR / "compile_config.py").read_text(encoding="utf-8")
    assert 'hasattr(block.attn, "processor")' not in src, (
        "compile_config.py 에 hasattr processor 가드 잔류 (큰따옴표 버전)"
    )
    assert "hasattr(block.attn, 'processor')" not in src, (
        "compile_config.py 에 hasattr processor 가드 잔류 (작은따옴표 버전)"
    )


# ---------------------------------------------------------------------------
# G7: sage 미설치 경로 — no-op + 로그 메시지 존재
# ---------------------------------------------------------------------------

def test_apply_sage_attention_unavailable_noop_and_logs(monkeypatch, caplog):
    """_SAGE_AVAILABLE=False 경로에서 no-op + skip 관련 로그 메시지가 있어야 한다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", False)
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=1, n_single=1)

    with caplog.at_level(logging.DEBUG, logger="models.compile_config"):
        cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    # use_sage 변경 없음
    assert all(b.attn.use_sage is False for b in all_blocks), (
        "sage 미설치 경로에서 use_sage 가 변경됐다"
    )
    # 로그에 skip 관련 키워드 존재
    all_log = " ".join(r.getMessage() for r in caplog.records).lower()
    assert any(kw in all_log for kw in ("unavailable", "skip", "not available", "disabled", "sage")), (
        f"sage 미설치 경로에서 skip/unavailable 관련 로그 없음. "
        f"기록된 로그: {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# G8: 빈 transformer (블록 수 0)
# ---------------------------------------------------------------------------

def test_apply_sage_attention_empty_transformer_no_crash(monkeypatch):
    """transformer_blocks 와 single_transformer_blocks 가 모두 빈 경우
    apply_sage_attention 이 예외 없이 정상 반환한다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=0, n_single=0)

    # 예외가 발생하면 테스트 실패
    cc.apply_sage_attention(t)


# ---------------------------------------------------------------------------
# G9: ENABLE 미설정 시 transformer에 블록이 있어도 루프 진입 안 함
#     (early-return 위치가 루프 진입 전인지 구조적으로 확인)
# ---------------------------------------------------------------------------

def test_apply_sage_attention_disable_env_no_use_sage_set_on_any_block(monkeypatch):
    """MOTIFVIDEO_ENABLE_SAGE 미설정 시 use_sage 세팅이 단 하나의 블록에도 일어나지 않는다.
    n_dual=5, n_single=5 로 통계적으로 확인한다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.delenv("MOTIFVIDEO_ENABLE_SAGE", raising=False)

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=5, n_single=5)
    cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    activated = [b for b in all_blocks if b.attn.use_sage is True]
    assert len(activated) == 0, (
        f"MOTIFVIDEO_ENABLE_SAGE 미설정에서 {len(activated)}개 블록에 use_sage=True 가 세팅됐다. "
        "early-return 이 루프 이전에 위치해야 한다."
    )


# ---------------------------------------------------------------------------
# G10: single_transformer_blocks 만 있는 transformer
# ---------------------------------------------------------------------------

def test_apply_sage_attention_single_blocks_only(monkeypatch):
    """transformer_blocks=[] 이고 single_transformer_blocks 만 있는 경우에도
    single 블록들에 use_sage=True 가 설정된다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=0, n_single=4)
    cc.apply_sage_attention(t)

    for b in t.single_transformer_blocks:
        assert b.attn.use_sage is True, (
            "single_transformer_blocks 만 있는 경우 use_sage=True 가 설정되지 않음"
        )


def test_apply_sage_attention_dual_blocks_only(monkeypatch):
    """single_transformer_blocks=[] 이고 transformer_blocks(dual) 만 있는 경우에도
    dual 블록들에 use_sage=True 가 설정된다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=4, n_single=0)
    cc.apply_sage_attention(t)

    for b in t.transformer_blocks:
        assert b.attn.use_sage is True, (
            "transformer_blocks(dual) 만 있는 경우 use_sage=True 가 설정되지 않음"
        )


# ---------------------------------------------------------------------------
# 소스 구조 최종 검증 — block.attn.processor 속성 대입 0건
# ---------------------------------------------------------------------------

def test_source_no_block_attn_processor_assignment():
    """compile_config.py 에 block.attn.processor = ... 대입이 없어야 한다.
    P4.1 에서 processor 대입 방식을 use_sage 플래그 방식으로 전환했으므로."""
    src = (_MODELS_DIR / "compile_config.py").read_text(encoding="utf-8")
    assert "block.attn.processor" not in src, (
        "compile_config.py 에 'block.attn.processor' 잔류.\n"
        "P4.1 에서 이 패턴은 block.attn.use_sage = True 로 대체되어야 한다."
    )
