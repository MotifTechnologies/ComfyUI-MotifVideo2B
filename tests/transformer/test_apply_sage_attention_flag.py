"""P4.1: apply_sage_attention 이 use_sage=True 플래그 방식으로 동작."""
import importlib.util
import os
import pathlib
import sys
import types

import pytest

# conftest 가 diffusers stub 자동 설치 (session-scoped)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_module(name: str, path: pathlib.Path):
    """spec_from_file_location 으로 모듈 로드 후 sys.modules 등록."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_sage_ops_loaded():
    """models.sage_ops 를 sys.modules 에 등록. compile_config 의 relative import 가
    from .sage_ops import _SAGE_AVAILABLE 를 통해 이 모듈을 참조하도록 한다.
    이미 로드된 경우 재로드 없이 반환."""
    if "models.sage_ops" not in sys.modules:
        _load_module("models.sage_ops", _REPO_ROOT / "models" / "sage_ops.py")
    return sys.modules["models.sage_ops"]


def _load_compile_config():
    """compile_config 모듈을 spec_from_file_location 으로 로드.
    sage_ops 가 먼저 sys.modules 에 등록되어야 relative import 가 성공한다.
    __package__ = "models" 설정으로 from .sage_ops import 가 동작하도록 한다."""
    _ensure_sage_ops_loaded()
    # 매 호출마다 fresh 로드 (monkeypatch 독립성 보장)
    mod_name = "models.compile_config"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        _REPO_ROOT / "models" / "compile_config.py",
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "models"
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _StubAttn:
    """MotifVideoAttention 대리. use_sage 속성만 노출."""
    def __init__(self):
        self.use_sage = False


class _StubBlock:
    def __init__(self):
        self.attn = _StubAttn()


class _StubTransformer:
    def __init__(self, n_dual=2, n_single=3):
        self.transformer_blocks = [_StubBlock() for _ in range(n_dual)]
        self.single_transformer_blocks = [_StubBlock() for _ in range(n_single)]


def test_apply_sage_attention_sets_use_sage_true_on_all_blocks(monkeypatch):
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0"))
    monkeypatch.delenv("MOTIFVIDEO_DISABLE_SAGE", raising=False)

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=2, n_single=3)
    cc.apply_sage_attention(t)

    for b in t.transformer_blocks:
        assert b.attn.use_sage is True
    for b in t.single_transformer_blocks:
        assert b.attn.use_sage is True


def test_apply_sage_attention_noop_when_env_disable(monkeypatch):
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setenv("MOTIFVIDEO_DISABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=1, n_single=1)
    cc.apply_sage_attention(t)

    for b in list(t.transformer_blocks) + list(t.single_transformer_blocks):
        assert b.attn.use_sage is False, "env disable 시 use_sage 건드리면 안 됨"


def test_apply_sage_attention_noop_when_sage_unavailable(monkeypatch):
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", False)
    monkeypatch.delenv("MOTIFVIDEO_DISABLE_SAGE", raising=False)

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=1, n_single=1)
    cc.apply_sage_attention(t)

    for b in list(t.transformer_blocks) + list(t.single_transformer_blocks):
        assert b.attn.use_sage is False


def test_no_legacy_hasattr_guard_in_compile_config():
    """P3.2 의 임시 hasattr 가드가 P4.1 에서 제거되었는지 소스 grep."""
    src = (_REPO_ROOT / "models" / "compile_config.py").read_text()
    assert 'hasattr(block.attn, "processor")' not in src
    assert "block.attn.processor = xDiTMotifVideoAttnProcessor()" not in src


def test_legacy_xdit_processor_import_possibly_removed():
    """compile_config 에서 xDiTMotifVideoAttnProcessor import 불필요 여부 체크.
    현재 P4.1 에서 제거 가능하나 P4.3 에서 attention_processor.py 전체 정리 시점이라
    아직 남아있어도 통과 (경고 형태)."""
    src = (_REPO_ROOT / "models" / "compile_config.py").read_text()
    if "from .attention_processor import" in src:
        # 유지해도 OK; P4.3 에서 정리
        import warnings
        warnings.warn("compile_config 에 xDiTMotifVideoAttnProcessor import 잔류 (P4.3 정리 예정)")
