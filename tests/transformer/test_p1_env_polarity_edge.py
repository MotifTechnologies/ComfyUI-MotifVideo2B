"""P1 tester: env polarity + strict opt-in gate edge cases.

요구사항:
  - 기본값 OFF: env 미설정 시 apply_sage_attention 이 no-op
  - opt-in: MOTIFVIDEO_ENABLE_SAGE=1 만 활성화
  - 기타 값("0", "", "true", "TRUE", "yes", "2") 은 기본 OFF 취급
  - sage_ops.py 보존: dispatch_optimized_attention import 경로 건드리지 않음
  - apply_sage_attention 함수 자체 삭제 실수 방어

블라인드 원칙: 구현 코드를 읽지 않고 요구사항만으로 작성.
"""
import importlib.util
import os
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 로더 헬퍼 (기존 test_apply_sage_attention_flag.py 패턴 미러)
# ---------------------------------------------------------------------------

def _ensure_sage_ops_loaded():
    if "models.sage_ops" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "models.sage_ops",
            _REPO_ROOT / "models" / "sage_ops.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["models.sage_ops"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["models.sage_ops"]


def _load_compile_config():
    _ensure_sage_ops_loaded()
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
    def __init__(self):
        self.use_sage = False


class _StubBlock:
    def __init__(self):
        self.attn = _StubAttn()


class _StubTransformer:
    def __init__(self, n_dual=2, n_single=2):
        self.transformer_blocks = [_StubBlock() for _ in range(n_dual)]
        self.single_transformer_blocks = [_StubBlock() for _ in range(n_single)]


# ---------------------------------------------------------------------------
# 기본값 OFF — 다양한 env 조합
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env_val", [
    "0",
    "",
    "true",
    "TRUE",
    "yes",
    "2",
    "enable",
    " 1",   # 공백 포함 (strict != "1")
    "1 ",   # 후행 공백
])
def test_apply_sage_attention_noop_for_non_one_values(monkeypatch, env_val):
    """MOTIFVIDEO_ENABLE_SAGE 값이 정확히 "1" 이 아니면 no-op 이어야 한다."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(
        sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0")
    )
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", env_val)

    cc = _load_compile_config()
    t = _StubTransformer()
    cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    for b in all_blocks:
        assert b.attn.use_sage is False, (
            f"env='{env_val}' 일 때 use_sage 가 True 로 바뀌면 안 됨 (strict '1' 비교 위반)"
        )


def test_apply_sage_attention_noop_when_env_unset(monkeypatch):
    """환경변수 자체 미설정 시 no-op."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.delenv("MOTIFVIDEO_ENABLE_SAGE", raising=False)

    cc = _load_compile_config()
    t = _StubTransformer()
    cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    for b in all_blocks:
        assert b.attn.use_sage is False


# ---------------------------------------------------------------------------
# opt-in: "1" 정확히 매칭
# ---------------------------------------------------------------------------

def test_apply_sage_attention_activates_on_exact_one(monkeypatch):
    """MOTIFVIDEO_ENABLE_SAGE=1 정확히일 때만 use_sage=True 로 전환."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", True)
    monkeypatch.setitem(
        sys.modules, "sageattention", types.SimpleNamespace(__version__="stub-0")
    )
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer(n_dual=3, n_single=2)
    cc.apply_sage_attention(t)

    all_blocks = list(t.transformer_blocks) + list(t.single_transformer_blocks)
    assert all(b.attn.use_sage is True for b in all_blocks), (
        "MOTIFVIDEO_ENABLE_SAGE=1 시 모든 블록 use_sage=True 여야 함"
    )


# ---------------------------------------------------------------------------
# apply_sage_attention 함수 존재 방어 (함수 삭제 실수 탐지)
# ---------------------------------------------------------------------------

def test_apply_sage_attention_function_exists_and_callable():
    """apply_sage_attention 함수가 compile_config 에 존재하고 호출 가능해야 한다."""
    cc = _load_compile_config()
    assert hasattr(cc, "apply_sage_attention"), (
        "apply_sage_attention 함수가 models/compile_config.py 에 없음 (삭제 실수)"
    )
    assert callable(cc.apply_sage_attention)


# ---------------------------------------------------------------------------
# sage_ops.py 보존: dispatch_optimized_attention import 경로 확인
# ---------------------------------------------------------------------------

def test_sage_ops_dispatch_optimized_attention_importable():
    """sage_ops.py 에서 dispatch_optimized_attention 이 import 가능한 심볼이어야 한다."""
    sage_ops = _ensure_sage_ops_loaded()
    assert hasattr(sage_ops, "dispatch_optimized_attention"), (
        "dispatch_optimized_attention 이 sage_ops 에서 사라짐 — 보존 위반"
    )
    assert callable(sage_ops.dispatch_optimized_attention)


def test_sage_ops_py_key_symbols_preserved():
    """sage_ops.py 가 P1 에서 보존 대상 — key 심볼 존재로 behavior 레벨 가드.

    git diff 대신 실제 심볼을 ast 로 확인 (로컬 작업 중 git 상태에 취약한 verify 제거).
    """
    import ast
    sage_ops_path = _REPO_ROOT / "models" / "sage_ops.py"
    tree = ast.parse(sage_ops_path.read_text())

    names_in_module = set()
    # try/except 등 블록 내부 assign 도 수집하기 위해 ast.walk 전체 순회.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names_in_module.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names_in_module.add(target.id)

    # _SAGE_AVAILABLE 플래그 (try/except import 블록 안에 assign) + dispatch_optimized_attention
    # dispatcher 는 P1 이 보존해야 한다.
    required_symbols = {"_SAGE_AVAILABLE", "dispatch_optimized_attention"}
    missing = required_symbols - names_in_module
    assert not missing, (
        f"sage_ops.py 의 핵심 심볼이 사라짐 (보존 위반): {missing}"
    )


# ---------------------------------------------------------------------------
# sage 미설치 환경에서도 기본 OFF 는 no-op (env=1 이어도 sage 없으면 skip)
# ---------------------------------------------------------------------------

def test_apply_sage_attention_sage_unavailable_noop_even_with_env_one(monkeypatch):
    """sageattention 미설치(_SAGE_AVAILABLE=False) + MOTIFVIDEO_ENABLE_SAGE=1 조합.
    sage 가 없으면 use_sage 를 True 로 설정해도 crash 없고, 최소한 use_sage 플래그를
    무결하게 처리(True or False 어느 쪽이든 crash 없어야 함)."""
    sage_ops = _ensure_sage_ops_loaded()
    monkeypatch.setattr(sage_ops, "_SAGE_AVAILABLE", False)
    # sageattention 모듈이 없는 환경 시뮬레이션
    monkeypatch.setitem(sys.modules, "sageattention", None)
    monkeypatch.setenv("MOTIFVIDEO_ENABLE_SAGE", "1")

    cc = _load_compile_config()
    t = _StubTransformer()
    # crash 없어야 함 (use_sage 값은 구현 재량)
    cc.apply_sage_attention(t)


# ---------------------------------------------------------------------------
# __init__.py 자동 호출 경로: apply_sage_attention import 가능
# ---------------------------------------------------------------------------

def test_init_module_sage_gate_ast_structure():
    """models/__init__.py 가 `MOTIFVIDEO_ENABLE_SAGE == "1"` 게이트 내부에서
    apply_sage_attention 을 호출하는 구조인지 ast 로 검증 (grep-only fake coverage 회피)."""
    import ast
    init_path = _REPO_ROOT / "models" / "__init__.py"
    tree = ast.parse(init_path.read_text())

    def _has_enable_sage_env_check(node):
        """os.environ.get("MOTIFVIDEO_ENABLE_SAGE") == "1" 또는 동등 표현을 찾는다."""
        if not isinstance(node, ast.Compare):
            return False
        left = node.left
        # 왼쪽이 os.environ.get("MOTIFVIDEO_ENABLE_SAGE") 패턴인지
        if not (isinstance(left, ast.Call)
                and isinstance(left.func, ast.Attribute)
                and left.func.attr == "get"
                and isinstance(left.func.value, ast.Attribute)
                and left.func.value.attr == "environ"
                and left.args
                and isinstance(left.args[0], ast.Constant)
                and left.args[0].value == "MOTIFVIDEO_ENABLE_SAGE"):
            return False
        # 오른쪽이 "1" Constant 인지
        if not (len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "1"):
            return False
        return True

    def _walk_if_with_env_gate_calls_apply_sage(node):
        """If 노드가 ENABLE_SAGE=="1" 게이트이고 body 안에서 apply_sage_attention 을 호출하는지."""
        if not isinstance(node, ast.If):
            return False
        if not _has_enable_sage_env_check(node.test):
            return False
        # body 에서 apply_sage_attention 호출 여부 (직접 Name 또는 Attribute)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id == "apply_sage_attention":
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "apply_sage_attention":
                    return True
        return False

    found = any(_walk_if_with_env_gate_calls_apply_sage(n) for n in ast.walk(tree))
    assert found, (
        "models/__init__.py 에서 `if os.environ.get('MOTIFVIDEO_ENABLE_SAGE') == '1':` "
        "블록 안에서 apply_sage_attention 을 호출하는 구조를 찾지 못함 "
        "(grep-only fake coverage → ast 구조 검증으로 강화)."
    )
