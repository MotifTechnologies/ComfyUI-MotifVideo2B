"""tests/test_init_syspath.py — 체크리스트 2.1~2.2: __init__.py sys.path 주입 제거 검증

요구사항:
  - __init__.py 에서 motif_core / motif_pipelines sys.path.insert 완전 제거
  - import sys 제거 (sys.path 주입 목적의 코드)
  - 프로덕션 코드에 motif_core 잔존 import 없음
  - sys.path 없이도 import 체인이 정상 동작

블라인드 테스트 원칙:
  소스를 구현 의도로 읽지 않고 요구사항 위반 여부만 검사한다.
"""

import ast
import os
import sys
import importlib
import importlib.util
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOP_INIT_PY = os.path.join(_ROOT, "__init__.py")


# ---------------------------------------------------------------------------
# 소스 텍스트 로드 (한 번만)
# ---------------------------------------------------------------------------

with open(_TOP_INIT_PY, encoding="utf-8") as _f:
    _TOP_INIT_SRC = _f.read()


# ===========================================================================
# 1. __init__.py 소스 정적 검증 — sys.path.insert 완전 제거
# ===========================================================================

class TestSysPathInsertRemoved:
    """__init__.py 에 sys.path.insert 호출이 없어야 한다."""

    def test_no_sys_path_insert_literal(self):
        """소스 텍스트에 'sys.path.insert' 문자열이 없어야 한다."""
        assert "sys.path.insert" not in _TOP_INIT_SRC, (
            "__init__.py 에 'sys.path.insert' 가 남아 있음. "
            "motif_core/motif_pipelines sys.path 주입이 제거되지 않았습니다."
        )

    def test_no_sys_path_append_literal(self):
        """sys.path.append 도 없어야 한다 (유사 패턴 방어)."""
        assert "sys.path.append" not in _TOP_INIT_SRC, (
            "__init__.py 에 'sys.path.append' 가 남아 있음."
        )

    def test_ast_no_sys_path_insert_call(self):
        """AST 수준에서 sys.path.insert() 호출 노드가 없어야 한다."""
        tree = ast.parse(_TOP_INIT_SRC)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # sys.path.insert(...)  →  Attribute(Attribute(Name('sys'), 'path'), 'insert')
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "insert"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
            ):
                pytest.fail(
                    f"AST에서 sys.path.insert 호출 발견 (line {node.lineno})"
                )

    def test_ast_no_sys_path_append_call(self):
        """AST 수준에서 sys.path.append() 호출 노드가 없어야 한다."""
        tree = ast.parse(_TOP_INIT_SRC)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "append"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
            ):
                pytest.fail(
                    f"AST에서 sys.path.append 호출 발견 (line {node.lineno})"
                )


# ===========================================================================
# 2. __init__.py 소스 정적 검증 — motif_core / motif_pipelines 문자열 제거
# ===========================================================================

class TestMotifPathStringsRemoved:
    """__init__.py 에 motif_core/motif_pipelines 경로 문자열이 없어야 한다."""

    def test_no_motif_core_path_string(self):
        """'motif_core' 가 sys.path 주입 맥락에서 사용되지 않아야 한다.

        Note: import 구문에 motif_core가 남아 있는지 여부는 별도 테스트(4번)에서
        검증. 여기서는 경로 문자열("/.../motif_core") 형태만 검사한다.
        """
        import re
        # 따옴표 안에 들어간 경로 문자열만 검사 (from motif_core import 는 제외)
        path_pattern = re.compile(r'["\'][^"\']*motif_core[^"\']*["\']')
        matches = path_pattern.findall(_TOP_INIT_SRC)
        assert len(matches) == 0, (
            f"__init__.py 에 motif_core 경로 문자열이 남아 있음: {matches}"
        )

    def test_no_motif_pipelines_path_string(self):
        """'motif_pipelines' 가 경로 문자열로 남아 있지 않아야 한다."""
        import re
        path_pattern = re.compile(r'["\'][^"\']*motif_pipelines[^"\']*["\']')
        matches = path_pattern.findall(_TOP_INIT_SRC)
        assert len(matches) == 0, (
            f"__init__.py 에 motif_pipelines 경로 문자열이 남아 있음: {matches}"
        )

    def test_ast_no_motif_string_constants_with_slashes(self):
        """AST Constant 노드에 'motif_core' 또는 'motif_pipelines' 를 포함하며
        경로처럼 슬래시를 포함한 문자열이 없어야 한다."""
        tree = ast.parse(_TOP_INIT_SRC)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            val = node.value
            if ("motif_core" in val or "motif_pipelines" in val) and "/" in val:
                pytest.fail(
                    f"AST에서 motif 경로 상수 발견 (line {node.lineno}): {val!r}"
                )


# ===========================================================================
# 3. 기존 테스트 실행 — import 체인 정상 동작 확인 (subprocess)
#
# sys.path 주입 없이 pytest tests/transformer/ tests/test_model_init.py 가
# exit code 0 으로 완료되어야 한다.
# ===========================================================================

class TestImportChainWithoutSysPath:
    """sys.path 없이도 기존 테스트(transformer/, test_model_init.py)가 통과해야 한다."""

    def test_existing_test_suites_pass(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/transformer/",
                "tests/test_model_init.py",
                "-v", "--tb=short", "-q",
            ],
            capture_output=True,
            text=True,
            cwd=_ROOT,
        )
        assert result.returncode == 0, (
            "기존 테스트(transformer/ + test_model_init.py)가 실패함.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


# ===========================================================================
# 4. 프로덕션 코드 잔존 motif_core import 최종 검증
#
# tests/ 폴더를 제외한 .py 파일에 'from motif_core' 또는 'import motif_core'
# 형태의 실행 가능한 import 구문이 없어야 한다.
# ===========================================================================

def _collect_production_py_files(root: str):
    """root 아래 tests/ 를 제외한 모든 .py 파일 경로를 반환한다."""
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        # tests/ 디렉토리 제외
        dirnames[:] = [
            d for d in dirnames
            if d not in ("tests", "__pycache__", ".git", "node_modules")
        ]
        for fname in filenames:
            if fname.endswith(".py"):
                result.append(os.path.join(dirpath, fname))
    return result


_PROD_FILES = _collect_production_py_files(_ROOT)


class TestNoMotifCoreImportInProduction:
    """프로덕션 .py 파일에 motif_core 잔존 import 없음."""

    @pytest.mark.parametrize("fpath", _PROD_FILES)
    def test_no_from_motif_core_import(self, fpath):
        """각 프로덕션 파일에 'from motif_core.' import 구문 없음."""
        import re
        with open(fpath, encoding="utf-8") as f:
            src = f.read()
        pattern = re.compile(r"^\s*from\s+motif_core\b", re.MULTILINE)
        matches = pattern.findall(src)
        rel = os.path.relpath(fpath, _ROOT)
        assert len(matches) == 0, (
            f"{rel} 에 'from motif_core' import 구문이 {len(matches)}개 남아 있음.\n"
            f"  발견: {matches}"
        )

    @pytest.mark.parametrize("fpath", _PROD_FILES)
    def test_no_import_motif_core(self, fpath):
        """각 프로덕션 파일에 'import motif_core' 구문 없음."""
        import re
        with open(fpath, encoding="utf-8") as f:
            src = f.read()
        pattern = re.compile(r"^\s*import\s+motif_core\b", re.MULTILINE)
        matches = pattern.findall(src)
        rel = os.path.relpath(fpath, _ROOT)
        assert len(matches) == 0, (
            f"{rel} 에 'import motif_core' 구문이 {len(matches)}개 남아 있음.\n"
            f"  발견: {matches}"
        )


# ===========================================================================
# 5. 경계값 및 엣지케이스 — __init__.py AST 구조 보조 검증
# ===========================================================================

class TestTopLevelInitAstSanity:
    """__init__.py AST 파싱 자체가 성공하고, 최상위 import sys 가 없어야 한다."""

    def test_ast_parses_without_error(self):
        """소스가 유효한 Python 이어야 한다."""
        try:
            ast.parse(_TOP_INIT_SRC)
        except SyntaxError as e:
            pytest.fail(f"__init__.py 구문 오류: {e}")

    def test_no_toplevel_import_sys_for_path_injection(self):
        """최상위 'import sys' 가 없어야 한다.

        sys.path 주입이 제거된 후라면 sys 를 최상위에서 import 할 이유가 없다.
        단, sys 가 다른 목적으로 쓰이는 경우 이 테스트는 조정 필요.
        """
        tree = ast.parse(_TOP_INIT_SRC)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sys":
                        pytest.fail(
                            f"최상위 'import sys' 발견 (line {node.lineno}). "
                            "sys.path 주입 제거 후 불필요한 import."
                        )

    def test_file_is_not_empty(self):
        """__init__.py 가 완전히 비어 있지 않아야 한다."""
        assert len(_TOP_INIT_SRC.strip()) > 0, "__init__.py 가 빈 파일입니다."

    def test_no_sys_reference_at_all_in_source(self):
        """소스 텍스트 전체에 'sys.path' 문자열이 없어야 한다."""
        assert "sys.path" not in _TOP_INIT_SRC, (
            "__init__.py 에 'sys.path' 참조가 남아 있음."
        )
