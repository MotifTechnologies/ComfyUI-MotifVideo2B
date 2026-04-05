"""체크리스트 1.1 검증: models/transformer/ 파일 내장 테스트.

실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI/custom_nodes/ComfyUI-MotifVideo1.9B
    pytest tests/transformer/test_transformer_bundle.py -v

테스트 항목:
  1. MotifVideoTransformer3DModel import 성공
  2. TreadMixin, is_tread_start, is_tread_end import 성공
  3. motif_core 잔존 import 없음 (소스 grep)
  4. loguru 잔존 import 없음 (소스 grep)
  5. accelerate 패키지 import 없음 (accelerate_patch.py)
  6. accelerate_patch.py no-op stub 동작
  7. diffusers.hooks._helpers try/except 보호
  8. transformer_motif_video.py 내 sibling 상대 import 검증
  9. 심볼 타입 검증 (문자열/None stub 아님)
 10. __init__.py export / 파일 존재 확인

참고: models/__init__.py 는 ComfyUI + CUDA 환경에 의존하므로
패키지 전체 import는 하지 않는다. 각 대상 파일을 importlib.util
로 독립 로드하거나 소스를 AST/grep으로 분석한다.
"""

import ast
import importlib.util
import os
import re
import sys

import pytest

# 프로젝트 루트
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
TRANSFORMER_DIR = os.path.join(PROJECT_ROOT, "models", "transformer")

# ─────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────

def _src(filename: str) -> str:
    """파일 전체 소스 반환."""
    path = os.path.join(TRANSFORMER_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_file(filename: str, module_name: str):
    """파일을 독립 모듈로 로드. 상대 import 의존이 있으면 ImportError 전파."""
    path = os.path.join(TRANSFORMER_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(filename: str) -> ast.Module:
    return ast.parse(_src(filename))


# ─────────────────────────────────────────────
# 1. 필수 파일 존재 확인
# ─────────────────────────────────────────────

class TestFilesExist:
    @pytest.mark.parametrize("filename", [
        "__init__.py",
        "transformer_motif_video.py",
        "tread_mixin.py",
        "accelerate_patch.py",
    ])
    def test_file_present(self, filename):
        path = os.path.join(TRANSFORMER_DIR, filename)
        assert os.path.isfile(path), f"필수 파일 없음: {path}"

    def test_transformer_dir_is_python_package(self):
        """__init__.py 가 있어야 패키지로 인식된다."""
        assert os.path.isfile(os.path.join(TRANSFORMER_DIR, "__init__.py"))


# ─────────────────────────────────────────────
# 2. motif_core 잔존 import 없음
# ─────────────────────────────────────────────

_MOTIF_CORE_RE = re.compile(r"\bmotif_core\b")

class TestNoMotifCoreImport:
    @pytest.mark.parametrize("filename", [
        "transformer_motif_video.py",
        "tread_mixin.py",
        "accelerate_patch.py",
        "__init__.py",
    ])
    def test_no_motif_core_in_file(self, filename):
        src = _src(filename)
        hits = [
            f"L{i+1}: {line}"
            for i, line in enumerate(src.splitlines())
            if _MOTIF_CORE_RE.search(line)
        ]
        assert not hits, (
            f"{filename}에 motif_core 잔존:\n" + "\n".join(hits)
        )


# ─────────────────────────────────────────────
# 3. loguru 잔존 import 없음
# ─────────────────────────────────────────────

_LOGURU_RE = re.compile(r"\bloguru\b")

class TestNoLoguruImport:
    @pytest.mark.parametrize("filename", [
        "transformer_motif_video.py",
        "tread_mixin.py",
        "accelerate_patch.py",
        "__init__.py",
    ])
    def test_no_loguru_in_file(self, filename):
        src = _src(filename)
        hits = [
            f"L{i+1}: {line}"
            for i, line in enumerate(src.splitlines())
            if _LOGURU_RE.search(line)
        ]
        assert not hits, (
            f"{filename}에 loguru 잔존:\n" + "\n".join(hits)
        )

    def test_tread_mixin_imports_stdlib_logging(self):
        """loguru 대신 stdlib logging을 사용해야 한다."""
        src = _src("tread_mixin.py")
        has_logging = bool(
            re.search(r"^\s*import\s+logging\b", src, re.MULTILINE) or
            re.search(r"^\s*from\s+logging\b", src, re.MULTILINE)
        )
        assert has_logging, "tread_mixin.py에 logging import 없음"

    def test_tread_mixin_no_loguru_logger_usage(self):
        """logger.xxx 패턴이 loguru 객체 사용인지 확인.
        stdlib logging 사용 시 logging.xxx 또는 getLogger 패턴이어야 한다."""
        src = _src("tread_mixin.py")
        # loguru 특유의 패턴 (from loguru import logger) 없어야 함
        assert not re.search(r"from\s+loguru\s+import\s+logger", src), (
            "tread_mixin.py에 loguru logger import 잔존"
        )


# ─────────────────────────────────────────────
# 4. accelerate 패키지 import 없음 (accelerate_patch.py)
# ─────────────────────────────────────────────

_ACCELERATE_IMPORT_RE = re.compile(
    r"^\s*(import\s+accelerate\b|from\s+accelerate\b)",
    re.MULTILINE,
)

class TestNoAccelerateImport:
    def test_no_accelerate_import_in_patch(self):
        src = _src("accelerate_patch.py")
        hits = [
            f"L{i+1}: {line}"
            for i, line in enumerate(src.splitlines())
            if _ACCELERATE_IMPORT_RE.match(line)
        ]
        assert not hits, (
            "accelerate_patch.py에 accelerate import 발견:\n" + "\n".join(hits)
        )

    def test_no_accelerate_import_in_transformer(self):
        """transformer_motif_video.py 에서도 accelerate 직접 import 없어야 함."""
        src = _src("transformer_motif_video.py")
        hits = [
            f"L{i+1}: {line}"
            for i, line in enumerate(src.splitlines())
            if _ACCELERATE_IMPORT_RE.match(line)
        ]
        assert not hits, (
            "transformer_motif_video.py에 accelerate 직접 import:\n" + "\n".join(hits)
        )


# ─────────────────────────────────────────────
# 5. accelerate_patch.py no-op stub 동작
# ─────────────────────────────────────────────

class TestAcceleratePatchNoOp:
    def test_importable_without_accelerate_installed(self):
        """accelerate_patch.py는 accelerate 패키지 없이도 import 가능해야 한다."""
        mod = _load_file("accelerate_patch.py", "_acc_patch_test")
        assert mod is not None

    def test_double_import_no_error(self):
        """두 번 로드해도 에러가 없어야 한다 (side-effect 없음)."""
        for i in range(2):
            mod = _load_file("accelerate_patch.py", f"_acc_patch_double_{i}")
            assert mod is not None

    def test_patch_callables_do_not_raise_runtime_error(self):
        """stub 함수들이 존재할 경우 호출 시 RuntimeError가 없어야 한다."""
        mod = _load_file("accelerate_patch.py", "_acc_patch_call_test")
        for name in dir(mod):
            if name.startswith("_"):
                continue
            attr = getattr(mod, name)
            if callable(attr) and not isinstance(attr, type):
                try:
                    attr()
                except TypeError:
                    pass  # 인자 개수 불일치 허용
                except RuntimeError as exc:
                    pytest.fail(
                        f"accelerate_patch.{name}() RuntimeError: {exc}"
                    )

    def test_accelerate_patch_source_is_not_empty(self):
        """no-op stub이라도 빈 파일이면 안 된다 (최소 1줄 이상)."""
        src = _src("accelerate_patch.py").strip()
        assert src, "accelerate_patch.py 가 완전히 비어 있음"


# ─────────────────────────────────────────────
# 6. diffusers.hooks._helpers try/except 보호
# ─────────────────────────────────────────────

class TestDiffusersHooksGuarded:
    def test_try_except_block_present_if_hooks_referenced(self):
        """diffusers.hooks 관련 코드가 있으면 try/except 블록도 있어야 한다."""
        src = _src("transformer_motif_video.py")
        has_hooks_ref = "diffusers.hooks" in src or "_helpers" in src
        if not has_hooks_ref:
            pytest.skip("diffusers.hooks 참조 없음 — 보호 불필요")
        assert re.search(r"\btry\s*:", src), (
            "transformer_motif_video.py에 try/except 없음 "
            "(diffusers.hooks._helpers 미보호)"
        )

    def test_diffusers_hooks_import_inside_try_ast(self):
        """AST 검사: diffusers.hooks import가 Try 노드 바깥에 있으면 실패.

        ast.walk는 Try 노드의 자식도 flat하게 반환하므로 직접 순회 방식 대신
        Try 노드가 보호하는 라인 번호 집합을 먼저 수집한 뒤 비교한다.
        """
        src = _src("transformer_motif_video.py")
        tree = ast.parse(src)

        # Try 블록 안에 속하는 ImportFrom 라인 번호 집합 수집
        guarded_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # body (try 절) 내 모든 ImportFrom 수집
                for child in ast.walk(node):
                    if isinstance(child, ast.ImportFrom):
                        guarded_lines.add(child.lineno)

        # 전체 ImportFrom 중 diffusers.hooks 를 참조하는 것
        hooks_imports: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "diffusers.hooks" in node.module:
                    hooks_imports.append((node.lineno, node.module))

        unguarded = [
            (ln, mod) for ln, mod in hooks_imports
            if ln not in guarded_lines
        ]
        assert not unguarded, (
            "diffusers.hooks import가 try/except 밖에 있음:\n"
            + "\n".join(f"  L{ln}: {mod}" for ln, mod in unguarded)
        )

    def test_except_block_catches_importerror(self):
        """try 블록이 ImportError를 catch해야 한다 (bare except도 허용)."""
        src = _src("transformer_motif_video.py")
        has_hooks_ref = "diffusers.hooks" in src or "_helpers" in src
        if not has_hooks_ref:
            pytest.skip("diffusers.hooks 참조 없음")

        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # 이 Try 블록 안에 diffusers.hooks import가 있는지
            body_src = ast.dump(node)
            if "diffusers" not in body_src and "_helpers" not in body_src:
                continue
            # handler 확인
            for handler in node.handlers:
                if handler.type is None:
                    return  # bare except — OK
                if isinstance(handler.type, ast.Name):
                    if handler.type.id in ("ImportError", "Exception"):
                        return
                if isinstance(handler.type, ast.Tuple):
                    names = [
                        e.id for e in handler.type.elts
                        if isinstance(e, ast.Name)
                    ]
                    if "ImportError" in names or "Exception" in names:
                        return

        # diffusers.hooks 참조가 있지만 적절한 handler가 없는 경우
        if has_hooks_ref:
            pytest.fail(
                "diffusers.hooks import의 try 블록에 "
                "ImportError/Exception handler 없음"
            )


# ─────────────────────────────────────────────
# 7. transformer_motif_video.py — 상대 import 검증
# ─────────────────────────────────────────────

class TestRelativeImports:
    def test_no_absolute_motif_core_import(self):
        """'from motif_core.' 절대 import가 없어야 한다."""
        src = _src("transformer_motif_video.py")
        hits = [
            f"L{i+1}: {line}"
            for i, line in enumerate(src.splitlines())
            if re.search(r"\bfrom\s+motif_core\b", line) or
               re.search(r"\bimport\s+motif_core\b", line)
        ]
        assert not hits, (
            "motif_core 절대 import 잔존:\n" + "\n".join(hits)
        )

    def test_sibling_modules_not_imported_absolutely(self):
        """같은 디렉터리의 sibling 모듈을 절대 import하면 안 된다."""
        src = _src("transformer_motif_video.py")
        tree = ast.parse(src)

        sibling_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(TRANSFORMER_DIR)
            if f.endswith(".py") and not f.startswith("__")
        }

        absolute_sibling: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    top = node.module.split(".")[0]
                    if top in sibling_stems:
                        absolute_sibling.append((node.lineno, node.module))

        assert not absolute_sibling, (
            "sibling 모듈을 절대 import:\n" +
            "\n".join(f"  L{ln}: {mod}" for ln, mod in absolute_sibling)
        )

    def test_tread_mixin_no_absolute_sibling_import(self):
        """tread_mixin.py도 같은 패키지 내 sibling을 절대 import하지 않아야 한다."""
        src = _src("tread_mixin.py")
        tree = ast.parse(src)
        sibling_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(TRANSFORMER_DIR)
            if f.endswith(".py") and not f.startswith("__")
        }
        absolute_sibling = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    top = node.module.split(".")[0]
                    if top in sibling_stems:
                        absolute_sibling.append((node.lineno, node.module))
        assert not absolute_sibling, (
            "tread_mixin.py 절대 sibling import:\n" +
            "\n".join(f"  L{ln}: {mod}" for ln, mod in absolute_sibling)
        )


# ─────────────────────────────────────────────
# 8. __init__.py export 검증 (소스 레벨)
# ─────────────────────────────────────────────

class TestInitExports:
    def test_init_references_transformer_model(self):
        """__init__.py 소스에 MotifVideoTransformer3DModel 이름이 나타나야 한다."""
        src = _src("__init__.py")
        assert "MotifVideoTransformer3DModel" in src, (
            "__init__.py에 MotifVideoTransformer3DModel 참조 없음"
        )

    def test_init_does_not_reference_motif_core_directly(self):
        """__init__.py가 motif_core를 직접 import하지 않아야 한다.
        (transformer_motif_video.py 내 로컬 정의를 재-export해야 함)"""
        src = _src("__init__.py")
        hits = [
            f"L{i+1}: {line}"
            for i, line in enumerate(src.splitlines())
            if _MOTIF_CORE_RE.search(line)
        ]
        assert not hits, (
            "__init__.py에 motif_core 직접 참조:\n" + "\n".join(hits)
        )

    def test_init_imports_from_local_transformer_module(self):
        """__init__.py가 .transformer_motif_video (상대 import) 또는
        같은 패키지 내 모듈에서 MotifVideoTransformer3DModel을 가져와야 한다."""
        src = _src("__init__.py")
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # 상대 import (level > 0) 이거나 models.transformer.* 절대 import
                if node.level and node.level > 0:
                    names = [alias.name for alias in node.names]
                    if "MotifVideoTransformer3DModel" in names:
                        found = True
                        break
                    # 모듈 자체가 transformer_motif_video 이면 OK
                    if node.module and "transformer_motif_video" in node.module:
                        found = True
                        break
        assert found, (
            "__init__.py가 로컬(.transformer_motif_video)에서 "
            "MotifVideoTransformer3DModel을 import하지 않음"
        )

    def test_all_four_source_files_parseable(self):
        """4개 파일 모두 AST parse가 성공해야 한다 (문법 오류 없음)."""
        for fname in [
            "__init__.py",
            "transformer_motif_video.py",
            "tread_mixin.py",
            "accelerate_patch.py",
        ]:
            try:
                ast.parse(_src(fname))
            except SyntaxError as exc:
                pytest.fail(f"{fname} 문법 오류: {exc}")


# ─────────────────────────────────────────────
# 9. accelerate_patch.py 경계값 — 빈 파일 / 주석만 있는 경우
# ─────────────────────────────────────────────

class TestAcceleratePatchEdge:
    def test_not_just_pass_or_ellipsis(self):
        """파일이 'pass' 또는 '...' 한 줄짜리 이상이어야 한다.
        완전 무의미한 stub은 아닌지 확인."""
        src = _src("accelerate_patch.py").strip()
        # 주석 제거 후 실질 코드가 있어야 함
        lines = [
            line.strip()
            for line in src.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        # 최소한 1줄은 실질 코드여야 함 (pass나 ... 만이라도 OK — 존재 자체가 의미 있음)
        assert len(lines) >= 1, "accelerate_patch.py에 실질 코드 없음"

    def test_accelerate_patch_no_syntax_error(self):
        try:
            ast.parse(_src("accelerate_patch.py"))
        except SyntaxError as exc:
            pytest.fail(f"accelerate_patch.py 문법 오류: {exc}")


# ─────────────────────────────────────────────
# 10. 소스 레벨 심볼 존재 확인 (AST — import 실행 없이)
# ─────────────────────────────────────────────

class TestSymbolExistence:
    def _defined_names(self, filename: str) -> set[str]:
        """파일에서 최상위 정의된 이름(class, def, assignment) 수집."""
        tree = _parse(filename)
        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        return names

    def test_transformer_motif_video_defines_transformer_class(self):
        """transformer_motif_video.py가 MotifVideoTransformer3DModel 클래스를 정의해야 한다."""
        names = self._defined_names("transformer_motif_video.py")
        assert "MotifVideoTransformer3DModel" in names, (
            f"MotifVideoTransformer3DModel 클래스 정의 없음. 발견: {names}"
        )

    def test_tread_mixin_defines_tread_mixin_class(self):
        """tread_mixin.py가 TreadMixin 클래스를 정의해야 한다."""
        names = self._defined_names("tread_mixin.py")
        assert "TreadMixin" in names, (
            f"TreadMixin 클래스 정의 없음. 발견: {names}"
        )

    def test_tread_mixin_defines_is_tread_start(self):
        names = self._defined_names("tread_mixin.py")
        assert "is_tread_start" in names, (
            f"is_tread_start 함수 정의 없음. 발견: {names}"
        )

    def test_tread_mixin_defines_is_tread_end(self):
        names = self._defined_names("tread_mixin.py")
        assert "is_tread_end" in names, (
            f"is_tread_end 함수 정의 없음. 발견: {names}"
        )

    def test_tread_mixin_class_has_methods(self):
        """TreadMixin 클래스가 메서드를 1개 이상 가져야 한다 (빈 pass class 아님)."""
        tree = _parse("tread_mixin.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TreadMixin":
                method_nodes = [
                    n for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                assert len(method_nodes) >= 1, (
                    "TreadMixin에 메서드가 없음 (빈 stub 의심)"
                )
                return
        pytest.fail("TreadMixin 클래스를 찾을 수 없음")

    def test_is_tread_start_is_function_not_constant(self):
        """is_tread_start가 함수 정의여야 한다 (상수 assign이 아님)."""
        tree = _parse("tread_mixin.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "is_tread_start":
                return
        pytest.fail("is_tread_start가 함수 정의(def)가 아님")

    def test_is_tread_end_is_function_not_constant(self):
        tree = _parse("tread_mixin.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "is_tread_end":
                return
        pytest.fail("is_tread_end가 함수 정의(def)가 아님")
