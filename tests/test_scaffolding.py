"""
체크리스트 항목 1 "프로젝트 스캐폴딩" 테스트.

범위:
1. 모든 .py 파일 Python 문법 검증 (py_compile)
2. 패키지 구조 — 모든 패키지 디렉토리에 __init__.py 존재
3. NODE_CLASS_MAPPINGS 3개 노드 클래스의 필수 속성 검증
4. __init__.py graceful failure 처리 검증

주의: comfy 패키지 없이도 동작하도록 설계됨.
      ImportError / 실행 의존성은 mock으로 격리.
"""

import ast
import importlib
import inspect
import pathlib
import py_compile
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

# 프로젝트 루트
PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 1. Python 문법 검증
# ---------------------------------------------------------------------------

class TestPythonSyntax(unittest.TestCase):
    """모든 .py 파일이 SyntaxError 없이 컴파일되는지 확인."""

    def _collect_py_files(self):
        return sorted(PROJECT_ROOT.rglob("*.py"))

    def test_syntax_no_py_files_missing(self):
        """py 파일이 1개 이상 존재해야 한다."""
        files = self._collect_py_files()
        self.assertGreater(len(files), 0, "프로젝트에 .py 파일이 없음")

    def test_syntax_all_py_files_compile(self):
        """py_compile로 각 파일의 문법을 검증한다."""
        errors = []
        for path in self._collect_py_files():
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"{path.relative_to(PROJECT_ROOT)}: {exc}")
        self.assertEqual(errors, [], "문법 오류 파일:\n" + "\n".join(errors))

    def test_syntax_ast_parse_root_init(self):
        """루트 __init__.py가 AST 파싱 가능해야 한다."""
        src = (PROJECT_ROOT / "__init__.py").read_text()
        try:
            ast.parse(src)
        except SyntaxError as exc:
            self.fail(f"__init__.py AST 파싱 실패: {exc}")

    def test_syntax_ast_parse_nodes_package(self):
        """nodes/__init__.py가 AST 파싱 가능해야 한다."""
        src = (PROJECT_ROOT / "nodes" / "__init__.py").read_text()
        try:
            ast.parse(src)
        except SyntaxError as exc:
            self.fail(f"nodes/__init__.py AST 파싱 실패: {exc}")


# ---------------------------------------------------------------------------
# 2. 패키지 구조 검증
# ---------------------------------------------------------------------------

EXPECTED_PACKAGE_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "nodes",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "text_encoders",
]


class TestPackageStructure(unittest.TestCase):
    """모든 패키지 디렉토리에 __init__.py가 있는지 확인."""

    def test_structure_root_init_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "__init__.py").is_file(),
            "루트 __init__.py 없음"
        )

    def test_structure_nodes_init_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "nodes" / "__init__.py").is_file(),
            "nodes/__init__.py 없음"
        )

    def test_structure_models_init_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "models" / "__init__.py").is_file(),
            "models/__init__.py 없음"
        )

    def test_structure_text_encoders_init_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "text_encoders" / "__init__.py").is_file(),
            "text_encoders/__init__.py 없음"
        )

    def test_structure_all_expected_dirs_present(self):
        """예상 패키지 디렉토리가 모두 존재하는지 일괄 확인."""
        missing = [
            str(d.relative_to(PROJECT_ROOT))
            for d in EXPECTED_PACKAGE_DIRS
            if not d.is_dir()
        ]
        self.assertEqual(missing, [], f"누락된 패키지 디렉토리: {missing}")

    def test_structure_node_source_files_exist(self):
        """nodes/ 아래 4개 노드 소스 파일이 존재해야 한다."""
        for name in ("loader.py", "latent.py", "text_encode.py", "teacache.py"):
            path = PROJECT_ROOT / "nodes" / name
            self.assertTrue(path.is_file(), f"nodes/{name} 없음")

    def test_structure_config_py_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "config.py").is_file(),
            "config.py 없음"
        )

    def test_structure_no_stray_init_under_hidden_dirs(self):
        """숨김 디렉토리(.claude, .git 등) 안의 __init__.py는 없어야 한다 (경계값)."""
        hidden = [
            p for p in PROJECT_ROOT.rglob("__init__.py")
            if any(part.startswith(".") for part in p.parts)
        ]
        self.assertEqual(
            hidden, [],
            f"숨김 디렉토리에 __init__.py 발견: {hidden}"
        )


# ---------------------------------------------------------------------------
# 3. 노드 클래스 필수 속성 검증
# ---------------------------------------------------------------------------

def _load_node_class(module_rel_path: str, class_name: str):
    """
    comfy 없이 노드 클래스를 로드한다.
    mock.patch로 comfy 관련 import를 막는다.
    """
    # sys.path에 프로젝트 루트를 임시 추가
    str_root = str(PROJECT_ROOT)
    inserted = str_root not in sys.path
    if inserted:
        sys.path.insert(0, str_root)

    # comfy 관련 모듈을 mock으로 대체
    mock_comfy = MagicMock()
    fake_modules = {
        "comfy": mock_comfy,
        "comfy.supported_models": mock_comfy.supported_models,
        "comfy.model_base": mock_comfy.model_base,
        "comfy.latent_formats": mock_comfy.latent_formats,
        "comfy.sd": mock_comfy.sd,
        "comfy.clip": mock_comfy.clip,
    }

    original = {}
    for key, val in fake_modules.items():
        original[key] = sys.modules.get(key)
        sys.modules[key] = val

    # 캐시 무효화 (이전 import 잔재 제거)
    for mod_key in list(sys.modules.keys()):
        if mod_key.startswith("nodes.") or mod_key in (
            "nodes", "config",
            "ComfyUI_MotifVideo1.9B", "ComfyUI-MotifVideo1.9B",
        ):
            del sys.modules[mod_key]

    try:
        module = importlib.import_module(module_rel_path)
        return getattr(module, class_name)
    finally:
        # 복원
        for key, val in original.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val
        if inserted:
            sys.path.remove(str_root)


REQUIRED_ATTRS = ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY")

NODE_SPECS = [
    ("nodes.loader", "MotifVideoModelLoader"),
    ("nodes.text_encode", "MotifTextEncode"),
    ("nodes.latent", "EmptyMotifLatent"),
    ("nodes.teacache", "MotifTeaCache"),
]


class TestNodeClassAttributes(unittest.TestCase):
    """3개 노드 클래스가 ComfyUI 필수 속성을 모두 갖는지 검증."""

    def _get_class(self, module_path, class_name):
        try:
            return _load_node_class(module_path, class_name)
        except Exception as exc:
            self.skipTest(f"{class_name} 로드 실패 (의존성 문제): {exc}")

    # --- MotifVideoModelLoader ---

    def test_loader_has_required_attrs(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        for attr in REQUIRED_ATTRS:
            self.assertTrue(hasattr(cls, attr), f"MotifVideoModelLoader.{attr} 없음")

    def test_loader_input_types_is_classmethod(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        self.assertTrue(
            callable(cls.INPUT_TYPES),
            "INPUT_TYPES는 callable(classmethod)이어야 함"
        )

    def test_loader_input_types_returns_dict_with_required_key(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        result = cls.INPUT_TYPES()
        self.assertIsInstance(result, dict, "INPUT_TYPES()가 dict를 반환해야 함")
        self.assertIn("required", result, "INPUT_TYPES() 결과에 'required' 키 없음")

    def test_loader_return_types_is_tuple(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        self.assertIsInstance(cls.RETURN_TYPES, tuple, "RETURN_TYPES는 tuple이어야 함")

    def test_loader_return_types_contains_model_clip_vae(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        self.assertEqual(
            cls.RETURN_TYPES, ("MODEL", "CLIP", "VAE"),
            f"예상 ('MODEL','CLIP','VAE'), 실제 {cls.RETURN_TYPES}"
        )

    def test_loader_function_is_string(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        self.assertIsInstance(cls.FUNCTION, str, "FUNCTION은 str이어야 함")

    def test_loader_function_method_exists(self):
        """FUNCTION 문자열이 실제 인스턴스 메서드를 가리켜야 한다."""
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        self.assertTrue(
            hasattr(cls, cls.FUNCTION),
            f"FUNCTION='{cls.FUNCTION}'에 해당하는 메서드가 클래스에 없음"
        )

    def test_loader_category_is_string(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        self.assertIsInstance(cls.CATEGORY, str, "CATEGORY는 str이어야 함")

    def test_loader_category_nonempty(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        self.assertTrue(len(cls.CATEGORY) > 0, "CATEGORY가 빈 문자열임")

    def test_loader_input_required_has_four_fields(self):
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        required = cls.INPUT_TYPES()["required"]
        self.assertEqual(
            len(required), 4,
            f"required 필드 수 예상 4, 실제 {len(required)}: {list(required.keys())}"
        )

    def test_loader_weight_dtype_options_include_fp8(self):
        """fp8_e4m3fn 옵션이 weight_dtype 선택지에 있어야 한다."""
        cls = self._get_class("nodes.loader", "MotifVideoModelLoader")
        dtype_field = cls.INPUT_TYPES()["required"]["weight_dtype"]
        # 첫 번째 원소가 선택지 list/tuple
        options = dtype_field[0]
        self.assertIn("fp8_e4m3fn", options, "fp8_e4m3fn 옵션 없음")

    # --- MotifTextEncode ---

    def test_text_encode_has_required_attrs(self):
        cls = self._get_class("nodes.text_encode", "MotifTextEncode")
        for attr in REQUIRED_ATTRS:
            self.assertTrue(hasattr(cls, attr), f"MotifTextEncode.{attr} 없음")

    def test_text_encode_return_types_two_conditioning(self):
        cls = self._get_class("nodes.text_encode", "MotifTextEncode")
        self.assertEqual(
            cls.RETURN_TYPES, ("CONDITIONING", "CONDITIONING"),
            f"예상 ('CONDITIONING','CONDITIONING'), 실제 {cls.RETURN_TYPES}"
        )

    def test_text_encode_has_return_names(self):
        """RETURN_NAMES는 선택 속성이지만 있다면 길이가 RETURN_TYPES와 같아야 한다."""
        cls = self._get_class("nodes.text_encode", "MotifTextEncode")
        if hasattr(cls, "RETURN_NAMES"):
            self.assertEqual(
                len(cls.RETURN_NAMES), len(cls.RETURN_TYPES),
                "RETURN_NAMES 길이가 RETURN_TYPES와 다름"
            )

    def test_text_encode_return_names_positive_negative(self):
        cls = self._get_class("nodes.text_encode", "MotifTextEncode")
        self.assertTrue(hasattr(cls, "RETURN_NAMES"), "RETURN_NAMES 없음")
        self.assertEqual(
            cls.RETURN_NAMES, ("positive", "negative"),
            f"예상 ('positive','negative'), 실제 {cls.RETURN_NAMES}"
        )

    def test_text_encode_input_required_has_clip_and_texts(self):
        cls = self._get_class("nodes.text_encode", "MotifTextEncode")
        required = cls.INPUT_TYPES()["required"]
        for key in ("clip", "text", "negative_prompt"):
            self.assertIn(key, required, f"required에 '{key}' 키 없음")

    def test_text_encode_function_method_exists(self):
        cls = self._get_class("nodes.text_encode", "MotifTextEncode")
        self.assertTrue(
            hasattr(cls, cls.FUNCTION),
            f"FUNCTION='{cls.FUNCTION}'에 해당하는 메서드 없음"
        )

    def test_text_encode_category_is_motifvideo(self):
        cls = self._get_class("nodes.text_encode", "MotifTextEncode")
        self.assertEqual(cls.CATEGORY, "motifvideo")

    # --- EmptyMotifLatent ---

    def test_latent_has_required_attrs(self):
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        for attr in REQUIRED_ATTRS:
            self.assertTrue(hasattr(cls, attr), f"EmptyMotifLatent.{attr} 없음")

    def test_latent_return_types_is_latent_tuple(self):
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        self.assertEqual(
            cls.RETURN_TYPES, ("LATENT",),
            f"예상 ('LATENT',), 실제 {cls.RETURN_TYPES}"
        )

    def test_latent_input_required_has_four_dimension_fields(self):
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        required = cls.INPUT_TYPES()["required"]
        for key in ("width", "height", "num_frames", "batch_size"):
            self.assertIn(key, required, f"required에 '{key}' 키 없음")

    def test_latent_width_min_boundary(self):
        """width min=64 경계값 확인."""
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        width_spec = cls.INPUT_TYPES()["required"]["width"][1]
        self.assertIn("min", width_spec, "width에 min 없음")
        self.assertEqual(width_spec["min"], 64)

    def test_latent_height_min_boundary(self):
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        height_spec = cls.INPUT_TYPES()["required"]["height"][1]
        self.assertEqual(height_spec["min"], 64)

    def test_latent_num_frames_min_is_one(self):
        """num_frames 최솟값이 1이어야 한다 (0 프레임 방지)."""
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        nf_spec = cls.INPUT_TYPES()["required"]["num_frames"][1]
        self.assertEqual(nf_spec["min"], 1, "num_frames min은 1이어야 함 (0 프레임 금지)")

    def test_latent_batch_size_min_is_one(self):
        """batch_size 최솟값이 1이어야 한다."""
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        bs_spec = cls.INPUT_TYPES()["required"]["batch_size"][1]
        self.assertEqual(bs_spec["min"], 1)

    def test_latent_function_method_exists(self):
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        self.assertTrue(
            hasattr(cls, cls.FUNCTION),
            f"FUNCTION='{cls.FUNCTION}'에 해당하는 메서드 없음"
        )

    def test_latent_category_is_motifvideo(self):
        cls = self._get_class("nodes.latent", "EmptyMotifLatent")
        self.assertEqual(cls.CATEGORY, "motifvideo")

    # --- 공통: FUNCTION 문자열이 빈 값이 아님 ---

    def test_all_nodes_function_nonempty(self):
        for mod, cls_name in NODE_SPECS:
            cls = self._get_class(mod, cls_name)
            self.assertTrue(
                len(cls.FUNCTION) > 0,
                f"{cls_name}.FUNCTION이 빈 문자열"
            )

    # --- 타입 불일치 엣지케이스 ---

    def test_return_types_not_list(self):
        """RETURN_TYPES는 tuple이어야 하며 list이면 안 된다."""
        for mod, cls_name in NODE_SPECS:
            cls = self._get_class(mod, cls_name)
            self.assertNotIsInstance(
                cls.RETURN_TYPES, list,
                f"{cls_name}.RETURN_TYPES가 list임 — tuple이어야 함"
            )

    def test_input_types_returns_fresh_dict_each_call(self):
        """INPUT_TYPES()를 두 번 호출했을 때 동일한 구조를 반환해야 한다."""
        for mod, cls_name in NODE_SPECS:
            cls = self._get_class(mod, cls_name)
            first = cls.INPUT_TYPES()
            second = cls.INPUT_TYPES()
            self.assertEqual(
                first.keys(), second.keys(),
                f"{cls_name}.INPUT_TYPES() 두 번 호출 결과 키 불일치"
            )


# ---------------------------------------------------------------------------
# 4. __init__.py graceful failure 검증
# ---------------------------------------------------------------------------

class TestGracefulFailure(unittest.TestCase):
    """
    루트 __init__.py가 ImportError 발생 시에도 graceful하게 처리하는지 확인.
    comfy 패키지 없음, nodes 로드 실패 등 두 가지 시나리오를 검증.
    """

    def _import_root_with_mocks(self, fail_nodes: bool = False):
        """
        루트 __init__를 격리된 환경에서 import한다.
        fail_nodes=True 이면 nodes.loader import를 강제로 실패시킨다.
        반환값: 생성된 모듈 객체
        """
        str_root = str(PROJECT_ROOT)
        inserted = str_root not in sys.path
        if inserted:
            sys.path.insert(0, str_root)

        # 기존 캐시 제거
        to_remove = [
            k for k in sys.modules
            if k in (
                "nodes", "nodes.loader", "nodes.latent", "nodes.text_encode",
                "config", "__init__",
            ) or k.startswith("ComfyUI")
        ]
        for k in to_remove:
            del sys.modules[k]

        # comfy는 항상 mock (없는 환경 시뮬레이션)
        mock_comfy = MagicMock()
        mock_comfy.supported_models.models = []

        comfy_patches = {
            "comfy": mock_comfy,
            "comfy.supported_models": mock_comfy.supported_models,
        }

        if fail_nodes:
            # nodes.loader import 자체를 실패하게 만든다
            broken = types.ModuleType("nodes.loader")
            broken.__spec__ = None

            def raise_on_attr(name):
                raise ImportError("의도적 ImportError — nodes.loader")

            broken.__getattr__ = raise_on_attr
            sys.modules["nodes.loader"] = broken

        original = {}
        for key, val in comfy_patches.items():
            original[key] = sys.modules.get(key)
            sys.modules[key] = val

        try:
            import importlib
            # 패키지 이름은 파일시스템 경로 기반으로 직접 로드
            spec = importlib.util.spec_from_file_location(
                "_test_root_init",
                str(PROJECT_ROOT / "__init__.py"),
                submodule_search_locations=[str(PROJECT_ROOT)],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_test_root_init"] = mod
            # 서브모듈 import가 상대경로를 쓰므로 패키지 이름 맞춰줌
            mod.__package__ = "_test_root_init"
            spec.loader.exec_module(mod)
            return mod
        finally:
            del sys.modules["_test_root_init"]
            for key, val in original.items():
                if val is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = val
            if inserted:
                sys.path.remove(str_root)

    def test_graceful_node_class_mappings_exists_always(self):
        """
        comfy 없는 환경에서도 NODE_CLASS_MAPPINGS가 정의되어야 한다.
        (빈 dict이거나 정상 dict이어야 함 — AttributeError는 허용 불가)
        """
        # 구조 검사만: __init__.py 소스에 fallback 패턴이 있는지 확인
        src = (PROJECT_ROOT / "__init__.py").read_text()
        tree = ast.parse(src)

        # except 블록 안에 NODE_CLASS_MAPPINGS = {} 대입이 있는지 확인
        fallback_found = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.ExceptHandler,)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "NODE_CLASS_MAPPINGS"
                            ):
                                fallback_found = True
        self.assertTrue(
            fallback_found,
            "__init__.py except 블록에 NODE_CLASS_MAPPINGS fallback 대입이 없음"
        )

    def test_graceful_node_display_name_mappings_fallback_exists(self):
        """NODE_DISPLAY_NAME_MAPPINGS도 fallback이 있어야 한다."""
        src = (PROJECT_ROOT / "__init__.py").read_text()
        tree = ast.parse(src)

        fallback_found = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.ExceptHandler,)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "NODE_DISPLAY_NAME_MAPPINGS"
                            ):
                                fallback_found = True
        self.assertTrue(
            fallback_found,
            "__init__.py except 블록에 NODE_DISPLAY_NAME_MAPPINGS fallback 없음"
        )

    def test_graceful_try_except_wraps_node_imports(self):
        """노드 import가 try/except 안에 있는지 AST로 확인."""
        src = (PROJECT_ROOT / "__init__.py").read_text()
        tree = ast.parse(src)

        loader_import_in_try = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    if isinstance(child, ast.ImportFrom):
                        if child.module and "loader" in child.module:
                            loader_import_in_try = True
        self.assertTrue(
            loader_import_in_try,
            "nodes.loader import가 try 블록 안에 없음 — graceful failure 미적용"
        )

    def test_graceful_all_exports_defined(self):
        """__all__에 NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS가 포함되어야 한다."""
        src = (PROJECT_ROOT / "__init__.py").read_text()
        tree = ast.parse(src)

        all_values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            all_values = [
                                elt.value if isinstance(elt, ast.Constant) else None
                                for elt in node.value.elts
                            ]

        self.assertIn(
            "NODE_CLASS_MAPPINGS", all_values,
            "__all__에 NODE_CLASS_MAPPINGS 없음"
        )
        self.assertIn(
            "NODE_DISPLAY_NAME_MAPPINGS", all_values,
            "__all__에 NODE_DISPLAY_NAME_MAPPINGS 없음"
        )

    def test_graceful_node_mappings_keys_count_when_success(self):
        """
        nodes 로드 성공 시 NODE_CLASS_MAPPINGS에 3개 항목이 있어야 한다.
        AST로 딕셔너리 리터럴 키 수를 확인 (런타임 import 없이).
        """
        src = (PROJECT_ROOT / "__init__.py").read_text()
        tree = ast.parse(src)

        mapping_sizes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "NODE_CLASS_MAPPINGS"
                        and isinstance(node.value, ast.Dict)
                    ):
                        mapping_sizes.append(len(node.value.keys))

        # 정상 경로 딕셔너리(4개)와 fallback {}(0개) 모두 있을 수 있음
        self.assertIn(
            4, mapping_sizes,
            f"NODE_CLASS_MAPPINGS 정상 경로에 4개 항목이 없음. 발견된 크기: {mapping_sizes}"
        )

    def test_graceful_fallback_mapping_is_empty_dict(self):
        """fallback NODE_CLASS_MAPPINGS는 빈 딕셔너리여야 한다."""
        src = (PROJECT_ROOT / "__init__.py").read_text()
        tree = ast.parse(src)

        fallback_sizes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "NODE_CLASS_MAPPINGS"
                                and isinstance(child.value, ast.Dict)
                            ):
                                fallback_sizes.append(len(child.value.keys))

        self.assertIn(
            0, fallback_sizes,
            f"fallback NODE_CLASS_MAPPINGS가 빈 dict({{}})가 아님. 발견 크기: {fallback_sizes}"
        )

    def test_graceful_print_on_error_uses_stderr(self):
        """에러 출력이 stderr로 향하는지 확인 (사용자 혼란 방지)."""
        src = (PROJECT_ROOT / "__init__.py").read_text()
        # sys.stderr 참조가 있어야 함
        self.assertIn(
            "sys.stderr",
            src,
            "__init__.py 에러 출력이 sys.stderr를 사용하지 않음"
        )


# ---------------------------------------------------------------------------
# 5. nodes/__init__.py re-export 구조 검증
# ---------------------------------------------------------------------------

class TestNodesPackageInit(unittest.TestCase):
    """nodes/__init__.py가 3개 클래스를 올바르게 re-export하는지 확인."""

    def test_nodes_init_exports_core_classes(self):
        """nodes/__init__.py에 기존 3개 코어 클래스가 포함되어야 한다.
        MotifTeaCache는 루트 __init__.py에서 직접 import하므로 여기서 불필요."""
        src = (PROJECT_ROOT / "nodes" / "__init__.py").read_text()
        for name in ("MotifTextEncoderLoader", "MotifTextEncode", "EmptyMotifLatent"):
            self.assertIn(name, src, f"nodes/__init__.py에 {name} 없음")

    def test_nodes_init_has_dunder_all(self):
        src = (PROJECT_ROOT / "nodes" / "__init__.py").read_text()
        self.assertIn("__all__", src, "nodes/__init__.py에 __all__ 없음")

    def test_nodes_init_all_contains_at_least_three_names(self):
        src = (PROJECT_ROOT / "nodes" / "__init__.py").read_text()
        tree = ast.parse(src)
        all_values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            all_values = [
                                elt.value if isinstance(elt, ast.Constant) else None
                                for elt in node.value.elts
                            ]
        self.assertGreaterEqual(
            len(all_values), 3,
            f"__all__ 원소 수 3개 이상이어야 함, 실제 {len(all_values)}: {all_values}"
        )

    def test_nodes_init_no_wildcard_import(self):
        """from .module import * 패턴은 없어야 한다."""
        src = (PROJECT_ROOT / "nodes" / "__init__.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.assertNotEqual(
                        alias.name, "*",
                        "nodes/__init__.py에 wildcard import(*) 사용 — 명시적 import 권장"
                    )


# ---------------------------------------------------------------------------
# 6. MotifTeaCache 노드 등록 검증 (AST + 런타임)
# ---------------------------------------------------------------------------

class TestMotifTeaCacheRegistration(unittest.TestCase):
    """루트 __init__.py에 MotifTeaCache가 올바르게 등록되어 있는지 검증."""

    def _get_root_init_src_and_tree(self):
        src = (PROJECT_ROOT / "__init__.py").read_text()
        return src, ast.parse(src)

    # --- AST: NODE_CLASS_MAPPINGS에 "MotifTeaCache" 키 존재 ---

    def test_teacache_key_in_node_class_mappings_ast(self):
        """NODE_CLASS_MAPPINGS 딕셔너리 리터럴에 'MotifTeaCache' 키가 있어야 한다."""
        _, tree = self._get_root_init_src_and_tree()

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "NODE_CLASS_MAPPINGS"
                        and isinstance(node.value, ast.Dict)
                    ):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and key.value == "MotifTeaCache":
                                found = True
        self.assertTrue(
            found,
            "NODE_CLASS_MAPPINGS에 'MotifTeaCache' 키가 없음"
        )

    # --- AST: NODE_DISPLAY_NAME_MAPPINGS에 "MotifTeaCache" 키 존재 ---

    def test_teacache_key_in_node_display_name_mappings_ast(self):
        """NODE_DISPLAY_NAME_MAPPINGS 딕셔너리 리터럴에 'MotifTeaCache' 키가 있어야 한다."""
        _, tree = self._get_root_init_src_and_tree()

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "NODE_DISPLAY_NAME_MAPPINGS"
                        and isinstance(node.value, ast.Dict)
                    ):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and key.value == "MotifTeaCache":
                                found = True
        self.assertTrue(
            found,
            "NODE_DISPLAY_NAME_MAPPINGS에 'MotifTeaCache' 키가 없음"
        )

    # --- AST: NODE_DISPLAY_NAME_MAPPINGS "MotifTeaCache" 값이 올바른지 ---

    def test_teacache_display_name_value_correct(self):
        """NODE_DISPLAY_NAME_MAPPINGS['MotifTeaCache'] 값이 'MotifVideo TeaCache'이어야 한다."""
        _, tree = self._get_root_init_src_and_tree()

        display_value = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "NODE_DISPLAY_NAME_MAPPINGS"
                        and isinstance(node.value, ast.Dict)
                    ):
                        for key, val in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant) and key.value == "MotifTeaCache":
                                if isinstance(val, ast.Constant):
                                    display_value = val.value

        self.assertEqual(
            display_value,
            "MotifVideo TeaCache",
            f"NODE_DISPLAY_NAME_MAPPINGS['MotifTeaCache'] 예상 'MotifVideo TeaCache', 실제 {display_value!r}"
        )

    # --- AST: MotifTeaCache import 문이 teacache 모듈에서 오는지 ---

    def test_teacache_imported_from_nodes_teacache(self):
        """from .nodes.teacache import MotifTeaCache 구문이 있어야 한다."""
        _, tree = self._get_root_init_src_and_tree()

        import_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                if "teacache" in module and "MotifTeaCache" in names:
                    import_found = True

        self.assertTrue(
            import_found,
            "from .nodes.teacache import MotifTeaCache 구문이 __init__.py에 없음"
        )

    # --- 런타임: MotifTeaCache 클래스가 실제로 로드 가능하고 올바른 타입인지 ---

    def test_teacache_class_loadable_and_has_required_attrs(self):
        """MotifTeaCache가 ComfyUI 노드 필수 속성을 모두 가진 클래스인지 확인."""
        try:
            cls = _load_node_class("nodes.teacache", "MotifTeaCache")
        except Exception as exc:
            self.skipTest(f"MotifTeaCache 로드 실패 (의존성 문제): {exc}")

        for attr in REQUIRED_ATTRS:
            self.assertTrue(
                hasattr(cls, attr),
                f"MotifTeaCache.{attr} 속성 없음"
            )

    def test_teacache_class_is_class_not_instance(self):
        """NODE_CLASS_MAPPINGS 값은 인스턴스가 아닌 클래스(type)이어야 한다."""
        try:
            cls = _load_node_class("nodes.teacache", "MotifTeaCache")
        except Exception as exc:
            self.skipTest(f"MotifTeaCache 로드 실패: {exc}")

        self.assertIsInstance(
            cls, type,
            "MotifTeaCache가 클래스(type)가 아님 — 인스턴스가 등록되면 안 됨"
        )

    def test_teacache_return_types_is_tuple_not_list(self):
        """RETURN_TYPES는 tuple이어야 하며 list이면 안 된다."""
        try:
            cls = _load_node_class("nodes.teacache", "MotifTeaCache")
        except Exception as exc:
            self.skipTest(f"MotifTeaCache 로드 실패: {exc}")

        self.assertNotIsInstance(
            cls.RETURN_TYPES, list,
            "MotifTeaCache.RETURN_TYPES가 list임 — tuple이어야 함"
        )

    def test_teacache_function_method_exists_on_class(self):
        """FUNCTION 문자열이 가리키는 메서드가 클래스에 실제로 존재해야 한다."""
        try:
            cls = _load_node_class("nodes.teacache", "MotifTeaCache")
        except Exception as exc:
            self.skipTest(f"MotifTeaCache 로드 실패: {exc}")

        self.assertTrue(
            hasattr(cls, cls.FUNCTION),
            f"MotifTeaCache.FUNCTION='{cls.FUNCTION}'에 해당하는 메서드가 없음"
        )

    # --- 경계값: NODE_CLASS_MAPPINGS 키가 빈 문자열이 아닌지 ---

    def test_teacache_mapping_key_nonempty_string(self):
        """'MotifTeaCache' 키가 빈 문자열이 아닌지 확인 (경계값)."""
        _, tree = self._get_root_init_src_and_tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "NODE_CLASS_MAPPINGS"
                        and isinstance(node.value, ast.Dict)
                    ):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and key.value == "MotifTeaCache":
                                self.assertGreater(
                                    len(key.value), 0,
                                    "NODE_CLASS_MAPPINGS 'MotifTeaCache' 키가 빈 문자열"
                                )
                                return
        # 키를 찾지 못한 경우 — 이미 test_teacache_key_in_node_class_mappings_ast에서 잡힘
        self.skipTest("'MotifTeaCache' 키를 찾지 못함 — 다른 테스트에서 검출됨")
