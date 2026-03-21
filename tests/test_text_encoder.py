"""항목 5+6 검증: Text Encoder 통합 + MotifTextEncoderLoader 정적 테스트.

    cd /lustrefs/team-multimodal/minsu/ComfyUI/custom_nodes/ComfyUI-MotifVideo1.9B
    python tests/test_text_encoder.py

GPU 테스트 (ComfyUI 루트에서):
    cd /lustrefs/team-multimodal/minsu/ComfyUI
    python custom_nodes/ComfyUI-MotifVideo1.9B/tests/test_text_encoder.py --gpu
"""

import ast
import os
import sys
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T5_GEMMA2_PY = os.path.join(_project_root, "text_encoders", "t5_gemma2.py")
LOADER_PY = os.path.join(_project_root, "nodes", "loader.py")
CONFIG_PY = os.path.join(_project_root, "config.py")
INIT_PY = os.path.join(_project_root, "__init__.py")


class TestT5Gemma2Syntax(unittest.TestCase):
    """AST 기반 정적 검증."""

    @classmethod
    def setUpClass(cls):
        with open(T5_GEMMA2_PY) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)
        cls.classes = {
            n.name: n for n in ast.walk(cls.tree) if isinstance(n, ast.ClassDef)
        }

    def test_syntax_valid(self):
        compile(self.source, T5_GEMMA2_PY, "exec")

    def test_tokenizer_class_exists(self):
        self.assertIn("MotifVideoTokenizer", self.classes)

    def test_sd1_tokenizer_class_exists(self):
        self.assertIn("MotifVideoSD1Tokenizer", self.classes)

    def test_model_class_exists(self):
        self.assertIn("MotifVideoT5Gemma2Model", self.classes)

    def test_sd1_clip_model_class_exists(self):
        self.assertIn("MotifVideoSD1ClipModel", self.classes)

    def test_te_factory_exists(self):
        funcs = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)]
        self.assertIn("te", funcs)

    def test_model_has_encode_token_weights(self):
        cls = self.classes["MotifVideoT5Gemma2Model"]
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("encode_token_weights", methods)

    def test_model_has_load_sd(self):
        cls = self.classes["MotifVideoT5Gemma2Model"]
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("load_sd", methods)

    def test_model_has_encode(self):
        cls = self.classes["MotifVideoT5Gemma2Model"]
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("encode", methods)

    def test_sd1clip_has_set_clip_options(self):
        cls = self.classes["MotifVideoSD1ClipModel"]
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("set_clip_options", methods)
        self.assertIn("reset_clip_options", methods)

    def test_no_hardcoded_checkpoint_path(self):
        """config_path 기본값이 None이어야 함 (하드코딩 금지)."""
        self.assertNotIn("/lustrefs/", self.source.split("model_options.get")[0]
                         if "model_options.get" in self.source else "")


class TestLoaderSyntax(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(LOADER_PY) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    def test_syntax_valid(self):
        compile(self.source, LOADER_PY, "exec")

    def test_loader_class_exists(self):
        classes = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("MotifTextEncoderLoader", classes)

    def test_loader_has_required_attrs(self):
        cls = next(n for n in ast.walk(self.tree)
                   if isinstance(n, ast.ClassDef) and n.name == "MotifTextEncoderLoader")
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("INPUT_TYPES", methods)
        self.assertIn("load_text_encoder", methods)

    def test_return_type_clip(self):
        self.assertIn('("CLIP",)', self.source)


class TestConfigClipTarget(unittest.TestCase):

    def test_clip_target_not_none(self):
        with open(CONFIG_PY) as f:
            source = f.read()
        # clip_target should import and return ClipTarget, not return None
        self.assertNotIn("return None", source.split("def clip_target")[1])

    def test_imports_te_factory(self):
        with open(CONFIG_PY) as f:
            source = f.read()
        self.assertIn("from .text_encoders.t5_gemma2 import", source)


class TestInitMappings(unittest.TestCase):

    def test_loader_registered(self):
        with open(INIT_PY) as f:
            source = f.read()
        self.assertIn("MotifTextEncoderLoader", source)

    def test_old_loader_removed(self):
        with open(INIT_PY) as f:
            source = f.read()
        self.assertNotIn("MotifVideoModelLoader", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
