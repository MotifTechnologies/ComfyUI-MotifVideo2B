"""항목 7+8 검증: MotifTextEncode + EmptyMotifLatent 정적 테스트.

    python tests/test_nodes.py
"""

import ast
import os
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_ENCODE_PY = os.path.join(_project_root, "nodes", "text_encode.py")
LATENT_PY = os.path.join(_project_root, "nodes", "latent.py")


class TestTextEncode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(TEXT_ENCODE_PY) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    def test_syntax(self):
        compile(self.source, TEXT_ENCODE_PY, "exec")

    def test_class_exists(self):
        classes = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("MotifTextEncode", classes)

    def test_return_types(self):
        self.assertIn('("CONDITIONING", "CONDITIONING")', self.source)

    def test_uses_encode_from_tokens_scheduled(self):
        """encode_from_tokens_scheduled 반환값을 직접 사용 (dict.pop 아님)."""
        self.assertIn("encode_from_tokens_scheduled", self.source)
        self.assertNotIn(".pop(", self.source)

    def test_no_not_implemented(self):
        self.assertNotIn("NotImplementedError", self.source)


class TestEmptyMotifLatent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(LATENT_PY) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    def test_syntax(self):
        compile(self.source, LATENT_PY, "exec")

    def test_class_exists(self):
        classes = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("EmptyMotifLatent", classes)

    def test_return_type(self):
        self.assertIn('("LATENT",)', self.source)

    def test_spatial_factor(self):
        self.assertIn("SPATIAL_FACTOR = 8", self.source)

    def test_temporal_factor(self):
        self.assertIn("TEMPORAL_FACTOR = 4", self.source)

    def test_latent_channels(self):
        self.assertIn("LATENT_CHANNELS = 16", self.source)

    def test_default_resolution(self):
        """Default 1280x736 matches MotifVideo training resolution."""
        self.assertIn('"default": 1280', self.source)
        self.assertIn('"default": 736', self.source)

    def test_default_frames(self):
        self.assertIn('"default": 121', self.source)

    def test_uses_intermediate_device(self):
        self.assertIn("intermediate_device()", self.source)

    def test_output_samples_key(self):
        self.assertIn('"samples"', self.source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
