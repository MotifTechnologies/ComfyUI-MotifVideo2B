"""항목 3 검증: MotifVideo19B model config 정적 검증 (CPU 가능) + 체크포인트 비교.

실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI/custom_nodes/ComfyUI-MotifVideo1.9B
    python tests/test_config.py

GPU 환경에서 comfy import 포함 전체 테스트:
    cd /lustrefs/team-multimodal/minsu/ComfyUI
    python custom_nodes/ComfyUI-MotifVideo1.9B/tests/test_config.py --gpu
"""

import ast
import json
import os
import sys
import unittest

TRANSFORMER_CONFIG = "/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/transformer/config.json"
SCHEDULER_CONFIG = "/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/scheduler/scheduler_config.json"
CONFIG_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")


class TestConfigStaticAnalysis(unittest.TestCase):
    """AST 기반 정적 검증 — comfy/CUDA 불필요."""

    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PY) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    def _find_class(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def _find_assign(self, class_node, attr):
        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == attr:
                        return node.value
        return None

    def test_syntax_valid(self):
        """config.py 문법 검증."""
        compile(self.source, CONFIG_PY, "exec")

    def test_class_exists(self):
        self.assertIsNotNone(self._find_class("MotifVideo19B"))

    def test_inherits_base(self):
        cls = self._find_class("MotifVideo19B")
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Attribute):
                base_names.append(base.attr)
            elif isinstance(base, ast.Name):
                base_names.append(base.id)
        self.assertIn("BASE", base_names)

    def test_has_unet_config(self):
        cls = self._find_class("MotifVideo19B")
        self.assertIsNotNone(self._find_assign(cls, "unet_config"))

    def test_has_optimizations(self):
        cls = self._find_class("MotifVideo19B")
        self.assertIsNotNone(self._find_assign(cls, "optimizations"))

    def test_has_model_type_method(self):
        cls = self._find_class("MotifVideo19B")
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("model_type", methods)

    def test_has_get_model_method(self):
        cls = self._find_class("MotifVideo19B")
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("get_model", methods)

    def test_has_clip_target_method(self):
        cls = self._find_class("MotifVideo19B")
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        self.assertIn("clip_target", methods)


class TestConfigCheckpointMatch(unittest.TestCase):
    """config.py의 하드코딩 값을 체크포인트 config와 비교."""

    @classmethod
    def setUpClass(cls):
        # config.py에서 unet_config dict를 직접 파싱
        with open(CONFIG_PY) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "MotifVideo19B":
                for item in ast.walk(node):
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == "unet_config":
                                cls.unet_config = ast.literal_eval(item.value)
                            if isinstance(target, ast.Name) and target.id == "sampling_settings":
                                cls.sampling_settings = ast.literal_eval(item.value)

    @unittest.skipUnless(os.path.exists(TRANSFORMER_CONFIG), "checkpoint not available")
    def test_matches_transformer_config(self):
        with open(TRANSFORMER_CONFIG) as f:
            ckpt = json.load(f)
        for key in ["in_channels", "out_channels", "num_attention_heads",
                     "attention_head_dim", "num_layers", "num_single_layers",
                     "num_decoder_layers", "text_embed_dim", "patch_size", "patch_size_t"]:
            self.assertEqual(
                self.unet_config[key], ckpt[key],
                f"config.py unet_config['{key}']={self.unet_config[key]} != checkpoint {ckpt[key]}"
            )

    @unittest.skipUnless(os.path.exists(SCHEDULER_CONFIG), "checkpoint not available")
    def test_shift_matches_scheduler(self):
        with open(SCHEDULER_CONFIG) as f:
            sched = json.load(f)
        expected = sched.get("shift", sched.get("global_shift"))
        self.assertEqual(self.sampling_settings["shift"], expected)

    def test_image_model_marker(self):
        self.assertEqual(self.unet_config["image_model"], "motif_video")

    def test_in_channels_33(self):
        self.assertEqual(self.unet_config["in_channels"], 33)


if __name__ == "__main__":
    unittest.main(verbosity=2)
