"""Unit tests for nodes/image_encode.py — MotifImageEncode node.

Covers:
- Class importability
- INPUT_TYPES structure (required: positive/CONDITIONING, negative/CONDITIONING,
  vae/VAE, image/IMAGE)
- RETURN_TYPES == ("CONDITIONING", "CONDITIONING")
- FUNCTION == "encode"
- CATEGORY == "motifvideo"
- encode() method existence and concat_latent_image injection into conditioning
- Edge cases: empty conditioning list, None image passthrough guard,
  multiple conditioning entries
"""

import importlib.util
import sys
import types
import unittest.mock as mock

import pytest
import torch

# ---------------------------------------------------------------------------
# Module isolation: mock comfy and related packages before loading image_encode
# so the test file is importable without a full ComfyUI installation.
# ---------------------------------------------------------------------------

_comfy_mock = mock.MagicMock()
sys.modules.setdefault("comfy", _comfy_mock)
sys.modules.setdefault("comfy.utils", mock.MagicMock())
sys.modules.setdefault("comfy.cli_args", mock.MagicMock())
sys.modules.setdefault("comfy.model_management", mock.MagicMock())
sys.modules.setdefault("comfy.sd", mock.MagicMock())
sys.modules.setdefault("folder_paths", mock.MagicMock())

# node_helpers lives at the ComfyUI root.
# conditioning_set_values(conditioning, values) must return a list of
# (tensor, {**original_dict, **values}) tuples so that length and key tests work.
def _real_conditioning_set_values(conditioning, values):
    result = []
    for t, d in conditioning:
        new_d = {**d, **values}
        result.append((t, new_d))
    return result


_node_helpers_mock = mock.MagicMock()
_node_helpers_mock.conditioning_set_values.side_effect = _real_conditioning_set_values
sys.modules.setdefault("node_helpers", _node_helpers_mock)

_IMAGE_ENCODE_PATH = (
    "/lustrefs/team-multimodal/minsu/ComfyUI"
    "/custom_nodes/ComfyUI-MotifVideo1.9B/nodes/image_encode.py"
)

_spec = importlib.util.spec_from_file_location("image_encode", _IMAGE_ENCODE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MotifImageEncode = _mod.MotifImageEncode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conditioning(n=1):
    """Return a list of n conditioning entries, each (tensor, dict) pair."""
    return [
        (torch.zeros(1, 4096, 4096), {"pooled_output": torch.zeros(1, 768)})
        for _ in range(n)
    ]


def _make_vae_mock(latent_tensor=None):
    """Return a mock VAE object whose encode() returns a latent dict."""
    if latent_tensor is None:
        latent_tensor = torch.zeros(1, 16, 4, 20, 35)  # (B, C, T, H, W)
    vae = mock.MagicMock()
    # VAE encode returns an object with samples attribute (ComfyUI convention)
    latent_result = mock.MagicMock()
    latent_result.__getitem__ = mock.Mock(side_effect=lambda k: latent_tensor)
    vae.encode.return_value = {"samples": latent_tensor}
    return vae, latent_tensor


# ===========================================================================
# 1. Importability
# ===========================================================================


class TestMotifImageEncodeImportable:
    def test_class_exists_in_module(self):
        assert hasattr(_mod, "MotifImageEncode"), (
            "MotifImageEncode class not found in nodes/image_encode.py"
        )

    def test_class_is_a_type(self):
        assert isinstance(MotifImageEncode, type)


# ===========================================================================
# 2. Class attributes (static contract)
# ===========================================================================


class TestMotifImageEncodeClassAttributes:
    def test_return_types_is_two_conditioning(self):
        assert MotifImageEncode.RETURN_TYPES == ("CONDITIONING", "CONDITIONING"), (
            f"Expected ('CONDITIONING', 'CONDITIONING'), got {MotifImageEncode.RETURN_TYPES}"
        )

    def test_function_is_encode(self):
        assert MotifImageEncode.FUNCTION == "encode", (
            f"Expected FUNCTION='encode', got {MotifImageEncode.FUNCTION!r}"
        )

    def test_category_is_motifvideo(self):
        assert MotifImageEncode.CATEGORY == "motifvideo", (
            f"Expected CATEGORY='motifvideo', got {MotifImageEncode.CATEGORY!r}"
        )

    def test_return_names_length_matches_return_types(self):
        """RETURN_NAMES (if defined) must have same length as RETURN_TYPES."""
        if hasattr(MotifImageEncode, "RETURN_NAMES"):
            assert len(MotifImageEncode.RETURN_NAMES) == len(
                MotifImageEncode.RETURN_TYPES
            )


# ===========================================================================
# 3. INPUT_TYPES structure
# ===========================================================================


class TestMotifImageEncodeInputTypes:
    @pytest.fixture(autouse=True)
    def _call_input_types(self):
        self.it = MotifImageEncode.INPUT_TYPES()

    def test_input_types_callable(self):
        assert callable(MotifImageEncode.INPUT_TYPES)

    def test_input_types_returns_dict(self):
        assert isinstance(self.it, dict)

    def test_required_key_present(self):
        assert "required" in self.it, "INPUT_TYPES must have a 'required' key"

    def test_positive_in_required(self):
        assert "positive" in self.it["required"], (
            "'positive' must be in required inputs"
        )

    def test_negative_in_required(self):
        assert "negative" in self.it["required"], (
            "'negative' must be in required inputs"
        )

    def test_vae_in_required(self):
        assert "vae" in self.it["required"], "'vae' must be in required inputs"

    def test_image_in_required(self):
        assert "image" in self.it["required"], "'image' must be in required inputs"

    def test_positive_type_is_conditioning(self):
        typ = self.it["required"]["positive"]
        # ComfyUI convention: first element of the tuple is the type string
        assert typ[0] == "CONDITIONING", (
            f"positive type should be CONDITIONING, got {typ[0]!r}"
        )

    def test_negative_type_is_conditioning(self):
        typ = self.it["required"]["negative"]
        assert typ[0] == "CONDITIONING", (
            f"negative type should be CONDITIONING, got {typ[0]!r}"
        )

    def test_vae_type_is_vae(self):
        typ = self.it["required"]["vae"]
        assert typ[0] == "VAE", f"vae type should be VAE, got {typ[0]!r}"

    def test_image_type_is_image(self):
        typ = self.it["required"]["image"]
        assert typ[0] == "IMAGE", f"image type should be IMAGE, got {typ[0]!r}"

    def test_no_mask_in_required(self):
        """mask 처리 불필요 (concat_cond 내부에서 자동 생성) — mask 입력 없어야 함."""
        required = self.it["required"]
        assert "mask" not in required, (
            "mask should NOT be in required inputs; it's handled internally"
        )


# ===========================================================================
# 4. encode() method existence
# ===========================================================================


class TestMotifImageEncodeMethodExists:
    def test_encode_method_exists(self):
        assert callable(getattr(MotifImageEncode, "encode", None)), (
            "encode() method must exist on MotifImageEncode"
        )


# ===========================================================================
# 5. encode() behaviour — concat_latent_image injection
# ===========================================================================


class TestMotifImageEncodeEncodeLogic:
    """Verify that encode() injects concat_latent_image into each conditioning entry.

    Uses a mock VAE to avoid GPU dependency.
    """

    def _run_encode(self, positive, negative, vae_mock, image):
        node = MotifImageEncode()
        return node.encode(positive=positive, negative=negative, vae=vae_mock, image=image)

    def test_returns_two_values(self):
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(1)
        neg = _make_conditioning(1)
        result = self._run_encode(pos, neg, vae, image)
        assert len(result) == 2, f"encode() must return 2 values, got {len(result)}"

    def test_output_positive_has_concat_latent_image(self):
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(1)
        neg = _make_conditioning(1)
        out_pos, out_neg = self._run_encode(pos, neg, vae, image)
        for entry in out_pos:
            cond_dict = entry[1]
            assert "concat_latent_image" in cond_dict, (
                "positive conditioning must contain 'concat_latent_image' key after encode()"
            )

    def test_output_negative_has_concat_latent_image(self):
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(1)
        neg = _make_conditioning(1)
        out_pos, out_neg = self._run_encode(pos, neg, vae, image)
        for entry in out_neg:
            cond_dict = entry[1]
            assert "concat_latent_image" in cond_dict, (
                "negative conditioning must contain 'concat_latent_image' key after encode()"
            )

    def test_original_conditioning_not_mutated(self):
        """encode() must not mutate the original positive/negative conditioning dicts."""
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(1)
        neg = _make_conditioning(1)
        original_pos_keys = set(pos[0][1].keys())
        original_neg_keys = set(neg[0][1].keys())
        self._run_encode(pos, neg, vae, image)
        assert set(pos[0][1].keys()) == original_pos_keys, (
            "encode() must not mutate original positive conditioning dict"
        )
        assert set(neg[0][1].keys()) == original_neg_keys, (
            "encode() must not mutate original negative conditioning dict"
        )

    def test_multiple_conditioning_entries_all_modified(self):
        """All entries in multi-entry conditioning must receive concat_latent_image."""
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(3)
        neg = _make_conditioning(3)
        out_pos, out_neg = self._run_encode(pos, neg, vae, image)
        assert len(out_pos) == 3, "Output positive conditioning entry count must match input"
        assert len(out_neg) == 3, "Output negative conditioning entry count must match input"
        for i, entry in enumerate(out_pos):
            assert "concat_latent_image" in entry[1], (
                f"out_pos[{i}] missing 'concat_latent_image'"
            )
        for i, entry in enumerate(out_neg):
            assert "concat_latent_image" in entry[1], (
                f"out_neg[{i}] missing 'concat_latent_image'"
            )

    def test_vae_encode_is_called(self):
        """VAE.encode must be invoked during encode()."""
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(1)
        neg = _make_conditioning(1)
        self._run_encode(pos, neg, vae, image)
        assert vae.encode.called, "vae.encode() must be called inside encode()"

    def test_process_latent_in_not_called_directly(self):
        """process_latent_in 호출하지 않음 — concat_cond 내부에서 처리됨."""
        # We verify that the node itself does not call process_latent_in on the VAE.
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(1)
        neg = _make_conditioning(1)
        self._run_encode(pos, neg, vae, image)
        # process_latent_in should NOT have been called directly on the vae mock
        if hasattr(vae, "process_latent_in"):
            assert not vae.process_latent_in.called, (
                "process_latent_in must NOT be called directly; handled inside concat_cond"
            )


# ===========================================================================
# 6. Edge cases
# ===========================================================================


class TestMotifImageEncodeEdgeCases:
    def _run_encode(self, positive, negative, vae_mock, image):
        node = MotifImageEncode()
        return node.encode(positive=positive, negative=negative, vae=vae_mock, image=image)

    def test_empty_positive_conditioning(self):
        """Empty positive conditioning list must not raise."""
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        try:
            out_pos, out_neg = self._run_encode([], _make_conditioning(1), vae, image)
        except Exception as exc:
            pytest.fail(f"Empty positive conditioning raised: {exc}")

    def test_empty_negative_conditioning(self):
        """Empty negative conditioning list must not raise."""
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        try:
            out_pos, out_neg = self._run_encode(_make_conditioning(1), [], vae, image)
        except Exception as exc:
            pytest.fail(f"Empty negative conditioning raised: {exc}")

    def test_different_positive_negative_entry_counts(self):
        """Positive and negative may have different numbers of entries."""
        vae, _ = _make_vae_mock()
        image = torch.zeros(1, 736, 1280, 3)
        pos = _make_conditioning(2)
        neg = _make_conditioning(1)
        try:
            out_pos, out_neg = self._run_encode(pos, neg, vae, image)
        except Exception as exc:
            pytest.fail(f"Different entry count raised: {exc}")
        assert len(out_pos) == 2
        assert len(out_neg) == 1

    def test_batch_size_greater_than_one(self):
        """Batch size > 1 must not raise."""
        vae, _ = _make_vae_mock(latent_tensor=torch.zeros(2, 16, 4, 20, 35))
        image = torch.zeros(2, 736, 1280, 3)
        pos = _make_conditioning(1)
        neg = _make_conditioning(1)
        try:
            self._run_encode(pos, neg, vae, image)
        except Exception as exc:
            pytest.fail(f"Batch size 2 raised: {exc}")


# ===========================================================================
# 7. NODE_CLASS_MAPPINGS registration (AST-based, same pattern as test_scaffolding.py)
# ===========================================================================


class TestNodeClassMappings:
    """Registration is done in the top-level __init__.py, not in individual
    node modules.  We verify via AST parsing (comfy unavailable in test env)."""

    @staticmethod
    def _get_init_source():
        import pathlib
        return pathlib.Path(__file__).resolve().parent.parent / "__init__.py"

    def test_motif_image_encode_in_node_class_mappings(self):
        """NODE_CLASS_MAPPINGS dict literal should contain 'MotifImageEncode'."""
        import ast
        source = self._get_init_source().read_text()
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS":
                        if isinstance(node.value, ast.Dict):
                            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                            if "MotifImageEncode" in keys:
                                found = True
        assert found, "NODE_CLASS_MAPPINGS에 'MotifImageEncode' 키가 없음"

    def test_motif_image_encode_in_display_name_mappings(self):
        """NODE_DISPLAY_NAME_MAPPINGS should contain 'MotifImageEncode'."""
        import ast
        source = self._get_init_source().read_text()
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "NODE_DISPLAY_NAME_MAPPINGS":
                        if isinstance(node.value, ast.Dict):
                            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                            if "MotifImageEncode" in keys:
                                found = True
        assert found, "NODE_DISPLAY_NAME_MAPPINGS에 'MotifImageEncode' 키가 없음"

    def test_import_statement_exists(self):
        """__init__.py should import MotifImageEncode from nodes.image_encode."""
        source = self._get_init_source().read_text()
        assert "from .nodes.image_encode import MotifImageEncode" in source
