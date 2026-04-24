"""Unit tests for nodes/loader.py — MotifTextEncoderLoader.load_clip.

Covers 4 contract assertions:
1. Normal return — (CLIP_instance,) tuple, comfy.sd.CLIP called exactly once.
2. Config embedded — model_options kwarg has no 'motifvideo_config_path' key.
3. Tokenizer bundled — tokenizer_data kwarg has no 'motifvideo_tokenizer_path' key.
4. Filename-agnostic — arbitrary clip_name produces the same return shape.

Design constraints (see P0.6 checklist):
- No absolute-path hardcoding; all paths derived from __file__.
- No module-level sys.modules mutation.  All mocks are fixture-scoped via
  monkeypatch.setitem / monkeypatch.delitem and are cleaned up automatically.
- loader module is imported inside the fixture *after* mocks are in place.
- Exactly 4 test cases.
- loader.py uses a relative import (from ..text_encoders.t5_gemma2).
  We satisfy it by registering a synthetic parent package 'motifnodes' in
  sys.modules so Python can resolve '..' correctly without touching the
  real ComfyUI package tree.
"""

import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Resolve loader.py path relative to this test file (no absolute hardcoding)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LOADER_PATH = _REPO_ROOT / "nodes" / "loader.py"

# Synthetic package name used only inside this test module.
# Must not collide with real installed packages.
_PKG = "motifnodes"


# ---------------------------------------------------------------------------
# Fixture: wire mocks into sys.modules, then load loader.py in isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def loader_env(monkeypatch):
    """Load nodes/loader.py with all external dependencies mocked.

    Uses monkeypatch.setitem / monkeypatch.delitem so every insertion is
    automatically reverted when the test ends — no cross-test pollution.

    Relative import resolution strategy
    ------------------------------------
    loader.py contains ``from ..text_encoders.t5_gemma2 import …``.
    When the module is assigned __package__ = '<PKG>.nodes', Python resolves
    '..' as '<PKG>' and then looks up '<PKG>.text_encoders.t5_gemma2' in
    sys.modules.  We create synthetic stub modules for the whole hierarchy
    so the lookup succeeds without touching the real ComfyUI tree.
    """

    # ---- synthetic parent package hierarchy (relative import plumbing) ------
    pkg_root = types.ModuleType(_PKG)
    pkg_nodes = types.ModuleType(f"{_PKG}.nodes")
    pkg_te = types.ModuleType(f"{_PKG}.text_encoders")

    t5_mock = MagicMock(name="text_encoders.t5_gemma2")
    t5_mock.MotifVideoSD1Tokenizer = MagicMock(name="MotifVideoSD1Tokenizer")
    t5_mock.te = MagicMock(name="te", return_value=MagicMock(name="TEModelCls"))

    for key, val in [
        (_PKG, pkg_root),
        (f"{_PKG}.nodes", pkg_nodes),
        (f"{_PKG}.text_encoders", pkg_te),
        (f"{_PKG}.text_encoders.t5_gemma2", t5_mock),
    ]:
        monkeypatch.setitem(sys.modules, key, val)

    # ---- folder_paths mock --------------------------------------------------
    fp_mock = MagicMock(name="folder_paths")
    fp_mock.get_filename_list.return_value = [
        "motifvideo_t5gemma2.safetensors",
        "custom_te_variant.safetensors",
    ]
    fp_mock.get_full_path_or_raise.side_effect = (
        lambda cat, name: f"/fake/{cat}/{name}"
    )
    fp_mock.get_folder_paths.return_value = ["/fake/embeddings"]

    monkeypatch.setitem(sys.modules, "folder_paths", fp_mock)

    # ---- safetensors mock ---------------------------------------------------
    st_torch_mock = MagicMock(name="safetensors.torch")
    st_torch_mock.load_file.return_value = {}  # empty state_dict

    st_mock = MagicMock(name="safetensors")
    st_mock.torch = st_torch_mock

    monkeypatch.setitem(sys.modules, "safetensors", st_mock)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st_torch_mock)

    # ---- comfy mock chain ---------------------------------------------------
    clip_instance = MagicMock(name="CLIP_instance")

    comfy_sd_mock = MagicMock(name="comfy.sd")
    comfy_sd_mock.CLIP.return_value = clip_instance

    comfy_supported_mock = MagicMock(name="comfy.supported_models_base")
    clip_target_obj = MagicMock(name="ClipTarget_instance")
    comfy_supported_mock.ClipTarget.return_value = clip_target_obj

    comfy_sd1_clip_mock = MagicMock(name="comfy.sd1_clip")

    comfy_mock = MagicMock(name="comfy")
    comfy_mock.sd = comfy_sd_mock
    comfy_mock.supported_models_base = comfy_supported_mock
    comfy_mock.sd1_clip = comfy_sd1_clip_mock

    monkeypatch.setitem(sys.modules, "comfy", comfy_mock)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_mock)
    monkeypatch.setitem(sys.modules, "comfy.supported_models_base", comfy_supported_mock)
    monkeypatch.setitem(sys.modules, "comfy.sd1_clip", comfy_sd1_clip_mock)

    # ---- load loader.py via spec (isolated from real package tree) ----------
    loader_module_key = f"{_PKG}.nodes.loader"
    monkeypatch.delitem(sys.modules, loader_module_key, raising=False)

    spec = importlib.util.spec_from_file_location(
        loader_module_key,
        str(_LOADER_PATH),
    )
    loader_mod = importlib.util.module_from_spec(spec)
    # __package__ = '<PKG>.nodes' makes '..' in the relative import resolve to
    # '<PKG>', which we have registered in sys.modules above.
    loader_mod.__package__ = f"{_PKG}.nodes"
    monkeypatch.setitem(sys.modules, loader_module_key, loader_mod)

    spec.loader.exec_module(loader_mod)

    return loader_mod, comfy_sd_mock, clip_instance


# ===========================================================================
# Test cases (exactly 4)
# ===========================================================================


class TestLoadClipContract:
    """Contract tests for MotifTextEncoderLoader.load_clip."""

    def test_normal_return_is_clip_tuple(self, loader_env):
        """Case 1: load_clip returns (CLIP_instance,) and CLIP called once."""
        loader_mod, comfy_sd_mock, clip_instance = loader_env

        node = loader_mod.MotifTextEncoderLoader()
        result = node.load_clip(
            clip_name="motifvideo_t5gemma2.safetensors",
            dtype="bfloat16",
        )

        assert result == (clip_instance,), (
            f"Expected (CLIP_instance,), got {result!r}"
        )
        comfy_sd_mock.CLIP.assert_called_once()

    def test_model_options_has_no_config_path_key(self, loader_env):
        """Case 2: model_options passed to comfy.sd.CLIP must not contain
        'motifvideo_config_path' — config is now a code constant (P0.1)."""
        loader_mod, comfy_sd_mock, _ = loader_env

        node = loader_mod.MotifTextEncoderLoader()
        node.load_clip(
            clip_name="motifvideo_t5gemma2.safetensors",
            dtype="bfloat16",
        )

        _, kwargs = comfy_sd_mock.CLIP.call_args
        model_options = kwargs.get("model_options", {})
        assert "motifvideo_config_path" not in model_options, (
            "model_options still contains 'motifvideo_config_path' — "
            "P0.1 removal may be incomplete"
        )

    def test_tokenizer_data_has_no_tokenizer_path_key(self, loader_env):
        """Case 3: tokenizer_data passed to comfy.sd.CLIP must not contain
        'motifvideo_tokenizer_path' — tokenizer defaults to bundled path (P0.4)."""
        loader_mod, comfy_sd_mock, _ = loader_env

        node = loader_mod.MotifTextEncoderLoader()
        node.load_clip(
            clip_name="motifvideo_t5gemma2.safetensors",
            dtype="bfloat16",
        )

        _, kwargs = comfy_sd_mock.CLIP.call_args
        tokenizer_data = kwargs.get("tokenizer_data", {})
        assert "motifvideo_tokenizer_path" not in tokenizer_data, (
            "tokenizer_data still contains 'motifvideo_tokenizer_path' — "
            "P0.4 removal may be incomplete"
        )

    def test_filename_agnostic_returns_clip_tuple(self, loader_env):
        """Case 4: load_clip with an arbitrary filename produces the same
        (CLIP_instance,) return — loader must not branch on clip_name."""
        loader_mod, comfy_sd_mock, clip_instance = loader_env

        comfy_sd_mock.CLIP.reset_mock()

        node = loader_mod.MotifTextEncoderLoader()
        result = node.load_clip(
            clip_name="custom_te_variant.safetensors",
            dtype="bfloat16",
        )

        assert result == (clip_instance,), (
            f"Expected (CLIP_instance,), got {result!r} for custom filename"
        )
        comfy_sd_mock.CLIP.assert_called_once()
