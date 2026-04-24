"""Unit tests for nodes/image_encode.py — Item 4 (i2v-shape-align).

Requirements tested:
1. INPUT_TYPES['required'] has 'latent' slot  → test_input_types_has_latent_slot
2. encode() resizes image tensor to (target_h, target_w)
   derived from latent samples shape             → test_encode_resizes_to_latent_shape
3. vae.encode() is called with RGB (3-channel)
   slice of the resized tensor                   → test_encode_calls_vae_with_rgb_slice

Environment: CPU-only pod.
- comfy.model_management is mocked (CUDA eager init crashes without GPU).
- comfy.utils is imported for real (works on CPU) — common_upscale genuine.
- VAE is always mocked (requires GPU + large checkpoint).

Import strategy:
  pytest.ini rootdir = /lustrefs/.../ComfyUI — 'comfy' is a real package there.
  We must NOT register a fake 'comfy' package stub before loading comfy.utils.
  Instead: import comfy.utils for real first, then patch comfy.model_management
  into the already-loaded comfy package so it doesn't trigger CUDA.
"""

import importlib.util
import pathlib
import sys
import types
import unittest.mock as mock

import pytest
import torch

# ---------------------------------------------------------------------------
# Step 1: import comfy.utils for real (CPU-safe), then patch model_management
# ---------------------------------------------------------------------------

# Import the real comfy.utils FIRST so that 'comfy' package is registered.
# After this, sys.modules['comfy'] is the real package object.
import comfy.utils as _real_comfy_utils  # noqa: E402  (real import, CPU-safe)

# Now replace model_management with a CPU stub to prevent CUDA crash.
_mm_stub = types.ModuleType("comfy.model_management")
_mm_stub.intermediate_device = lambda: torch.device("cpu")
_mm_stub.get_torch_device = lambda: torch.device("cpu")
sys.modules["comfy.model_management"] = _mm_stub
# Attach to the real comfy package object so attribute access also works.
import comfy as _comfy_pkg
_comfy_pkg.model_management = _mm_stub

# ---------------------------------------------------------------------------
# Step 2: install node_helpers stub
# ---------------------------------------------------------------------------

def _install_node_helpers_mock():
    nh = types.ModuleType("node_helpers")

    def _conditioning_set_values(cond, values):
        result = []
        for entry in cond:
            new_entry = [entry[0], {**entry[1], **values}]
            result.append(new_entry)
        return result

    nh.conditioning_set_values = _conditioning_set_values
    sys.modules.setdefault("node_helpers", nh)


_install_node_helpers_mock()

# ---------------------------------------------------------------------------
# Step 3: load nodes/image_encode.py via spec_from_file_location
# ---------------------------------------------------------------------------

_MODULE_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "nodes" / "image_encode.py"
)

_spec = importlib.util.spec_from_file_location("nodes.image_encode", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MotifImageEncode = _mod.MotifImageEncode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_latent(b=1, c=16, t=31, h=92, w=160):
    """Return a ComfyUI LATENT dict with given samples shape."""
    return {"samples": torch.zeros(b, c, t, h, w)}


def _make_image(b=1, H=500, W=1000, C=3):
    """Return a BHWC float tensor in [0, 1] (ComfyUI IMAGE convention)."""
    return torch.rand(b, H, W, C)


def _make_conditioning():
    """Minimal ComfyUI conditioning list: list of [tensor, dict] pairs."""
    cond_tensor = torch.zeros(1, 1, 4)
    return [[cond_tensor, {}]]


def _make_vae_mock():
    """Return a MagicMock VAE that records encode() calls."""
    vae = mock.MagicMock()
    vae.encode.return_value = torch.zeros(1, 16, 31, 92, 160)
    return vae


# ===========================================================================
# Test 1 — INPUT_TYPES declares 'latent' slot
# ===========================================================================

class TestInputTypesContract:
    def test_input_types_has_latent_slot(self):
        """INPUT_TYPES['required'] must contain 'latent' key (ComfyUI node contract)."""
        it = MotifImageEncode.INPUT_TYPES()
        required = it.get("required", {})
        assert "latent" in required, (
            f"'latent' slot missing from INPUT_TYPES required: {list(required.keys())}"
        )

    def test_latent_slot_type_is_LATENT(self):
        """The 'latent' slot must declare type 'LATENT'."""
        it = MotifImageEncode.INPUT_TYPES()
        latent_spec = it["required"]["latent"]
        assert latent_spec[0] == "LATENT", (
            f"'latent' slot type must be 'LATENT', got {latent_spec[0]!r}"
        )


# ===========================================================================
# Test 2 — encode() resizes image to match latent shape
# ===========================================================================

class TestEncodeResizesToLatentShape:
    """Verify that the image fed to vae.encode has the correct spatial dims."""

    def test_encode_resizes_to_latent_shape(self):
        """image [1,500,1000,3] + latent [1,16,31,92,160]
        → vae.encode receives image with H=736, W=1280.

        target_h = 92 * 8 = 736
        target_w = 160 * 8 = 1280
        """
        latent = _make_latent(b=1, c=16, t=31, h=92, w=160)
        image = _make_image(b=1, H=500, W=1000, C=3)
        pos = _make_conditioning()
        neg = _make_conditioning()
        vae = _make_vae_mock()

        node = MotifImageEncode()
        node.encode(pos, neg, vae, image, latent)

        assert vae.encode.call_count == 1, (
            f"vae.encode expected 1 call, got {vae.encode.call_count}"
        )

        called_image = vae.encode.call_args[0][0]
        target_h = latent["samples"].shape[-2] * 8   # 92 * 8 = 736
        target_w = latent["samples"].shape[-1] * 8   # 160 * 8 = 1280

        assert called_image.shape[1] == target_h, (
            f"vae.encode image H expected {target_h}, got {called_image.shape[1]}"
        )
        assert called_image.shape[2] == target_w, (
            f"vae.encode image W expected {target_w}, got {called_image.shape[2]}"
        )

    def test_encode_square_latent_resize(self):
        """Square latent [1,16,10,8,8] → target 64×64.

        Verifies the formula works for non-representative shapes.
        """
        latent = _make_latent(b=1, c=16, t=10, h=8, w=8)
        image = _make_image(b=1, H=300, W=200, C=3)
        pos = _make_conditioning()
        neg = _make_conditioning()
        vae = _make_vae_mock()

        node = MotifImageEncode()
        node.encode(pos, neg, vae, image, latent)

        assert vae.encode.call_count == 1
        called_image = vae.encode.call_args[0][0]
        target_h = 8 * 8   # 64
        target_w = 8 * 8   # 64
        assert called_image.shape[1] == target_h, (
            f"H expected {target_h}, got {called_image.shape[1]}"
        )
        assert called_image.shape[2] == target_w, (
            f"W expected {target_w}, got {called_image.shape[2]}"
        )


# ===========================================================================
# Test 3 — vae.encode receives 3-channel (RGB) slice
# ===========================================================================

class TestEncodePassesRGBSlice:
    def test_encode_calls_vae_with_rgb_slice(self):
        """vae.encode must receive an image tensor with exactly 3 channels (C=3).

        Provides a 4-channel image (RGBA) to expose missing RGB slice.
        """
        latent = _make_latent(b=1, c=16, t=31, h=92, w=160)
        image = _make_image(b=1, H=500, W=1000, C=4)
        pos = _make_conditioning()
        neg = _make_conditioning()
        vae = _make_vae_mock()

        node = MotifImageEncode()
        node.encode(pos, neg, vae, image, latent)

        assert vae.encode.call_count == 1
        called_image = vae.encode.call_args[0][0]
        assert called_image.shape[-1] == 3, (
            f"vae.encode must receive 3-channel image (RGB slice), "
            f"got shape {tuple(called_image.shape)}"
        )

    def test_encode_3ch_input_also_passes_rgb(self):
        """3-channel input should also yield 3-channel argument to vae.encode."""
        latent = _make_latent(b=1, c=16, t=31, h=92, w=160)
        image = _make_image(b=1, H=500, W=1000, C=3)
        pos = _make_conditioning()
        neg = _make_conditioning()
        vae = _make_vae_mock()

        node = MotifImageEncode()
        node.encode(pos, neg, vae, image, latent)

        called_image = vae.encode.call_args[0][0]
        assert called_image.shape[-1] == 3, (
            f"3-ch input: vae.encode arg must still be 3-channel, "
            f"got {tuple(called_image.shape)}"
        )
