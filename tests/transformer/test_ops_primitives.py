# tests/transformer/test_ops_primitives.py
#
# Verifies P1.1 checklist criteria:
#   1. state_dict key parity between local ops_primitives classes and diffusers originals
#   2. forward output parity when weights are copied from diffusers to local instances
#
# CUDA-free design: comfy.ops is replaced with a lightweight mock so the test
# suite can run without a GPU.  The production load path always has CUDA available
# and uses real comfy.ops — only the test environment needs this workaround.

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — all relative to this file, no absolute path strings.
# ---------------------------------------------------------------------------
# tests/transformer/test_ops_primitives.py → parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
# ComfyUI/custom_nodes/<repo> → parents[1] = ComfyUI root
_COMFYUI_ROOT = _REPO_ROOT.parent.parent

# Insert ComfyUI root so diffusers (installed in the active venv) is importable.
# No explicit site-packages path needed: the Python interpreter that runs pytest
# already has its venv's site-packages on sys.path.
if str(_COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMFYUI_ROOT))

# ---------------------------------------------------------------------------
# comfy.ops mock — installed via autouse module-scoped fixture so that
# sys.modules is properly restored after the test module finishes.
#
# Detection logic:
#   - If CUDA is available, attempt a real comfy.ops import.
#     If it succeeds, use it (production-equivalent path).
#   - Otherwise inject the lightweight mock.
#   - Environment variable MOTIF_FORCE_MOCK_COMFY=1 forces mock regardless
#     of CUDA presence (useful for debugging on GPU pods without comfy).
# ---------------------------------------------------------------------------

def _make_comfy_ops_mock():
    """Create a minimal comfy.ops mock where Linear/LayerNorm are plain nn classes."""

    class _MockOps:
        class Linear(nn.Linear):
            def __init__(self, *args, dtype=None, device=None, **kwargs):
                super().__init__(*args, **kwargs)

        class LayerNorm(nn.LayerNorm):
            def __init__(self, *args, dtype=None, device=None, **kwargs):
                super().__init__(*args, **kwargs)

    mock_comfy = types.ModuleType("comfy")
    mock_ops = types.ModuleType("comfy.ops")
    mock_ops.disable_weight_init = _MockOps
    mock_comfy.ops = mock_ops
    return mock_comfy, mock_ops


def _should_use_real_comfy() -> bool:
    """Return True only when CUDA is available and real comfy.ops can be imported."""
    import os
    if os.environ.get("MOTIF_FORCE_MOCK_COMFY") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import comfy.ops  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def comfy_mock_fixture():
    """Install comfy mock for the module; restore sys.modules on teardown.

    Teardown correctness: the pop/restore logic after `yield` is exercised at
    pytest module-scope finalization. It cannot be observed from within the same
    pytest session because the fixture is still active when test functions run.
    If teardown were broken, other test modules that import real `comfy` would
    start seeing the mock instead — making that failure self-reporting without
    needing a dedicated test.
    """
    if _should_use_real_comfy():
        # Real comfy is available — nothing to mock, nothing to restore.
        yield
        return

    # Back up existing entries (may be None/absent).
    _prev_comfy = sys.modules.get("comfy", None)
    _prev_comfy_ops = sys.modules.get("comfy.ops", None)

    mock_comfy, mock_ops = _make_comfy_ops_mock()
    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.ops"] = mock_ops

    yield

    # Teardown: restore original state.
    if _prev_comfy is None:
        sys.modules.pop("comfy", None)
    else:
        sys.modules["comfy"] = _prev_comfy

    if _prev_comfy_ops is None:
        sys.modules.pop("comfy.ops", None)
    else:
        sys.modules["comfy.ops"] = _prev_comfy_ops


# Import ops_primitives directly via importlib to avoid triggering models/__init__.py
# which imports comfy.model_base and would conflict with our lightweight mock.
import importlib.util as _ilu

_PRIMITIVES_PATH = _REPO_ROOT / "models" / "transformer" / "ops_primitives.py"
_spec = _ilu.spec_from_file_location("ops_primitives", _PRIMITIVES_PATH)
_ops_primitives_mod = _ilu.module_from_spec(_spec)
# Reset _DEFAULT_OPS cache so lazy init picks up our mock.
_ops_primitives_mod._DEFAULT_OPS = None
_spec.loader.exec_module(_ops_primitives_mod)

# Expose classes under convenient names
LocalTimestepEmbedding = _ops_primitives_mod.TimestepEmbedding
LocalPixArtAlpha = _ops_primitives_mod.PixArtAlphaTextProjection
LocalAdaLayerNormZero = _ops_primitives_mod.AdaLayerNormZero
LocalAdaLayerNormZeroSingle = _ops_primitives_mod.AdaLayerNormZeroSingle
LocalAdaLayerNormContinuous = _ops_primitives_mod.AdaLayerNormContinuous
LocalFeedForward = _ops_primitives_mod.FeedForward

from diffusers.models.embeddings import (
    TimestepEmbedding as DiffTimestepEmbedding,
    PixArtAlphaTextProjection as DiffPixArtAlpha,
)
from diffusers.models.normalization import (
    AdaLayerNormZero as DiffAdaLayerNormZero,
    AdaLayerNormZeroSingle as DiffAdaLayerNormZeroSingle,
    AdaLayerNormContinuous as DiffAdaLayerNormContinuous,
)
from diffusers.models.attention import FeedForward as DiffFeedForward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42


def _set_seed():
    torch.manual_seed(SEED)


def _randn(*shape):
    _set_seed()
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestStateDictKeyParity(unittest.TestCase):
    """P1.1 Verify: local.state_dict().keys() == diffusers.state_dict().keys() (set-equal)."""

    def _assert_key_parity(self, local_inst, diff_inst, label: str):
        local_keys = set(local_inst.state_dict().keys())
        diff_keys = set(diff_inst.state_dict().keys())
        extra = local_keys - diff_keys
        missing = diff_keys - local_keys
        self.assertEqual(
            local_keys,
            diff_keys,
            msg=(
                f"{label}: key mismatch.\n"
                f"  Extra in local:   {sorted(extra)}\n"
                f"  Missing in local: {sorted(missing)}"
            ),
        )

    def test_timestep_embedding(self):
        local = LocalTimestepEmbedding(256, 3072)
        diff = DiffTimestepEmbedding(256, 3072)
        self._assert_key_parity(local, diff, "TimestepEmbedding(256,3072)")

    def test_pixart_alpha_silu(self):
        local = LocalPixArtAlpha(256, 3072, act_fn="silu")
        diff = DiffPixArtAlpha(256, 3072, act_fn="silu")
        self._assert_key_parity(local, diff, "PixArtAlphaTextProjection(silu)")

    def test_pixart_alpha_gelu_tanh(self):
        local = LocalPixArtAlpha(4096, 3072)
        diff = DiffPixArtAlpha(4096, 3072)
        self._assert_key_parity(local, diff, "PixArtAlphaTextProjection(gelu_tanh)")

    def test_ada_layer_norm_zero(self):
        local = LocalAdaLayerNormZero(3072)
        diff = DiffAdaLayerNormZero(3072)
        self._assert_key_parity(local, diff, "AdaLayerNormZero(3072)")

    def test_ada_layer_norm_zero_single(self):
        local = LocalAdaLayerNormZeroSingle(3072)
        diff = DiffAdaLayerNormZeroSingle(3072)
        self._assert_key_parity(local, diff, "AdaLayerNormZeroSingle(3072)")

    def test_ada_layer_norm_continuous(self):
        local = LocalAdaLayerNormContinuous(3072, 3072, elementwise_affine=False, eps=1e-6)
        diff = DiffAdaLayerNormContinuous(3072, 3072, elementwise_affine=False, eps=1e-6)
        self._assert_key_parity(local, diff, "AdaLayerNormContinuous(no affine)")

    def test_ada_layer_norm_continuous_with_affine_bias_true(self):
        """state_dict keys must match diffusers when elementwise_affine=True, bias=True."""
        local = LocalAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=True, eps=1e-6)
        diff = DiffAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=True, eps=1e-6)
        self._assert_key_parity(local, diff, "AdaLayerNormContinuous(affine+bias=True)")

    def test_ada_layer_norm_continuous_with_affine_bias_false(self):
        """state_dict keys must match diffusers when elementwise_affine=True, bias=False."""
        local = LocalAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=False, eps=1e-6)
        diff = DiffAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=False, eps=1e-6)
        self._assert_key_parity(local, diff, "AdaLayerNormContinuous(affine+bias=False)")

    def test_feedforward_gelu_approximate(self):
        local = LocalFeedForward(3072, mult=4, activation_fn="gelu-approximate")
        diff = DiffFeedForward(3072, mult=4, activation_fn="gelu-approximate")
        self._assert_key_parity(local, diff, "FeedForward(gelu-approximate)")


class TestForwardParity(unittest.TestCase):
    """P1.1 Verify: same input + copied weights → torch.allclose(local, diffusers, rtol=1e-5)."""

    rtol = 1e-5
    atol = 1e-5

    def _check_close(self, local_out, diff_out, label: str):
        if isinstance(local_out, (tuple, list)):
            self.assertEqual(
                len(local_out),
                len(diff_out),
                msg=(
                    f"{label}: output tuple length mismatch. "
                    f"local={len(local_out)}, diffusers={len(diff_out)}"
                ),
            )
            for i, (lo, do) in enumerate(zip(local_out, diff_out)):
                if lo is None:
                    continue
                self.assertTrue(
                    torch.allclose(lo.float(), do.float(), rtol=self.rtol, atol=self.atol),
                    msg=f"{label} output[{i}] not close. max diff={( lo.float()-do.float()).abs().max().item():.6f}",
                )
        else:
            self.assertTrue(
                torch.allclose(local_out.float(), diff_out.float(), rtol=self.rtol, atol=self.atol),
                msg=f"{label} output not close. max diff={(local_out.float()-diff_out.float()).abs().max().item():.6f}",
            )

    def test_timestep_embedding(self):
        diff = DiffTimestepEmbedding(256, 3072)
        local = LocalTimestepEmbedding(256, 3072)
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        x = _randn(2, 256)
        with torch.no_grad():
            self._check_close(local(x), diff(x), "TimestepEmbedding")

    def test_pixart_alpha_silu(self):
        diff = DiffPixArtAlpha(256, 3072, act_fn="silu")
        local = LocalPixArtAlpha(256, 3072, act_fn="silu")
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        x = _randn(2, 256)
        with torch.no_grad():
            self._check_close(local(x), diff(x), "PixArtAlpha(silu)")

    def test_pixart_alpha_gelu_tanh(self):
        diff = DiffPixArtAlpha(4096, 3072)
        local = LocalPixArtAlpha(4096, 3072)
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        x = _randn(2, 4096)
        with torch.no_grad():
            self._check_close(local(x), diff(x), "PixArtAlpha(gelu_tanh)")

    def test_ada_layer_norm_zero(self):
        diff = DiffAdaLayerNormZero(3072)
        local = LocalAdaLayerNormZero(3072)
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        # x: (batch, seq, dim), emb: (batch, dim)
        x = _randn(2, 16, 3072)
        _set_seed()
        emb = torch.randn(2, 3072)
        with torch.no_grad():
            local_out = local(x, emb=emb)
            diff_out = diff(x, emb=emb)
        self._check_close(local_out, diff_out, "AdaLayerNormZero")

    def test_ada_layer_norm_zero_single(self):
        diff = DiffAdaLayerNormZeroSingle(3072)
        local = LocalAdaLayerNormZeroSingle(3072)
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        x = _randn(2, 16, 3072)
        _set_seed()
        emb = torch.randn(2, 3072)
        with torch.no_grad():
            self._check_close(local(x, emb=emb), diff(x, emb=emb), "AdaLayerNormZeroSingle")

    def test_ada_layer_norm_continuous(self):
        diff = DiffAdaLayerNormContinuous(3072, 3072, elementwise_affine=False, eps=1e-6)
        local = LocalAdaLayerNormContinuous(3072, 3072, elementwise_affine=False, eps=1e-6)
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        # x: (batch, seq, dim), conditioning_embedding: (batch, dim)
        x = _randn(2, 16, 3072)
        _set_seed()
        cond = torch.randn(2, 3072)
        with torch.no_grad():
            self._check_close(local(x, cond), diff(x, cond), "AdaLayerNormContinuous")

    def test_ada_layer_norm_continuous_bias_true(self):
        """Forward parity when elementwise_affine=True, bias=True (norm has weight+bias)."""
        diff = DiffAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=True, eps=1e-6)
        local = LocalAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=True, eps=1e-6)
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        x = _randn(2, 16, 3072)
        _set_seed()
        cond = torch.randn(2, 3072)
        with torch.no_grad():
            self._check_close(local(x, cond), diff(x, cond), "AdaLayerNormContinuous(bias=True)")

    def test_ada_layer_norm_continuous_bias_false(self):
        """Forward parity and state_dict key match when elementwise_affine=True, bias=False.

        Diffusers AdaLayerNormContinuous with bias=False has norm.weight but no norm.bias.
        Our implementation must match — bias=False is forwarded to ops.LayerNorm.
        """
        diff = DiffAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=False, eps=1e-6)
        local = LocalAdaLayerNormContinuous(3072, 3072, elementwise_affine=True, bias=False, eps=1e-6)
        # Verify norm has weight but no bias (mirrors diffusers behaviour)
        self.assertIsNotNone(local.norm.weight, "norm.weight should exist when elementwise_affine=True")
        self.assertIsNone(local.norm.bias, "norm.bias should be None when bias=False")
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        x = _randn(2, 16, 3072)
        _set_seed()
        cond = torch.randn(2, 3072)
        with torch.no_grad():
            self._check_close(local(x, cond), diff(x, cond), "AdaLayerNormContinuous(bias=False)")

    def test_feedforward_gelu_approximate(self):
        diff = DiffFeedForward(3072, mult=4, activation_fn="gelu-approximate")
        local = LocalFeedForward(3072, mult=4, activation_fn="gelu-approximate")
        local.load_state_dict(diff.state_dict())
        diff.eval(); local.eval()

        x = _randn(2, 16, 3072)
        with torch.no_grad():
            self._check_close(local(x), diff(x), "FeedForward(gelu-approximate)")


class TestUnsupportedBranches(unittest.TestCase):
    """Verify that out-of-subset branches raise NotImplementedError explicitly."""

    def test_ada_layer_norm_zero_fp32_layer_norm_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            LocalAdaLayerNormZero(3072, norm_type="fp32_layer_norm")
        self.assertIn("fp32_layer_norm", str(ctx.exception))

    def test_ada_layer_norm_continuous_rms_norm_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            LocalAdaLayerNormContinuous(3072, 3072, norm_type="rms_norm")
        self.assertIn("rms_norm", str(ctx.exception))

    def test_feedforward_geglu_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            LocalFeedForward(3072, activation_fn="geglu")
        self.assertIn("geglu", str(ctx.exception))

    def test_feedforward_default_raises(self):
        """FeedForward() with no activation_fn arg must raise (default is 'geglu')."""
        with self.assertRaises(NotImplementedError):
            LocalFeedForward(3072)


if __name__ == "__main__":
    unittest.main()
