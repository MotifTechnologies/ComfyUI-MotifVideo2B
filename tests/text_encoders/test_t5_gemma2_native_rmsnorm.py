"""
Tests for T5Gemma2RMSNorm in text_encoders/t5_gemma2_native.py.

Scope: T5Gemma2RMSNorm class only (checklist item 2).
Other classes (Attention/MLP/Layer/Encoder) are out of scope for this file.

Environment: must be run with ComfyUI venv python (has comfy_aimdo dep):
  $COMFYUI_VENV_PYTHON -m pytest tests/text_encoders/test_t5_gemma2_native_rmsnorm.py -v
where COMFYUI_VENV_PYTHON points to ComfyUI's venv python (e.g.
`<comfy_root>/.venv/bin/python`). The bootstrap below derives `<comfy_root>`
from this file's location (assumes layout `<comfy_root>/custom_nodes/<repo>/tests/...`).
"""

import math
import os
import sys
from pathlib import Path

# --- ComfyUI environment bootstrap (must precede any comfy import) ---
# Layout: <comfy_root>/custom_nodes/<repo>/tests/text_encoders/<this_file>
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_COMFYUI_ROOT_GUESS = _THIS_FILE.parents[4]
_COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", _COMFYUI_ROOT_GUESS)).resolve()

# enable args parsing FIRST and replace sys.argv BEFORE any comfy submodule
# (especially comfy.cli_args) gets imported transitively.
sys.argv = ["x", "--cpu"]
for p in (str(_COMFYUI_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import pytest
import torch

from text_encoders.t5_gemma2_native import T5Gemma2RMSNorm  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reference_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure-python reference: RMS normalize along last dim, return float32."""
    x_f = x.float()
    rms = x_f.pow(2).mean(-1, keepdim=True)
    return x_f * torch.rsqrt(rms + eps)


# ---------------------------------------------------------------------------
# 1. initialisation tests
# ---------------------------------------------------------------------------

class TestT5Gemma2RMSNormInit:

    def test_init_shape(self):
        """weight.shape == (dim,), zero-init, eps == 1e-6."""
        dim = 2560
        norm = T5Gemma2RMSNorm(dim)

        assert norm.weight.shape == (dim,), (
            f"Expected weight shape ({dim},), got {norm.weight.shape}"
        )
        assert torch.all(norm.weight == 0.0), (
            "weight must be zero-initialised (Gemma 2 convention)"
        )
        assert norm.eps == pytest.approx(1e-6), (
            f"Default eps should be 1e-6, got {norm.eps}"
        )

    def test_init_custom_eps(self):
        """Custom eps value is stored correctly."""
        norm = T5Gemma2RMSNorm(8, eps=1e-5)
        assert norm.eps == pytest.approx(1e-5)

    def test_weight_is_nn_parameter(self):
        """weight must be an nn.Parameter so it participates in state_dict."""
        import torch.nn as nn
        norm = T5Gemma2RMSNorm(16)
        assert isinstance(norm.weight, nn.Parameter), (
            "weight should be nn.Parameter"
        )

    def test_weight_small_dim(self):
        """Degenerate dim=1: weight.shape == (1,), zero-init."""
        norm = T5Gemma2RMSNorm(1)
        assert norm.weight.shape == (1,)
        assert float(norm.weight[0].detach()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. forward — shape and dtype preservation
# ---------------------------------------------------------------------------

class TestT5Gemma2RMSNormForward:

    def test_forward_shape_preserved(self):
        """Output shape must equal input shape (B, S, D)."""
        B, S, D = 2, 17, 128
        norm = T5Gemma2RMSNorm(D)
        x = torch.randn(B, S, D)
        out = norm(x)
        assert out.shape == (B, S, D), (
            f"Shape mismatch: expected {(B, S, D)}, got {out.shape}"
        )

    def test_forward_dtype_preserved_bfloat16(self):
        """bfloat16 input -> bfloat16 output."""
        D = 64
        norm = T5Gemma2RMSNorm(D)
        x = torch.randn(1, 4, D).to(torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got {out.dtype}"
        )

    def test_forward_dtype_preserved_float32(self):
        """float32 input -> float32 output."""
        D = 64
        norm = T5Gemma2RMSNorm(D)
        x = torch.randn(1, 4, D)  # default float32
        out = norm(x)
        assert out.dtype == torch.float32, (
            f"Expected float32 output, got {out.dtype}"
        )


# ---------------------------------------------------------------------------
# 3. forward — numerical correctness
# ---------------------------------------------------------------------------

class TestT5Gemma2RMSNormNumerics:

    def test_forward_zero_weight_equals_norm(self):
        """
        At zero-init (default), (1 + weight) == 1, so forward = _norm(x).
        Verify: output.pow(2).mean(-1) ~= 1.0 for any non-trivial x.
        """
        D = 128
        norm = T5Gemma2RMSNorm(D)  # weight == zeros
        x = torch.randn(3, 7, D)
        out = norm(x)

        rms_sq = out.float().pow(2).mean(-1)  # shape (3, 7)
        assert torch.allclose(
            rms_sq, torch.ones_like(rms_sq), atol=1e-3
        ), f"RMS^2 should be ~1.0 for zero-init weight, got max={rms_sq.max():.6f}"

    def test_forward_nonzero_weight_scales_correctly(self):
        """
        weight = 0.5 everywhere => output = _norm(x) * (1 + 0.5) = _norm(x) * 1.5.
        Verify with RMS-normalised input so ground truth is easy to compute.
        """
        D = 64
        norm = T5Gemma2RMSNorm(D)
        # Set weight to 0.5
        with torch.no_grad():
            norm.weight.fill_(0.5)

        # Construct x whose RMS per-vector is exactly 1 (already RMS-normalised)
        x = torch.randn(2, 4, D)
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).sqrt()
        x_normed = (x_f / rms).to(x.dtype)  # unit-RMS input

        out = norm(x_normed)
        expected = _reference_norm(x_normed, eps=1e-6) * 1.5

        assert torch.allclose(out.float(), expected, atol=1e-4), (
            "weight=0.5 should scale output by factor 1.5 vs _norm"
        )

    def test_gemma2_specific_formula_weight_minus_one(self):
        """
        Gemma-2 formula: output = _norm(x) * (1.0 + weight).
        When weight = -1.0: output = _norm(x) * 0 = 0.

        Standard RMSNorm (output = _norm(x) * weight) would give -_norm(x).
        This test distinguishes the two formulas unambiguously.
        """
        D = 32
        norm = T5Gemma2RMSNorm(D)
        with torch.no_grad():
            norm.weight.fill_(-1.0)

        x = torch.randn(1, 3, D)
        out = norm(x)

        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
            "weight=-1 must give zero output under Gemma-2 (1+weight) formula, "
            f"but got max abs = {out.abs().max():.6f}"
        )

    def test_gemma2_not_standard_rmsnorm(self):
        """
        Confirm the implementation is NOT standard RMSNorm.
        Standard: output = _norm(x) * weight.
        Gemma-2:  output = _norm(x) * (1 + weight).
        With weight=-1: standard gives -_norm(x), Gemma-2 gives 0.
        We already check == 0 above; here we additionally verify it is not
        the negated _norm (which would be the standard-formula wrong output).
        """
        D = 32
        norm = T5Gemma2RMSNorm(D)
        with torch.no_grad():
            norm.weight.fill_(-1.0)

        x = torch.randn(1, 3, D)
        out = norm(x)
        standard_wrong = -_reference_norm(x, eps=1e-6)

        # If implementation were standard, out ~= standard_wrong (non-zero).
        # Gemma-2 gives 0, so they must NOT be close.
        assert not torch.allclose(out.float(), standard_wrong, atol=1e-4), (
            "Output matches standard RMSNorm(weight=-1) = -_norm(x), "
            "but Gemma-2 formula requires zero output — implementation is wrong."
        )

    def test_eps_numerical_stability(self):
        """Very small input must not produce NaN or Inf."""
        D = 8
        norm = T5Gemma2RMSNorm(D, eps=1e-6)
        x = torch.zeros(1, 1, D) + 1e-30  # near-zero but non-zero

        out = norm(x)

        assert not torch.isnan(out).any(), "NaN detected for near-zero input"
        assert not torch.isinf(out).any(), "Inf detected for near-zero input"

    def test_dim_1_degenerate(self):
        """dim=1: mean over last dim is the single element itself. Must not crash."""
        norm = T5Gemma2RMSNorm(1)
        x = torch.tensor([[[3.0]]])  # shape (1, 1, 1)
        out = norm(x)
        assert out.shape == (1, 1, 1)
        # _norm: x / sqrt(x^2 + eps) ~= 1.0 for large x; exact: 3/sqrt(9+1e-6)~=1
        expected = 3.0 / math.sqrt(9.0 + 1e-6)
        assert abs(float(out[0, 0, 0].detach()) - expected) < 1e-4

    def test_with_negative_input(self):
        """Negative inputs: pow(2) removes sign, output must be finite."""
        D = 16
        norm = T5Gemma2RMSNorm(D)
        x = -torch.abs(torch.randn(2, 5, D)) - 1.0  # all negative

        out = norm(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 4. extra_repr
# ---------------------------------------------------------------------------

class TestT5Gemma2RMSNormExtraRepr:

    def test_extra_repr_format(self):
        """extra_repr() should return '(dim,), eps=<value>' string."""
        norm = T5Gemma2RMSNorm(2560)
        r = norm.extra_repr()
        assert "(2560,)" in r, f"extra_repr missing dim tuple: {r!r}"
        assert "eps=" in r, f"extra_repr missing 'eps=': {r!r}"
        assert "1e-06" in r or "1e-6" in r, (
            f"extra_repr eps value not matching 1e-6: {r!r}"
        )
