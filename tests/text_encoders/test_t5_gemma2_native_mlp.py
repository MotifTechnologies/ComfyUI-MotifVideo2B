"""
Tests for T5Gemma2MLP in text_encoders/t5_gemma2_native.py.

Scope: T5Gemma2MLP class only (checklist item 4).
Other classes (RMSNorm, SelfAttention) are covered in separate files.

Blind-test principle: tests are written from the spec only — implementation
code was NOT read before writing these tests.

Spec reference:
  - gate_proj, up_proj: Linear(hidden_size=2560, intermediate_size=10240, bias=False)
  - down_proj: Linear(intermediate_size=10240, hidden_size=2560, bias=False)
  - act_fn: F.gelu(x, approximate="tanh") for hidden_activation="gelu_pytorch_tanh"
  - dropout: nn.Dropout(config.dropout_rate=0.0)
  - forward: down_proj(dropout(act_fn(gate_proj(x)) * up_proj(x)))
  - unsupported hidden_activation -> NotImplementedError

Environment: must be run with ComfyUI venv python:
  /lustrefs/team-multimodal/minsu/ComfyUI/.venv/bin/python \\
      -m pytest tests/text_encoders/test_t5_gemma2_native_mlp.py -v

Layout assumption: <comfy_root>/custom_nodes/<repo>/tests/text_encoders/<this>
"""

import copy
import os
import sys
from pathlib import Path

# --- ComfyUI environment bootstrap (must precede any comfy import) ----------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_COMFYUI_ROOT_GUESS = _THIS_FILE.parents[4]
_COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", _COMFYUI_ROOT_GUESS)).resolve()

sys.argv = ["x", "--cpu"]
for _p in (str(_COMFYUI_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.ops  # noqa: E402

from text_encoders.t5_gemma2_native import T5Gemma2MLP  # noqa: E402


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _make_config():
    """Return T5Gemma2TextConfig built from the project's bundled defaults."""
    from text_encoders.t5_gemma2_config import T5_GEMMA2_CONFIG
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2EncoderConfig,
    )
    return T5Gemma2EncoderConfig(**T5_GEMMA2_CONFIG).text_config


def _make_config_with(hidden_activation: str):
    """Return a config copy with a replaced hidden_activation field."""
    cfg = _make_config()
    # Use a mutable copy; config objects are typically simple dataclasses/Namespace
    cfg_copy = copy.copy(cfg)
    cfg_copy.hidden_activation = hidden_activation
    return cfg_copy


# Expected model dimensions (from spec)
_HIDDEN_SIZE = 2560
_INTERMEDIATE_SIZE = 10240


# ---------------------------------------------------------------------------
# 1. Init / attribute existence
# ---------------------------------------------------------------------------

class TestT5Gemma2MLPInit:

    def test_init_attribute_existence(self):
        """gate_proj, up_proj, down_proj, act_fn, dropout must all exist."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)

        assert hasattr(mlp, "gate_proj"), "missing attribute: gate_proj"
        assert hasattr(mlp, "up_proj"), "missing attribute: up_proj"
        assert hasattr(mlp, "down_proj"), "missing attribute: down_proj"
        assert hasattr(mlp, "act_fn"), "missing attribute: act_fn"
        assert hasattr(mlp, "dropout"), "missing attribute: dropout"

    def test_init_weight_shapes(self):
        """gate_proj and up_proj weight (10240, 2560); down_proj weight (2560, 10240)."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)

        assert mlp.gate_proj.weight.shape == (_INTERMEDIATE_SIZE, _HIDDEN_SIZE), (
            f"gate_proj weight shape mismatch: {mlp.gate_proj.weight.shape}"
        )
        assert mlp.up_proj.weight.shape == (_INTERMEDIATE_SIZE, _HIDDEN_SIZE), (
            f"up_proj weight shape mismatch: {mlp.up_proj.weight.shape}"
        )
        assert mlp.down_proj.weight.shape == (_HIDDEN_SIZE, _INTERMEDIATE_SIZE), (
            f"down_proj weight shape mismatch: {mlp.down_proj.weight.shape}"
        )

    def test_init_no_bias(self):
        """All three projection layers must have bias=None (bias=False at init)."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)

        assert mlp.gate_proj.bias is None, "gate_proj.bias must be None"
        assert mlp.up_proj.bias is None, "up_proj.bias must be None"
        assert mlp.down_proj.bias is None, "down_proj.bias must be None"

    def test_init_dropout_rate(self):
        """dropout.p must match config.dropout_rate (expected 0.0)."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)

        assert mlp.dropout.p == pytest.approx(cfg.dropout_rate), (
            f"dropout.p={mlp.dropout.p} != config.dropout_rate={cfg.dropout_rate}"
        )
        # Explicitly confirm it is 0.0 for the default config
        assert mlp.dropout.p == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Activation function
# ---------------------------------------------------------------------------

class TestT5Gemma2MLPActivation:

    def test_act_fn_is_gelu_tanh(self):
        """act_fn(x) must match F.gelu(x, approximate='tanh') element-wise (abs diff < 1e-7)."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)

        x = torch.linspace(-3.0, 3.0, steps=64)
        result = mlp.act_fn(x)
        expected = F.gelu(x, approximate="tanh")

        assert torch.allclose(result, expected, atol=1e-7), (
            f"act_fn deviates from gelu_pytorch_tanh; max diff={( result - expected).abs().max():.2e}"
        )

    def test_init_unsupported_activation_raises(self):
        """hidden_activation='silu' must raise NotImplementedError at instantiation."""
        cfg = _make_config_with("silu")
        with pytest.raises(NotImplementedError):
            T5Gemma2MLP(cfg)

    def test_init_unsupported_activation_relu_raises(self):
        """hidden_activation='relu' must raise NotImplementedError at instantiation."""
        cfg = _make_config_with("relu")
        with pytest.raises(NotImplementedError):
            T5Gemma2MLP(cfg)


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------

class TestT5Gemma2MLPForward:

    def test_forward_shape_preserved(self):
        """Input (B, S, hidden_size) -> output same shape."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)
        mlp.eval()

        B, S = 2, 7
        x = torch.randn(B, S, _HIDDEN_SIZE)
        with torch.no_grad():
            out = mlp(x)

        assert out.shape == (B, S, _HIDDEN_SIZE), (
            f"Output shape mismatch: expected {(B, S, _HIDDEN_SIZE)}, got {out.shape}"
        )

    def test_forward_dtype_bf16(self):
        """dtype=bfloat16 build + bf16 input -> bf16 output."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg, dtype=torch.bfloat16)
        mlp.eval()

        x = torch.randn(1, 4, _HIDDEN_SIZE, dtype=torch.bfloat16)
        with torch.no_grad():
            out = mlp(x)

        assert out.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got {out.dtype}"
        )

    def test_forward_swiglu_formula(self):
        """
        Output must equal down_proj(act_fn(gate_proj(x)) * up_proj(x)).
        dropout.p == 0.0 so dropout is identity — no randomness in comparison.
        """
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)
        mlp.eval()

        x = torch.randn(1, 3, _HIDDEN_SIZE)
        with torch.no_grad():
            out = mlp(x)
            # Re-compute manually using the same weights
            gate = mlp.act_fn(mlp.gate_proj(x))
            up = mlp.up_proj(x)
            expected = mlp.down_proj(gate * up)

        assert torch.allclose(out, expected, atol=1e-6), (
            f"forward output does not match down_proj(act_fn(gate_proj(x)) * up_proj(x)); "
            f"max diff={( out - expected).abs().max():.2e}"
        )

    def test_forward_zero_input(self):
        """
        Zero input -> zero output.
        act_fn(0) * 0 = 0, down_proj(zeros) = zeros (no bias).
        """
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg)
        mlp.eval()

        x = torch.zeros(1, 5, _HIDDEN_SIZE)
        with torch.no_grad():
            out = mlp(x)

        assert torch.allclose(out, torch.zeros_like(out), atol=1e-9), (
            f"Zero input must yield zero output; max abs={out.abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# 4. operations injection
# ---------------------------------------------------------------------------

class TestT5Gemma2MLPOperations:

    def test_operations_manual_cast_comfy_cast_weights(self):
        """operations=manual_cast -> all three projections have comfy_cast_weights=True."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg, operations=comfy.ops.manual_cast)

        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, name)
            assert getattr(proj, "comfy_cast_weights", False) is True, (
                f"{name}.comfy_cast_weights expected True with manual_cast operations"
            )

    def test_operations_none_default_disable_weight_init(self):
        """
        operations=None -> falls back to disable_weight_init.
        Projections must NOT have comfy_cast_weights=True
        (either attribute is absent or False).
        """
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg, operations=None)

        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, name)
            cast = getattr(proj, "comfy_cast_weights", False)
            assert cast is not True, (
                f"{name}.comfy_cast_weights should be False/absent with default operations, got {cast}"
            )


# ---------------------------------------------------------------------------
# 5. dtype propagation
# ---------------------------------------------------------------------------

class TestT5Gemma2MLPDtype:

    def test_dtype_propagated_to_weights(self):
        """dtype=bfloat16 -> all projection weights are bfloat16."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg, dtype=torch.bfloat16)

        assert mlp.gate_proj.weight.dtype == torch.bfloat16, (
            f"gate_proj.weight.dtype={mlp.gate_proj.weight.dtype}, expected bfloat16"
        )
        assert mlp.up_proj.weight.dtype == torch.bfloat16, (
            f"up_proj.weight.dtype={mlp.up_proj.weight.dtype}, expected bfloat16"
        )
        assert mlp.down_proj.weight.dtype == torch.bfloat16, (
            f"down_proj.weight.dtype={mlp.down_proj.weight.dtype}, expected bfloat16"
        )

    def test_device_cpu_propagated(self):
        """device='cpu' explicit -> all projection weights on CPU."""
        cfg = _make_config()
        mlp = T5Gemma2MLP(cfg, device=torch.device("cpu"))

        for name in ("gate_proj", "up_proj", "down_proj"):
            w = getattr(mlp, name).weight
            assert w.device.type == "cpu", (
                f"{name}.weight expected on cpu, got {w.device}"
            )
