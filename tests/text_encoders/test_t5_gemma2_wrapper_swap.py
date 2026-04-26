"""
Tests for checklist item 6: wrapper swap in MotifVideoT5Gemma2Model.__init__

Scope: smoke tests verifying that the HF T5Gemma2Encoder import was replaced
with the native module. Deep behavioural regression (load_sd two-format compat,
encode_token_weights batch processing, numerical parity) is tested in items 8-9.

Blind-test principle: item-6 diff was NOT read before writing these tests.
Tests derived from the spec/checklist only.

Run:
  /lustrefs/team-multimodal/minsu/ComfyUI/.venv/bin/python \\
      -m pytest tests/text_encoders/test_t5_gemma2_wrapper_swap.py -v
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ComfyUI environment bootstrap (must precede any comfy import)
# ---------------------------------------------------------------------------
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

import pytest          # noqa: E402
import torch           # noqa: E402
import comfy.ops       # noqa: E402

from text_encoders.t5_gemma2 import (  # noqa: E402
    MotifVideoT5Gemma2Model,
    MotifVideoTokenizer,
    MotifVideoSD1Tokenizer,
    MotifVideoSD1ClipModel,
    te,
)
from text_encoders import t5_gemma2_native  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture: default-constructed wrapper model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_default():
    """MotifVideoT5Gemma2Model with all defaults (device='cpu', dtype=None)."""
    return MotifVideoT5Gemma2Model(device="cpu", dtype=None, model_options={})


# ===========================================================================
# 1. test_encoder_is_native
# ===========================================================================

class TestEncoderIsNative:
    """m.encoder must be the native T5Gemma2Encoder, not HF's."""

    def test_encoder_is_native(self, model_default):
        """m.encoder must be an instance of text_encoders.t5_gemma2_native.T5Gemma2Encoder."""
        assert isinstance(model_default.encoder, t5_gemma2_native.T5Gemma2Encoder), (
            f"Expected t5_gemma2_native.T5Gemma2Encoder, got {type(model_default.encoder)}"
        )

    def test_encoder_is_not_hf_t5gemma2encoder(self, model_default):
        """m.encoder must NOT be the HF transformers.models.t5gemma2 T5Gemma2Encoder."""
        try:
            from transformers.models.t5gemma2.modeling_t5gemma2 import (
                T5Gemma2Encoder as HF_T5Gemma2Encoder,
            )
            assert not isinstance(model_default.encoder, HF_T5Gemma2Encoder), (
                "m.encoder must not be an instance of HF T5Gemma2Encoder after swap"
            )
        except ImportError:
            pytest.skip("HF T5Gemma2Encoder not importable; skipping HF type check")


# ===========================================================================
# 2. test_encoder_has_text_model
# ===========================================================================

class TestEncoderHasTextModel:
    """m.encoder.text_model must exist with the expected structure."""

    def test_encoder_has_text_model(self, model_default):
        """m.encoder.text_model must exist."""
        assert hasattr(model_default.encoder, "text_model"), (
            "m.encoder must have attribute 'text_model'"
        )

    def test_text_model_layers_count_34(self, model_default):
        """text_model.layers must contain exactly 34 layers."""
        layers = model_default.encoder.text_model.layers
        assert len(layers) == 34, (
            f"Expected 34 layers, got {len(layers)}"
        )

    def test_text_model_has_embed_tokens(self, model_default):
        """text_model.embed_tokens must exist."""
        assert hasattr(model_default.encoder.text_model, "embed_tokens"), (
            "text_model must have attribute 'embed_tokens'"
        )

    def test_text_model_has_norm(self, model_default):
        """text_model.norm must exist."""
        assert hasattr(model_default.encoder.text_model, "norm"), (
            "text_model must have attribute 'norm'"
        )

    def test_text_model_has_rotary_emb(self, model_default):
        """text_model.rotary_emb must exist."""
        assert hasattr(model_default.encoder.text_model, "rotary_emb"), (
            "text_model must have attribute 'rotary_emb'"
        )


# ===========================================================================
# 3. test_default_dtype_bfloat16
# ===========================================================================

class TestDefaultDtypeBfloat16:
    """dtype=None must resolve to torch.bfloat16."""

    def test_dtype_attribute_is_bfloat16(self, model_default):
        """m.dtype must be torch.bfloat16 when constructed with dtype=None."""
        assert model_default.dtype == torch.bfloat16, (
            f"Expected m.dtype=torch.bfloat16, got {model_default.dtype}"
        )

    def test_dtypes_set_contains_bfloat16(self, model_default):
        """m.dtypes must be {torch.bfloat16} when constructed with dtype=None."""
        assert model_default.dtypes == {torch.bfloat16}, (
            f"Expected m.dtypes={{torch.bfloat16}}, got {model_default.dtypes}"
        )

    def test_explicit_float32_respected(self):
        """dtype=torch.float32 must be stored as-is (not overridden to bfloat16)."""
        m = MotifVideoT5Gemma2Model(device="cpu", dtype=torch.float32, model_options={})
        assert m.dtype == torch.float32, (
            f"dtype=torch.float32 not preserved; got {m.dtype}"
        )

    def test_explicit_bfloat16_respected(self):
        """dtype=torch.bfloat16 (explicit) must also work without error."""
        m = MotifVideoT5Gemma2Model(device="cpu", dtype=torch.bfloat16, model_options={})
        assert m.dtype == torch.bfloat16


# ===========================================================================
# 4. test_eval_mode_and_no_grad
# ===========================================================================

class TestEvalModeAndNoGrad:
    """After __init__, encoder must be in eval mode and all params frozen."""

    def test_encoder_not_training(self, model_default):
        """m.encoder.training must be False."""
        assert model_default.encoder.training is False, (
            "m.encoder must be in eval mode after __init__"
        )

    def test_all_parameters_no_grad(self, model_default):
        """Every parameter in the wrapper must have requires_grad=False."""
        failing = [
            name
            for name, p in model_default.named_parameters()
            if p.requires_grad is True
        ]
        assert len(failing) == 0, (
            "The following parameters still have requires_grad=True:\n"
            + "\n".join(failing[:10])
        )

    def test_freeze_method_idempotent(self):
        """Calling freeze() on a freshly constructed model must not raise."""
        m = MotifVideoT5Gemma2Model(device="cpu", dtype=None, model_options={})
        m.freeze()  # second freeze after __init__ freeze
        assert m.encoder.training is False
        failing = [name for name, p in m.named_parameters() if p.requires_grad is True]
        assert len(failing) == 0


# ===========================================================================
# 5. test_operations_propagated_to_encoder
# ===========================================================================

class TestOperationsPropagatedToEncoder:
    """model_options['operations'] must be forwarded to the native encoder."""

    def test_manual_cast_propagated_to_q_proj(self):
        """
        model_options={'operations': comfy.ops.manual_cast} ->
        m.encoder.text_model.layers[0].self_attn.q_proj must have comfy_cast_weights=True.
        """
        m = MotifVideoT5Gemma2Model(
            device="cpu",
            dtype=None,
            model_options={"operations": comfy.ops.manual_cast},
        )
        q_proj = m.encoder.text_model.layers[0].self_attn.q_proj
        val = getattr(q_proj, "comfy_cast_weights", None)
        assert val is True, (
            f"q_proj.comfy_cast_weights expected True with manual_cast, got {val}"
        )

    def test_default_operations_no_cast(self):
        """
        model_options={} (no 'operations' key) -> layers[0].self_attn.q_proj
        must NOT have comfy_cast_weights=True (disable_weight_init default).
        """
        m = MotifVideoT5Gemma2Model(device="cpu", dtype=None, model_options={})
        q_proj = m.encoder.text_model.layers[0].self_attn.q_proj
        val = getattr(q_proj, "comfy_cast_weights", None)
        # disable_weight_init gives False or None
        assert val is not True, (
            f"Without manual_cast, comfy_cast_weights must not be True, got {val}"
        )

    def test_manual_cast_propagated_to_embed_tokens(self):
        """embed_tokens must also carry comfy_cast_weights=True under manual_cast."""
        m = MotifVideoT5Gemma2Model(
            device="cpu",
            dtype=None,
            model_options={"operations": comfy.ops.manual_cast},
        )
        embed_tokens = m.encoder.text_model.embed_tokens
        val = getattr(embed_tokens, "comfy_cast_weights", None)
        assert val is True, (
            f"embed_tokens.comfy_cast_weights expected True with manual_cast, got {val}"
        )


# ===========================================================================
# 6. test_no_hf_t5gemma2encoder_import
# ===========================================================================

class TestNoHFT5Gemma2EncoderImport:
    """The HF T5Gemma2Encoder must not be imported at module level in t5_gemma2.py."""

    _SOURCE_FILE = _REPO_ROOT / "text_encoders" / "t5_gemma2.py"

    def _get_source(self):
        return self._SOURCE_FILE.read_text(encoding="utf-8")

    def test_no_hf_import_via_grep(self):
        """
        Static text check: the line
          from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Encoder
        must NOT appear anywhere in t5_gemma2.py.
        """
        source = self._get_source()
        hf_import_pattern = (
            "from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Encoder"
        )
        assert hf_import_pattern not in source, (
            "HF T5Gemma2Encoder import found in t5_gemma2.py — wrapper swap not applied"
        )

    def test_no_hf_import_via_ast(self):
        """
        AST-level check: no ImportFrom node imports T5Gemma2Encoder from
        transformers.models.t5gemma2.modeling_t5gemma2 at module scope.
        """
        source = self._get_source()
        tree = ast.parse(source)

        hf_module = "transformers.models.t5gemma2.modeling_t5gemma2"
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == hf_module:
                imported_names = [alias.name for alias in node.names]
                assert "T5Gemma2Encoder" not in imported_names, (
                    f"AST found HF T5Gemma2Encoder import from {hf_module}: {imported_names}"
                )


# ===========================================================================
# 7. test_other_classes_still_exposed
# ===========================================================================

class TestOtherClassesStillExposed:
    """Non-changed symbols must still be importable from text_encoders.t5_gemma2."""

    def test_motif_video_tokenizer_exposed(self):
        """MotifVideoTokenizer must be importable."""
        assert MotifVideoTokenizer is not None
        assert isinstance(MotifVideoTokenizer, type)

    def test_motif_video_sd1_tokenizer_exposed(self):
        """MotifVideoSD1Tokenizer must be importable."""
        assert MotifVideoSD1Tokenizer is not None
        assert isinstance(MotifVideoSD1Tokenizer, type)

    def test_motif_video_sd1_clip_model_exposed(self):
        """MotifVideoSD1ClipModel must be importable."""
        assert MotifVideoSD1ClipModel is not None
        assert isinstance(MotifVideoSD1ClipModel, type)

    def test_te_factory_exposed(self):
        """te() factory function must be importable and callable."""
        assert te is not None
        assert callable(te)

    def test_te_factory_returns_class(self):
        """te() with no args must return a class (type), not raise."""
        result = te()
        assert isinstance(result, type), (
            f"te() must return a class, got {type(result)}"
        )


# ===========================================================================
# 8. test_existing_methods_intact
# ===========================================================================

class TestExistingMethodsIntact:
    """All wrapper methods must exist and preserve their expected signatures."""

    def test_freeze_method_exists(self, model_default):
        """freeze must be an attribute of the model."""
        assert hasattr(model_default, "freeze")
        assert callable(model_default.freeze)

    def test_encode_method_exists(self, model_default):
        """encode must exist with parameter 'tokens'."""
        assert hasattr(model_default, "encode")
        sig = inspect.signature(model_default.encode)
        params = list(sig.parameters.keys())
        assert "tokens" in params, (
            f"encode() missing 'tokens' parameter; got {params}"
        )

    def test_encode_token_weights_method_exists(self, model_default):
        """encode_token_weights must exist with parameter 'token_weight_pairs'."""
        assert hasattr(model_default, "encode_token_weights")
        sig = inspect.signature(model_default.encode_token_weights)
        params = list(sig.parameters.keys())
        assert "token_weight_pairs" in params, (
            f"encode_token_weights() missing 'token_weight_pairs' parameter; got {params}"
        )

    def test_load_sd_method_exists(self, model_default):
        """load_sd must exist with parameter 'sd'."""
        assert hasattr(model_default, "load_sd")
        sig = inspect.signature(model_default.load_sd)
        params = list(sig.parameters.keys())
        assert "sd" in params, (
            f"load_sd() missing 'sd' parameter; got {params}"
        )

    def test_set_clip_options_method_exists(self, model_default):
        """set_clip_options must exist with parameter 'options'."""
        assert hasattr(model_default, "set_clip_options")
        sig = inspect.signature(model_default.set_clip_options)
        params = list(sig.parameters.keys())
        assert "options" in params, (
            f"set_clip_options() missing 'options' parameter; got {params}"
        )

    def test_reset_clip_options_method_exists(self, model_default):
        """reset_clip_options must exist and be callable with no args."""
        assert hasattr(model_default, "reset_clip_options")
        sig = inspect.signature(model_default.reset_clip_options)
        # Only 'self' — no extra positional params
        params = [
            k for k, v in sig.parameters.items()
            if v.default is inspect.Parameter.empty
        ]
        assert len(params) == 0, (
            f"reset_clip_options() must take no arguments; required params: {params}"
        )

    def test_get_input_embeddings_method_exists(self, model_default):
        """get_input_embeddings must exist and return encoder.text_model.embed_tokens."""
        assert hasattr(model_default, "get_input_embeddings")
        result = model_default.get_input_embeddings()
        assert result is model_default.encoder.text_model.embed_tokens, (
            "get_input_embeddings() must return encoder.text_model.embed_tokens"
        )

    def test_set_clip_options_smoke(self, model_default):
        """set_clip_options({}) must not raise."""
        model_default.set_clip_options({})

    def test_reset_clip_options_smoke(self, model_default):
        """reset_clip_options() must not raise."""
        model_default.reset_clip_options()
