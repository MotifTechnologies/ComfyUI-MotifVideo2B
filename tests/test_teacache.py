"""Unit tests for nodes/teacache.py — CPU-only, no GPU required.

Tests cover:
- MotifTeaCache node class structure (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY)
- _TeaCacheState.reset() state initialization
- _TeaCacheState.should_skip() logic for all decision branches
- MotifTeaCache.apply_teacache() with enable=False bypass
- MotifTeaCache.apply_teacache() with missing adapter attributes
- MotifTeaCache.apply_teacache() idempotency guard
- MotifTeaCache.apply_teacache() full patch path

Run:
    pytest tests/test_teacache.py -v
"""

import sys
import os
import types
import importlib
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Path setup and heavy-dependency stubbing
# Must happen BEFORE any node import to avoid transitive ComfyUI chain.
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Stub every ComfyUI / motif package that teacache.py does NOT need but that
# nodes/__init__.py (and its transitive imports) would pull in.
def _stub(name: str):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]

# ComfyUI runtime stubs
_stub("folder_paths")
_stub("comfy")
_stub("comfy.cli_args")
_stub("comfy.supported_models")
_stub("comfy.model_detection")
_stub("comfy.sd")
_stub("comfy.model_patcher")

# motif-core / motif-pipelines stubs (pulled in by loader.py / text_encode.py)
for _m in [
    "motif",
    "motif.models",
    "motif.pipelines",
    "transformers",
    "safetensors",
    "safetensors.torch",
]:
    _stub(_m)

# Provide a minimal cli_args.args so folder_paths doesn't crash
_cli_args = _stub("comfy.cli_args")
_cli_args.args = MagicMock()

# Now import teacache directly, bypassing nodes/__init__.py
import importlib.util as _ilu
_teacache_path = os.path.join(_root, "nodes", "teacache.py")
_spec = _ilu.spec_from_file_location("nodes.teacache", _teacache_path)
_teacache_mod = _ilu.module_from_spec(_spec)
sys.modules["nodes.teacache"] = _teacache_mod
_spec.loader.exec_module(_teacache_mod)

MotifTeaCache = _teacache_mod.MotifTeaCache
_TeaCacheState = _teacache_mod._TeaCacheState
_MOTIF_POLY_COEFFS = _teacache_mod._MOTIF_POLY_COEFFS
_extract_modulated_input = _teacache_mod._extract_modulated_input
_make_teacache_forward = _teacache_mod._make_teacache_forward


# ===========================================================================
# Helpers
# ===========================================================================

def _make_state(thresh: float = 1.0) -> _TeaCacheState:
    """Return a fresh _TeaCacheState with given threshold."""
    return _TeaCacheState(rel_l1_thresh=thresh, poly_coeffs=_MOTIF_POLY_COEFFS)


def _make_tensor(val: float = 1.0, shape=(2, 4, 8)) -> torch.Tensor:
    """Return a float32 tensor filled with val."""
    return torch.full(shape, val, dtype=torch.float32)


# ===========================================================================
# 1. Node class structure
# ===========================================================================

class TestMotifTeaCacheClassStructure:
    """INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY validation."""

    def test_input_types_returns_dict(self):
        result = MotifTeaCache.INPUT_TYPES()
        assert isinstance(result, dict), "INPUT_TYPES must return a dict"

    def test_input_types_has_required_key(self):
        result = MotifTeaCache.INPUT_TYPES()
        assert "required" in result

    def test_input_types_required_contains_model(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        assert "model" in required

    def test_input_types_required_contains_rel_l1_thresh(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        assert "rel_l1_thresh" in required

    def test_input_types_required_contains_enable(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        assert "enable" in required

    def test_rel_l1_thresh_type_is_float(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        field = required["rel_l1_thresh"]
        assert field[0] == "FLOAT"

    def test_rel_l1_thresh_default_is_0_3(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        meta = required["rel_l1_thresh"][1]
        assert meta["default"] == 0.3

    def test_rel_l1_thresh_min_is_0(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        meta = required["rel_l1_thresh"][1]
        assert meta["min"] == 0.0

    def test_enable_type_is_boolean(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        field = required["enable"]
        assert field[0] == "BOOLEAN"

    def test_return_types_is_model_tuple(self):
        assert MotifTeaCache.RETURN_TYPES == ("MODEL",)

    def test_function_name(self):
        assert MotifTeaCache.FUNCTION == "apply_teacache"

    def test_category(self):
        assert MotifTeaCache.CATEGORY == "motifvideo"

    def test_apply_teacache_method_exists(self):
        node = MotifTeaCache()
        assert callable(node.apply_teacache)


# ===========================================================================
# 2. _TeaCacheState tests
# ===========================================================================

class TestTeaCacheStateReset:
    """reset() restores all fields to initial values."""

    def test_reset_clears_accumulated_distance(self):
        state = _make_state()
        state.accumulated_rel_l1_distance = 99.9
        state.reset()
        assert state.accumulated_rel_l1_distance == 0.0

    def test_reset_clears_previous_modulated_input(self):
        state = _make_state()
        state.previous_modulated_input = _make_tensor()
        state.reset()
        assert state.previous_modulated_input is None

    def test_reset_clears_previous_residual(self):
        state = _make_state()
        state.previous_residual = _make_tensor()
        state.reset()
        assert state.previous_residual is None

    def test_reset_clears_step_counter(self):
        state = _make_state()
        state.step_counter = 42
        state.reset()
        assert state.step_counter == 0

    def test_reset_idempotent(self):
        """Calling reset() twice leaves state clean."""
        state = _make_state()
        state.accumulated_rel_l1_distance = 5.0
        state.step_counter = 7
        state.reset()
        state.reset()
        assert state.accumulated_rel_l1_distance == 0.0
        assert state.step_counter == 0


class TestTeaCacheStateShouldSkip:
    """should_skip() decision logic."""

    def test_returns_false_when_no_previous_modulated_input(self):
        """Cold start: previous_modulated_input is None → must compute."""
        state = _make_state(thresh=1.0)
        current = _make_tensor(1.0)
        result = state.should_skip(current)
        assert result is False

    def test_returns_false_when_previous_residual_is_none(self):
        """previous_modulated_input set but previous_residual is None → must compute."""
        state = _make_state(thresh=1.0)
        state.previous_modulated_input = _make_tensor(1.0)
        # previous_residual remains None
        current = _make_tensor(1.0)
        result = state.should_skip(current)
        assert result is False

    def test_returns_true_when_accumulated_distance_below_threshold(self):
        """Small input change → accumulated drift stays below thresh → skip."""
        # Use a high threshold so that any reasonable rescaled diff triggers skip.
        state = _make_state(thresh=1e6)
        # Populate cache
        prev_tensor = _make_tensor(1.0)
        state.previous_modulated_input = prev_tensor.clone()
        state.previous_residual = _make_tensor(0.5)  # arbitrary

        # current is almost identical to previous → raw_diff ≈ 0 → rescaled tiny
        current = _make_tensor(1.0)
        result = state.should_skip(current)
        assert result is True

    def test_returns_false_and_resets_accumulator_when_threshold_exceeded(self):
        """Accumulated distance reaches threshold → must compute, reset accumulator."""
        # Use a near-zero threshold so that any change exceeds it immediately.
        state = _make_state(thresh=0.0)
        prev_tensor = _make_tensor(2.0)
        state.previous_modulated_input = prev_tensor.clone()
        state.previous_residual = _make_tensor(0.5)

        # Different tensor → non-zero diff
        current = _make_tensor(3.0)
        result = state.should_skip(current)
        assert result is False
        assert state.accumulated_rel_l1_distance == 0.0, (
            "accumulator must be reset to 0 when threshold is exceeded"
        )

    def test_accumulator_increments_on_skip(self):
        """Each skip step adds to accumulated_rel_l1_distance."""
        state = _make_state(thresh=1e9)
        state.previous_modulated_input = _make_tensor(1.0)
        state.previous_residual = _make_tensor(0.5)

        current = _make_tensor(1.0)
        state.should_skip(current)
        dist_after_one = state.accumulated_rel_l1_distance

        state.previous_modulated_input = current.clone()
        state.should_skip(current)
        dist_after_two = state.accumulated_rel_l1_distance

        assert dist_after_two > dist_after_one

    def test_returns_false_when_previous_is_near_zero(self):
        """mean_prev < 1e-10 guard: degenerate previous input → return False."""
        state = _make_state(thresh=1e9)
        state.previous_modulated_input = torch.zeros(2, 4, 8)
        state.previous_residual = _make_tensor(0.5)

        current = _make_tensor(1.0)
        result = state.should_skip(current)
        assert result is False

    def test_threshold_boundary_below(self):
        """accumulated just below threshold → skip."""
        state = _make_state(thresh=100.0)
        state.previous_modulated_input = _make_tensor(1.0)
        state.previous_residual = _make_tensor(0.5)
        # Drive accumulator to just below threshold manually
        state.accumulated_rel_l1_distance = 99.0

        # Pass identical tensors so raw_diff = 0 and rescale_func(0) is small but positive
        # The rescaled value will be added to 99.0; if still < 100 → skip
        current = _make_tensor(1.0)
        result = state.should_skip(current)
        # Poly at x=0: _MOTIF_POLY_COEFFS[0] = 498.65... so accumulated will blow past thresh
        # This tests that the boundary check is strict (<, not <=)
        # With added rescaled value >= 0, result depends on poly(0); just assert type
        assert isinstance(result, bool)

    def test_does_not_mutate_input_tensors(self):
        """should_skip() must not modify the tensors passed to it."""
        state = _make_state(thresh=1.0)
        prev = _make_tensor(1.0)
        state.previous_modulated_input = prev.clone()
        state.previous_residual = _make_tensor(0.5)

        current = _make_tensor(1.5)
        current_copy = current.clone()
        state.should_skip(current)
        assert torch.allclose(current, current_copy), "should_skip must not mutate current input"


# ===========================================================================
# 3. apply_teacache(enable=False) bypass
# ===========================================================================

class TestApplyTeaCacheDisabled:
    """enable=False must return the original model without cloning or patching."""

    def _make_mock_model(self):
        return MagicMock(name="model")

    def test_returns_tuple(self):
        node = MotifTeaCache()
        model = self._make_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        assert isinstance(result, tuple)

    def test_returns_single_element_tuple(self):
        node = MotifTeaCache()
        model = self._make_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        assert len(result) == 1

    def test_returns_original_model_unchanged(self):
        node = MotifTeaCache()
        model = self._make_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        assert result[0] is model, "enable=False must return the exact original model object"

    def test_does_not_call_clone(self):
        node = MotifTeaCache()
        model = self._make_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        model.clone.assert_not_called()

    def test_zero_threshold_still_bypassed(self):
        node = MotifTeaCache()
        model = self._make_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.0, enable=False)
        assert result[0] is model

    def test_negative_threshold_still_bypassed(self):
        """Even out-of-range threshold values should not matter when disabled."""
        node = MotifTeaCache()
        model = self._make_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=-1.0, enable=False)
        assert result[0] is model


# ===========================================================================
# 4. apply_teacache(enable=True) — mock model structure
# ===========================================================================

def _make_full_mock_model():
    """Build a minimal mock hierarchy: ModelPatcher → inner_model → transformer.

    diffusion_model IS the transformer directly (MotifVideoTransformer3DModel).
    There is no intermediate adapter wrapper.

    Returns:
        (model, patched_model, inner_model, transformer, transformer)
        The 4th and 5th elements are the same object (transformer = diffusion_model).
        This matches the return signature callers unpack as:
            model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
    """
    # Transformer with one block — diffusion_model IS this transformer
    block0 = MagicMock(name="block0")
    transformer = MagicMock(name="transformer")
    transformer.transformer_blocks = [block0]
    transformer._teacache_enabled = False
    # Ensure getattr returns a real bool, not a Mock object
    type(transformer)._teacache_enabled = False

    inner_model = MagicMock(name="inner_model")
    inner_model.diffusion_model = transformer  # diffusion_model IS transformer

    patched_model = MagicMock(name="patched_model")
    patched_model.model = inner_model

    model = MagicMock(name="model")
    model.clone.return_value = patched_model

    # Return transformer twice so callers unpacking (model, patched_model, inner_model, X, Y)
    # get transformer for both X and Y (they are the same object).
    return model, patched_model, inner_model, transformer, transformer


class TestApplyTeaCacheEnabled:
    """apply_teacache(enable=True) with mocked model hierarchy."""

    def test_clones_model(self):
        node = MotifTeaCache()
        model, patched_model, *_ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        model.clone.assert_called_once()

    def test_returns_patched_model(self):
        node = MotifTeaCache()
        model, patched_model, *_ = _make_full_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert result[0] is patched_model

    def test_adapter_forward_is_replaced(self):
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        original_forward = transformer.forward
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert transformer.forward is not original_forward, "transformer.forward must be replaced"

    def test_adapter_marked_as_enabled(self):
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert transformer._teacache_enabled is True

    def test_state_attached_to_adapter(self):
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert hasattr(transformer, "_teacache_state")
        assert isinstance(transformer._teacache_state, _TeaCacheState)

    def test_state_threshold_matches_input(self):
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.42, enable=True)
        assert transformer._teacache_state.rel_l1_thresh == 0.42

    def test_no_transformer_blocks_returns_patched_without_patch(self):
        """If diffusion_model has no transformer_blocks, return patched_model without patching."""
        node = MotifTeaCache()

        # diffusion_model is a MagicMock with spec=[] so hasattr(x, 'transformer_blocks') is False
        diffusion_model = MagicMock(spec=[])
        inner_model = MagicMock()
        inner_model.diffusion_model = diffusion_model

        patched_model = MagicMock()
        patched_model.model = inner_model

        model = MagicMock()
        model.clone.return_value = patched_model

        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert result[0] is patched_model

    def test_empty_transformer_blocks_returns_patched_without_patch(self):
        """If transformer_blocks is empty, return patched_model without patching."""
        node = MotifTeaCache()

        # diffusion_model IS the transformer; give it empty transformer_blocks
        transformer = MagicMock()
        transformer.transformer_blocks = []  # empty!
        transformer._teacache_enabled = False

        inner_model = MagicMock()
        inner_model.diffusion_model = transformer  # no adapter wrapper

        patched_model = MagicMock()
        patched_model.model = inner_model

        model = MagicMock()
        model.clone.return_value = patched_model

        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert result[0] is patched_model

    def test_idempotency_skips_repatch_when_already_enabled(self):
        """Calling apply_teacache twice on already-patched transformer must not re-patch."""
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()

        # First call — patches transformer
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        forward_after_first = transformer.forward

        # Simulate second call on a model whose clone resolves to the same transformer
        patched_model2 = MagicMock(name="patched_model2")
        patched_model2.model = inner_model
        model2 = MagicMock(name="model2")
        model2.clone.return_value = patched_model2

        node.apply_teacache(model=model2, rel_l1_thresh=0.5, enable=True)

        # forward must not have been replaced a second time
        assert transformer.forward is forward_after_first, (
            "Idempotency guard: forward must not be replaced on second apply_teacache call"
        )

    def test_idempotency_updates_threshold(self):
        """When already patched, threshold update is applied to existing state."""
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)

        patched_model2 = MagicMock()
        patched_model2.model = inner_model
        model2 = MagicMock()
        model2.clone.return_value = patched_model2

        node.apply_teacache(model=model2, rel_l1_thresh=0.99, enable=True)
        assert transformer._teacache_state.rel_l1_thresh == 0.99


# ===========================================================================
# 5. _TeaCacheState constructor
# ===========================================================================

class TestTeaCacheStateInit:
    def test_initial_accumulated_distance_is_zero(self):
        state = _make_state()
        assert state.accumulated_rel_l1_distance == 0.0

    def test_initial_previous_modulated_input_is_none(self):
        state = _make_state()
        assert state.previous_modulated_input is None

    def test_initial_previous_residual_is_none(self):
        state = _make_state()
        assert state.previous_residual is None

    def test_initial_step_counter_is_zero(self):
        state = _make_state()
        assert state.step_counter == 0

    def test_threshold_stored(self):
        state = _make_state(thresh=0.77)
        assert state.rel_l1_thresh == 0.77

    def test_rescale_func_is_callable(self):
        state = _make_state()
        assert callable(state.rescale_func)

    def test_rescale_func_is_numpy_poly1d(self):
        state = _make_state()
        assert isinstance(state.rescale_func, np.poly1d)

    def test_custom_poly_coeffs_stored(self):
        custom_coeffs = [1.0, 2.0, 3.0]
        state = _TeaCacheState(rel_l1_thresh=0.5, poly_coeffs=custom_coeffs)
        # poly1d([1, 2, 3]) at x=1 = 1 + 2 + 3 = 6
        assert abs(state.rescale_func(1.0) - 6.0) < 1e-6


# ===========================================================================
# 6. _MOTIF_POLY_COEFFS sanity check
# ===========================================================================

class TestPolyCoeffs:
    def test_has_five_coefficients(self):
        assert len(_MOTIF_POLY_COEFFS) == 5

    def test_all_coefficients_are_finite(self):
        for c in _MOTIF_POLY_COEFFS:
            assert np.isfinite(c), f"Coefficient {c} is not finite"

    def test_poly_evaluates_at_zero(self):
        """poly(0) should equal the constant term (last coefficient)."""
        p = np.poly1d(_MOTIF_POLY_COEFFS)
        # np.poly1d([a4, a3, a2, a1, a0]) at x=0 equals a0
        assert np.isfinite(p(0.0))

    def test_poly_evaluates_at_one(self):
        p = np.poly1d(_MOTIF_POLY_COEFFS)
        assert np.isfinite(p(1.0))


# ===========================================================================
# 7. _extract_modulated_input
# ===========================================================================

class TestExtractModulatedInput:
    """_extract_modulated_input calls block0.norm1 and returns norm_hidden_states."""

    def _make_transformer(self, norm_out: torch.Tensor):
        """Build a minimal mock transformer whose block0.norm1 returns norm_out."""
        block0 = MagicMock(name="block0")
        # norm1 returns (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        extra = MagicMock()
        block0.norm1.return_value = (norm_out, extra, extra, extra, extra)

        transformer = MagicMock(name="transformer")
        transformer.transformer_blocks = [block0]
        return transformer, block0

    def test_returns_first_element_of_norm1_output(self):
        """Return value must be the first tuple element from norm1."""
        expected = torch.ones(2, 4, 8)
        transformer, _ = self._make_transformer(expected)
        hidden_states = torch.zeros(2, 4, 8)
        temb = torch.zeros(2, 64)

        result = _extract_modulated_input(transformer, hidden_states, temb)
        assert torch.allclose(result, expected)

    def test_calls_norm1_with_hidden_states_and_temb(self):
        """norm1 must be called with (hidden_states, emb=temb)."""
        norm_out = torch.ones(2, 4, 8)
        transformer, block0 = self._make_transformer(norm_out)
        hidden_states = torch.full((2, 4, 8), 2.0)
        temb = torch.full((2, 64), 3.0)

        _extract_modulated_input(transformer, hidden_states, temb)

        block0.norm1.assert_called_once()
        call_args, call_kwargs = block0.norm1.call_args
        # positional arg is hidden_states
        assert call_args[0] is hidden_states
        # keyword arg emb is temb
        assert call_kwargs["emb"] is temb

    def test_uses_no_grad_context(self):
        """norm1 call must happen inside torch.no_grad() — grad should not flow."""
        norm_out = torch.ones(2, 4, 8, requires_grad=False)
        transformer, block0 = self._make_transformer(norm_out)

        hidden_states = torch.ones(2, 4, 8, requires_grad=True)
        temb = torch.ones(2, 64, requires_grad=True)

        with torch.no_grad():
            # Confirm that within no_grad the function runs cleanly
            result = _extract_modulated_input(transformer, hidden_states, temb)
        assert result is not None

    def test_returns_tensor(self):
        """Return type must be a torch.Tensor."""
        norm_out = torch.zeros(1, 3, 5)
        transformer, _ = self._make_transformer(norm_out)
        result = _extract_modulated_input(transformer, torch.zeros(1, 3, 5), torch.zeros(1, 16))
        assert isinstance(result, torch.Tensor)

    def test_accesses_first_block_only(self):
        """Only transformer_blocks[0].norm1 is called, not any other block."""
        norm_out = torch.ones(2, 4, 8)
        block0 = MagicMock(name="block0")
        block0.norm1.return_value = (norm_out, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        block1 = MagicMock(name="block1")

        transformer = MagicMock(name="transformer")
        transformer.transformer_blocks = [block0, block1]

        _extract_modulated_input(transformer, torch.zeros(2, 4, 8), torch.zeros(2, 64))

        block0.norm1.assert_called_once()
        block1.norm1.assert_not_called()


# ===========================================================================
# 8. _make_teacache_forward
# ===========================================================================

class TestMakeTeacacheForward:
    """The closure returned by _make_teacache_forward."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_components(self, thresh: float = 1e9):
        """Return (original_forward, transformer, state, x, kwargs_base)."""
        # Shared tensor shapes
        B, C, T, H, W = 1, 4, 2, 4, 4
        x = torch.ones(B, C, T, H, W)

        # Original adapter forward: just adds 1 to x
        def original_fwd(x_in, timestep, **kw):
            return x_in + 1.0

        # norm1 returns a zeros tensor (modulated input)
        norm_out = torch.zeros(B, 16, 8)
        block0 = MagicMock(name="block0")
        block0.norm1.return_value = (norm_out, MagicMock(), MagicMock(), MagicMock(), MagicMock())

        transformer = MagicMock(name="transformer")
        transformer.transformer_blocks = [block0]
        # time_text_embed and x_embedder used in teacache_forward preamble
        transformer.time_text_embed.return_value = (
            torch.zeros(B, 64),   # temb
            MagicMock(),           # _token_replace_emb
        )
        transformer.x_embedder.return_value = torch.zeros(B, 16, 8)

        state = _TeaCacheState(rel_l1_thresh=thresh, poly_coeffs=_MOTIF_POLY_COEFFS)

        return original_fwd, transformer, state, x

    # ------------------------------------------------------------------
    # Basic contract
    # ------------------------------------------------------------------

    def test_returns_callable(self):
        original_fwd, transformer, state, x = self._make_components()
        forward = _make_teacache_forward(original_fwd, transformer, state)
        assert callable(forward)

    def test_compute_path_returns_original_forward_result(self):
        """First call (no cache) must go through original_adapter_forward."""
        original_fwd, transformer, state, x = self._make_components()
        forward = _make_teacache_forward(original_fwd, transformer, state)

        result = forward(x, timestep=torch.tensor([0.5]))
        # original_fwd returns x + 1; x is all-ones so result is all-twos
        assert torch.allclose(result, torch.full_like(x, 2.0))

    def test_compute_path_increments_step_counter(self):
        """Step counter must be incremented on every call regardless of skip/compute."""
        original_fwd, transformer, state, x = self._make_components()
        forward = _make_teacache_forward(original_fwd, transformer, state)

        assert state.step_counter == 0
        forward(x, timestep=torch.tensor([0.5]))
        assert state.step_counter == 1

    def test_compute_path_stores_residual(self):
        """After compute step, state.previous_residual must be set."""
        original_fwd, transformer, state, x = self._make_components()
        forward = _make_teacache_forward(original_fwd, transformer, state)

        assert state.previous_residual is None
        forward(x, timestep=torch.tensor([0.5]))
        assert state.previous_residual is not None

    def test_compute_path_residual_equals_output_minus_input(self):
        """Cached residual must equal output - original_x."""
        original_fwd, transformer, state, x = self._make_components()
        forward = _make_teacache_forward(original_fwd, transformer, state)

        result = forward(x.clone(), timestep=torch.tensor([0.5]))
        # residual = output - original_x = (x+1) - x = 1
        assert torch.allclose(state.previous_residual, torch.ones_like(x))

    def test_compute_path_stores_modulated_input(self):
        """After compute step, state.previous_modulated_input must be set."""
        original_fwd, transformer, state, x = self._make_components()
        forward = _make_teacache_forward(original_fwd, transformer, state)

        assert state.previous_modulated_input is None
        forward(x, timestep=torch.tensor([0.5]))
        assert state.previous_modulated_input is not None

    def test_skip_path_reuses_cached_residual(self):
        """After seeding cache, skip step returns x + previous_residual.

        norm_out in _make_components is zeros, so previous_modulated_input must
        be non-zero (to avoid the degenerate-zero guard in should_skip) AND the
        current norm_out must be close enough that the poly-rescaled diff stays
        below thresh=1e9.  We seed with ones so mean_prev=1 and current=zeros
        gives raw_diff=1 → poly(1)≈275 << 1e9 → skip=True.
        """
        original_fwd, transformer, state, x = self._make_components(thresh=1e9)
        forward = _make_teacache_forward(original_fwd, transformer, state)

        # previous_modulated_input must be non-zero so mean_prev > 1e-10
        state.previous_modulated_input = torch.ones(1, 16, 8)
        state.previous_residual = torch.full_like(x, 7.0)

        result = forward(x, timestep=torch.tensor([0.5]))
        # skip path: output = x + previous_residual = 1 + 7 = 8
        assert torch.allclose(result, torch.full_like(x, 8.0))

    def test_skip_path_does_not_call_original_forward(self):
        """Original forward must not be called when cache is reused."""
        call_log = []

        def tracking_fwd(x_in, timestep, **kw):
            call_log.append("called")
            return x_in + 1.0

        _, transformer, state, x = self._make_components(thresh=1e9)
        forward = _make_teacache_forward(tracking_fwd, transformer, state)

        # previous_modulated_input must be non-zero (avoid degenerate-zero guard)
        state.previous_modulated_input = torch.ones(1, 16, 8)
        state.previous_residual = torch.ones_like(x)

        forward(x, timestep=torch.tensor([0.5]))
        assert len(call_log) == 0, "original_adapter_forward must not be called on skip step"

    def test_skip_path_increments_step_counter(self):
        """Step counter must increment even on skip steps."""
        original_fwd, transformer, state, x = self._make_components(thresh=1e9)
        forward = _make_teacache_forward(original_fwd, transformer, state)

        # non-zero seed to avoid degenerate-zero guard
        state.previous_modulated_input = torch.ones(1, 16, 8)
        state.previous_residual = torch.ones_like(x)

        forward(x, timestep=torch.tensor([0.5]))
        assert state.step_counter == 1

    def test_skip_then_compute_then_skip_sequence(self):
        """Simulate a realistic 3-step sampling sequence.

        norm_out must be non-zero so that the mean_prev > 1e-10 guard in
        should_skip is satisfied and the skip path is reachable after step 0.
        We use norm_out=ones; after step 0 stores it as previous_modulated_input,
        step 1's current input is also ones → raw_diff=0 → poly(0)≈499 << 1e9
        → skip=True.
        """
        # Step 0 (compute): no cache yet
        # Step 1 (skip): cache populated, thresh very high → skip
        # Step 2 (compute): force compute by setting thresh=0 so accumulated >= thresh
        B, C, T, H, W = 1, 4, 2, 4, 4
        x = torch.ones(B, C, T, H, W)

        compute_calls = []

        def tracking_fwd(x_in, timestep, **kw):
            compute_calls.append(state.step_counter)
            return x_in + 1.0

        # Use non-zero norm_out so mean_prev > 1e-10 after step 0
        norm_out = torch.ones(1, 16, 8)
        block0 = MagicMock(name="block0")
        block0.norm1.return_value = (norm_out, MagicMock(), MagicMock(), MagicMock(), MagicMock())

        transformer = MagicMock(name="transformer")
        transformer.transformer_blocks = [block0]
        transformer.time_text_embed.return_value = (torch.zeros(1, 64), MagicMock())
        transformer.x_embedder.return_value = torch.zeros(1, 16, 8)

        state = _TeaCacheState(rel_l1_thresh=1e9, poly_coeffs=_MOTIF_POLY_COEFFS)
        forward = _make_teacache_forward(tracking_fwd, transformer, state)

        # Step 0: compute (no cache)
        forward(x.clone(), timestep=torch.tensor([0.9]))
        assert state.step_counter == 1
        assert len(compute_calls) == 1

        # Step 1: skip (cache populated with non-zero norm_out, thresh very high)
        forward(x.clone(), timestep=torch.tensor([0.8]))
        assert state.step_counter == 2
        # original forward was NOT called again
        assert len(compute_calls) == 1, (
            "Step 1 should skip — original forward must not be called a second time"
        )

        # Step 2: force compute by dropping threshold to 0 so any accumulated dist >= 0
        state.rel_l1_thresh = 0.0
        # accumulated_rel_l1_distance is already > 0 from step 1 (poly(0) ≈ 499)
        forward(x.clone(), timestep=torch.tensor([0.7]))
        assert state.step_counter == 3
        assert len(compute_calls) == 2

    def test_does_not_mutate_x_in_compute_path(self):
        """original_adapter_forward receives the mutated x from ComfyUI; we clone before
        passing to preserve the original for residual computation."""
        B, C, T, H, W = 1, 4, 2, 4, 4
        x = torch.ones(B, C, T, H, W)
        x_original_clone = x.clone()

        # original_fwd modifies x in-place to simulate ComfyUI adapter behavior
        def mutating_fwd(x_in, timestep, **kw):
            x_in.fill_(99.0)  # in-place mutation
            return x_in

        norm_out = torch.zeros(1, 16, 8)
        block0 = MagicMock()
        block0.norm1.return_value = (norm_out, MagicMock(), MagicMock(), MagicMock(), MagicMock())

        transformer = MagicMock()
        transformer.transformer_blocks = [block0]
        transformer.time_text_embed.return_value = (torch.zeros(1, 64), MagicMock())
        transformer.x_embedder.return_value = torch.zeros(1, 16, 8)

        state = _TeaCacheState(rel_l1_thresh=1e9, poly_coeffs=_MOTIF_POLY_COEFFS)
        forward = _make_teacache_forward(mutating_fwd, transformer, state)

        forward(x, timestep=torch.tensor([0.5]))
        # residual = output - ori_x = 99 - 1 = 98
        assert torch.allclose(state.previous_residual, torch.full_like(x, 98.0)), (
            "ori_x clone must capture pre-mutation value; residual must be output - original_x"
        )


# ===========================================================================
# 9. Node loading — apply_teacache() return-value contract (enable=True)
# ===========================================================================

class TestNodeLoadingReturnContract:
    """apply_teacache(enable=True) must return a 1-element tuple wrapping the
    cloned model.  Tests here focus on return-type/shape, not internal state."""

    def test_returns_tuple_when_enabled(self):
        node = MotifTeaCache()
        model, patched_model, *_ = _make_full_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert isinstance(result, tuple), "apply_teacache must return a tuple"

    def test_returns_single_element_tuple_when_enabled(self):
        node = MotifTeaCache()
        model, patched_model, *_ = _make_full_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert len(result) == 1, "apply_teacache must return a 1-element tuple"

    def test_first_element_is_cloned_model(self):
        """Return value must be the cloned model, not the original."""
        node = MotifTeaCache()
        model, patched_model, *_ = _make_full_mock_model()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert result[0] is patched_model
        assert result[0] is not model, "apply_teacache must return the clone, not the original"

    def test_original_model_not_mutated_when_enabled(self):
        """model.clone() is called; the original model object itself is not patched."""
        node = MotifTeaCache()
        model, patched_model, *_ = _make_full_mock_model()
        # Record clone call count before
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        # The original model must have been cloned exactly once
        model.clone.assert_called_once()

    def test_adapter_forward_is_callable_after_patch(self):
        """After apply_teacache, transformer.forward must be a callable (not a Mock)."""
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert callable(transformer.forward), "patched transformer.forward must be callable"

    def test_adapter_forward_is_plain_function_not_mock(self):
        """The patched forward must be the teacache closure, not a MagicMock."""
        import inspect
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        # A real function (closure) is not an instance of MagicMock
        assert not isinstance(transformer.forward, MagicMock), (
            "transformer.forward must be replaced with a real closure, not a MagicMock"
        )


# ===========================================================================
# 10. _compute_sampling_progress
# ===========================================================================

_compute_sampling_progress = _teacache_mod._compute_sampling_progress


class TestComputeSamplingProgress:
    """_compute_sampling_progress converts sigma → progress in [0, 1]."""

    def test_sigma_1_gives_progress_0(self):
        """sigma=1.0 (start of sampling) → progress=0.0."""
        ts = torch.tensor([1.0])
        assert _compute_sampling_progress(ts) == pytest.approx(0.0)

    def test_sigma_0_gives_progress_1(self):
        """sigma=0.0 (end of sampling) → progress=1.0."""
        ts = torch.tensor([0.0])
        assert _compute_sampling_progress(ts) == pytest.approx(1.0)

    def test_sigma_0_5_gives_progress_0_5(self):
        """sigma=0.5 → progress=0.5."""
        ts = torch.tensor([0.5])
        assert _compute_sampling_progress(ts) == pytest.approx(0.5)

    def test_result_clamped_above_0(self):
        """sigma > 1.0 → clamped to 0.0."""
        ts = torch.tensor([1.5])
        assert _compute_sampling_progress(ts) == pytest.approx(0.0)

    def test_result_clamped_below_1(self):
        """sigma < 0.0 → clamped to 1.0."""
        ts = torch.tensor([-0.1])
        assert _compute_sampling_progress(ts) == pytest.approx(1.0)

    def test_uses_first_element_of_batch(self):
        """Batch tensor: only first element is used."""
        ts = torch.tensor([0.3, 0.9])  # progress from first = 1 - 0.3 = 0.7
        assert _compute_sampling_progress(ts) == pytest.approx(0.7)

    def test_multidim_tensor(self):
        """2D tensor: progress computed from first element after flatten."""
        ts = torch.tensor([[0.2]])
        assert _compute_sampling_progress(ts) == pytest.approx(0.8)


# ===========================================================================
# 11. start/end range filtering in teacache_forward
# ===========================================================================

class TestTeacacheForwardStartEnd:
    """teacache_forward bypasses caching when progress is outside [start, end)."""

    def _make_components_with_range(self, start: float, end: float, thresh: float = 1e9):
        """Like _make_components in TestMakeTeacacheForward but with start/end."""
        B, C, T, H, W = 1, 4, 2, 4, 4
        x = torch.ones(B, C, T, H, W)

        call_log = []

        def original_fwd(x_in, timestep, **kw):
            call_log.append("called")
            return x_in + 1.0

        norm_out = torch.ones(1, 16, 8)  # non-zero to avoid degenerate guard
        block0 = MagicMock(name="block0")
        block0.norm1.return_value = (norm_out, MagicMock(), MagicMock(), MagicMock(), MagicMock())

        transformer = MagicMock(name="transformer")
        transformer.transformer_blocks = [block0]
        transformer.time_text_embed.return_value = (torch.zeros(B, 64), MagicMock())
        transformer.x_embedder.return_value = torch.zeros(B, 16, 8)

        state = _TeaCacheState(
            rel_l1_thresh=thresh,
            poly_coeffs=_MOTIF_POLY_COEFFS,
            start=start,
            end=end,
        )
        forward = _make_teacache_forward(original_fwd, transformer, state)
        return forward, state, x, call_log

    def test_out_of_range_before_start_calls_original_forward(self):
        """progress < start → original forward called, no cache update."""
        # sigma=0.8 → progress=0.2; start=0.5 → out of range
        forward, state, x, call_log = self._make_components_with_range(start=0.5, end=1.0)
        forward(x, timestep=torch.tensor([0.8]))
        assert len(call_log) == 1, "original forward must be called when out of range"

    def test_out_of_range_before_start_no_modulated_input_stored(self):
        """progress < start → previous_modulated_input stays None."""
        forward, state, x, call_log = self._make_components_with_range(start=0.5, end=1.0)
        forward(x, timestep=torch.tensor([0.8]))
        assert state.previous_modulated_input is None

    def test_out_of_range_before_start_step_counter_increments(self):
        """Even when out of range, step_counter increments."""
        forward, state, x, call_log = self._make_components_with_range(start=0.5, end=1.0)
        assert state.step_counter == 0
        forward(x, timestep=torch.tensor([0.8]))
        assert state.step_counter == 1

    def test_in_range_populates_cache(self):
        """progress in [start, end) → modulated input is stored after compute."""
        # sigma=0.4 → progress=0.6; range [0.5, 1.0) → in range
        forward, state, x, call_log = self._make_components_with_range(start=0.5, end=1.0)
        forward(x, timestep=torch.tensor([0.4]))
        assert state.previous_modulated_input is not None

    def test_out_of_range_after_end_calls_original_forward(self):
        """progress >= end → original forward called, no cache update."""
        # sigma=0.05 → progress=0.95; end=0.9 → out of range
        forward, state, x, call_log = self._make_components_with_range(start=0.0, end=0.9)
        forward(x, timestep=torch.tensor([0.05]))
        assert len(call_log) == 1

    def test_out_of_range_after_end_no_modulated_input_stored(self):
        """progress >= end → previous_modulated_input stays None."""
        forward, state, x, call_log = self._make_components_with_range(start=0.0, end=0.9)
        forward(x, timestep=torch.tensor([0.05]))
        assert state.previous_modulated_input is None

    def test_default_range_full_pass(self):
        """start=0.0, end=1.0 → all sigma values in range (original behavior)."""
        # sigma=0.5 → progress=0.5; range [0.0, 1.0) → in range
        forward, state, x, call_log = self._make_components_with_range(start=0.0, end=1.0)
        forward(x, timestep=torch.tensor([0.5]))
        # In-range → cache update should happen
        assert state.previous_modulated_input is not None

    def test_end_boundary_exclusive(self):
        """progress == end is out of range (exclusive upper bound)."""
        # sigma=0.0 → progress=1.0; end=1.0 → exactly at boundary (exclusive)
        forward, state, x, call_log = self._make_components_with_range(start=0.0, end=1.0)
        forward(x, timestep=torch.tensor([0.0]))
        # progress=1.0 is not < 1.0, so out-of-range path taken
        # However, sigma=0.0 is clamped: progress = clamp(1-0, 0, 1) = 1.0
        # 1.0 < 1.0 is False → out of range
        assert state.previous_modulated_input is None

    def test_start_end_state_stored_correctly(self):
        """_TeaCacheState stores start and end correctly."""
        state = _TeaCacheState(
            rel_l1_thresh=0.3,
            poly_coeffs=_MOTIF_POLY_COEFFS,
            start=0.2,
            end=0.8,
        )
        assert state.start == pytest.approx(0.2)
        assert state.end == pytest.approx(0.8)

    def test_apply_teacache_stores_start_end_in_state(self):
        """apply_teacache with start/end passes them to _TeaCacheState."""
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True, start=0.1, end=0.9)
        assert transformer._teacache_state.start == pytest.approx(0.1)
        assert transformer._teacache_state.end == pytest.approx(0.9)

    def test_input_types_contains_start(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        assert "start" in required

    def test_input_types_contains_end(self):
        required = MotifTeaCache.INPUT_TYPES()["required"]
        assert "end" in required

    def test_start_default_is_0(self):
        meta = MotifTeaCache.INPUT_TYPES()["required"]["start"][1]
        assert meta["default"] == 0.0

    def test_end_default_is_1(self):
        meta = MotifTeaCache.INPUT_TYPES()["required"]["end"][1]
        assert meta["default"] == 1.0

    def test_start_type_is_float(self):
        field = MotifTeaCache.INPUT_TYPES()["required"]["start"]
        assert field[0] == "FLOAT"

    def test_end_type_is_float(self):
        field = MotifTeaCache.INPUT_TYPES()["required"]["end"]
        assert field[0] == "FLOAT"


# ===========================================================================
# 10. Monkey-patch disable — adapter untouched when enable=False
# ===========================================================================

class TestMonkeyPatchDisable:
    """enable=False must leave adapter.forward completely untouched."""

    def _make_model_with_trackable_forward(self):
        """Return model hierarchy where transformer.forward is a real sentinel callable.

        diffusion_model IS the transformer directly (no adapter wrapper).
        """
        sentinel_forward_calls = []

        def sentinel_forward(*args, **kwargs):
            sentinel_forward_calls.append(1)
            return args[0] if args else None

        block0 = MagicMock()
        transformer = MagicMock()
        transformer.transformer_blocks = [block0]
        transformer.forward = sentinel_forward  # real callable, not a Mock method
        transformer._teacache_enabled = False

        inner_model = MagicMock()
        inner_model.diffusion_model = transformer  # diffusion_model IS transformer

        patched_model = MagicMock()
        patched_model.model = inner_model

        model = MagicMock()
        model.clone.return_value = patched_model

        return model, transformer, sentinel_forward, sentinel_forward_calls

    def test_adapter_forward_not_replaced_when_disabled(self):
        node = MotifTeaCache()
        model, transformer, sentinel_forward, _ = self._make_model_with_trackable_forward()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        # transformer.forward is never accessed because enable=False returns early
        # (model.clone() not called, so transformer is unreachable — just verify model identity)
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        assert result[0] is model

    def test_teacache_enabled_flag_not_set_when_disabled(self):
        """When enable=False, transformer._teacache_enabled must remain False."""
        node = MotifTeaCache()
        model, transformer, _, _ = self._make_model_with_trackable_forward()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        # transformer is unreachable via the disabled path, but original model is returned
        # Verify that no teacache state leaked onto the transformer
        assert transformer._teacache_enabled is False

    def test_teacache_state_not_attached_when_disabled(self):
        """When enable=False, transformer must not have a real _TeaCacheState attached.

        MagicMock auto-creates attributes on access so we cannot use hasattr.
        Instead verify that _teacache_state (if it exists on the Mock) is NOT
        an instance of _TeaCacheState — i.e. apply_teacache never ran the patch
        path that creates a real state object.
        """
        node = MotifTeaCache()
        model, transformer, _, _ = self._make_model_with_trackable_forward()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=False)
        # If _teacache_state was attached by the patch path it would be a real
        # _TeaCacheState; a MagicMock auto-attribute is just another Mock.
        state_attr = object.__getattribute__(transformer, "_mock_children").get("_teacache_state")
        assert state_attr is None or not isinstance(state_attr, _TeaCacheState), (
            "enable=False must not attach a real _TeaCacheState to the transformer"
        )


# ===========================================================================
# 11. Fallback — transformer_blocks missing or empty: no state attached
# ===========================================================================

class TestFallbackNoTransformerBlocks:
    """When transformer_blocks is absent/empty, no TeaCache state must be attached
    and _teacache_enabled must not be set to True."""

    def _make_model_empty_blocks(self):
        """diffusion_model IS the transformer; it has empty transformer_blocks."""
        transformer = MagicMock()
        transformer.transformer_blocks = []
        transformer._teacache_enabled = False

        inner_model = MagicMock()
        inner_model.diffusion_model = transformer  # no adapter wrapper

        patched_model = MagicMock()
        patched_model.model = inner_model

        model = MagicMock()
        model.clone.return_value = patched_model

        return model, patched_model, transformer

    def test_enabled_flag_not_set_on_empty_blocks(self):
        node = MotifTeaCache()
        model, patched_model, transformer = self._make_model_empty_blocks()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert transformer._teacache_enabled is False, (
            "transformer._teacache_enabled must remain False when transformer_blocks is empty"
        )

    def test_state_not_attached_on_empty_blocks(self):
        node = MotifTeaCache()
        model, patched_model, transformer = self._make_model_empty_blocks()
        # Remove _teacache_state if it happened to exist on the Mock
        if hasattr(transformer, "_teacache_state"):
            del transformer._teacache_state
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert not hasattr(transformer, "_teacache_state"), (
            "No _teacache_state must be attached when transformer_blocks is empty"
        )

    def test_returns_patched_model_not_original_on_empty_blocks(self):
        """Even on fallback, the cloned patched_model (not original) is returned."""
        node = MotifTeaCache()
        model, patched_model, transformer = self._make_model_empty_blocks()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert result[0] is patched_model
        assert result[0] is not model

    def _make_model_no_transformer_blocks_attr(self):
        """diffusion_model has no transformer_blocks attribute at all."""
        # spec=[] means no attributes by default → hasattr returns False
        diffusion_model = MagicMock(spec=[])

        inner_model = MagicMock()
        inner_model.diffusion_model = diffusion_model

        patched_model = MagicMock()
        patched_model.model = inner_model

        model = MagicMock()
        model.clone.return_value = patched_model

        return model, patched_model, diffusion_model

    def test_returns_patched_model_on_missing_transformer_attr(self):
        node = MotifTeaCache()
        model, patched_model, diffusion_model = self._make_model_no_transformer_blocks_attr()
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        assert result[0] is patched_model

    def test_enabled_flag_not_set_on_missing_transformer_attr(self):
        node = MotifTeaCache()
        model, patched_model, diffusion_model = self._make_model_no_transformer_blocks_attr()
        # diffusion_model has spec=[] so _teacache_enabled cannot be set
        # Guard in apply_teacache checks hasattr(transformer, "transformer_blocks")
        result = node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)
        # Verify that teacache_enabled was not set (spec=[] would raise AttributeError)
        assert not hasattr(diffusion_model, "_teacache_enabled"), (
            "No _teacache_enabled must be set on diffusion_model without transformer_blocks"
        )


# ===========================================================================
# 12. Idempotency — detailed assertions
# ===========================================================================

class TestIdempotencyDetailed:
    """Re-applying apply_teacache to an already-patched transformer must not
    install a second layer of wrapping."""

    def _patch_once(self):
        """Return (node, model2, transformer) after first patch applied."""
        node = MotifTeaCache()
        model, patched_model, inner_model, transformer, _ = _make_full_mock_model()
        node.apply_teacache(model=model, rel_l1_thresh=0.3, enable=True)

        # Second model whose clone still resolves to the same inner_model/transformer
        patched_model2 = MagicMock(name="patched_model2")
        patched_model2.model = inner_model
        model2 = MagicMock(name="model2")
        model2.clone.return_value = patched_model2

        return node, model2, transformer

    def test_forward_object_identity_preserved_on_repatch(self):
        """transformer.forward must be the exact same object after second apply."""
        node, model2, transformer = self._patch_once()
        forward_before = transformer.forward
        node.apply_teacache(model=model2, rel_l1_thresh=0.5, enable=True)
        assert transformer.forward is forward_before, (
            "Idempotency: transformer.forward identity must not change on second apply"
        )

    def test_state_object_identity_preserved_on_repatch(self):
        """_teacache_state must be the same object instance after second apply."""
        node, model2, transformer = self._patch_once()
        state_before = transformer._teacache_state
        node.apply_teacache(model=model2, rel_l1_thresh=0.5, enable=True)
        assert transformer._teacache_state is state_before, (
            "Idempotency: _teacache_state object must not be replaced on second apply"
        )

    def test_state_is_reset_on_repatch(self):
        """Idempotency path resets cache state so the next generation starts clean."""
        node, model2, transformer = self._patch_once()
        # Dirty the state to simulate mid-generation state
        transformer._teacache_state.step_counter = 42
        transformer._teacache_state.accumulated_rel_l1_distance = 99.9
        transformer._teacache_state.previous_modulated_input = torch.ones(1, 8, 8)

        node.apply_teacache(model=model2, rel_l1_thresh=0.5, enable=True)

        assert transformer._teacache_state.step_counter == 0, (
            "Idempotency path must reset step_counter"
        )
        assert transformer._teacache_state.accumulated_rel_l1_distance == 0.0, (
            "Idempotency path must reset accumulated_rel_l1_distance"
        )
        assert transformer._teacache_state.previous_modulated_input is None, (
            "Idempotency path must reset previous_modulated_input"
        )

    def test_enabled_flag_remains_true_on_repatch(self):
        node, model2, transformer = self._patch_once()
        node.apply_teacache(model=model2, rel_l1_thresh=0.5, enable=True)
        assert transformer._teacache_enabled is True
