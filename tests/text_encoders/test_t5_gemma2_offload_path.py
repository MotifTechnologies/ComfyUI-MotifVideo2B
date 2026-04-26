"""
Tests for checklist item 11: Partial offload — ModelPatcher vbar branch entry
(C) unit tests.

Verification strategy:
  Approach A (unit): test the *branch predicate* directly.
  ModelPatcherDynamic.patch_model() inlines setup_param; the only discriminant
  for the vbar vs force-load path is `hasattr(m, "comfy_cast_weights")` where
  the attribute is truthy.  ModelPatcher._load_list() also gates on this
  predicate and is callable on CPU without GPU resident weights.

  Equivalence argued in item-11 scope:
    hasattr(m, "comfy_cast_weights") AND truthy
    <=> ModelPatcherDynamic would enter vbar branch (not force-load/backup)
  This is guaranteed by the ModelPatcherDynamic.patch_model source at lines
  1591-1608 of comfy/model_patcher.py:
      if hasattr(m, "comfy_cast_weights"):
          ...
          if vbar is not None and not hasattr(m, "_v"):
              m._v = vbar.alloc(v_weight_size)    # vbar path
      else:
          self.backup[key] = ...                  # force-load path

Blind-test principle: item-11 diff was NOT read before writing these tests.
Tests derive from spec + ModelPatcher source only.

Run:
  /lustrefs/team-multimodal/minsu/ComfyUI/.venv/bin/python \\
      -m pytest tests/text_encoders/test_t5_gemma2_offload_path.py -v
"""

from __future__ import annotations

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
import torch.nn as nn  # noqa: E402
import comfy.ops       # noqa: E402
import comfy.model_patcher  # noqa: E402

from text_encoders.t5_gemma2_native import (  # noqa: E402
    T5Gemma2TextEncoder,
    T5Gemma2Encoder,
)
from text_encoders.t5_gemma2_config import T5_GEMMA2_CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _make_encoder_config():
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2EncoderConfig,
    )
    return T5Gemma2EncoderConfig(**T5_GEMMA2_CONFIG)


def _make_text_config():
    return _make_encoder_config().text_config


# Layers with comfy_cast_weights in manual_cast: Linear + Embedding
# Per layer: q_proj, k_proj, v_proj, o_proj (4) + gate_proj, up_proj, down_proj (3) = 7
# 34 layers x 7 = 238, plus embed_tokens (1 Embedding) = 239
# RMSNorm uses raw nn.Parameter (not comfy.ops.Linear/Embedding) -> excluded
_NUM_LAYERS = 34
_LINEARS_PER_LAYER = 7   # 4 attn + 3 mlp
_EXPECTED_CCW_COUNT = _NUM_LAYERS * _LINEARS_PER_LAYER + 1  # +1 embed_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_leaf_modules(model: nn.Module):
    """Return list of (name, module) pairs for leaf modules with parameters."""
    result = []
    for name, m in model.named_modules():
        params = list(m.parameters(recurse=False))
        if params:
            result.append((name, m))
    return result


def _modules_with_cast_weights(model: nn.Module):
    return [(n, m) for n, m in _collect_leaf_modules(model)
            if hasattr(m, "comfy_cast_weights")]


def _modules_without_cast_weights(model: nn.Module):
    return [(n, m) for n, m in _collect_leaf_modules(model)
            if not hasattr(m, "comfy_cast_weights")]


# ===========================================================================
# 1. manual_cast encoder: every Linear/Embedding hits vbar predicate
# ===========================================================================

class TestVbarBranchUnderManualCast:
    """manual_cast build -> comfy_cast_weights truthy on all trainable layers."""

    @pytest.fixture(scope="class")
    def encoder(self):
        cfg = _make_text_config()
        return T5Gemma2TextEncoder(
            cfg, dtype=None, device="cpu", operations=comfy.ops.manual_cast,
        )

    def test_native_layers_hit_vbar_branch_under_manual_cast(self, encoder):
        """All leaf modules with comfy_cast_weights have the attribute set to True.

        ModelPatcherDynamic.patch_model() evaluates `hasattr(m, "comfy_cast_weights")`
        as the sole discriminant for vbar vs force-load.  A truthy value at this
        attribute is the necessary-and-sufficient condition for vbar path entry.
        """
        ccw_modules = _modules_with_cast_weights(encoder)
        assert len(ccw_modules) > 0, (
            "No modules with comfy_cast_weights found — manual_cast build broken"
        )
        failures = [
            f"{n}: comfy_cast_weights={m.comfy_cast_weights!r}"
            for n, m in ccw_modules
            if not m.comfy_cast_weights
        ]
        assert not failures, (
            "Modules have comfy_cast_weights attribute but value is falsy "
            "(would NOT enter vbar branch):\n" + "\n".join(failures)
        )

    def test_no_force_load_branch_modules_under_manual_cast(self, encoder):
        """Under manual_cast, no leaf-with-params module falls into the else branch.

        The else branch (force-load / backup) is taken when `hasattr(m, "comfy_cast_weights")`
        is False.  With manual_cast all comfy-layer types carry the attribute,
        so the else branch must not be taken for any of them.
        """
        no_ccw = _modules_without_cast_weights(encoder)
        # Only modules that have parameters directly on them matter for the branch
        # (named_parameters recurse=False check).  Raw nn.Parameter holders such
        # as RMSNorm weights appear as parameters of the *parent* module, not as
        # leaf-module parameters in the model_patcher sense.  We assert that no
        # ops.Linear / ops.Embedding equivalent is missing the attribute.
        bad = [
            (n, type(m).__name__)
            for n, m in no_ccw
            if issubclass(type(m), (nn.Linear, nn.Embedding))
        ]
        assert not bad, (
            "nn.Linear/Embedding subclasses missing comfy_cast_weights "
            "(would fall into force-load else branch):\n"
            + "\n".join(f"  {n}: {cls}" for n, cls in bad)
        )

    def test_load_list_classifies_ccw_modules_correctly(self, encoder):
        """ModelPatcher._load_list marks comfy_cast_weights modules for vbar.

        _load_list uses `hasattr(m, "comfy_cast_weights")` to append entries
        with the comfy_cast_weights flag.  Verify that our encoder produces
        entries that carry the flag when wrapped in a plain ModelPatcher.
        """
        cpu = torch.device("cpu")
        patcher = comfy.model_patcher.ModelPatcher(
            model=encoder,
            load_device=cpu,
            offload_device=cpu,
        )
        load_list = patcher._load_list(for_dynamic=True, default_device=cpu)
        # Each entry: (criteria..., module_mem, name, module, params)
        ccw_in_list = [
            (name, m)
            for *_, name, m, params in load_list
            if hasattr(m, "comfy_cast_weights")
        ]
        assert len(ccw_in_list) > 0, (
            "ModelPatcher._load_list produced zero entries with comfy_cast_weights; "
            "encoder modules would never reach vbar allocation path"
        )


# ===========================================================================
# 2. ModelPatcher branch predicate: hasattr-only, force-load reserved for raw nn.*
# ===========================================================================

class TestForceLoadVsVbarBranchPredicate:
    """Verify the *actual* ModelPatcher branch rule (`hasattr(m, comfy_cast_weights)`).

    Original assumption that disable_weight_init falls into the else (force-load)
    branch was wrong: comfy.ops base class CastWeightBiasOp defines
    `comfy_cast_weights = False` at the class level, so hasattr is True for both
    disable_weight_init and manual_cast variants. Both enter the vbar branch;
    manual_cast additionally has the value True, which gates the per-call cast.

    The else (force-load) branch is reserved for *raw* nn.Linear / nn.Embedding,
    which carry no comfy_cast_weights attribute at all. The native impl never
    constructs raw nn.Linear / nn.Embedding (verified separately by static grep
    in the item-11 verify command), so production native modules always take
    the vbar branch — the goal of this checklist item.
    """

    def test_native_disable_weight_init_modules_have_ccw_attr(self):
        """disable_weight_init build: every Linear/Embedding still has the attr.

        Under disable_weight_init the value is False (no per-call cast), but
        the attribute itself is present, so ModelPatcher.hasattr branch sends
        these modules into the vbar path — not force-load.
        """
        cfg = _make_text_config()
        encoder = T5Gemma2TextEncoder(
            cfg, dtype=None, device="cpu",
            operations=comfy.ops.disable_weight_init,
        )
        linear_embedding = [
            (n, m) for n, m in encoder.named_modules()
            if isinstance(m, (nn.Linear, nn.Embedding))
        ]
        assert len(linear_embedding) > 0, "No Linear/Embedding modules found"
        missing_attr = [
            (n, type(m).__name__) for n, m in linear_embedding
            if not hasattr(m, "comfy_cast_weights")
        ]
        assert missing_attr == [], (
            "disable_weight_init build must still carry comfy_cast_weights "
            "(class-level default False) on every Linear/Embedding so "
            "ModelPatcher sends them through the vbar branch. Missing: "
            + str(missing_attr)
        )
        # Class-level value is False (no cast), but the attribute exists.
        false_ccw = [(n, m) for n, m in linear_embedding if not m.comfy_cast_weights]
        true_ccw = [(n, m) for n, m in linear_embedding if m.comfy_cast_weights]
        assert len(false_ccw) > 0, "Expected at least one False ccw under disable_weight_init"
        assert len(true_ccw) == 0, (
            "disable_weight_init modules unexpectedly have ccw=True at build time: "
            + str([(n, type(m).__name__) for n, m in true_ccw])
        )

    def test_raw_nn_linear_lacks_ccw_attr_force_load_branch(self):
        """Negative control: raw nn.Linear / nn.Embedding lack comfy_cast_weights.

        ModelPatcher's `hasattr` predicate returns False for these, so they
        would land in the else (force-load) branch. The native impl must
        therefore avoid constructing raw nn.Linear / nn.Embedding directly —
        item-11's static grep verify enforces this; this test pins the
        contract that makes that grep meaningful.
        """
        raw_linear = nn.Linear(8, 8)
        raw_embedding = nn.Embedding(8, 8)
        assert not hasattr(raw_linear, "comfy_cast_weights"), (
            "raw nn.Linear unexpectedly has comfy_cast_weights — ModelPatcher "
            "branch predicate would no longer distinguish raw vs comfy.ops."
        )
        assert not hasattr(raw_embedding, "comfy_cast_weights"), (
            "raw nn.Embedding unexpectedly has comfy_cast_weights — same as above."
        )


# ===========================================================================
# 3. Module count sanity: manual_cast -> expected 239 comfy_cast_weights=True
# ===========================================================================

class TestModuleCountUnderManualCast:
    """Sanity: correct number of comfy_cast_weights=True modules in manual_cast encoder."""

    @pytest.fixture(scope="class")
    def encoder(self):
        cfg = _make_text_config()
        return T5Gemma2TextEncoder(
            cfg, dtype=None, device="cpu", operations=comfy.ops.manual_cast,
        )

    def test_count_of_modules_with_comfy_cast_weights(self, encoder):
        """manual_cast encoder must have exactly 239 comfy_cast_weights=True modules.

        Calculation:
          34 layers * (q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj)
          = 34 * 7 = 238 Linear modules
          + 1 embed_tokens (Embedding)
          = 239

        RMSNorm layers store weight as raw nn.Parameter on a plain nn.Module
        (no ops.Linear/Embedding wrapper), so they appear in named_modules but
        do NOT carry comfy_cast_weights.  They must NOT be counted.
        """
        ccw_true = [
            (n, m) for n, m in encoder.named_modules()
            if hasattr(m, "comfy_cast_weights") and m.comfy_cast_weights
        ]
        count = len(ccw_true)
        assert count == _EXPECTED_CCW_COUNT, (
            f"Expected {_EXPECTED_CCW_COUNT} modules with comfy_cast_weights=True, "
            f"got {count}.\n"
            f"  Linear per layer: {_LINEARS_PER_LAYER}, layers: {_NUM_LAYERS}, "
            f"embed: 1.\n"
            f"  Actual modules:\n"
            + "\n".join(f"    {n}: {type(m).__name__}" for n, m in ccw_true[:10])
            + (f"\n    ... and {count - 10} more" if count > 10 else "")
        )

    def test_rmsnorm_modules_not_counted(self, encoder):
        """RMSNorm layers must NOT carry comfy_cast_weights=True.

        RMSNorm weight is a raw nn.Parameter; it is not wrapped in comfy.ops.*,
        so it always takes the backup/force-load branch.  This is expected and
        acceptable — the norm layers are small and fully reside in VRAM.
        """
        rmsnorm_with_ccw_true = [
            (n, m) for n, m in encoder.named_modules()
            if "norm" in n.lower()
            and hasattr(m, "comfy_cast_weights")
            and m.comfy_cast_weights
        ]
        assert not rmsnorm_with_ccw_true, (
            "RMSNorm-like modules unexpectedly have comfy_cast_weights=True "
            "(would cause double counting): "
            + str([(n, type(m).__name__) for n, m in rmsnorm_with_ccw_true])
        )


# ===========================================================================
# 4. T5Gemma2Encoder (outer wrapper): same predicate holds end-to-end
# ===========================================================================

class TestOuterEncoderVbarPredicate:
    """T5Gemma2Encoder (the outer ComfyUI wrapper) propagates manual_cast correctly."""

    @pytest.fixture(scope="class")
    def outer_encoder(self):
        cfg = _make_encoder_config()
        return T5Gemma2Encoder(
            cfg, dtype=None, device="cpu", operations=comfy.ops.manual_cast,
        )

    def test_outer_encoder_manual_cast_propagates_ccw(self, outer_encoder):
        """T5Gemma2Encoder with manual_cast: inner text_model modules all have CCW=True."""
        ccw_true = [
            (n, m) for n, m in outer_encoder.named_modules()
            if hasattr(m, "comfy_cast_weights") and m.comfy_cast_weights
        ]
        assert len(ccw_true) >= _EXPECTED_CCW_COUNT, (
            f"T5Gemma2Encoder (outer) expected >= {_EXPECTED_CCW_COUNT} "
            f"comfy_cast_weights=True modules, got {len(ccw_true)}"
        )

    def test_outer_encoder_default_keeps_attr_value_false(self):
        """T5Gemma2Encoder default (disable_weight_init): attr present, value False.

        This is the build-time state. Both manual_cast and disable_weight_init
        builds enter the ModelPatcher vbar branch (hasattr is True for both),
        but only manual_cast has the attribute set to True at build time so
        that the per-call cast actually fires. Default operations leaves the
        attribute at the class-level False; ModelPatcher promotes it to True
        only after entering the branch when it wants to dispatch via vbar.
        """
        cfg = _make_encoder_config()
        encoder = T5Gemma2Encoder(
            cfg, dtype=None, device="cpu",
            # operations=None -> falls back to disable_weight_init inside
        )
        linear_embedding = [
            (n, m) for n, m in encoder.named_modules()
            if isinstance(m, (nn.Linear, nn.Embedding))
        ]
        assert len(linear_embedding) > 0
        # Build-time: attr is present (class-level), value is False.
        attr_missing = [(n, type(m).__name__) for n, m in linear_embedding
                        if not hasattr(m, "comfy_cast_weights")]
        attr_true = [(n, type(m).__name__) for n, m in linear_embedding
                     if getattr(m, "comfy_cast_weights", False)]
        assert attr_missing == [], (
            "Default encoder must still carry comfy_cast_weights attribute "
            "(class-level) on every Linear/Embedding. Missing: " + str(attr_missing)
        )
        assert attr_true == [], (
            "Default encoder unexpectedly has ccw=True at build time "
            "(should be False until ModelPatcher promotes it): " + str(attr_true)
        )


# ===========================================================================
# 5. ModelPatcher API compatibility (signature regression guard)
# ===========================================================================

class TestModelPatcherApiCompat:
    """Guard: ModelPatcher API matches what this plan was written against."""

    def test_setup_param_signature_compat(self):
        """ModelPatcher._load_list and patch_model must accept for_dynamic kwarg.

        _load_list is the entry point that feeds the vbar branch loop.
        If ComfyUI updates remove or rename this method, tests above would
        silently pass with empty lists — this guard catches that early.
        """
        assert hasattr(comfy.model_patcher.ModelPatcher, "_load_list"), (
            "comfy.model_patcher.ModelPatcher._load_list not found; "
            "ComfyUI may have refactored the dynamic loading path"
        )
        sig = inspect.signature(comfy.model_patcher.ModelPatcher._load_list)
        params = list(sig.parameters)
        assert "for_dynamic" in params, (
            f"ModelPatcher._load_list missing 'for_dynamic' param; got {params}"
        )
        assert "default_device" in params, (
            f"ModelPatcher._load_list missing 'default_device' param; got {params}"
        )

    def test_model_patcher_dynamic_class_exists(self):
        """ModelPatcherDynamic must exist (vbar path lives there)."""
        assert hasattr(comfy.model_patcher, "ModelPatcherDynamic"), (
            "comfy.model_patcher.ModelPatcherDynamic not found; "
            "the vbar partial-offload implementation may have moved"
        )

    def test_cast_weight_bias_op_class_default(self):
        """CastWeightBiasOp.comfy_cast_weights must default to False.

        manual_cast overrides this to True.  If the default changes to True,
        disable_weight_init would also enter the vbar branch unexpectedly.
        """
        from comfy.ops import CastWeightBiasOp
        assert CastWeightBiasOp.comfy_cast_weights is False, (
            "CastWeightBiasOp.comfy_cast_weights default changed from False — "
            "disable_weight_init vs manual_cast branch distinction may be broken"
        )

    def test_manual_cast_linear_has_ccw_true(self):
        """comfy.ops.manual_cast.Linear.comfy_cast_weights must be True (class level)."""
        assert comfy.ops.manual_cast.Linear.comfy_cast_weights is True, (
            "manual_cast.Linear.comfy_cast_weights is not True — "
            "vbar branch would never be entered for Linear layers"
        )

    def test_manual_cast_embedding_has_ccw_true(self):
        """comfy.ops.manual_cast.Embedding.comfy_cast_weights must be True (class level)."""
        assert comfy.ops.manual_cast.Embedding.comfy_cast_weights is True, (
            "manual_cast.Embedding.comfy_cast_weights is not True — "
            "vbar branch would never be entered for Embedding layers"
        )
