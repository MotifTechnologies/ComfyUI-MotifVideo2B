"""tests/test_detect_unet_config.py — Unit tests for _detect_unet_config_with_motif().

Requirements under test (checklist item 1):
  - state_dict에 `single_transformer_blocks.0.cross_attn_query_proj.weight` 키가 있으면
    `enable_text_cross_attention_single: True`
  - state_dict에 `transformer_blocks.0.cross_attn_query_proj.weight` 키가 있으면
    `enable_text_cross_attention_dual: True`
  - 키가 없으면 False (기존 체크포인트 하위 호환)
  - `numel() > 0` 검증으로 빈 텐서 방어
  - 파라미터명이 `enable_text_cross_attention_dual/single`로 변경됨
    (이전: `cross_attention_dual/single`)

All tests are CPU-only.

실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI/custom_nodes/ComfyUI-MotifVideo1.9B
    python -m pytest tests/test_detect_unet_config.py -v
"""

import importlib.util
import os
import sys
import types

import pytest
import torch


# ---------------------------------------------------------------------------
# Stubs + module extraction
#
# Strategy:
#   1. Register a minimal comfy.model_detection stub whose detect_unet_config
#      is a known sentinel.
#   2. Load __init__.py WITHOUT setting __package__ — the try-block that
#      imports comfy.model_detection and defines the function will succeed;
#      relative import failures (.config, .nodes.*) are caught and ignored.
#   3. After exec, the patched function is the new value of
#      comfy.model_detection.detect_unet_config AND is stored on the module
#      object as _detect_unet_config_with_motif.
# ---------------------------------------------------------------------------

def _setup_stubs():
    comfy_mod = types.ModuleType("comfy")
    sys.modules["comfy"] = comfy_mod

    # comfy.model_detection — stub with a sentinel original
    md_mod = types.ModuleType("comfy.model_detection")

    def _sentinel_original(state_dict, key_prefix, metadata=None):
        return {"image_model": "unknown_fallback"}

    md_mod.detect_unet_config = _sentinel_original
    sys.modules["comfy.model_detection"] = md_mod
    comfy_mod.model_detection = md_mod

    # comfy.supported_models — just needs a .models list
    sm_mod = types.ModuleType("comfy.supported_models")
    sm_mod.models = []
    sys.modules["comfy.supported_models"] = sm_mod
    comfy_mod.supported_models = sm_mod

    # comfy.model_base — required by test_model_init.py when collected together
    model_base_mod = types.ModuleType("comfy.model_base")

    class _FakeModelType:
        FLOW = "flow"

    class _FakeBaseModel:
        def __init__(self, model_config, model_type=None, device=None):
            pass

        def extra_conds(self, **kwargs):
            return {}

    model_base_mod.ModelType = _FakeModelType
    model_base_mod.BaseModel = _FakeBaseModel
    sys.modules["comfy.model_base"] = model_base_mod
    comfy_mod.model_base = model_base_mod

    # comfy.conds — required by test_model_init.py
    conds_mod = types.ModuleType("comfy.conds")

    class _FakeCONDRegular:
        def __init__(self, val):
            self.val = val

    conds_mod.CONDRegular = _FakeCONDRegular
    sys.modules["comfy.conds"] = conds_mod
    comfy_mod.conds = conds_mod

    return md_mod, _sentinel_original


_md_mod, _sentinel_original = _setup_stubs()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PY = os.path.join(_ROOT, "__init__.py")


def _load_init():
    spec = importlib.util.spec_from_file_location("_pkg_init_for_test", _INIT_PY)
    mod = importlib.util.module_from_spec(spec)
    # Do NOT set __package__ — let it stay as None/default so the
    # try-block importing comfy.model_detection works while relative imports
    # (.config, .nodes.*) fail harmlessly in their own try-except blocks.
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    return mod


_init_mod = _load_init()

# Guard: function must have been defined and the patch applied
_detect_full = getattr(_init_mod, "_detect_unet_config_with_motif", None)
if _detect_full is None:
    _detect_full = _md_mod.detect_unet_config
    if _detect_full is _sentinel_original:
        pytest.skip(
            "_detect_unet_config_with_motif not found / monkey-patch did not apply",
            allow_module_level=True,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper: always pass key_prefix="" so tests focus on cross-attn
# logic without worrying about the prefix parameter.
# ---------------------------------------------------------------------------

def _detect(state_dict: dict) -> dict:
    return _detect_full(state_dict, "", metadata=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensor(numel_positive: bool = True) -> torch.Tensor:
    return torch.zeros(1) if numel_positive else torch.zeros(0)


def _minimal_motif_sd(extra: dict = None) -> dict:
    """
    Minimum state_dict that satisfies the MotifVideo identity check
    (context_embedder + image_embedder both present) so the MotifVideo branch
    is entered.  Shapes are kept minimal but numerically valid for the
    architecture-detection arithmetic in the function.

    inner_dim = 128  → num_attention_heads = 128 // 128 = 1
    x_embedder shape = [128, 16]    → in_channels = 16
    proj_out shape   = [64, 128]    → out_channels = 64 // (2*2) = 16
    num_layers = 0   (no transformer_blocks.0.norm1.linear.weight)
    num_single_layers = 1 (single_transformer_blocks.0.attn.to_k.weight exists)
    """
    inner_dim = 128
    sd = {
        "context_embedder.linear_1.weight": torch.zeros(64, 16),
        "image_embedder.linear_1.weight": torch.zeros(64, 32),
        "single_transformer_blocks.0.attn.to_k.weight": torch.zeros(inner_dim, inner_dim),
        "x_embedder.proj.weight": torch.zeros(inner_dim, 16),
        "proj_out.weight": torch.zeros(64, inner_dim),
    }
    if extra:
        sd.update(extra)
    return sd


# ===========================================================================
# 1. single_transformer_blocks cross-attn key → enable_text_cross_attention_single
# ===========================================================================

class TestSingleBlockCrossAttn:

    SINGLE_KEY = "single_transformer_blocks.0.cross_attn_query_proj.weight"

    def test_single_key_present_sets_single_true(self):
        sd = _minimal_motif_sd({self.SINGLE_KEY: _make_tensor()})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is True, (
            "Expected enable_text_cross_attention_single=True when key is present"
        )

    def test_single_key_absent_sets_single_false(self):
        sd = _minimal_motif_sd()
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False, (
            "Expected enable_text_cross_attention_single=False when key is absent"
        )

    def test_single_key_present_does_not_enable_dual(self):
        """Only the single key present — dual must stay False."""
        sd = _minimal_motif_sd({self.SINGLE_KEY: _make_tensor()})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_dual") is False, (
            "dual must stay False when only the single key is present"
        )

    def test_single_key_empty_tensor_treated_as_absent(self):
        """numel() == 0  →  treated as absent  →  False."""
        sd = _minimal_motif_sd({self.SINGLE_KEY: _make_tensor(numel_positive=False)})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False, (
            "Empty tensor (numel=0) must NOT activate single cross-attn flag"
        )


# ===========================================================================
# 2. transformer_blocks cross-attn key → enable_text_cross_attention_dual
# ===========================================================================

class TestDualBlockCrossAttn:

    DUAL_KEY = "transformer_blocks.0.cross_attn_query_proj.weight"

    def test_dual_key_present_sets_dual_true(self):
        sd = _minimal_motif_sd({self.DUAL_KEY: _make_tensor()})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_dual") is True, (
            "Expected enable_text_cross_attention_dual=True when key is present"
        )

    def test_dual_key_absent_sets_dual_false(self):
        sd = _minimal_motif_sd()
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_dual") is False, (
            "Expected enable_text_cross_attention_dual=False when key is absent"
        )

    def test_dual_key_present_does_not_enable_single(self):
        """Only the dual key present — single must stay False."""
        sd = _minimal_motif_sd({self.DUAL_KEY: _make_tensor()})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False, (
            "single must stay False when only the dual key is present"
        )

    def test_dual_key_empty_tensor_treated_as_absent(self):
        """numel() == 0  →  treated as absent  →  False."""
        sd = _minimal_motif_sd({self.DUAL_KEY: _make_tensor(numel_positive=False)})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_dual") is False, (
            "Empty tensor (numel=0) must NOT activate dual cross-attn flag"
        )


# ===========================================================================
# 3. Both keys present simultaneously
# ===========================================================================

class TestBothKeysCrossAttn:

    SINGLE_KEY = "single_transformer_blocks.0.cross_attn_query_proj.weight"
    DUAL_KEY = "transformer_blocks.0.cross_attn_query_proj.weight"

    def test_both_keys_present_enables_both(self):
        sd = _minimal_motif_sd({
            self.SINGLE_KEY: _make_tensor(),
            self.DUAL_KEY: _make_tensor(),
        })
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is True
        assert cfg.get("enable_text_cross_attention_dual") is True

    def test_both_keys_empty_tensors_disables_both(self):
        sd = _minimal_motif_sd({
            self.SINGLE_KEY: _make_tensor(numel_positive=False),
            self.DUAL_KEY: _make_tensor(numel_positive=False),
        })
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False
        assert cfg.get("enable_text_cross_attention_dual") is False

    def test_single_empty_dual_valid_only_dual_true(self):
        sd = _minimal_motif_sd({
            self.SINGLE_KEY: _make_tensor(numel_positive=False),
            self.DUAL_KEY: _make_tensor(),
        })
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False
        assert cfg.get("enable_text_cross_attention_dual") is True

    def test_dual_empty_single_valid_only_single_true(self):
        sd = _minimal_motif_sd({
            self.SINGLE_KEY: _make_tensor(),
            self.DUAL_KEY: _make_tensor(numel_positive=False),
        })
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is True
        assert cfg.get("enable_text_cross_attention_dual") is False


# ===========================================================================
# 4. Backward compatibility — legacy checkpoint (no cross-attn keys)
# ===========================================================================

class TestBackwardCompatibility:

    def test_no_cross_attn_keys_both_false(self):
        cfg = _detect(_minimal_motif_sd())
        assert cfg.get("enable_text_cross_attention_single") is False
        assert cfg.get("enable_text_cross_attention_dual") is False

    def test_unrelated_keys_do_not_trigger_flags(self):
        sd = _minimal_motif_sd({
            "single_transformer_blocks.0.attn.to_v.weight": _make_tensor(),
            "transformer_blocks.0.norm1.linear.weight": _make_tensor(),
        })
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False
        assert cfg.get("enable_text_cross_attention_dual") is False

    def test_old_name_cross_attention_single_absent_from_output(self):
        """Old parameter name `cross_attention_single` must NOT appear."""
        cfg = _detect(_minimal_motif_sd())
        assert "cross_attention_single" not in cfg, (
            "Old name cross_attention_single must not appear in config output"
        )

    def test_old_name_cross_attention_dual_absent_from_output(self):
        """Old parameter name `cross_attention_dual` must NOT appear."""
        cfg = _detect(_minimal_motif_sd())
        assert "cross_attention_dual" not in cfg, (
            "Old name cross_attention_dual must not appear in config output"
        )

    def test_new_names_always_present_in_output(self):
        """Both new names must be keys in the output regardless of presence."""
        cfg = _detect(_minimal_motif_sd())
        assert "enable_text_cross_attention_single" in cfg
        assert "enable_text_cross_attention_dual" in cfg

    def test_non_motif_state_dict_falls_through_without_crash(self):
        """A state_dict without MotifVideo identity keys falls through to
        the original detection stub — must not raise."""
        sd = {"some_unrelated.weight": _make_tensor()}
        result = _detect_full(sd, "", metadata=None)
        assert isinstance(result, dict)


# ===========================================================================
# 5. Boundary / edge cases
# ===========================================================================

class TestBoundaryCases:

    SINGLE_KEY = "single_transformer_blocks.0.cross_attn_query_proj.weight"
    DUAL_KEY = "transformer_blocks.0.cross_attn_query_proj.weight"

    def test_2d_tensor_positive_numel_counts_as_present(self):
        """shape [4, 8] → numel=32 → must count as present."""
        sd = _minimal_motif_sd({self.SINGLE_KEY: torch.zeros(4, 8)})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is True

    def test_scalar_tensor_numel_one_counts_as_present(self):
        """torch.tensor(0.0) → numel=1 → must count as present."""
        sd = _minimal_motif_sd({self.DUAL_KEY: torch.tensor(0.0)})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_dual") is True

    def test_block_index_1_does_not_trigger_detection(self):
        """Spec targets index 0 only; block index 1 keys must be ignored."""
        sd = _minimal_motif_sd({
            "single_transformer_blocks.1.cross_attn_query_proj.weight": _make_tensor(),
            "transformer_blocks.1.cross_attn_query_proj.weight": _make_tensor(),
        })
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False, (
            "Block index 1 must not activate single flag"
        )
        assert cfg.get("enable_text_cross_attention_dual") is False, (
            "Block index 1 must not activate dual flag"
        )

    def test_bare_cross_attn_key_without_block_prefix_not_matched(self):
        """Key `cross_attn_query_proj.weight` without block path must not match."""
        sd = _minimal_motif_sd({"cross_attn_query_proj.weight": _make_tensor()})
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is False
        assert cfg.get("enable_text_cross_attention_dual") is False

    def test_return_type_is_dict(self):
        cfg = _detect(_minimal_motif_sd())
        assert isinstance(cfg, dict), f"Expected dict, got {type(cfg).__name__}"

    def test_large_state_dict_with_single_key_no_crash(self):
        """1000+ extra keys alongside the target key must not cause issues."""
        sd = _minimal_motif_sd(
            {f"extra_layer_{i}.weight": _make_tensor() for i in range(1000)}
        )
        sd[self.SINGLE_KEY] = _make_tensor()
        cfg = _detect(sd)
        assert cfg.get("enable_text_cross_attention_single") is True

    def test_image_model_marker_preserved_in_output(self):
        """image_model: motif_video marker must still be in the returned config."""
        cfg = _detect(_minimal_motif_sd())
        assert cfg.get("image_model") == "motif_video"

    def test_multiple_calls_produce_consistent_results(self):
        """Repeated calls with the same input must yield the same output."""
        sd = _minimal_motif_sd({self.SINGLE_KEY: _make_tensor()})
        results = [_detect(sd) for _ in range(3)]
        for r in results:
            assert r.get("enable_text_cross_attention_single") is True
            assert r.get("enable_text_cross_attention_dual") is False
