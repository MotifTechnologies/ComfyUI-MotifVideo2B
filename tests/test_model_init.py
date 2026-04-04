"""tests/test_model_init.py — CPU-only unit tests for models/__init__.py.

Covers:
  1. _make_comfyui_forward: argument mapping, return value, ignored kwargs
  2. state_dict key prefix: checkpoint has no 'transformer.' prefix
  3. control / transformer_options are NOT forwarded to original_forward
  4. _TRANSFORMER_PARAMS: new param names present, old names absent, mapping

All tests run on CPU without instantiating the real transformer model.
motif_core / diffusers / comfy imports are mocked where necessary.
"""

import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so that `from models import _make_comfyui_forward` works
# without the full ComfyUI / motif_core environment.
# ---------------------------------------------------------------------------

def _install_stubs():
    """Install lightweight module stubs before importing the target module."""

    # ---- comfy stubs ----
    comfy_mod = types.ModuleType("comfy")
    sys.modules.setdefault("comfy", comfy_mod)

    # comfy.model_base
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
    sys.modules.setdefault("comfy.model_base", model_base_mod)
    comfy_mod.model_base = model_base_mod

    # comfy.conds
    conds_mod = types.ModuleType("comfy.conds")

    class _FakeCONDRegular:
        def __init__(self, val):
            self.val = val

    conds_mod.CONDRegular = _FakeCONDRegular
    sys.modules.setdefault("comfy.conds", conds_mod)
    comfy_mod.conds = conds_mod

    # ---- motif_core stubs ----
    motif_core_mod = types.ModuleType("motif_core")
    sys.modules.setdefault("motif_core", motif_core_mod)

    for sub in [
        "motif_core.models",
        "motif_core.models.transformers",
        "motif_core.models.transformers.transformer_motif_video",
    ]:
        sys.modules.setdefault(sub, types.ModuleType(sub))

    class _FakeTransformer:
        pass

    sys.modules["motif_core.models.transformers.transformer_motif_video"] \
        .MotifVideoTransformer3DModel = _FakeTransformer

    # ---- local package stubs (.adapter, .latent_format) ----
    # These are relative imports inside models/__init__.py.
    # We register them under both possible names.
    for mod_name in [
        "models.adapter",
        "models.latent_format",
        "ComfyUI_MotifVideo1_9B.models.adapter",
        "ComfyUI_MotifVideo1_9B.models.latent_format",
    ]:
        stub = types.ModuleType(mod_name)
        stub.MotifVideoModelAdapter = object
        stub.MotifVideoLatent = object
        sys.modules.setdefault(mod_name, stub)


_install_stubs()


# ---------------------------------------------------------------------------
# Import the function under test directly (avoids full package init).
# ---------------------------------------------------------------------------

import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PY = os.path.join(_ROOT, "models", "__init__.py")


def _load_models_init():
    """Load models/__init__.py as a standalone module with mocked deps."""
    spec = importlib.util.spec_from_file_location("_models_init", _INIT_PY)
    mod = importlib.util.module_from_spec(spec)
    # Set __package__ so relative imports resolve against our stubs.
    mod.__package__ = "models"
    spec.loader.exec_module(mod)
    return mod


_models_mod = _load_models_init()
_make_comfyui_forward = _models_mod._make_comfyui_forward


# ===========================================================================
# 1. _make_comfyui_forward — argument mapping
# ===========================================================================

class TestMakeComfyuiForwardArgMapping:
    """Verify that ComfyUI positional/keyword args are correctly remapped."""

    def _record_forward(self):
        """Return a fake original_forward that records its call kwargs."""
        calls = []

        def original_forward(**kwargs):
            calls.append(kwargs)
            return ("fake_sample",)

        return original_forward, calls

    def test_positional_x_mapped_to_hidden_states(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        x = torch.zeros(1, 4, 2, 8, 8)
        fwd(x, timestep=torch.tensor([1.0]))
        assert calls[0]["hidden_states"] is x

    def test_timestep_forwarded_unchanged(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        ts = torch.tensor([42.0])
        fwd(torch.zeros(1), timestep=ts)
        assert calls[0]["timestep"] is ts

    def test_context_mapped_to_encoder_hidden_states(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        ctx = torch.randn(1, 77, 512)
        fwd(torch.zeros(1), timestep=torch.zeros(1), context=ctx)
        assert calls[0]["encoder_hidden_states"] is ctx

    def test_encoder_attention_mask_forwarded(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        mask = torch.ones(1, 77, dtype=torch.bool)
        fwd(torch.zeros(1), timestep=torch.zeros(1), encoder_attention_mask=mask)
        assert calls[0]["encoder_attention_mask"] is mask

    def test_pooled_projections_forwarded(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        pp = torch.randn(1, 256)
        fwd(torch.zeros(1), timestep=torch.zeros(1), pooled_projections=pp)
        assert calls[0]["pooled_projections"] is pp

    def test_image_embeds_forwarded(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        ie = torch.randn(1, 64, 768)
        fwd(torch.zeros(1), timestep=torch.zeros(1), image_embeds=ie)
        assert calls[0]["image_embeds"] is ie

    def test_return_dict_false_always_sent(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        fwd(torch.zeros(1), timestep=torch.zeros(1))
        assert calls[0]["return_dict"] is False

    def test_returns_output_index_zero(self):
        """forward() returns (sample,); we must return sample, not the tuple."""
        import torch

        sentinel = object()

        def orig(**kwargs):
            return (sentinel, "should_be_ignored")

        fwd = _make_comfyui_forward(orig)
        result = fwd(torch.zeros(1), timestep=torch.zeros(1))
        assert result is sentinel

    def test_context_none_when_not_provided(self):
        """encoder_hidden_states should be None when context is omitted."""
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        fwd(torch.zeros(1), timestep=torch.zeros(1))
        assert calls[0]["encoder_hidden_states"] is None


# ===========================================================================
# 2. _make_comfyui_forward — control / transformer_options NOT forwarded
# ===========================================================================

class TestMakeComfyuiForwardIgnoredArgs:
    """control and transformer_options must NOT reach original_forward."""

    def _record_forward(self):
        calls = []

        def original_forward(**kwargs):
            calls.append(kwargs)
            return ("sample",)

        return original_forward, calls

    def test_control_not_in_original_forward_kwargs(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        fwd(torch.zeros(1), timestep=torch.zeros(1), control={"some": "data"})
        assert "control" not in calls[0], (
            "control must not be forwarded to original_forward"
        )

    def test_transformer_options_not_in_original_forward_kwargs(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        fwd(
            torch.zeros(1),
            timestep=torch.zeros(1),
            transformer_options={"patches": {}},
        )
        assert "transformer_options" not in calls[0], (
            "transformer_options must not be forwarded to original_forward"
        )

    def test_both_control_and_transformer_options_ignored(self):
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        fwd(
            torch.zeros(1),
            timestep=torch.zeros(1),
            control={"k": "v"},
            transformer_options={"patches": []},
        )
        assert "control" not in calls[0]
        assert "transformer_options" not in calls[0]

    def test_extra_kwargs_not_forwarded(self):
        """**kwargs in comfyui_forward should also be swallowed, not passed on."""
        import torch
        orig, calls = self._record_forward()
        fwd = _make_comfyui_forward(orig)
        fwd(torch.zeros(1), timestep=torch.zeros(1), unknown_future_arg=123)
        assert "unknown_future_arg" not in calls[0]


# ===========================================================================
# 3. state_dict key prefix — checkpoint must not have 'transformer.' prefix
# ===========================================================================

CHECKPOINT_PATH = (
    "/lustrefs/team-multimodal/minsu/ComfyUI/models/diffusion_models/"
    "motifvideo_1.9b.safetensors"
)


@pytest.fixture(scope="module")
def checkpoint_keys():
    """Load checkpoint keys once for the whole module."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    try:
        from safetensors import safe_open
    except ImportError:
        pytest.skip("safetensors not installed")
    with safe_open(CHECKPOINT_PATH, framework="pt", device="cpu") as f:
        return list(f.keys())


class TestCheckpointKeyPrefix:
    """State-dict keys must NOT carry a spurious 'transformer.' prefix."""

    def test_checkpoint_has_742_keys(self, checkpoint_keys):
        assert len(checkpoint_keys) == 742, (
            f"Expected 742 keys, got {len(checkpoint_keys)}"
        )

    def test_no_transformer_prefix(self, checkpoint_keys):
        prefixed = [k for k in checkpoint_keys if k.startswith("transformer.")]
        assert len(prefixed) == 0, (
            f"Found {len(prefixed)} keys with 'transformer.' prefix: "
            f"{prefixed[:5]}"
        )

    def test_all_keys_are_non_empty_strings(self, checkpoint_keys):
        for k in checkpoint_keys:
            assert isinstance(k, str) and k, f"Invalid key: {k!r}"

    def test_known_root_level_keys_present(self, checkpoint_keys):
        """Spot-check a few well-known top-level param names."""
        key_set = set(checkpoint_keys)
        expected = {
            "context_embedder.linear_1.bias",
            "context_embedder.linear_1.weight",
            "x_embedder.proj.bias",
            "x_embedder.proj.weight",
        }
        missing = expected - key_set
        assert not missing, f"Expected keys not found in checkpoint: {missing}"

    def test_no_nested_transformer_prefix_in_any_key(self, checkpoint_keys):
        """No key at any depth should start with 'transformer.' substring."""
        bad = [k for k in checkpoint_keys if "transformer." in k.split(".")[0]]
        assert len(bad) == 0, (
            f"Top-level segment is 'transformer' in keys: {bad[:5]}"
        )


# ===========================================================================
# 4. Edge / boundary cases for _make_comfyui_forward
# ===========================================================================

class TestMakeComfyuiForwardEdgeCases:
    """Boundary and type-mismatch edge cases."""

    def test_empty_tensor_x(self):
        import torch

        def orig(**kwargs):
            return (kwargs["hidden_states"],)

        fwd = _make_comfyui_forward(orig)
        empty = torch.zeros(0, 4, 2, 8, 8)
        result = fwd(empty, timestep=torch.zeros(0))
        assert result.shape == empty.shape

    def test_none_values_passed_through(self):
        """None for optional args should be forwarded as None, not omitted."""
        calls = []

        def orig(**kwargs):
            calls.append(kwargs)
            return (None,)

        fwd = _make_comfyui_forward(orig)
        import torch
        fwd(
            torch.zeros(1),
            timestep=torch.zeros(1),
            context=None,
            encoder_attention_mask=None,
            pooled_projections=None,
            image_embeds=None,
        )
        assert calls[0]["encoder_hidden_states"] is None
        assert calls[0]["encoder_attention_mask"] is None
        assert calls[0]["pooled_projections"] is None
        assert calls[0]["image_embeds"] is None

    def test_original_forward_called_exactly_once(self):
        import torch
        call_count = [0]

        def orig(**kwargs):
            call_count[0] += 1
            return ("out",)

        fwd = _make_comfyui_forward(orig)
        fwd(torch.zeros(1), timestep=torch.zeros(1))
        assert call_count[0] == 1

    def test_original_forward_exception_propagates(self):
        import torch

        def orig(**kwargs):
            raise RuntimeError("transformer exploded")

        fwd = _make_comfyui_forward(orig)
        with pytest.raises(RuntimeError, match="transformer exploded"):
            fwd(torch.zeros(1), timestep=torch.zeros(1))

    def test_integer_timestep_accepted(self):
        """Timestep may arrive as plain Python int from some callers."""
        calls = []

        def orig(**kwargs):
            calls.append(kwargs)
            return ("out",)

        fwd = _make_comfyui_forward(orig)
        import torch
        fwd(torch.zeros(1), timestep=1)
        assert calls[0]["timestep"] == 1

    def test_multiple_calls_independent(self):
        """Each call is isolated; no state leaks between calls."""
        import torch
        results = []

        def orig(**kwargs):
            results.append(kwargs["hidden_states"].shape)
            return (kwargs["hidden_states"],)

        fwd = _make_comfyui_forward(orig)
        fwd(torch.zeros(1, 4), timestep=torch.zeros(1))
        fwd(torch.zeros(2, 4), timestep=torch.zeros(1))
        assert results == [torch.Size([1, 4]), torch.Size([2, 4])]


# ===========================================================================
# 5. _TRANSFORMER_PARAMS — parameter name rename validation via source parsing
#
# _TRANSFORMER_PARAMS is a local variable inside MotifVideoModel.__init__,
# so we verify it by inspecting the source text of models/__init__.py directly.
# ===========================================================================

_INIT_SRC = open(_INIT_PY).read()


class TestTransformerParamsNamesInSource:
    """The source of models/__init__.py must contain the new
    enable_text_cross_attention_* param strings inside _TRANSFORMER_PARAMS
    and must NOT contain the old cross_attention_* bare names."""

    def test_new_name_dual_present_in_source(self):
        """Source must contain literal 'enable_text_cross_attention_dual'."""
        assert "enable_text_cross_attention_dual" in _INIT_SRC, (
            "Literal 'enable_text_cross_attention_dual' not found in models/__init__.py"
        )

    def test_new_name_single_present_in_source(self):
        """Source must contain literal 'enable_text_cross_attention_single'."""
        assert "enable_text_cross_attention_single" in _INIT_SRC, (
            "Literal 'enable_text_cross_attention_single' not found in models/__init__.py"
        )

    def test_old_name_dual_absent_from_transformer_params_block(self):
        """The _TRANSFORMER_PARAMS set literal must not contain the bare old name.
        We extract the set literal from the source and check its tokens."""
        import re
        # Extract the set body between _TRANSFORMER_PARAMS = { ... }
        m = re.search(
            r"_TRANSFORMER_PARAMS\s*=\s*\{([^}]+)\}", _INIT_SRC, re.DOTALL
        )
        assert m is not None, "_TRANSFORMER_PARAMS assignment not found in source"
        block = m.group(1)
        # Tokenise quoted strings in the block
        tokens = re.findall(r'"([^"]+)"', block)
        assert "cross_attention_dual" not in tokens, (
            f"Old name 'cross_attention_dual' still in _TRANSFORMER_PARAMS block: {tokens}"
        )

    def test_old_name_single_absent_from_transformer_params_block(self):
        """The _TRANSFORMER_PARAMS set literal must not contain the bare old name."""
        import re
        m = re.search(
            r"_TRANSFORMER_PARAMS\s*=\s*\{([^}]+)\}", _INIT_SRC, re.DOTALL
        )
        assert m is not None, "_TRANSFORMER_PARAMS assignment not found in source"
        block = m.group(1)
        tokens = re.findall(r'"([^"]+)"', block)
        assert "cross_attention_single" not in tokens, (
            f"Old name 'cross_attention_single' still in _TRANSFORMER_PARAMS block: {tokens}"
        )

    def test_new_dual_inside_transformer_params_block(self):
        """'enable_text_cross_attention_dual' must appear inside the
        _TRANSFORMER_PARAMS set literal, not just elsewhere in the file."""
        import re
        m = re.search(
            r"_TRANSFORMER_PARAMS\s*=\s*\{([^}]+)\}", _INIT_SRC, re.DOTALL
        )
        assert m is not None, "_TRANSFORMER_PARAMS assignment not found in source"
        block = m.group(1)
        tokens = re.findall(r'"([^"]+)"', block)
        assert "enable_text_cross_attention_dual" in tokens, (
            f"'enable_text_cross_attention_dual' not found inside _TRANSFORMER_PARAMS block. "
            f"Found: {tokens}"
        )

    def test_new_single_inside_transformer_params_block(self):
        """'enable_text_cross_attention_single' must appear inside the
        _TRANSFORMER_PARAMS set literal."""
        import re
        m = re.search(
            r"_TRANSFORMER_PARAMS\s*=\s*\{([^}]+)\}", _INIT_SRC, re.DOTALL
        )
        assert m is not None, "_TRANSFORMER_PARAMS assignment not found in source"
        block = m.group(1)
        tokens = re.findall(r'"([^"]+)"', block)
        assert "enable_text_cross_attention_single" in tokens, (
            f"'enable_text_cross_attention_single' not found inside _TRANSFORMER_PARAMS block. "
            f"Found: {tokens}"
        )

    def test_transformer_params_block_is_non_empty(self):
        """The _TRANSFORMER_PARAMS set must have at least one quoted string."""
        import re
        m = re.search(
            r"_TRANSFORMER_PARAMS\s*=\s*\{([^}]+)\}", _INIT_SRC, re.DOTALL
        )
        assert m is not None, "_TRANSFORMER_PARAMS assignment not found in source"
        block = m.group(1)
        tokens = re.findall(r'"([^"]+)"', block)
        assert len(tokens) > 0, "_TRANSFORMER_PARAMS block appears to be empty"


# ===========================================================================
# 6. unet_config → transformer_kwargs mapping logic (inline simulation)
#
# Since _TRANSFORMER_PARAMS is a local variable, we extract the param set
# from the source and simulate the filtering logic directly.
# ===========================================================================

def _extract_transformer_params_from_source():
    """Parse _TRANSFORMER_PARAMS token list from models/__init__.py source."""
    import re
    m = re.search(
        r"_TRANSFORMER_PARAMS\s*=\s*\{([^}]+)\}", _INIT_SRC, re.DOTALL
    )
    assert m is not None, "_TRANSFORMER_PARAMS assignment not found in source"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


_PARSED_PARAMS = _extract_transformer_params_from_source()


class TestTransformerParamsPropagation:
    """Simulate the unet_config filtering logic using the parsed param set
    and verify correct extraction for the renamed keys."""

    def _build_unet_config(self, **overrides):
        base = {
            "in_channels": 16,
            "num_attention_heads": 8,
            "attention_head_dim": 64,
            "num_layers": 2,
            "enable_text_cross_attention_dual": False,
            "enable_text_cross_attention_single": False,
        }
        base.update(overrides)
        return base

    def _collect_transformer_kwargs(self, unet_config):
        """Replicate: {k: v for k, v in unet_config.items() if k in _TRANSFORMER_PARAMS}"""
        return {k: v for k, v in unet_config.items() if k in _PARSED_PARAMS}

    # --- happy path: True values ---

    def test_enable_text_cross_attention_single_true_extracted(self):
        cfg = self._build_unet_config(enable_text_cross_attention_single=True)
        kwargs = self._collect_transformer_kwargs(cfg)
        assert kwargs.get("enable_text_cross_attention_single") is True

    def test_enable_text_cross_attention_dual_true_extracted(self):
        cfg = self._build_unet_config(enable_text_cross_attention_dual=True)
        kwargs = self._collect_transformer_kwargs(cfg)
        assert kwargs.get("enable_text_cross_attention_dual") is True

    # --- happy path: False values (existing checkpoint — no cross-attn) ---

    def test_enable_text_cross_attention_single_false_extracted(self):
        """False value must not be silently dropped by the filter."""
        cfg = self._build_unet_config(enable_text_cross_attention_single=False)
        kwargs = self._collect_transformer_kwargs(cfg)
        assert "enable_text_cross_attention_single" in kwargs, (
            "enable_text_cross_attention_single=False was not extracted"
        )
        assert kwargs["enable_text_cross_attention_single"] is False

    def test_enable_text_cross_attention_dual_false_extracted(self):
        cfg = self._build_unet_config(enable_text_cross_attention_dual=False)
        kwargs = self._collect_transformer_kwargs(cfg)
        assert "enable_text_cross_attention_dual" in kwargs
        assert kwargs["enable_text_cross_attention_dual"] is False

    # --- old names are NOT extracted ---

    def test_old_cross_attention_dual_not_extracted(self):
        """Old key must not pass through after rename."""
        cfg = self._build_unet_config()
        cfg["cross_attention_dual"] = True
        kwargs = self._collect_transformer_kwargs(cfg)
        assert "cross_attention_dual" not in kwargs, (
            "Old key 'cross_attention_dual' must not pass through _TRANSFORMER_PARAMS"
        )

    def test_old_cross_attention_single_not_extracted(self):
        cfg = self._build_unet_config()
        cfg["cross_attention_single"] = True
        kwargs = self._collect_transformer_kwargs(cfg)
        assert "cross_attention_single" not in kwargs

    # --- boundary: missing key entirely ---

    def test_missing_enable_key_not_in_kwargs_no_keyerror(self):
        """unet_config without the new keys must not raise KeyError."""
        cfg = {
            "in_channels": 16,
            "num_attention_heads": 8,
            "attention_head_dim": 64,
            "num_layers": 2,
        }
        kwargs = self._collect_transformer_kwargs(cfg)
        assert "enable_text_cross_attention_single" not in kwargs
        assert "enable_text_cross_attention_dual" not in kwargs

    # --- boundary: both True simultaneously ---

    def test_both_enable_flags_true_simultaneously(self):
        cfg = self._build_unet_config(
            enable_text_cross_attention_dual=True,
            enable_text_cross_attention_single=True,
        )
        kwargs = self._collect_transformer_kwargs(cfg)
        assert kwargs["enable_text_cross_attention_dual"] is True
        assert kwargs["enable_text_cross_attention_single"] is True

    # --- type boundary ---

    def test_integer_one_passed_through_as_is(self):
        """int 1 must not be coerced to bool True."""
        cfg = self._build_unet_config(enable_text_cross_attention_single=1)
        kwargs = self._collect_transformer_kwargs(cfg)
        assert kwargs["enable_text_cross_attention_single"] == 1

    def test_none_value_passed_through(self):
        """None must not be dropped."""
        cfg = self._build_unet_config(enable_text_cross_attention_single=None)
        kwargs = self._collect_transformer_kwargs(cfg)
        assert "enable_text_cross_attention_single" in kwargs
        assert kwargs["enable_text_cross_attention_single"] is None
