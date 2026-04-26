"""
Unit tests for T5Gemma2Encoder native implementation (checklist item 7).

Tests: state_dict key structure, forward smoke, comfy_cast_weights injection.

Blind-test principle: item-7 diff was NOT read before writing these tests.
Tests are derived from the spec / checklist only.

Environment: must be run with ComfyUI venv python:
  /lustrefs/team-multimodal/minsu/ComfyUI/.venv/bin/python \\
      -m pytest tests/text_encoders/test_t5_gemma2_native.py -v

Layout: <comfy_root>/custom_nodes/<repo>/tests/text_encoders/<this_file>
"""

from __future__ import annotations

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
import torch.nn as nn  # noqa: E402
import comfy.ops  # noqa: E402

from text_encoders.t5_gemma2_config import T5_GEMMA2_CONFIG  # noqa: E402
from text_encoders.t5_gemma2_native import T5Gemma2Encoder  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cfg():
    """Return a *reduced* T5Gemma2EncoderConfig for fast unit tests.

    Production config (34 layers, hidden=2560, intermediate=10240, vocab=262144)
    is far too large for routine CPU unit tests — building the encoder once
    allocates multi-GB and cloning state_dict via torch.zeros_like compounds
    that across every test. We override the dimensions that drive parameter
    count to small values while keeping the *structural* surface (layer
    types pattern, RoPE per-type config, GQA grouping, EOI token offset)
    intact so the tests still exercise the same code paths.

    Numerical parity vs HF and real checkpoint loading are validated in
    items 8–9 with the production config.
    """
    from copy import deepcopy
    from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2EncoderConfig

    cfg_dict = deepcopy(T5_GEMMA2_CONFIG)
    text_cfg = cfg_dict["text_config"]
    text_cfg["num_hidden_layers"] = 4  # 4 layers exercises sliding+full mix
    text_cfg["hidden_size"] = 64
    text_cfg["intermediate_size"] = 128
    text_cfg["num_attention_heads"] = 4
    text_cfg["num_key_value_heads"] = 2  # GQA 2:1 preserved
    text_cfg["head_dim"] = 16  # hidden_size / num_attention_heads
    text_cfg["query_pre_attn_scalar"] = 16
    text_cfg["vocab_size"] = 256
    text_cfg["max_position_embeddings"] = 64
    text_cfg["sliding_window"] = 8
    # Layer types: alternate sliding/full for compact mix. 4 layers total.
    text_cfg["layer_types"] = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]
    # boi/eoi/image token indices need to fit in the reduced vocab.
    text_cfg["bos_token_id"] = 2
    text_cfg["eos_token_id"] = 1
    text_cfg["pad_token_id"] = 0
    cfg_dict["boi_token_index"] = 250
    cfg_dict["eoi_token_index"] = 251
    cfg_dict["image_token_index"] = 252
    cfg_dict["vocab_size"] = 256

    # Vision tower is unused by the native encoder but the config object
    # still constructs SiglipVisionConfig — shrink to keep the cost low.
    vision_cfg = cfg_dict["vision_config"]
    vision_cfg["hidden_size"] = 32
    vision_cfg["intermediate_size"] = 64
    vision_cfg["num_hidden_layers"] = 1
    vision_cfg["num_attention_heads"] = 2
    vision_cfg["image_size"] = 32
    vision_cfg["patch_size"] = 8
    vision_cfg["vocab_size"] = 256

    return T5Gemma2EncoderConfig(**cfg_dict)


# Number of layers in the reduced fixture above — used to derive the
# expected state_dict key count (13 per layer + 3 module-level keys).
_REDUCED_NUM_LAYERS = 4
_REDUCED_TOTAL_KEYS = 13 * _REDUCED_NUM_LAYERS + 3


# ---------------------------------------------------------------------------
# Required test cases (exact names required by verify command)
# ---------------------------------------------------------------------------

def test_state_dict_keys():
    """
    T5Gemma2Encoder(cfg).state_dict() must expose 13*N+3 keys (N layers),
    covering all text_model tensors with correct naming convention.

    Uses the reduced fixture (4 layers) for speed — production config (34
    layers, 445 keys) is verified by item 8 wrapper compatibility tests
    and item 9 parity tests.

    Verifies:
      - Top-level anchors: embed_tokens.weight, embed_tokens.eoi_embedding,
        norm.weight
      - Per-layer keys (13 per layer): mlp.{gate,up,down}_proj.weight,
        self_attn.{q,k,v,o}_proj.weight, self_attn.{q,k}_norm.weight,
        pre/post_self_attn_layernorm.weight, pre/post_feedforward_layernorm.weight
      - Absence of vision_tower / multi_modal_projector / inv_freq / embed_scale keys
      - load_state_dict(strict=True) with correctly-shaped dummy tensors: 0 missing, 0 unexpected
    """
    cfg = _make_cfg()
    encoder = T5Gemma2Encoder(cfg)
    sd = encoder.state_dict()
    keys = set(sd.keys())

    # ---- total key count ----
    assert len(keys) == _REDUCED_TOTAL_KEYS, (
        f"Expected {_REDUCED_TOTAL_KEYS} state_dict keys, got {len(keys)}.\n"
        f"Keys: {sorted(keys)}"
    )

    # ---- top-level anchors ----
    for required in (
        "text_model.embed_tokens.weight",
        "text_model.embed_tokens.eoi_embedding",
        "text_model.norm.weight",
    ):
        assert required in keys, f"Required key missing: {required}"

    # ---- per-layer keys for all 34 layers ----
    PER_LAYER_SUFFIXES = [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "pre_self_attn_layernorm.weight",
        "post_self_attn_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
    ]
    for layer_idx in range(_REDUCED_NUM_LAYERS):
        for suffix in PER_LAYER_SUFFIXES:
            key = f"text_model.layers.{layer_idx}.{suffix}"
            assert key in keys, (
                f"Layer {layer_idx} missing expected key: {key}"
            )

    # ---- keys that must NOT be present ----
    for forbidden_prefix in (
        "vision_tower",
        "multi_modal_projector",
    ):
        offenders = [k for k in keys if k.startswith(forbidden_prefix)]
        assert not offenders, (
            f"Native encoder must not expose '{forbidden_prefix}' keys: {offenders}"
        )

    for forbidden_suffix in ("inv_freq", "embed_scale"):
        offenders = [k for k in keys if k.endswith(forbidden_suffix)]
        assert not offenders, (
            f"key ending with '{forbidden_suffix}' must not be in state_dict "
            f"(persistent=False): {offenders}"
        )

    # ---- load_state_dict strict=True with dummy tensors ----
    dummy_sd = {k: torch.zeros_like(v) for k, v in sd.items()}
    missing_keys, unexpected_keys = encoder.load_state_dict(dummy_sd, strict=True)
    assert len(missing_keys) == 0, (
        f"load_state_dict(strict=True) reported missing keys: {missing_keys}"
    )
    assert len(unexpected_keys) == 0, (
        f"load_state_dict(strict=True) reported unexpected keys: {unexpected_keys}"
    )


def test_forward_smoke():
    """
    T5Gemma2Encoder forward with (B=1, S=16) input must:
      - return output with .last_hidden_state of shape (1, 16, cfg.text_config.hidden_size)
      - produce no NaN or Inf

    Init strategy: fill all parameters with a small positive constant (0.01)
    rather than orthogonal init (which performs per-parameter SVD — prohibitively
    slow on CPU for 34 layers × large weight matrices). Small-constant init
    avoids zero rows that trigger NaN in attention softmax while running quickly.
    """
    cfg = _make_cfg()
    encoder = T5Gemma2Encoder(cfg)

    # Small constant init: avoids zero rows (NaN risk) without the SVD cost
    # of orthogonal_ on large weight matrices
    torch.manual_seed(2025)
    with torch.no_grad():
        for p in encoder.parameters():
            p.fill_(0.01)

    encoder.eval()

    input_ids = torch.tensor([[2, 100, 200, 1] + [0] * 12], dtype=torch.long)  # (1, 16)
    attention_mask = torch.ones(1, 16, dtype=torch.long)

    torch.manual_seed(2025)
    with torch.no_grad():
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)

    # Shape check
    assert hasattr(out, "last_hidden_state"), (
        "Output must expose .last_hidden_state attribute"
    )
    assert out.last_hidden_state.shape == (1, 16, cfg.text_config.hidden_size), (
        f"Expected last_hidden_state.shape == (1, 16, cfg.text_config.hidden_size), got {out.last_hidden_state.shape}"
    )

    # Numerical validity
    assert not torch.isnan(out.last_hidden_state).any(), (
        "NaN detected in last_hidden_state"
    )
    assert not torch.isinf(out.last_hidden_state).any(), (
        "Inf detected in last_hidden_state"
    )


def test_comfy_cast_weights():
    """
    With operations=comfy.ops.manual_cast, every nn.Linear and nn.Embedding
    submodule inside T5Gemma2Encoder must have comfy_cast_weights == True.

    Additionally verifies that the default (operations=None, i.e.
    disable_weight_init) path uses disable_weight_init.Linear / Embedding
    rather than bare nn.Linear / nn.Embedding.
    """
    cfg = _make_cfg()

    # ---- manual_cast: comfy_cast_weights must be True on every leaf ----
    encoder_mc = T5Gemma2Encoder(cfg, operations=comfy.ops.manual_cast)

    missing_cast = []
    for name, module in encoder_mc.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            val = getattr(module, "comfy_cast_weights", None)
            if val is not True:
                missing_cast.append(
                    f"{name} ({type(module).__name__}): comfy_cast_weights={val!r}"
                )

    assert not missing_cast, (
        "The following Linear/Embedding modules are missing comfy_cast_weights=True "
        "when operations=manual_cast:\n" + "\n".join(missing_cast)
    )

    # ---- default path: no bare nn.Linear / nn.Embedding instances ----
    encoder_default = T5Gemma2Encoder(cfg)
    raw_nn_modules = []
    for name, module in encoder_default.named_modules():
        if type(module) in (nn.Linear, nn.Embedding):
            raw_nn_modules.append(f"{name} ({type(module).__name__})")

    assert not raw_nn_modules, (
        "Default (disable_weight_init) build must not produce bare nn.Linear/Embedding. "
        "Found:\n" + "\n".join(raw_nn_modules)
    )


# ---------------------------------------------------------------------------
# Item-7 verify-traceable extensions
# ---------------------------------------------------------------------------
# Each remaining test directly extends one of the three checklist requirements
# above (state_dict keys / forward smoke / comfy_cast_weights). Tracing comment
# on each test names the requirement it strengthens.


def test_load_state_dict_with_extra_vision_keys_strict_false():
    """
    Strengthens *test_state_dict_keys*: the verify above checks that the
    native encoder's state_dict has no vision keys; this test additionally
    proves that an HF-style state_dict carrying vision_tower /
    multi_modal_projector keys still loads cleanly with strict=False (real
    checkpoints from `MotifVideoT5Gemma2Model.load_sd()` arrive in this form).

    Verifies:
      - text_model.* keys are NOT in missing_keys
      - vision_tower.* / multi_modal_projector.* land in unexpected_keys
    """
    cfg = _make_cfg()
    encoder = T5Gemma2Encoder(cfg)
    sd = encoder.state_dict()

    # Add fake HF-style vision keys that the native encoder does not declare
    extra_keys = {
        "vision_tower.encoder.layers.0.weight": torch.zeros(8, 8),
        "multi_modal_projector.linear_1.weight": torch.zeros(8, 8),
        "language_model.embed_tokens.weight": torch.zeros(8, 8),
    }
    hf_style_sd = {**sd, **extra_keys}

    missing_keys, unexpected_keys = encoder.load_state_dict(hf_style_sd, strict=False)

    # No text_model keys should be missing (all were in the base sd)
    text_model_missing = [k for k in missing_keys if k.startswith("text_model.")]
    assert not text_model_missing, (
        f"text_model.* keys unexpectedly missing after strict=False load: {text_model_missing}"
    )

    # All extra keys should appear in unexpected_keys
    for extra_key in extra_keys:
        assert extra_key in unexpected_keys, (
            f"Expected '{extra_key}' in unexpected_keys, but it was not: {unexpected_keys}"
        )


def test_forward_without_attention_mask():
    """
    Strengthens *test_forward_smoke*: that test always passes a mask. This
    verifies the mask=None branch (`_build_bidirectional_mask` returns None
    → SDPA full-attend) still produces the expected shape.
    """
    cfg = _make_cfg()
    encoder = T5Gemma2Encoder(cfg)
    torch.manual_seed(42)
    with torch.no_grad():
        for p in encoder.parameters():
            p.fill_(0.01)
    encoder.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out = encoder(input_ids=input_ids)  # no attention_mask

    assert out.last_hidden_state.shape == (1, 4, cfg.text_config.hidden_size), (
        f"Shape mismatch without attention_mask: {out.last_hidden_state.shape}"
    )


def test_forward_pixel_values_raises_not_implemented():
    """
    Strengthens *test_forward_smoke*: forward smoke covers pixel_values=None.
    This verifies the documented contract that non-None pixel_values raises
    NotImplementedError instead of silently degrading to a text-only encode.
    """
    cfg = _make_cfg()
    encoder = T5Gemma2Encoder(cfg)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    dummy_pixels = torch.randn(1, 3, 224, 224)

    with pytest.raises(NotImplementedError):
        encoder(input_ids=input_ids, pixel_values=dummy_pixels)


def test_state_dict_key_count_per_layer_is_13():
    """
    Strengthens *test_state_dict_keys*: that test enumerates per-layer keys
    by name; this one cross-checks the *count* per layer is exactly 13 so an
    accidental bias re-introduction or shape drift would be caught even if a
    key happens to land in the expected naming pattern by coincidence.

    The reduced fixture has 4 layers; production has 34. Item 8 (wrapper
    compatibility) re-validates against the production-config 445-key shape.
    """
    cfg = _make_cfg()
    encoder = T5Gemma2Encoder(cfg)
    sd = encoder.state_dict()
    keys = list(sd.keys())

    for layer_idx in range(_REDUCED_NUM_LAYERS):
        prefix = f"text_model.layers.{layer_idx}."
        layer_keys = [k for k in keys if k.startswith(prefix)]
        assert len(layer_keys) == 13, (
            f"Layer {layer_idx} expected 13 keys, got {len(layer_keys)}: {layer_keys}"
        )
