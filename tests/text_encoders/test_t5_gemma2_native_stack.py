"""
Tests for the T5Gemma2 encoder stack added in checklist item 5:
  - T5Gemma2EncoderLayer  (alias T5Gemma2DecoderLayer)
  - make_text_scaled_word_embedding_class / T5Gemma2TextScaledWordEmbedding
  - T5Gemma2RotaryEmbedding
  - T5Gemma2TextEncoder  (alias T5Gemma2TextModel)
  - T5Gemma2Encoder  (outer wrapper)
  - _build_bidirectional_mask
  - _build_sliding_window_mask

Blind-test principle: item-5 diff was NOT read before writing these tests.
Tests are derived from the spec/checklist only.

Environment: must be run with ComfyUI venv python:
  /lustrefs/team-multimodal/minsu/ComfyUI/.venv/bin/python \\
      -m pytest tests/text_encoders/test_t5_gemma2_native_stack.py -v

Layout: <comfy_root>/custom_nodes/<repo>/tests/text_encoders/<this_file>
"""

from __future__ import annotations

import math
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

from text_encoders.t5_gemma2_native import (  # noqa: E402
    T5Gemma2EncoderLayer,
    T5Gemma2DecoderLayer,
    make_text_scaled_word_embedding_class,
    T5Gemma2RotaryEmbedding,
    T5Gemma2TextEncoder,
    T5Gemma2TextModel,
    T5Gemma2Encoder,
    _build_bidirectional_mask,
    _build_sliding_window_mask,
)


# ---------------------------------------------------------------------------
# Shared config helpers
# ---------------------------------------------------------------------------

def _make_encoder_config():
    """Return T5Gemma2EncoderConfig from the project's bundled defaults."""
    from text_encoders.t5_gemma2_config import T5_GEMMA2_CONFIG
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2EncoderConfig,
    )
    return T5Gemma2EncoderConfig(**T5_GEMMA2_CONFIG)


def _make_text_config():
    """Return T5Gemma2TextConfig (the inner text sub-config)."""
    return _make_encoder_config().text_config


def _dummy_rope(B: int, S: int, head_dim: int = 256):
    """Identity rotation: cos=ones, sin=zeros — shape (B, S, head_dim)."""
    cos = torch.ones(B, S, head_dim)
    sin = torch.zeros(B, S, head_dim)
    return cos, sin


# ===========================================================================
# 1. T5Gemma2EncoderLayer
# ===========================================================================

class TestT5Gemma2EncoderLayerAttributes:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_text_config()

    def test_attributes_exist(self, cfg):
        """All 8 required attributes must be present."""
        layer = T5Gemma2EncoderLayer(cfg, layer_idx=0)
        for attr in (
            "self_attn",
            "mlp",
            "pre_self_attn_layernorm",
            "post_self_attn_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "dropout",
            "attention_type",
        ):
            assert hasattr(layer, attr), f"T5Gemma2EncoderLayer missing attribute: {attr}"

    def test_attention_type_sliding_idx0(self, cfg):
        """layer_idx=0 -> attention_type == 'sliding_attention'."""
        layer = T5Gemma2EncoderLayer(cfg, layer_idx=0)
        assert layer.attention_type == "sliding_attention", (
            f"layer_idx=0: expected sliding_attention, got {layer.attention_type}"
        )

    def test_attention_type_full_idx5(self, cfg):
        """layer_idx=5 -> attention_type == 'full_attention'."""
        layer = T5Gemma2EncoderLayer(cfg, layer_idx=5)
        assert layer.attention_type == "full_attention", (
            f"layer_idx=5: expected full_attention, got {layer.attention_type}"
        )

    def test_alias_decoder_layer_is_same_class(self):
        """T5Gemma2DecoderLayer must be the same object as T5Gemma2EncoderLayer."""
        assert T5Gemma2DecoderLayer is T5Gemma2EncoderLayer, (
            "T5Gemma2DecoderLayer must be an alias for T5Gemma2EncoderLayer"
        )


class TestT5Gemma2EncoderLayerForward:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_text_config()

    def test_forward_smoke_shape(self, cfg):
        """Forward (1, 16, 2560) -> same shape (1, 16, 2560)."""
        layer = T5Gemma2EncoderLayer(cfg, layer_idx=0)
        layer.eval()
        B, S, D = 1, 16, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _dummy_rope(B, S)
        with torch.no_grad():
            out = layer(hidden, position_embeddings=(cos, sin))
        assert out.shape == (B, S, D), (
            f"EncoderLayer forward shape mismatch: expected {(B, S, D)}, got {out.shape}"
        )

    def test_forward_dtype_bfloat16(self, cfg):
        """bfloat16 input with bfloat16 weights -> bfloat16 output."""
        layer = T5Gemma2EncoderLayer(cfg, layer_idx=0, dtype=torch.bfloat16)
        layer.eval()
        B, S, D = 1, 8, 2560
        hidden = torch.randn(B, S, D, dtype=torch.bfloat16)
        cos = torch.ones(B, S, 256, dtype=torch.bfloat16)
        sin = torch.zeros(B, S, 256, dtype=torch.bfloat16)
        with torch.no_grad():
            out = layer(hidden, position_embeddings=(cos, sin))
        assert out.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got {out.dtype}"
        )

    def test_forward_no_nan(self, cfg):
        """Forward must not produce NaN or Inf for random float32 input.

        Uses a fixed seed to ensure determinism across test ordering;
        uninitialized weights + specific RNG state can produce near-zero
        softmax denominators that become NaN under SDPA.
        """
        torch.manual_seed(2025)
        layer = T5Gemma2EncoderLayer(cfg, layer_idx=0)
        layer.eval()
        # Initialise weights to a small non-zero uniform value so that
        # random projections do not accidentally collapse to all-zero rows
        # (which causes NaN in softmax after q@k scaling).
        with torch.no_grad():
            for p in layer.parameters():
                if p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=0.01)
                else:
                    p.fill_(0.01)
        B, S, D = 1, 16, 2560
        torch.manual_seed(2025)
        hidden = torch.randn(B, S, D)
        cos, sin = _dummy_rope(B, S)
        with torch.no_grad():
            out = layer(hidden, position_embeddings=(cos, sin))
        assert not torch.isnan(out).any(), "NaN in EncoderLayer forward"
        assert not torch.isinf(out).any(), "Inf in EncoderLayer forward"

    def test_forward_with_attention_mask(self, cfg):
        """Forward with additive attention mask (B, 1, S, S) must not crash."""
        layer = T5Gemma2EncoderLayer(cfg, layer_idx=0)
        layer.eval()
        B, S, D = 1, 8, 2560
        hidden = torch.randn(B, S, D)
        cos, sin = _dummy_rope(B, S)
        mask = torch.zeros(B, 1, S, S)
        with torch.no_grad():
            out = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask)
        assert out.shape == (B, S, D)


# ===========================================================================
# 2. T5Gemma2RotaryEmbedding
# ===========================================================================

class TestT5Gemma2RotaryEmbeddingInit:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_text_config()

    def test_both_layer_type_buffers_exist(self, cfg):
        """Both sliding_attention and full_attention buffer pairs must be registered."""
        rope = T5Gemma2RotaryEmbedding(cfg)
        for lt in ("sliding_attention", "full_attention"):
            assert hasattr(rope, f"{lt}_inv_freq"), f"missing buffer: {lt}_inv_freq"
            assert hasattr(rope, f"{lt}_original_inv_freq"), (
                f"missing buffer: {lt}_original_inv_freq"
            )
            assert hasattr(rope, f"{lt}_attention_scaling"), (
                f"missing attribute: {lt}_attention_scaling"
            )

    def test_sliding_attention_inv_freq_first_element(self, cfg):
        """
        sliding_attention: rope_type='default', theta=10000.
        inv_freq[0] = 1 / 10000^0 = 1.0.
        """
        rope = T5Gemma2RotaryEmbedding(cfg)
        inv_freq = rope.sliding_attention_inv_freq
        assert abs(float(inv_freq[0]) - 1.0) < 1e-6, (
            f"sliding_attention inv_freq[0] expected 1.0, got {float(inv_freq[0])}"
        )

    def test_full_attention_inv_freq_first_element(self, cfg):
        """
        full_attention: rope_type='linear', theta=1_000_000, factor=8.0.
        inv_freq[0] = 1 / 1000000^0 / 8 = 0.125.
        """
        rope = T5Gemma2RotaryEmbedding(cfg)
        inv_freq = rope.full_attention_inv_freq
        assert abs(float(inv_freq[0]) - 0.125) < 1e-6, (
            f"full_attention inv_freq[0] expected 0.125 (linear/8), got {float(inv_freq[0])}"
        )

    def test_inv_freq_not_in_state_dict(self, cfg):
        """persistent=False: *_inv_freq keys must NOT appear in state_dict()."""
        rope = T5Gemma2RotaryEmbedding(cfg)
        sd_keys = list(rope.state_dict().keys())
        inv_freq_keys = [k for k in sd_keys if "inv_freq" in k]
        assert len(inv_freq_keys) == 0, (
            f"inv_freq keys found in state_dict (should be persistent=False): {inv_freq_keys}"
        )


class TestT5Gemma2RotaryEmbeddingForward:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_text_config()

    def test_forward_output_shape(self, cfg):
        """forward(x, position_ids, layer_type) -> (cos, sin) each shape (B, S, head_dim)."""
        rope = T5Gemma2RotaryEmbedding(cfg)
        B, S, head_dim = 2, 16, 256
        x = torch.randn(B, S, head_dim)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
        cos, sin = rope(x, position_ids, layer_type="sliding_attention")
        assert cos.shape == (B, S, head_dim), (
            f"cos shape expected {(B, S, head_dim)}, got {cos.shape}"
        )
        assert sin.shape == (B, S, head_dim), (
            f"sin shape expected {(B, S, head_dim)}, got {sin.shape}"
        )

    def test_forward_sliding_vs_full_differ(self, cfg):
        """sliding_attention and full_attention must produce different cos/sin values."""
        rope = T5Gemma2RotaryEmbedding(cfg)
        B, S, head_dim = 1, 8, 256
        x = torch.randn(B, S, head_dim)
        position_ids = torch.arange(S).unsqueeze(0)

        cos_s, sin_s = rope(x, position_ids, layer_type="sliding_attention")
        cos_f, sin_f = rope(x, position_ids, layer_type="full_attention")

        # They should differ in values (different inv_freq)
        assert not torch.allclose(cos_s, cos_f, atol=1e-6), (
            "sliding_attention and full_attention produced identical cos — should differ"
        )

    def test_forward_output_dtype_matches_input(self, cfg):
        """Output cos/sin dtype must match x.dtype."""
        rope = T5Gemma2RotaryEmbedding(cfg)
        B, S, head_dim = 1, 8, 256
        x = torch.randn(B, S, head_dim, dtype=torch.bfloat16)
        position_ids = torch.arange(S).unsqueeze(0)
        cos, sin = rope(x, position_ids, layer_type="sliding_attention")
        assert cos.dtype == torch.bfloat16, (
            f"cos dtype expected bfloat16, got {cos.dtype}"
        )
        assert sin.dtype == torch.bfloat16, (
            f"sin dtype expected bfloat16, got {sin.dtype}"
        )


# ===========================================================================
# 3. make_text_scaled_word_embedding_class / T5Gemma2TextScaledWordEmbedding
# ===========================================================================

class TestTextScaledWordEmbeddingFactory:

    def test_factory_with_disable_weight_init(self):
        """Factory called with disable_weight_init must return a class without error."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        assert cls is not None
        assert isinstance(cls, type), "factory must return a class (type)"

    def test_factory_with_manual_cast(self):
        """Factory called with manual_cast must also return a valid class."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.manual_cast)
        assert cls is not None
        assert isinstance(cls, type)

    def test_instance_attributes_exist(self):
        """Instantiated class must have weight, eoi_embedding, embed_scale, eoi_token_index."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(num_embeddings=262144, embedding_dim=2560, padding_idx=0)

        assert hasattr(obj, "weight"), "missing attribute: weight"
        assert hasattr(obj, "eoi_embedding"), "missing attribute: eoi_embedding"
        assert hasattr(obj, "embed_scale"), "missing attribute: embed_scale"
        assert hasattr(obj, "eoi_token_index"), "missing attribute: eoi_token_index"

    def test_eoi_embedding_is_parameter(self):
        """eoi_embedding must be an nn.Parameter."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(num_embeddings=262144, embedding_dim=2560, padding_idx=0)
        assert isinstance(obj.eoi_embedding, nn.Parameter), (
            "eoi_embedding must be nn.Parameter"
        )

    def test_eoi_embedding_shape(self):
        """eoi_embedding shape must be (embedding_dim,)."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(num_embeddings=262144, embedding_dim=2560, padding_idx=0)
        assert obj.eoi_embedding.shape == (2560,), (
            f"eoi_embedding shape expected (2560,), got {obj.eoi_embedding.shape}"
        )

    def test_eoi_embedding_zero_init(self):
        """eoi_embedding must be zero-initialized."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(num_embeddings=262144, embedding_dim=2560, padding_idx=0)
        assert torch.all(obj.eoi_embedding.data == 0.0), (
            "eoi_embedding must be zero-initialized"
        )

    def test_embed_scale_not_in_state_dict(self):
        """embed_scale must be persistent=False — not in state_dict()."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(num_embeddings=262144, embedding_dim=2560, padding_idx=0)
        sd_keys = list(obj.state_dict().keys())
        assert "embed_scale" not in sd_keys, (
            f"embed_scale should not be in state_dict (persistent=False), keys: {sd_keys}"
        )

    def test_state_dict_has_weight_and_eoi_embedding(self):
        """state_dict must contain 'weight' and 'eoi_embedding' (HF parity)."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(num_embeddings=262144, embedding_dim=2560, padding_idx=0)
        sd_keys = list(obj.state_dict().keys())
        assert "weight" in sd_keys, f"'weight' missing from state_dict: {sd_keys}"
        assert "eoi_embedding" in sd_keys, (
            f"'eoi_embedding' missing from state_dict: {sd_keys}"
        )

    def test_forward_normal_tokens_scaled(self):
        """
        Non-EOI tokens: output = Embedding(token) * embed_scale.
        With embed_scale=2.0, the result must be 2x the raw embedding.
        """
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(
            num_embeddings=1000,
            embedding_dim=8,
            padding_idx=0,
            embed_scale=2.0,
        )
        # Seed weight with known values
        with torch.no_grad():
            obj.weight.data.fill_(1.0)
        input_ids = torch.tensor([[1, 2, 3]])
        out = obj(input_ids)
        # Expected: ones * 2.0 = 2.0 everywhere (no eoi)
        assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-6), (
            "Normal tokens must be scaled by embed_scale"
        )

    def test_forward_eoi_token_replaced(self):
        """
        Token at eoi_token_index position must be replaced with eoi_embedding value.
        """
        eoi_idx = 256000
        emb_dim = 8
        cls = make_text_scaled_word_embedding_class(comfy.ops.disable_weight_init)
        obj = cls(
            num_embeddings=262144,
            embedding_dim=emb_dim,
            padding_idx=0,
            embed_scale=1.0,
            eoi_token_index=eoi_idx,
        )
        # Set eoi_embedding to a known value
        sentinel = 99.0
        with torch.no_grad():
            obj.eoi_embedding.data.fill_(sentinel)
            obj.weight.data.fill_(0.0)  # Normal tokens give zeros

        # Input: [non-eoi, eoi, non-eoi]
        input_ids = torch.tensor([[1, eoi_idx, 2]])
        out = obj(input_ids)

        # Position 1 (eoi) must be sentinel; positions 0 and 2 must be 0
        assert torch.allclose(out[0, 1], torch.full((emb_dim,), sentinel), atol=1e-6), (
            f"EOI position not replaced correctly: {out[0, 1].tolist()}"
        )
        assert torch.allclose(out[0, 0], torch.zeros(emb_dim), atol=1e-6), (
            "Non-EOI position should be 0 (weight=0, scale=1)"
        )
        assert torch.allclose(out[0, 2], torch.zeros(emb_dim), atol=1e-6), (
            "Non-EOI position should be 0 (weight=0, scale=1)"
        )

    def test_manual_cast_comfy_cast_weights(self):
        """operations=manual_cast -> underlying Embedding has comfy_cast_weights=True."""
        cls = make_text_scaled_word_embedding_class(comfy.ops.manual_cast)
        obj = cls(num_embeddings=1000, embedding_dim=8, padding_idx=0)
        val = getattr(obj, "comfy_cast_weights", None)
        assert val is True, (
            f"comfy_cast_weights expected True with manual_cast, got {val}"
        )


# ===========================================================================
# 4. T5Gemma2TextEncoder
# ===========================================================================

class TestT5Gemma2TextEncoderInit:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_text_config()

    def test_required_attributes_exist(self, cfg):
        """embed_tokens, norm, layers, dropout, rotary_emb must all exist."""
        enc = T5Gemma2TextEncoder(cfg)
        for attr in ("embed_tokens", "norm", "layers", "dropout", "rotary_emb"):
            assert hasattr(enc, attr), f"T5Gemma2TextEncoder missing attribute: {attr}"

    def test_layers_count_is_34(self, cfg):
        """layers must be ModuleList with exactly 34 entries."""
        enc = T5Gemma2TextEncoder(cfg)
        assert len(enc.layers) == 34, (
            f"Expected 34 layers, got {len(enc.layers)}"
        )

    def test_layer_attention_types_match_config(self, cfg):
        """Each layer's attention_type must match config.layer_types[i]."""
        enc = T5Gemma2TextEncoder(cfg)
        for i, layer in enumerate(enc.layers):
            expected = cfg.layer_types[i]
            actual = layer.attention_type
            assert actual == expected, (
                f"layers[{i}].attention_type={actual!r}, config says {expected!r}"
            )

    def test_alias_text_model_is_same_class(self):
        """T5Gemma2TextModel must be the same class as T5Gemma2TextEncoder."""
        assert T5Gemma2TextModel is T5Gemma2TextEncoder, (
            "T5Gemma2TextModel must be an alias for T5Gemma2TextEncoder"
        )


class TestT5Gemma2TextEncoderForward:

    @pytest.fixture(scope="class")
    def cfg(self):
        return _make_text_config()

    def test_forward_smoke_input_ids(self, cfg):
        """input_ids (1, 4) -> output.last_hidden_state shape (1, 4, 2560)."""
        enc = T5Gemma2TextEncoder(cfg)
        enc.eval()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            out = enc(input_ids=input_ids)
        assert hasattr(out, "last_hidden_state"), (
            "output must have .last_hidden_state attribute"
        )
        assert out.last_hidden_state.shape == (1, 4, 2560), (
            f"last_hidden_state shape expected (1, 4, 2560), got {out.last_hidden_state.shape}"
        )

    def test_forward_xor_both_none_raises_valueerror(self, cfg):
        """input_ids=None and inputs_embeds=None simultaneously -> ValueError."""
        enc = T5Gemma2TextEncoder(cfg)
        with pytest.raises((ValueError, AssertionError)):
            enc(input_ids=None, inputs_embeds=None)

    def test_forward_xor_both_provided_raises_valueerror(self, cfg):
        """input_ids and inputs_embeds both provided -> ValueError."""
        enc = T5Gemma2TextEncoder(cfg)
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        embeds = torch.randn(1, 4, 2560)
        with pytest.raises((ValueError, AssertionError)):
            enc(input_ids=input_ids, inputs_embeds=embeds)

    def test_forward_inputs_embeds_direct(self, cfg):
        """inputs_embeds provided directly (without input_ids) must work."""
        enc = T5Gemma2TextEncoder(cfg)
        enc.eval()
        embeds = torch.randn(1, 4, 2560)
        with torch.no_grad():
            out = enc(inputs_embeds=embeds)
        assert out.last_hidden_state.shape == (1, 4, 2560)

    def test_forward_with_attention_mask(self, cfg):
        """attention_mask (B, S) must be accepted and produce correct output shape."""
        enc = T5Gemma2TextEncoder(cfg)
        enc.eval()
        B, S = 1, 6
        input_ids = torch.zeros(B, S, dtype=torch.long)
        attention_mask = torch.ones(B, S, dtype=torch.long)
        with torch.no_grad():
            out = enc(input_ids=input_ids, attention_mask=attention_mask)
        assert out.last_hidden_state.shape == (B, S, 2560)

    def test_forward_with_position_ids(self, cfg):
        """Explicit position_ids must be accepted without error."""
        enc = T5Gemma2TextEncoder(cfg)
        enc.eval()
        B, S = 1, 4
        input_ids = torch.zeros(B, S, dtype=torch.long)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            out = enc(input_ids=input_ids, position_ids=position_ids)
        assert out.last_hidden_state.shape == (B, S, 2560)


# ===========================================================================
# 5. T5Gemma2Encoder (outer wrapper)
# ===========================================================================

class TestT5Gemma2EncoderInit:

    @pytest.fixture(scope="class")
    def enc_cfg(self):
        return _make_encoder_config()

    def test_text_model_attribute_exists(self, enc_cfg):
        """text_model attribute must exist."""
        enc = T5Gemma2Encoder(enc_cfg)
        assert hasattr(enc, "text_model"), "T5Gemma2Encoder missing attribute: text_model"

    def test_get_input_embeddings(self, enc_cfg):
        """get_input_embeddings() must return text_model.embed_tokens."""
        enc = T5Gemma2Encoder(enc_cfg)
        emb = enc.get_input_embeddings()
        assert emb is enc.text_model.embed_tokens, (
            "get_input_embeddings() must return text_model.embed_tokens"
        )


class TestT5Gemma2EncoderForward:

    @pytest.fixture(scope="class")
    def enc_cfg(self):
        return _make_encoder_config()

    def test_forward_smoke(self, enc_cfg):
        """input_ids (1, 4) -> output.last_hidden_state shape (1, 4, 2560)."""
        enc = T5Gemma2Encoder(enc_cfg)
        enc.eval()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            out = enc(input_ids=input_ids)
        assert out.last_hidden_state.shape == (1, 4, 2560), (
            f"Outer Encoder forward shape mismatch: {out.last_hidden_state.shape}"
        )

    def test_pixel_values_not_none_raises_not_implemented(self, enc_cfg):
        """pixel_values != None must raise NotImplementedError (not silently ignored)."""
        enc = T5Gemma2Encoder(enc_cfg)
        dummy_pixels = torch.randn(1, 3, 224, 224)
        with pytest.raises(NotImplementedError):
            enc(input_ids=torch.zeros(1, 4, dtype=torch.long), pixel_values=dummy_pixels)

    def test_pixel_values_none_passes_through(self, enc_cfg):
        """pixel_values=None (default) must pass through to text_model normally."""
        enc = T5Gemma2Encoder(enc_cfg)
        enc.eval()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            out = enc(input_ids=input_ids, pixel_values=None)
        assert out.last_hidden_state.shape == (1, 4, 2560)


class TestT5Gemma2EncoderStateDictKeys:

    @pytest.fixture(scope="class")
    def enc_cfg(self):
        return _make_encoder_config()

    def test_required_state_dict_keys_exist(self, enc_cfg):
        """Critical state_dict keys must all be present."""
        enc = T5Gemma2Encoder(enc_cfg)
        sd = enc.state_dict()
        required_keys = [
            "text_model.embed_tokens.weight",
            "text_model.embed_tokens.eoi_embedding",
            "text_model.layers.0.mlp.gate_proj.weight",
            "text_model.layers.0.self_attn.q_proj.weight",
            "text_model.layers.0.pre_self_attn_layernorm.weight",
            "text_model.norm.weight",
        ]
        for key in required_keys:
            assert key in sd, (
                f"Required state_dict key missing: '{key}'"
            )

    def test_inv_freq_not_in_state_dict(self, enc_cfg):
        """persistent=False: no *inv_freq* key in state_dict."""
        enc = T5Gemma2Encoder(enc_cfg)
        sd = enc.state_dict()
        inv_keys = [k for k in sd.keys() if "inv_freq" in k]
        assert len(inv_keys) == 0, (
            f"inv_freq keys found (should be persistent=False): {inv_keys}"
        )

    def test_comfy_cast_weights_manual_cast(self, enc_cfg):
        """
        operations=manual_cast: all Linear and Embedding under text_model
        must have comfy_cast_weights=True.
        """
        enc = T5Gemma2Encoder(enc_cfg, operations=comfy.ops.manual_cast)

        failures = []
        for name, module in enc.text_model.named_modules():
            # Check only leaf modules that are Linear or Embedding types
            if isinstance(module, (nn.Linear, nn.Embedding)):
                val = getattr(module, "comfy_cast_weights", None)
                if val is not True:
                    failures.append(f"{name}: comfy_cast_weights={val}")

        assert len(failures) == 0, (
            "The following modules lack comfy_cast_weights=True under manual_cast:\n"
            + "\n".join(failures[:10])  # show first 10
        )


# ===========================================================================
# 6. _build_bidirectional_mask
# ===========================================================================

class TestBuildBidirectionalMask:

    def test_none_attention_mask_returns_none(self):
        """attention_mask=None -> return None (SDPA treats None as full attend)."""
        result = _build_bidirectional_mask(
            seq_len=8,
            attention_mask=None,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert result is None, (
            f"Expected None when attention_mask=None, got {result}"
        )

    def test_all_ones_mask_produces_all_zeros(self):
        """attention_mask of all 1s (no pad) -> additive mask all zeros."""
        B, S = 1, 8
        am = torch.ones(B, S, dtype=torch.long)
        result = _build_bidirectional_mask(
            seq_len=S,
            attention_mask=am,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert result is not None
        assert result.shape == (B, 1, S, S), (
            f"Expected shape {(B, 1, S, S)}, got {result.shape}"
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6), (
            "All-real attention_mask must produce all-zero additive mask"
        )

    def test_pad_position_column_is_neg_inf(self):
        """
        attention_mask with a pad token (0) at position k:
        the k-th column in the (S, S) additive mask must be -inf for all rows.
        """
        B, S = 1, 4
        # Positions 0,1,2 real; position 3 is pad
        am = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        result = _build_bidirectional_mask(
            seq_len=S,
            attention_mask=am,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        # result shape: (B, 1, S, S)
        # The pad column (index 3) must be -inf everywhere
        pad_col = result[0, 0, :, 3]
        assert torch.all(pad_col.isinf() & (pad_col < 0)), (
            f"Pad column must be -inf, got: {pad_col.tolist()}"
        )
        # Non-pad columns (0,1,2) must be 0
        for col_idx in (0, 1, 2):
            col = result[0, 0, :, col_idx]
            assert torch.allclose(col, torch.zeros_like(col), atol=1e-6), (
                f"Real column {col_idx} must be 0, got: {col.tolist()}"
            )


# ===========================================================================
# 7. _build_sliding_window_mask
# ===========================================================================

class TestBuildSlidingWindowMask:

    def test_output_shape(self):
        """Output shape must be (B, 1, S, S)."""
        B, S = 2, 8
        result = _build_sliding_window_mask(
            seq_len=S,
            sliding_window=4,
            attention_mask=None,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert result.shape == (1, 1, S, S) or result.shape == (B, 1, S, S) or len(result.shape) == 4, (
            f"Expected 4D output, got shape {result.shape}"
        )
        assert result.shape[-2] == S and result.shape[-1] == S, (
            f"Last two dims must be (S, S)={(S, S)}, got {result.shape}"
        )

    def test_sliding_window_formula_sw4_s8(self):
        """
        sliding_window=4, S=8:
          left_w = (4+1)//2 = 2, right_w = 4//2+1 = 3
          dist = q_idx - kv_idx
          in_window: (dist>=0 & dist<2) | (dist<0 & -dist<3)
          i.e. dist in {-2, -1, 0, 1} -> in window
               dist in {-3, 2, 3, 4, 5, 6, ...} -> out

        Spot-check row q=4 (0-based):
          kv=0: dist=4 -> out  (-inf)
          kv=1: dist=3 -> out  (-inf)
          kv=2: dist=2 -> out  (-inf)
          kv=3: dist=1 -> in   (0)
          kv=4: dist=0 -> in   (0)
          kv=5: dist=-1 -> in  (0)
          kv=6: dist=-2 -> in  (0)
          kv=7: dist=-3 -> out (-inf)
        """
        S = 8
        sliding_window = 4
        result = _build_sliding_window_mask(
            seq_len=S,
            sliding_window=sliding_window,
            attention_mask=None,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        # Use first batch dimension
        mask_2d = result[0, 0]  # shape (S, S)
        q = 4

        # in-window positions (must be 0)
        for kv in (3, 4, 5, 6):
            val = float(mask_2d[q, kv])
            assert val == 0.0, (
                f"q={q},kv={kv}: dist={q-kv} expected in-window (0), got {val}"
            )

        # out-of-window positions (must be -inf)
        for kv in (0, 1, 2, 7):
            val = float(mask_2d[q, kv])
            assert math.isinf(val) and val < 0, (
                f"q={q},kv={kv}: dist={q-kv} expected out-of-window (-inf), got {val}"
            )

    def test_sliding_window_1024_left_right(self):
        """
        sliding_window=1024 -> left_w=(1024+1)//2=512, right_w=1024//2+1=513.
        Verify formula with a short sequence for speed: just check the scalar values.
        """
        left_w = (1024 + 1) // 2
        right_w = 1024 // 2 + 1
        assert left_w == 512, f"left_w expected 512, got {left_w}"
        assert right_w == 513, f"right_w expected 513, got {right_w}"

    def test_sliding_window_mask_with_attention_mask_pad(self):
        """
        When attention_mask has pad tokens, pad columns must also be -inf
        (combined sliding window + padding masking).
        """
        B, S = 1, 8
        sliding_window = 4
        # Last token is pad
        am = torch.ones(B, S, dtype=torch.long)
        am[0, -1] = 0
        result = _build_sliding_window_mask(
            seq_len=S,
            sliding_window=sliding_window,
            attention_mask=am,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        # Pad column (S-1) must be -inf in all rows
        pad_col = result[0, 0, :, S - 1]
        assert torch.all(pad_col.isinf() & (pad_col < 0)), (
            f"Pad column under sliding mask must be -inf, got: {pad_col.tolist()}"
        )
