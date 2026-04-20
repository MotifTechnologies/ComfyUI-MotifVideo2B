# tests/transformer/test_ops_injection.py
#
# Verifies P2.1 checklist criteria:
#   1. MotifVideoPatchEmbed.proj is created via operations.Conv3d when operations is provided
#   2. MotifVideoAdaNorm.linear is created via operations.Linear when operations is provided
#   3. Both classes fall back correctly when operations=None (default ops path)
#   4. dtype/device are propagated to weights on the default-fallback path

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup — production-path convention (mirrors other ComfyUI custom nodes).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Add ComfyUI root so `import comfy.*` works (production convention).
_COMFYUI_ROOT = os.path.abspath(os.path.join(_REPO_ROOT, "..", ".."))
if _COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, _COMFYUI_ROOT)

import torch
import pytest

# ---------------------------------------------------------------------------
# GPU skip guard — MUST precede any import of transformer_motif_video.
# comfy.model_management calls torch.cuda.current_device() at import time,
# so attempting the import on a CPU-only host crashes at collection.
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    pytest.skip("requires real comfy runtime (GPU)", allow_module_level=True)

import torch.nn as nn
from unittest.mock import patch

import comfy.ops  # noqa: F401 — available now that the skip guard passed

from models.transformer.transformer_motif_video import (
    MotifVideoPatchEmbed,
    MotifVideoAdaNorm,
    MotifVideoImageProjection,
    MotifVideoSingleTransformerBlock,
    MotifVideoTransformerBlock,
    MotifVideoConditionEmbedding,
    MotifVideoTransformer3DModel,
)
from models.transformer import transformer_motif_video as _tmv

# ---------------------------------------------------------------------------
# Marker subclasses — isinstance() confirms which ops was used.
# ---------------------------------------------------------------------------

class _MarkerConv3d(nn.Conv3d):
    pass

class _MarkerLinear(nn.Linear):
    pass

class _MockOps:
    Conv3d = _MarkerConv3d
    Linear = _MarkerLinear


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_patchembed_ops_injection():
    """MotifVideoPatchEmbed.proj must use Conv3d from the injected operations object."""
    embed = MotifVideoPatchEmbed(
        patch_size=(1, 2, 2),
        in_chans=33,
        embed_dim=3072,
        operations=_MockOps,
    )
    assert isinstance(embed.proj, _MarkerConv3d), (
        f"Expected proj to be _MarkerConv3d, got {type(embed.proj)}"
    )


def test_adanorm_ops_injection():
    """MotifVideoAdaNorm.linear must use Linear from the injected operations object."""
    norm = MotifVideoAdaNorm(
        in_features=3072,
        operations=_MockOps,
    )
    assert isinstance(norm.linear, _MarkerLinear), (
        f"Expected linear to be _MarkerLinear, got {type(norm.linear)}"
    )


def test_patchembed_default_fallback_and_dtype():
    """operations=None must fall back to comfy.ops.disable_weight_init.Conv3d with correct dtype/device."""
    with patch.object(_tmv, "_get_default_ops", wraps=_tmv._get_default_ops) as spy:
        embed = MotifVideoPatchEmbed(
            patch_size=(1, 2, 2),
            in_chans=33,
            embed_dim=3072,
            operations=None,
            dtype=torch.float16,
            device="cuda",
        )
        assert spy.call_count == 1, (
            f"_get_default_ops must be called exactly once, got {spy.call_count}"
        )
    assert type(embed.proj) is comfy.ops.disable_weight_init.Conv3d, (
        f"Expected comfy.ops.disable_weight_init.Conv3d, got {type(embed.proj)}"
    )
    assert embed.proj.weight.dtype == torch.float16, (
        f"Expected weight dtype float16, got {embed.proj.weight.dtype}"
    )
    assert embed.proj.weight.device.type == "cuda", (
        f"Expected weight on cuda, got {embed.proj.weight.device.type}"
    )


def test_adanorm_default_fallback_and_dtype():
    """operations=None must fall back to comfy.ops.disable_weight_init.Linear with correct dtype/device."""
    with patch.object(_tmv, "_get_default_ops", wraps=_tmv._get_default_ops) as spy:
        norm = MotifVideoAdaNorm(
            in_features=3072,
            operations=None,
            dtype=torch.float16,
            device="cuda",
        )
        assert spy.call_count == 1, (
            f"_get_default_ops must be called exactly once, got {spy.call_count}"
        )
    assert type(norm.linear) is comfy.ops.disable_weight_init.Linear, (
        f"Expected comfy.ops.disable_weight_init.Linear, got {type(norm.linear)}"
    )
    assert norm.linear.weight.dtype == torch.float16, (
        f"Expected weight dtype float16, got {norm.linear.weight.dtype}"
    )
    assert norm.linear.weight.device.type == "cuda", (
        f"Expected weight on cuda, got {norm.linear.weight.device.type}"
    )


def test_image_projection_default_fallback_and_dtype():
    """MotifVideoImageProjection.{norm_in,linear_1,linear_2,norm_out} must use
    _get_default_ops() fallback with dtype/device propagated when operations=None."""
    default_ops = comfy.ops.disable_weight_init
    proj = MotifVideoImageProjection(
        in_features=768, hidden_size=3072,
        dtype=torch.float16, device="cuda",
    )
    for attr, expected_cls in [
        ("norm_in", default_ops.LayerNorm),
        ("linear_1", default_ops.Linear),
        ("linear_2", default_ops.Linear),
        ("norm_out", default_ops.LayerNorm),
    ]:
        layer = getattr(proj, attr)
        assert type(layer) is expected_cls, f"{attr}: expected {expected_cls}, got {type(layer)}"
        assert layer.weight.dtype == torch.float16, f"{attr}.weight.dtype"
        assert layer.weight.device.type == "cuda", f"{attr}.weight.device"


def test_image_projection_ops_injection():
    """MotifVideoImageProjection must use injected operations' classes."""
    class _MarkerLayerNorm(nn.LayerNorm):
        pass
    class _MarkerLinear(nn.Linear):
        pass
    class _MockOps:
        LayerNorm = _MarkerLayerNorm
        Linear = _MarkerLinear
    proj = MotifVideoImageProjection(in_features=768, hidden_size=3072, operations=_MockOps)
    assert isinstance(proj.norm_in, _MarkerLayerNorm)
    assert isinstance(proj.linear_1, _MarkerLinear)
    assert isinstance(proj.linear_2, _MarkerLinear)
    assert isinstance(proj.norm_out, _MarkerLayerNorm)


# ---------------------------------------------------------------------------
# P2.3 — MotifVideoSingleTransformerBlock ops injection
# ---------------------------------------------------------------------------

def test_single_transformer_block_default_fallback_and_dtype():
    """operations=None must fall back to comfy.ops.disable_weight_init with dtype/device propagated.

    Tests enable_text_cross_attention=True to cover all injected layers.
    self.attn must remain a diffusers Attention instance (not replaced).
    """
    from diffusers.models.attention_processor import Attention as DiffusersAttention
    from models.transformer.ops_primitives import AdaLayerNormZeroSingle as LocalAdaLNZeroSingle

    default_ops = comfy.ops.disable_weight_init
    block = MotifVideoSingleTransformerBlock(
        num_attention_heads=24,
        attention_head_dim=128,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
        norm_type="layer_norm",
        enable_text_cross_attention=True,
        dtype=torch.float16,
        device="cuda",
    )

    # cross_attn_query_proj / cross_attn_out_proj → default_ops.Linear
    for attr in ("cross_attn_query_proj", "cross_attn_out_proj"):
        layer = getattr(block, attr)
        assert type(layer) is default_ops.Linear, f"{attr}: expected {default_ops.Linear}, got {type(layer)}"
        assert layer.weight.dtype == torch.float16, f"{attr}.weight.dtype"
        assert layer.weight.device.type == "cuda", f"{attr}.weight.device"

    # cross_attn_query_norm → default_ops.LayerNorm
    layer = block.cross_attn_query_norm
    assert type(layer) is default_ops.LayerNorm, f"cross_attn_query_norm: expected {default_ops.LayerNorm}, got {type(layer)}"

    # proj_mlp / proj_out → default_ops.Linear
    for attr in ("proj_mlp", "proj_out"):
        layer = getattr(block, attr)
        assert type(layer) is default_ops.Linear, f"{attr}: expected {default_ops.Linear}, got {type(layer)}"
        assert layer.weight.dtype == torch.float16, f"{attr}.weight.dtype"
        assert layer.weight.device.type == "cuda", f"{attr}.weight.device"

    # norm → local ops_primitives.AdaLayerNormZeroSingle (not diffusers)
    assert isinstance(block.norm, LocalAdaLNZeroSingle), (
        f"block.norm must be local AdaLayerNormZeroSingle, got {type(block.norm)}"
    )

    # self.attn must remain diffusers Attention (not replaced — #18 scope)
    assert isinstance(block.attn, DiffusersAttention), (
        f"block.attn must remain diffusers Attention, got {type(block.attn)}"
    )


def test_single_transformer_block_ops_injection():
    """Explicit _MockOps injection: layers must be marker types.

    Uses enable_text_cross_attention=True to cover the widest code path.
    """
    class _MarkerLayerNorm2(nn.LayerNorm):
        pass

    class _MarkerLinear2(nn.Linear):
        pass

    class _MockOps2:
        LayerNorm = _MarkerLayerNorm2
        Linear = _MarkerLinear2

    block = MotifVideoSingleTransformerBlock(
        num_attention_heads=4,
        attention_head_dim=64,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
        norm_type="layer_norm",
        enable_text_cross_attention=True,
        operations=_MockOps2,
    )

    assert type(block.cross_attn_query_proj) is _MarkerLinear2, (
        f"cross_attn_query_proj: expected _MarkerLinear2, got {type(block.cross_attn_query_proj)}"
    )
    assert type(block.cross_attn_query_norm) is _MarkerLayerNorm2, (
        f"cross_attn_query_norm: expected _MarkerLayerNorm2, got {type(block.cross_attn_query_norm)}"
    )
    assert type(block.cross_attn_out_proj) is _MarkerLinear2, (
        f"cross_attn_out_proj: expected _MarkerLinear2, got {type(block.cross_attn_out_proj)}"
    )
    assert type(block.proj_mlp) is _MarkerLinear2, (
        f"proj_mlp: expected _MarkerLinear2, got {type(block.proj_mlp)}"
    )
    assert type(block.proj_out) is _MarkerLinear2, (
        f"proj_out: expected _MarkerLinear2, got {type(block.proj_out)}"
    )


# ---------------------------------------------------------------------------
# P2.4 — MotifVideoTransformerBlock ops injection
# ---------------------------------------------------------------------------

def test_transformer_block_default_fallback_and_dtype():
    """operations=None must fall back to comfy.ops.disable_weight_init with dtype/device propagated.

    Uses enable_text_cross_attention=True to cover all injected layers.
    self.attn must remain a diffusers Attention instance (not replaced by ops).
    """
    from diffusers.models.attention_processor import Attention as DiffusersAttention
    from models.transformer.ops_primitives import (
        AdaLayerNormZero as LocalAdaLNZero,
        FeedForward as LocalFeedForward,
    )

    default_ops = comfy.ops.disable_weight_init
    block = MotifVideoTransformerBlock(
        num_attention_heads=24,
        attention_head_dim=128,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
        norm_type="layer_norm",
        enable_text_cross_attention=True,
        dtype=torch.float16,
        device="cuda",
    )

    # norm1 / norm1_context → local ops_primitives.AdaLayerNormZero
    for attr in ("norm1", "norm1_context"):
        layer = getattr(block, attr)
        assert isinstance(layer, LocalAdaLNZero), (
            f"{attr}: expected local AdaLayerNormZero, got {type(layer)}"
        )

    # cross_attn_query_proj / cross_attn_out_proj → default_ops.Linear
    for attr in ("cross_attn_query_proj", "cross_attn_out_proj"):
        layer = getattr(block, attr)
        assert type(layer) is default_ops.Linear, (
            f"{attr}: expected {default_ops.Linear}, got {type(layer)}"
        )
        assert layer.weight.dtype == torch.float16, f"{attr}.weight.dtype"
        assert layer.weight.device.type == "cuda", f"{attr}.weight.device"

    # cross_attn_query_norm → default_ops.LayerNorm
    layer = block.cross_attn_query_norm
    assert type(layer) is default_ops.LayerNorm, (
        f"cross_attn_query_norm: expected {default_ops.LayerNorm}, got {type(layer)}"
    )

    # norm2 / norm2_context → default_ops.LayerNorm (elementwise_affine=False, no weight)
    for attr in ("norm2", "norm2_context"):
        layer = getattr(block, attr)
        assert type(layer) is default_ops.LayerNorm, (
            f"{attr}: expected {default_ops.LayerNorm}, got {type(layer)}"
        )

    # ff / ff_context → local ops_primitives.FeedForward
    for attr in ("ff", "ff_context"):
        layer = getattr(block, attr)
        assert isinstance(layer, LocalFeedForward), (
            f"{attr}: expected local FeedForward, got {type(layer)}"
        )

    # self.attn must remain diffusers Attention (not replaced — #18 scope)
    assert isinstance(block.attn, DiffusersAttention), (
        f"block.attn must remain diffusers Attention, got {type(block.attn)}"
    )
    # dtype propagated to attn via .to() — sample weight check
    assert block.attn.to_q.weight.dtype == torch.float16, (
        f"attn.to_q.weight.dtype expected float16, got {block.attn.to_q.weight.dtype}"
    )
    assert block.attn.to_q.weight.device.type == "cuda", (
        f"attn.to_q.weight must be on cuda"
    )


def test_transformer_block_ops_injection():
    """Explicit _MockOps injection: norm2/ff layers must use marker types.

    Uses enable_text_cross_attention=True to cover all injected layers.
    """
    class _MarkerLayerNorm3(nn.LayerNorm):
        pass

    class _MarkerLinear3(nn.Linear):
        pass

    class _MockOps3:
        LayerNorm = _MarkerLayerNorm3
        Linear = _MarkerLinear3

    from models.transformer.ops_primitives import (
        AdaLayerNormZero as LocalAdaLNZero,
        FeedForward as LocalFeedForward,
    )

    block = MotifVideoTransformerBlock(
        num_attention_heads=4,
        attention_head_dim=64,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
        norm_type="layer_norm",
        enable_text_cross_attention=True,
        operations=_MockOps3,
    )

    # norm1 / norm1_context — local AdaLayerNormZero (uses ops internally)
    for attr in ("norm1", "norm1_context"):
        layer = getattr(block, attr)
        assert isinstance(layer, LocalAdaLNZero), (
            f"{attr}: expected local AdaLayerNormZero, got {type(layer)}"
        )

    # cross_attn_query_proj / cross_attn_out_proj → _MarkerLinear3
    for attr in ("cross_attn_query_proj", "cross_attn_out_proj"):
        assert type(getattr(block, attr)) is _MarkerLinear3, (
            f"{attr}: expected _MarkerLinear3, got {type(getattr(block, attr))}"
        )

    # cross_attn_query_norm → _MarkerLayerNorm3
    assert type(block.cross_attn_query_norm) is _MarkerLayerNorm3, (
        f"cross_attn_query_norm: expected _MarkerLayerNorm3, got {type(block.cross_attn_query_norm)}"
    )

    # norm2 / norm2_context → _MarkerLayerNorm3
    for attr in ("norm2", "norm2_context"):
        assert type(getattr(block, attr)) is _MarkerLayerNorm3, (
            f"{attr}: expected _MarkerLayerNorm3, got {type(getattr(block, attr))}"
        )

    # ff / ff_context — local FeedForward with ops propagated to inner Linear layers.
    # FeedForward structure: net[0] = GELU wrapper (contains proj), net[1] = Dropout, net[2] = Linear
    for attr in ("ff", "ff_context"):
        layer = getattr(block, attr)
        assert isinstance(layer, LocalFeedForward), (
            f"{attr}: expected local FeedForward, got {type(layer)}"
        )
        # Verify operations propagated into FeedForward internals.
        assert type(layer.net[0].proj) is _MarkerLinear3, (
            f"{attr}.net[0].proj: expected _MarkerLinear3, got {type(layer.net[0].proj)}"
        )
        assert type(layer.net[2]) is _MarkerLinear3, (
            f"{attr}.net[2]: expected _MarkerLinear3, got {type(layer.net[2])}"
        )


# ---------------------------------------------------------------------------
# P3.1 — MotifVideoConditionEmbedding ops injection
# ---------------------------------------------------------------------------

def test_condition_embedding_default_fallback_and_dtype():
    """operations=None must fall back to comfy.ops.disable_weight_init with dtype/device propagated.

    Covers pooled_projection_dim=<int> branch (text_embedder created).
    The pooled_projection_dim=None branch is covered by test_condition_embedding_no_pooled_projection.
    time_proj must remain diffusers Timesteps (sinusoidal, no parameters).
    """
    from diffusers.models.embeddings import Timesteps as DiffusersTimesteps
    from models.transformer.ops_primitives import (
        TimestepEmbedding as LocalTimestepEmbedding,
        PixArtAlphaTextProjection as LocalPixArtAlphaTextProjection,
    )

    default_ops = comfy.ops.disable_weight_init

    cond = MotifVideoConditionEmbedding(
        embedding_dim=3072,
        pooled_projection_dim=1536,
        dtype=torch.float16,
        device="cuda",
    )

    # time_proj must remain diffusers Timesteps (no parameters, not replaced)
    assert isinstance(cond.time_proj, DiffusersTimesteps), (
        f"time_proj must be diffusers Timesteps, got {type(cond.time_proj)}"
    )

    # timestep_embedder must be local ops_primitives.TimestepEmbedding
    assert isinstance(cond.timestep_embedder, LocalTimestepEmbedding), (
        f"timestep_embedder must be local TimestepEmbedding, got {type(cond.timestep_embedder)}"
    )
    for attr in ("linear_1", "linear_2"):
        layer = getattr(cond.timestep_embedder, attr)
        assert type(layer) is default_ops.Linear, (
            f"timestep_embedder.{attr}: expected {default_ops.Linear}, got {type(layer)}"
        )
        assert layer.weight.dtype == torch.float16, f"timestep_embedder.{attr}.weight.dtype"
        assert layer.weight.device.type == "cuda", f"timestep_embedder.{attr}.weight.device"

    # text_embedder must be local ops_primitives.PixArtAlphaTextProjection
    assert isinstance(cond.text_embedder, LocalPixArtAlphaTextProjection), (
        f"text_embedder must be local PixArtAlphaTextProjection, got {type(cond.text_embedder)}"
    )
    for attr in ("linear_1", "linear_2"):
        layer = getattr(cond.text_embedder, attr)
        assert type(layer) is default_ops.Linear, (
            f"text_embedder.{attr}: expected {default_ops.Linear}, got {type(layer)}"
        )
        assert layer.weight.dtype == torch.float16, f"text_embedder.{attr}.weight.dtype"
        assert layer.weight.device.type == "cuda", f"text_embedder.{attr}.weight.device"


def test_condition_embedding_ops_injection():
    """Explicit _MockOpsCond injection: timestep_embedder and text_embedder
    must be local ops_primitives instances with _MarkerLinear inner layers."""
    from models.transformer.ops_primitives import (
        TimestepEmbedding as LocalTimestepEmbedding,
        PixArtAlphaTextProjection as LocalPixArtAlphaTextProjection,
    )

    class _MarkerLinearCond(nn.Linear):
        pass

    class _MockOpsCond:
        Linear = _MarkerLinearCond

    cond = MotifVideoConditionEmbedding(
        embedding_dim=512,
        pooled_projection_dim=256,
        operations=_MockOpsCond,
    )

    # timestep_embedder is local ops_primitives class instance
    assert isinstance(cond.timestep_embedder, LocalTimestepEmbedding), (
        f"timestep_embedder must be local TimestepEmbedding, got {type(cond.timestep_embedder)}"
    )
    assert type(cond.timestep_embedder.linear_1) is _MarkerLinearCond, (
        f"timestep_embedder.linear_1: expected _MarkerLinearCond, got {type(cond.timestep_embedder.linear_1)}"
    )
    assert type(cond.timestep_embedder.linear_2) is _MarkerLinearCond, (
        f"timestep_embedder.linear_2: expected _MarkerLinearCond, got {type(cond.timestep_embedder.linear_2)}"
    )

    # text_embedder is local ops_primitives class instance
    assert isinstance(cond.text_embedder, LocalPixArtAlphaTextProjection), (
        f"text_embedder must be local PixArtAlphaTextProjection, got {type(cond.text_embedder)}"
    )
    assert type(cond.text_embedder.linear_1) is _MarkerLinearCond, (
        f"text_embedder.linear_1: expected _MarkerLinearCond, got {type(cond.text_embedder.linear_1)}"
    )
    assert type(cond.text_embedder.linear_2) is _MarkerLinearCond, (
        f"text_embedder.linear_2: expected _MarkerLinearCond, got {type(cond.text_embedder.linear_2)}"
    )


def test_condition_embedding_no_pooled_projection():
    """pooled_projection_dim=None must not create text_embedder (diffusers parity)."""
    cond = MotifVideoConditionEmbedding(embedding_dim=512, pooled_projection_dim=None)
    assert not hasattr(cond, "text_embedder"), (
        "text_embedder must not be created when pooled_projection_dim=None"
    )


# ---------------------------------------------------------------------------
# P3.2a — MotifVideoTransformer3DModel.__init__ signature + ignore_for_config
# ---------------------------------------------------------------------------

def test_transformer3d_register_to_config_sanity():
    """MotifVideoTransformer3DModel must exclude dtype/device/operations from self.config.

    These args are non-JSONable (torch.dtype / torch.device / class objects) and must
    be handled by ComfyUI at runtime; they must not appear in the checkpoint config
    saved via diffusers save_config() or to_config_dict().
    """
    from models.transformer.transformer_motif_video import MotifVideoTransformer3DModel

    model = MotifVideoTransformer3DModel(
        num_attention_heads=4,
        attention_head_dim=64,
        num_layers=1,
        num_single_layers=1,
        num_decoder_layers=0,
        text_embed_dim=256,
        image_embed_dim=None,
        pooled_projection_dim=None,
        rope_axes_dim=(16, 24, 24),
        operations=comfy.ops.disable_weight_init,
        dtype=torch.float16,
        device="cuda",
    )

    # Verify exclusion across all 3 config surfaces: self.config, to_config_dict(),
    # and save_config() (the actual JSON serialization path).
    from pathlib import Path
    import json, tempfile

    config_dict = dict(model.config)
    to_config = dict(model.to_config_dict())
    with tempfile.TemporaryDirectory() as td:
        model.save_config(td)
        with open(Path(td) / "config.json") as f:
            saved = json.load(f)

    for surface_name, surface in [("self.config", config_dict), ("to_config_dict", to_config), ("save_config", saved)]:
        for key in ("dtype", "device", "operations"):
            assert key not in surface, (
                f"{key!r} must NOT be in {surface_name}, but found: {surface.get(key)}"
            )

    # Positive case: regular args must still be present
    assert config_dict.get("num_attention_heads") == 4
    assert config_dict.get("attention_head_dim") == 64
    assert to_config.get("num_attention_heads") == 4
    assert saved.get("num_attention_heads") == 4


# ---------------------------------------------------------------------------
# P3.2b — MotifVideoTransformer3DModel top-level embedder/norm/projection ops
# ---------------------------------------------------------------------------

def _make_small_transformer3d(**kwargs):
    """Helper: minimal-cost MotifVideoTransformer3DModel for injection tests."""
    defaults = dict(
        num_attention_heads=4,
        attention_head_dim=64,
        num_layers=1,
        num_single_layers=1,
        num_decoder_layers=0,
        text_embed_dim=256,
        image_embed_dim=None,
        pooled_projection_dim=None,
        rope_axes_dim=(16, 24, 24),
    )
    defaults.update(kwargs)
    return MotifVideoTransformer3DModel(**defaults)


def test_transformer3d_top_level_default_fallback_and_dtype():
    """Top-level embedder/norm/proj must use _get_default_ops() with dtype/device propagated.

    Covers: x_embedder, context_embedder, time_text_embed, norm_out, proj_out.
    image_embed_dim=None so image_embedder is not created.
    blocks (transformer_blocks / single_transformer_blocks) are P3.2c scope.
    """
    from models.transformer.ops_primitives import (
        PixArtAlphaTextProjection as LocalPixArtProj,
        AdaLayerNormContinuous as LocalAdaLNCont,
    )
    default_ops = comfy.ops.disable_weight_init

    model = _make_small_transformer3d(
        dtype=torch.float16,
        device="cuda",
    )

    # x_embedder.proj → default_ops.Conv3d
    assert type(model.x_embedder.proj) is default_ops.Conv3d, (
        f"x_embedder.proj: expected {default_ops.Conv3d}, got {type(model.x_embedder.proj)}"
    )
    assert model.x_embedder.proj.weight.dtype == torch.float16
    assert model.x_embedder.proj.weight.device.type == "cuda"

    # context_embedder → local PixArtAlphaTextProjection instance
    assert isinstance(model.context_embedder, LocalPixArtProj), (
        f"context_embedder must be local PixArtAlphaTextProjection, got {type(model.context_embedder)}"
    )
    for attr in ("linear_1", "linear_2"):
        layer = getattr(model.context_embedder, attr)
        assert type(layer) is default_ops.Linear, f"context_embedder.{attr}: {type(layer)}"
        assert layer.weight.dtype == torch.float16, f"context_embedder.{attr}.weight.dtype"
        assert layer.weight.device.type == "cuda", f"context_embedder.{attr}.weight.device"

    # image_embedder must NOT exist (image_embed_dim=None)
    assert not hasattr(model, "image_embedder"), "image_embedder must not exist when image_embed_dim=None"

    # time_text_embed.timestep_embedder linear layers → default_ops.Linear
    for attr in ("linear_1", "linear_2"):
        layer = getattr(model.time_text_embed.timestep_embedder, attr)
        assert type(layer) is default_ops.Linear, (
            f"time_text_embed.timestep_embedder.{attr}: expected {default_ops.Linear}, got {type(layer)}"
        )
        assert layer.weight.dtype == torch.float16
        assert layer.weight.device.type == "cuda"

    # norm_out → local AdaLayerNormContinuous instance
    assert isinstance(model.norm_out, LocalAdaLNCont), (
        f"norm_out must be local AdaLayerNormContinuous, got {type(model.norm_out)}"
    )
    assert type(model.norm_out.linear) is default_ops.Linear, (
        f"norm_out.linear: expected {default_ops.Linear}, got {type(model.norm_out.linear)}"
    )

    # proj_out → default_ops.Linear with dtype/device propagated
    assert type(model.proj_out) is default_ops.Linear, (
        f"proj_out: expected {default_ops.Linear}, got {type(model.proj_out)}"
    )
    assert model.proj_out.weight.dtype == torch.float16, "proj_out.weight.dtype"
    assert model.proj_out.weight.device.type == "cuda", "proj_out.weight.device"


def test_transformer3d_top_level_ops_injection():
    """Explicit _MockOps3D injection: top-level layers must use marker types.

    Verifies x_embedder, context_embedder, time_text_embed, norm_out, proj_out.
    blocks internal layers are P3.2c scope — not checked here.
    """
    from models.transformer.ops_primitives import (
        PixArtAlphaTextProjection as LocalPixArtProj,
        AdaLayerNormContinuous as LocalAdaLNCont,
    )

    class _MarkerConv3d3D(nn.Conv3d):
        pass

    class _MarkerLinear3D(nn.Linear):
        pass

    class _MarkerLayerNorm3D(nn.LayerNorm):
        pass

    class _MockOps3D:
        Conv3d = _MarkerConv3d3D
        Linear = _MarkerLinear3D
        LayerNorm = _MarkerLayerNorm3D

    model = _make_small_transformer3d(operations=_MockOps3D)

    # x_embedder.proj → _MarkerConv3d3D
    assert isinstance(model.x_embedder.proj, _MarkerConv3d3D), (
        f"x_embedder.proj: expected _MarkerConv3d3D, got {type(model.x_embedder.proj)}"
    )

    # context_embedder → local PixArtAlphaTextProjection with _MarkerLinear3D inner layers
    assert isinstance(model.context_embedder, LocalPixArtProj), (
        f"context_embedder must be local PixArtAlphaTextProjection, got {type(model.context_embedder)}"
    )
    assert type(model.context_embedder.linear_1) is _MarkerLinear3D, (
        f"context_embedder.linear_1: expected _MarkerLinear3D, got {type(model.context_embedder.linear_1)}"
    )
    assert type(model.context_embedder.linear_2) is _MarkerLinear3D, (
        f"context_embedder.linear_2: expected _MarkerLinear3D, got {type(model.context_embedder.linear_2)}"
    )

    # time_text_embed.timestep_embedder → _MarkerLinear3D inner layers
    assert type(model.time_text_embed.timestep_embedder.linear_1) is _MarkerLinear3D, (
        f"timestep_embedder.linear_1: expected _MarkerLinear3D, got "
        f"{type(model.time_text_embed.timestep_embedder.linear_1)}"
    )
    assert type(model.time_text_embed.timestep_embedder.linear_2) is _MarkerLinear3D, (
        f"timestep_embedder.linear_2: expected _MarkerLinear3D, got "
        f"{type(model.time_text_embed.timestep_embedder.linear_2)}"
    )

    # norm_out → local AdaLayerNormContinuous with _MarkerLinear3D
    assert isinstance(model.norm_out, LocalAdaLNCont), (
        f"norm_out must be local AdaLayerNormContinuous, got {type(model.norm_out)}"
    )
    assert type(model.norm_out.linear) is _MarkerLinear3D, (
        f"norm_out.linear: expected _MarkerLinear3D, got {type(model.norm_out.linear)}"
    )

    # proj_out → _MarkerLinear3D
    assert type(model.proj_out) is _MarkerLinear3D, (
        f"proj_out: expected _MarkerLinear3D, got {type(model.proj_out)}"
    )


def test_transformer3d_image_embedder_branch():
    """image_embed_dim=<int> branch must create image_embedder with ops propagated.

    Covers the conditional at line 985-990 of MotifVideoTransformer3DModel.__init__
    that the default-fallback test skips (it uses image_embed_dim=None).
    """
    default_ops = comfy.ops.disable_weight_init
    model = _make_small_transformer3d(
        image_embed_dim=512,
        operations=None,
        dtype=torch.float16,
        device="cuda",
    )
    assert hasattr(model, "image_embedder"), "image_embedder must be created when image_embed_dim is not None"
    # image_embedder is a MotifVideoImageProjection instance — check its inner layers got ops propagated
    for attr in ("norm_in", "norm_out"):
        layer = getattr(model.image_embedder, attr)
        assert type(layer) is default_ops.LayerNorm, f"image_embedder.{attr}: expected LayerNorm, got {type(layer)}"
        assert layer.weight.dtype == torch.float16
        assert layer.weight.device.type == "cuda"
    for attr in ("linear_1", "linear_2"):
        layer = getattr(model.image_embedder, attr)
        assert type(layer) is default_ops.Linear, f"image_embedder.{attr}: expected Linear, got {type(layer)}"
        assert layer.weight.dtype == torch.float16
        assert layer.weight.device.type == "cuda"
