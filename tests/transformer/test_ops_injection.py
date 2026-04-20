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

from models.transformer.transformer_motif_video import MotifVideoPatchEmbed, MotifVideoAdaNorm
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
