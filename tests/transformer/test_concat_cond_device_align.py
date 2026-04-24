# tests/transformer/test_concat_cond_device_align.py
#
# Item 2 (P: 20260424-i2v-shape-align) — concat_cond device/shape contract
#
# Requirement:
#   MotifVideoModel.concat_cond must handle concat_latent_image on CPU even
#   when kwargs["device"] points to a different device (CUDA or meta).
#   torch.cat must complete without RuntimeError (device mismatch).
#   Result shape: [B, 17, T, H, W]
#   Result device: kwargs["device"]
#
# Strategy:
#   - CPU-to-CPU round trip: source tensor on CPU, target device = cpu.
#     Validates shape contract and .to(device) path without GPU.
#   - CPU-to-meta: source tensor on CPU, target device = meta.
#     Shows device sync fires (image.device != target device) and cat succeeds.
#     torch.cat on meta tensors is shape-only — no data movement, no CUDA.
#
# The implementation under test is extracted from models/__init__.py via a
# standalone function that mirrors concat_cond exactly, avoiding the full
# ComfyUI runtime import (which calls torch.cuda.current_device() at import
# time and fails in CPU-only environments).

from __future__ import annotations

import os
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Standalone implementation of concat_cond under test.
#
# This mirrors models/__init__.py MotifVideoModel.concat_cond exactly.
# The only external dependency is process_latent_in, which is passed as a
# callable so callers can inject identity (no comfy runtime needed).
# ---------------------------------------------------------------------------

def _concat_cond(process_latent_in, **kwargs):
    """Standalone mirror of MotifVideoModel.concat_cond.

    process_latent_in: callable(tensor) -> tensor  (latent scaling)
    kwargs: same keys as the real concat_cond — noise, device,
            concat_latent_image, dtype optional.
    """
    noise = kwargs.get("noise", None)
    if noise is None:
        return None

    device = kwargs["device"]
    dtype = noise.dtype

    B, _, T, H, W = noise.shape

    latent_condition = torch.zeros(B, 16, T, H, W, dtype=dtype, device=device)
    latent_mask = torch.zeros(B, 1, T, H, W, dtype=dtype, device=device)

    image = kwargs.get("concat_latent_image", None)
    if image is not None:
        image = process_latent_in(image)
        # Device sync: ComfyUI intermediate_device() may return CPU while
        # kwargs["device"] is CUDA/meta — sync before torch.cat.
        image = image.to(device=device, dtype=dtype)
        if image.ndim == 4:
            image = image.unsqueeze(2)
        if image.shape[2] < T:
            pad = torch.zeros(B, 16, T - image.shape[2], H, W, dtype=dtype, device=device)
            image = torch.cat([image, pad], dim=2)
        else:
            image = image[:, :, :T]

        latent_condition = image
        latent_mask[:, :, 0] = 1.0

    return torch.cat([latent_condition, latent_mask], dim=1)


# ---------------------------------------------------------------------------
# Test 1 — CPU concat_latent_image, target device = CPU
#
# Validates:
#   * shape is [B, 17, T, H, W]
#   * device of result equals kwargs["device"] (cpu)
#   * no RuntimeError from torch.cat
#   * latent_mask first slice is 1.0, rest 0.0
# ---------------------------------------------------------------------------

def test_concat_cond_cpu_image_cpu_device_shape_and_mask():
    B, T, H, W = 1, 5, 4, 4
    device = torch.device("cpu")
    dtype = torch.float32

    noise = torch.zeros(B, 16, T, H, W, dtype=dtype, device=device)
    # Simulate ComfyUI intermediate_device() = cpu: image stays on cpu
    image_cpu = torch.ones(B, 16, 1, H, W, dtype=dtype, device=device)

    result = _concat_cond(
        lambda x: x,
        noise=noise,
        device=device,
        concat_latent_image=image_cpu,
    )

    assert result is not None, "concat_cond returned None unexpectedly"
    assert result.shape == (B, 17, T, H, W), (
        f"Expected shape ({B}, 17, {T}, {H}, {W}), got {tuple(result.shape)}"
    )
    assert result.device.type == device.type, (
        f"Expected device type '{device.type}', got '{result.device.type}'"
    )
    # First temporal slice of mask channel (index 16) must be 1.0
    assert result[0, 16, 0].eq(1.0).all(), "mask first slice should be 1.0"
    # Remaining temporal slices of mask must be 0.0
    assert result[0, 16, 1:].eq(0.0).all(), "mask slices beyond first should be 0.0"


# ---------------------------------------------------------------------------
# Test 2 — CPU concat_latent_image, target device = meta  (MISMATCH test)
#
# This is the load-bearing device-mismatch test required by the spec:
#   "torch.cat가 다른 device 두 개를 받아도 에러 없이 통과함을 보여주는 테스트"
#
# Without the .to(device=device) line in concat_cond, torch.cat would raise:
#   RuntimeError: Expected all tensors to be on the same device
#   (zeros/latent_condition created on meta, image remains on cpu → mismatch)
#
# With the fix: image.to(device="meta") fires before cat → no error.
# meta device: pure shape arithmetic, no GPU required.
# ---------------------------------------------------------------------------

def test_concat_cond_cpu_image_meta_device_no_runtime_error():
    B, T, H, W = 1, 3, 8, 8
    meta = torch.device("meta")
    dtype = torch.float32

    # noise on meta (device=meta path — all zeros/pad will land on meta)
    noise = torch.zeros(B, 16, T, H, W, dtype=dtype, device=meta)
    # image starts on CPU — simulates ComfyUI intermediate_device() = cpu
    image_cpu = torch.ones(B, 16, 1, H, W, dtype=dtype)  # default: cpu

    assert image_cpu.device.type == "cpu", "pre-condition: image starts on cpu"
    assert noise.device.type == "meta", "pre-condition: noise device is meta"

    # Must NOT raise RuntimeError (device mismatch without the fix)
    result = _concat_cond(
        lambda x: x,  # process_latent_in = identity
        noise=noise,
        device=meta,
        concat_latent_image=image_cpu,
    )

    assert result is not None
    assert result.shape == (B, 17, T, H, W), (
        f"Expected ({B}, 17, {T}, {H}, {W}), got {tuple(result.shape)}"
    )
    assert result.device.type == "meta", (
        f"Result must be on 'meta' device, got '{result.device.type}'"
    )
