# tests/transformer/test_motif_video_model_wiring.py
#
# Verifies P4.1 checklist criteria:
#   1. MotifVideoModel passes operations/dtype/device to MotifVideoTransformer3DModel
#      via comfy.ops.pick_operations (wiring test, GPU only).
#   2. .to(dtype=torch.bfloat16) forced cast is absent from models/__init__.py
#      (static grep, no GPU required — in test_motif_video_model_no_cast.py).

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
# GPU skip guard — must precede any import of transformer_motif_video or
# comfy.model_management (calls torch.cuda.current_device() at import time).
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    pytest.skip("requires real comfy runtime (GPU)", allow_module_level=True)


import comfy.ops
from unittest.mock import patch, MagicMock

from models import MotifVideoModel


# ---------------------------------------------------------------------------
# Minimal mock model config — mirrors what MotifVideo19B produces at runtime.
# We mock the parts BaseModel.__init__ reads without pulling in the full
# supported_models machinery.
# ---------------------------------------------------------------------------

def _make_mock_config(*, manual_cast_dtype=None):
    """Return a minimal mock object compatible with BaseModel.__init__."""
    import comfy.latent_formats

    unet_config = {
        # MotifVideoTransformer3DModel minimal valid params
        "in_channels": 33,
        "out_channels": 16,
        "num_attention_heads": 4,
        "attention_head_dim": 64,
        "num_layers": 1,
        "num_single_layers": 1,
        "num_decoder_layers": 0,
        "text_embed_dim": 256,
        "image_embed_dim": None,
        "pooled_projection_dim": None,
        "rope_axes_dim": (16, 24, 24),
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "norm_type": "layer_norm",
        "enable_text_cross_attention_dual": False,
        "enable_text_cross_attention_single": False,
        # ComfyUI sets this at checkpoint load time (weight storage dtype).
        # Mock it so the wiring test verifies propagation of a real value
        # rather than accidentally asserting None-passthrough.
        "dtype": torch.float16,
    }

    cfg = MagicMock()
    cfg.unet_config = unet_config
    cfg.latent_format = comfy.latent_formats.SD15()
    cfg.manual_cast_dtype = manual_cast_dtype
    cfg.custom_operations = None
    cfg.optimizations = {}
    cfg.scaled_fp8 = None
    cfg.model_type = None
    cfg.sampling_settings = {}

    return cfg


# ---------------------------------------------------------------------------
# Test 1: wiring test — pick_operations is called and ops reach the transformer.
# ---------------------------------------------------------------------------

def test_motif_video_model_wiring_bf16():
    """MotifVideoModel.__init__ must call pick_operations and inject the resulting
    operations into MotifVideoTransformer3DModel.

    Verification strategy:
    - Patch comfy.ops.pick_operations to return a sentinel ops class with
      marker subclasses, then assert x_embedder.proj is that marker type.
    - This confirms the wiring: pick_operations → operations arg → transformer
      internal layers.

    manual_cast_dtype=None → pick_operations returns disable_weight_init
    (weight_dtype == compute_dtype == bfloat16 path).
    """
    import torch.nn as nn

    class _MarkerConv3d(nn.Conv3d):
        pass

    class _SentinelOps:
        Conv3d = _MarkerConv3d
        # Provide all ops attributes that MotifVideoTransformer3DModel uses.
        Linear = comfy.ops.disable_weight_init.Linear
        LayerNorm = comfy.ops.disable_weight_init.LayerNorm

    captured_calls = []

    def _mock_pick_operations(*args, **kwargs):
        captured_calls.append((args, kwargs))
        return _SentinelOps

    cfg = _make_mock_config(manual_cast_dtype=None)

    with patch("comfy.ops.pick_operations", side_effect=_mock_pick_operations):
        model = MotifVideoModel(cfg, device="cuda")

    # pick_operations must have been called exactly once.
    assert len(captured_calls) == 1, (
        f"pick_operations must be called exactly once, got {len(captured_calls)}"
    )

    # The sentinel ops must have reached x_embedder.proj (Conv3d injection).
    assert isinstance(model.diffusion_model.x_embedder.proj, _MarkerConv3d), (
        f"x_embedder.proj must be _MarkerConv3d (sentinel ops injected), "
        f"got {type(model.diffusion_model.x_embedder.proj)}"
    )


# ---------------------------------------------------------------------------
# Test 2: dtype/device/operations reach MotifVideoTransformer3DModel.__init__
# ---------------------------------------------------------------------------

def test_motif_video_model_wiring_args_captured():
    """MotifVideoTransformer3DModel.__init__ must receive dtype, device, and
    operations from MotifVideoModel.__init__.

    Verification strategy (approach B):
    - Monkey-patch MotifVideoTransformer3DModel.__init__ to capture the three
      keyword arguments before delegating to the original.
    - Assert dtype is not None, device matches the requested device, and
      operations is the sentinel returned by the mocked pick_operations.
    """
    import torch.nn as nn
    from models.transformer.transformer_motif_video import MotifVideoTransformer3DModel

    class _MarkerConv3d(nn.Conv3d):
        pass

    class _SentinelOps:
        Conv3d = _MarkerConv3d
        Linear = comfy.ops.disable_weight_init.Linear
        LayerNorm = comfy.ops.disable_weight_init.LayerNorm

    captured = {}
    _original_init = MotifVideoTransformer3DModel.__init__

    def _capture_init(self, *args, dtype=None, device=None, operations=None, **kwargs):
        captured["dtype"] = dtype
        captured["device"] = device
        captured["operations"] = operations
        return _original_init(self, *args, dtype=dtype, device=device, operations=operations, **kwargs)

    cfg = _make_mock_config(manual_cast_dtype=None)
    expected_device = "cuda"

    with patch("comfy.ops.pick_operations", return_value=_SentinelOps), \
         patch.object(MotifVideoTransformer3DModel, "__init__", _capture_init):
        MotifVideoModel(cfg, device=expected_device)

    assert captured.get("operations") is _SentinelOps, (
        f"operations must be the sentinel from pick_operations, got {captured.get('operations')}"
    )
    assert captured.get("dtype") is not None, "dtype must be passed to transformer __init__"
    assert captured.get("device") == expected_device, (
        f"device must be '{expected_device}', got {captured.get('device')}"
    )


# ---------------------------------------------------------------------------
# Test 3: custom_operations override — pick_operations must be bypassed
# ---------------------------------------------------------------------------

def test_motif_video_model_wiring_custom_operations_override():
    """When model_config.custom_operations is set, pick_operations must NOT be
    called and the custom ops must reach MotifVideoTransformer3DModel.__init__.

    Verification strategy:
    - Set cfg.custom_operations to a sentinel ops class.
    - Assert pick_operations is never called.
    - Assert transformer receives that exact sentinel as its operations arg.
    """
    import torch.nn as nn
    from models.transformer.transformer_motif_video import MotifVideoTransformer3DModel

    class _MockCustomConv3d(nn.Conv3d):
        pass

    class _MockCustomOps:
        Conv3d = _MockCustomConv3d
        Linear = comfy.ops.disable_weight_init.Linear
        LayerNorm = comfy.ops.disable_weight_init.LayerNorm

    captured = {}
    _original_init = MotifVideoTransformer3DModel.__init__

    def _capture_init(self, *args, dtype=None, device=None, operations=None, **kwargs):
        captured["operations"] = operations
        return _original_init(self, *args, dtype=dtype, device=device, operations=operations, **kwargs)

    cfg = _make_mock_config(manual_cast_dtype=None)
    cfg.custom_operations = _MockCustomOps  # override path

    pick_call_count = []

    def _should_not_be_called(*args, **kwargs):
        pick_call_count.append(1)
        return comfy.ops.disable_weight_init  # fallback to avoid crash

    with patch("comfy.ops.pick_operations", side_effect=_should_not_be_called), \
         patch.object(MotifVideoTransformer3DModel, "__init__", _capture_init):
        MotifVideoModel(cfg, device="cuda")

    assert len(pick_call_count) == 0, (
        f"pick_operations must NOT be called when custom_operations is set, "
        f"called {len(pick_call_count)} time(s)"
    )
    assert captured.get("operations") is _MockCustomOps, (
        f"transformer must receive custom_operations directly, "
        f"got {captured.get('operations')}"
    )
