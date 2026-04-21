"""Shared helpers for P5.* GPU smoke scripts.

Assumed execution context: GPU pod with CUDA available.
Import path: tests.transformer.gpu_scripts._common (via PYTHONPATH=<ComfyUI>)
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import time
from typing import Tuple

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PLAN_LOGS = REPO_ROOT / ".plans/20260421-fp8-phase2-attention-replace/logs"
INPUT_SEED_PT = PLAN_LOGS / "p5_input_seed.pt"

CHECKPOINT_PATH = pathlib.Path(os.environ.get(
    "MOTIF_CHECKPOINT_PATH",
    "/lustrefs/team-multimodal/checkpoints/base_checkpoint/model_cross_attn_18_550/transformer/diffusion_pytorch_model.safetensors",
))


def _checkpoint_sig() -> str:
    """Small signature of the checkpoint for staleness detection.

    Uses file path + size + mtime_ns as the fingerprint so that any re-save
    of the checkpoint (even with identical content) invalidates a stale baseline.
    Returns a 16-char hex prefix of the SHA-256 digest.
    """
    path = pathlib.Path(str(CHECKPOINT_PATH))
    st = path.stat()
    raw = f"{path}:{st.st_size}:{st.st_mtime_ns}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _input_seed_sig() -> str:
    """Signature of the shared input seed tensors.

    Captures seed value, shapes, dtypes, and element sums so that any accidental
    regeneration of the seed file (shape/dtype/value change) invalidates baselines
    that relied on a different seed.  Returns a 16-char hex prefix.
    """
    hs, eh, seed = generate_or_load_input_seed()
    raw = (
        f"{seed}:{tuple(hs.shape)}:{hs.dtype}:{hs.sum().item():.6f}"
        f":{tuple(eh.shape)}:{eh.dtype}:{eh.sum().item():.6f}"
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _log_line(f, key: str, value) -> None:
    """Write key=value to both stdout and the open log file."""
    line = f"{key}={value}"
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def ensure_logs_dir() -> pathlib.Path:
    PLAN_LOGS.mkdir(parents=True, exist_ok=True)
    return PLAN_LOGS


def generate_or_load_input_seed() -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (hidden_states, encoder_hidden_states, seed).

    If INPUT_SEED_PT already exists (written by P5.1), load it so that P5.2/P5.4/P5.5
    use the identical input tensors — required for bit-comparable cosine similarity.
    Otherwise, generate with a fixed seed and persist to INPUT_SEED_PT.

    Tensor shapes match the MotifVideo production forward signature:
      hidden_states       : [B=1, C_noise=33, T=33, H=40, W=72]  (noise latent)
      encoder_hidden_states: [B=1, L=77, D=3584]                   (text encoder dim)
    """
    ensure_logs_dir()
    if INPUT_SEED_PT.exists():
        blob = torch.load(INPUT_SEED_PT, map_location="cpu", weights_only=True)
        return blob["hidden_states"], blob["encoder_hidden_states"], int(blob["seed"])
    seed = 42
    g = torch.Generator().manual_seed(seed)
    B, C_noise, T, H, W = 1, 33, 33, 40, 72
    hs = torch.randn(B, C_noise, T, H, W, generator=g, dtype=torch.float32)
    eh = torch.randn(B, 77, 3584, generator=g, dtype=torch.float32)
    torch.save({"hidden_states": hs, "encoder_hidden_states": eh, "seed": seed}, INPUT_SEED_PT)
    return hs, eh, seed


def count_nan_inf(t: torch.Tensor) -> Tuple[int, int]:
    """Return (nan_count, inf_count) for a tensor."""
    nan = int(torch.isnan(t).sum().item())
    inf = int(torch.isinf(t).sum().item())
    return nan, inf


def shape_str(t: torch.Tensor) -> str:
    """Return shape as '[d0,d1,...]' (no spaces — grep-friendly)."""
    return "[" + ",".join(str(d) for d in t.shape) + "]"


def assert_attn_weights_loaded(model_keys: set, ckpt_keys: set, f) -> None:
    """Fail if any transformer attention weight is missing from the checkpoint.

    Checks repo-wide — all blocks (transformer_blocks.N and
    single_transformer_blocks.N), not just block 0.
    """
    expected_attn_keys = [k for k in model_keys if ".attn." in k]
    attn_missing = [k for k in expected_attn_keys if k not in ckpt_keys]
    _log_line(f, "attn_expected_count", len(expected_attn_keys))
    _log_line(f, "attn_missing_keys_count", len(attn_missing))
    if attn_missing:
        _log_line(f, "attn_missing_sample", attn_missing[:5])
    assert len(attn_missing) == 0, f"missing attn keys (first 5): {attn_missing[:5]}"


def _load_transformer_via_config(
    checkpoint_path: pathlib.Path,
    weight_dtype: torch.dtype,
    device: str = "cuda",
):
    """Internal helper: load transformer via MotifVideo19B config path.

    Uses MotifVideo19B directly (not comfy.sd.load_diffusion_model with
    model_options) so that config.optimizations propagates correctly through
    MotifVideoModel.__init__ → pick_operations.

    Returns (transformer, sd) where sd is the raw safetensors dict (CPU tensors).

    Memory note: checkpoint is loaded to CPU first to avoid duplicate VRAM
    residency during load_model_weights (cuda sd + cuda model params ≈ 2×VRAM
    on the 19B model).  After weights are transferred to the model, the internal
    sd reference is released and the CUDA cache is flushed; the returned sd
    contains CPU tensors only.
    """
    import gc
    from safetensors.torch import load_file
    from config import MotifVideo19B
    from models import MotifVideoModel

    # Load checkpoint bytes to CPU first to avoid duplicate VRAM residency.
    sd = load_file(str(checkpoint_path), device="cpu")

    # Inject weight storage dtype so pick_operations sees it.
    unet_config_orig = dict(MotifVideo19B.unet_config)
    MotifVideo19B.unet_config = dict(unet_config_orig)
    MotifVideo19B.unet_config["dtype"] = weight_dtype

    try:
        model_config = MotifVideo19B(unet_config=MotifVideo19B.unet_config)
        model = MotifVideoModel(model_config, device=device)
        model.load_model_weights(sd, unet_prefix="", assign=False)
    finally:
        MotifVideo19B.unet_config = unet_config_orig

    transformer = model.diffusion_model

    # Release the large tensor dict; weights are now owned by the model.
    # Callers only need sd.keys() (for assert_attn_weights_loaded), so we
    # hand back a lightweight key-only proxy instead of the full tensor dict.
    # This frees the CPU-side checkpoint copy and triggers a CUDA cache flush.
    sd_keys = list(sd.keys())
    del sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Wrap keys in a minimal dict-like object so callers can still do sd.keys()
    # and sd is not None checks without holding any tensor memory.
    class _KeysOnly(dict):
        """Minimal dict that holds only keys (no tensor values)."""
        pass

    sd_proxy = _KeysOnly({k: None for k in sd_keys})
    return transformer, sd_proxy


def load_transformer_bf16(checkpoint_path: pathlib.Path, device: str = "cuda"):
    """Load MotifVideo transformer with weight_dtype=bfloat16 via config path.

    Returns (transformer, sd). Uses MotifVideo19B config directly so that
    config.optimizations propagates through pick_operations — consistent with
    the production ComfyUI load path.
    """
    return _load_transformer_via_config(checkpoint_path, torch.bfloat16, device)


def load_transformer_fp16(checkpoint_path: pathlib.Path, device: str = "cuda"):
    """Load MotifVideo transformer with weight_dtype=float16 via config path."""
    return _load_transformer_via_config(checkpoint_path, torch.float16, device)


def load_transformer_fp8(checkpoint_path: pathlib.Path, device: str = "cuda"):
    """Load MotifVideo transformer via the production fp8 config flag path.

    Production fp8 routing:
      config.py → MotifVideo19B.optimizations["fp8"] == True
      → MotifVideoModel.__init__ calls comfy.ops.pick_operations(..., fp8_optimizations=True)
      → comfy.ops.fp8_ops wraps all ops.Linear / ops.RMSNorm / ops.Conv3d with manual_cast
      Weight storage dtype is fp8_e4m3fn; compute dtype is bf16 via fp8_ops.

    Prereq: P5.3 must have flipped config.MotifVideo19B.optimizations = {"fp8": True}.
    Using model_options={"dtype": torch.float8_e4m3fn} is wrong — it only changes the
    weight-load dtype and bypasses pick_operations entirely.
    """
    # Verify the config flag is True before proceeding.
    from config import MotifVideo19B
    assert MotifVideo19B.optimizations.get("fp8", False), (
        "P5.4 prereq failed: config.py optimizations['fp8'] must be True. "
        "Run P5.3 first (flip MotifVideo19B.optimizations = {\"fp8\": True})."
    )

    import comfy.sd  # noqa: F401

    # Load via the standard ComfyUI diffusion model path. ComfyUI will call
    # config.get_model() → MotifVideoModel.__init__, which reads
    # config.optimizations["fp8"] and calls pick_operations with fp8_optimizations=True,
    # resulting in comfy.ops.fp8_ops wrapping all injected layers.
    model = comfy.sd.load_diffusion_model(str(checkpoint_path))
    transformer = model.model.diffusion_model
    return transformer, None
