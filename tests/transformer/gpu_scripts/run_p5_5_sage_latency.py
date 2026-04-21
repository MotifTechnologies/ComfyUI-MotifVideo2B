"""P5.5 sage latency entrypoint. Runs on a GPU pod.

Measures approximate end-to-end sampling wall-clock time by running
forward × STEP_COUNT iterations with sage attention enabled.

Success condition (R2): elapsed_per_step <= baseline_per_step * 1.10 (step-scaled threshold).
Failure (2-hour regression): sage path not activating — check use_sage flags.

Required log fields (checked by checklist verify commands):
  elapsed_seconds=<int>
  use_sage_all_true=True
  STATUS=OK

Optional (populated when MOTIF_BASELINE_SECONDS env var is provided):
  baseline_seconds=<int>
  delta_seconds=<int>
  delta_pct=<float>
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import datetime

import torch

from tests.transformer.gpu_scripts._common import (
    CHECKPOINT_PATH,
    ensure_logs_dir,
    generate_or_load_input_seed,
    count_nan_inf,
    shape_str,
    _log_line,
    load_transformer_bf16,
    assert_attn_weights_loaded,
)

# Number of forward steps to simulate (proxy for sampler loop).
# Flow-matching Euler default is 50 steps; override with MOTIF_STEP_COUNT.
_STEP_COUNT = int(os.environ.get("MOTIF_STEP_COUNT", "50"))
if _STEP_COUNT <= 0:
    raise ValueError(f"MOTIF_STEP_COUNT must be positive, got {_STEP_COUNT}")

# Baseline reference: 20 min (1200 s) at 50 steps.
# Per-step thresholds scale with _STEP_COUNT so that e.g. 10 steps still pass
# when each step is within the same wall-clock budget.
_BASELINE_SECONDS = float(os.environ.get("MOTIF_BASELINE_SECONDS", "1200"))
_BASELINE_PER_STEP = _BASELINE_SECONDS / 50.0   # always 50-step reference
_UPPER_PER_STEP = _BASELINE_PER_STEP * 1.10     # +10% tolerance


def main() -> int:
    logs = ensure_logs_dir()
    log_path = logs / "p5_5_sage_latency.log"

    with open(log_path, "w") as f:
        try:
            _log_line(f, "script", "p5_5_sage_latency")
            _log_line(f, "step_count", _STEP_COUNT)
            _log_line(f, "checkpoint_path", str(CHECKPOINT_PATH))
            _log_line(f, "cuda_available", torch.cuda.is_available())

            assert torch.cuda.is_available(), "GPU required for P5.5"
            _log_line(f, "device", torch.cuda.get_device_name(0))

            transformer, sd = load_transformer_bf16(CHECKPOINT_PATH, device="cuda")
            _log_line(f, "model_loaded", True)

            # Sanity-check: no attn.* missing keys across all blocks.
            if sd is not None:
                model_keys = set(transformer.state_dict().keys())
                ckpt_keys = set(sd.keys())
                assert_attn_weights_loaded(model_keys, ckpt_keys, f)
            else:
                _log_line(f, "attn_missing_keys_count", "skipped")

            # Apply sage attention (sets use_sage=True on all blocks).
            # compile_config.py::apply_sage_attention must be importable via PYTHONPATH.
            from models.compile_config import apply_sage_attention  # noqa: PLC0415
            apply_sage_attention(transformer)
            _log_line(f, "apply_sage_attention_called", True)

            # Verify all blocks have use_sage=True.
            dual_blocks = list(getattr(transformer, "transformer_blocks", []))
            single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
            all_blocks = dual_blocks + single_blocks
            assert all_blocks, "No transformer blocks found — check model structure."

            all_use_sage = all(getattr(b.attn, "use_sage", False) for b in all_blocks)
            _log_line(f, "use_sage_all_true", all_use_sage)
            assert all_use_sage, (
                "apply_sage_attention did not set use_sage=True on all blocks. "
                "Check MOTIFVIDEO_DISABLE_SAGE env var and sage availability."
            )

            # Dump per-block use_sage (first 4 + last 4 for brevity; full dump expensive).
            sample_blocks = all_blocks[:4] + (all_blocks[-4:] if len(all_blocks) > 8 else [])
            for i, b in enumerate(sample_blocks):
                idx = i if i < 4 else len(all_blocks) - (len(sample_blocks) - i)
                _log_line(f, f"block_{idx}_use_sage", getattr(b.attn, "use_sage", None))
            _log_line(f, "total_blocks_checked", len(all_blocks))

            # Load shared seeded input.
            hs, eh, seed = generate_or_load_input_seed()
            hs = hs.to("cuda", dtype=torch.bfloat16)
            eh = eh.to("cuda", dtype=torch.bfloat16)
            timestep = torch.tensor([500], device="cuda", dtype=torch.float32)
            _log_line(f, "seed", seed)

            transformer.eval()

            # Warm-up: 1 step excluded from timing.
            with torch.no_grad():
                _ = transformer(hs, timestep=timestep, encoder_hidden_states=eh)
            torch.cuda.synchronize()

            # Timed loop: _STEP_COUNT forward passes.
            start_ts = datetime.datetime.utcnow().isoformat() + "Z"
            _log_line(f, "sampling_start_utc", start_ts)
            t0 = time.time()

            with torch.no_grad():
                for step in range(_STEP_COUNT):
                    out = transformer(hs, timestep=timestep, encoder_hidden_states=eh)
                    # Vary timestep as a real sampler would (simple linear schedule).
                    t_val = max(1, 500 - step * (500 // _STEP_COUNT))
                    timestep = torch.tensor([t_val], device="cuda", dtype=torch.float32)

            torch.cuda.synchronize()
            total_elapsed = time.time() - t0

            end_ts = datetime.datetime.utcnow().isoformat() + "Z"
            _log_line(f, "sampling_end_utc", end_ts)

            elapsed_seconds = int(total_elapsed)
            elapsed_per_step = total_elapsed / _STEP_COUNT
            _log_line(f, "elapsed_seconds", elapsed_seconds)
            _log_line(f, "elapsed_per_step", f"{elapsed_per_step:.3f}")
            _log_line(f, "baseline_per_step", f"{_BASELINE_PER_STEP:.3f}")
            _log_line(f, "upper_per_step", f"{_UPPER_PER_STEP:.3f}")

            # Baseline comparison (optional — requires MOTIF_BASELINE_SECONDS env override).
            baseline_str = os.environ.get("MOTIF_BASELINE_SECONDS", "")
            if baseline_str.strip():
                baseline_secs = int(baseline_str.strip())
                delta = elapsed_seconds - baseline_secs
                delta_pct = (delta / baseline_secs) * 100.0 if baseline_secs > 0 else float("nan")
                _log_line(f, "baseline_seconds", baseline_secs)
                _log_line(f, "delta_seconds", delta)
                _log_line(f, "delta_pct", f"{delta_pct:.1f}")

            # Validate final output for NaN/Inf.
            nan, inf_c = count_nan_inf(out)
            _log_line(f, "final_output_shape", shape_str(out))
            _log_line(f, "final_nan_count", nan)
            _log_line(f, "final_inf_count", inf_c)

            assert nan == 0 and inf_c == 0, f"NaN/Inf in final output: nan={nan}, inf={inf_c}"
            assert elapsed_per_step <= _UPPER_PER_STEP, (
                f"sage latency regression: {elapsed_per_step:.2f}s/step > "
                f"{_UPPER_PER_STEP:.2f}s/step "
                f"(baseline {_BASELINE_PER_STEP:.2f}s/step + 10%)"
            )

            _log_line(f, "STATUS", "OK")
            return 0

        except Exception as exc:
            _log_line(f, "error", type(exc).__name__)
            _log_line(f, "error_message", str(exc).replace("\n", " "))
            f.write(traceback.format_exc() + "\n")
            f.write("STATUS=FAIL\n")
            return 1


if __name__ == "__main__":
    sys.exit(main())
