"""P5.4 fp8 smoke entrypoint. Runs on a GPU pod.

Prerequisites:
  - P5.1 ran successfully (p5_input_seed.pt and p5_1_bf16_output.pt exist in logs/).
  - P5.3 flipped config.py: optimizations = {"fp8": True}.

Loads the MotifVideo transformer with fp8 ops, runs one forward pass using the
SAME seeded input as P5.1, and computes cosine similarity against the bf16
baseline output.

Writes:
  - logs/p5_4_fp8_smoke.log   — required verify fields
  - logs/p5_4_fp8_output.pt   — fp8 output tensor

Required log fields (checked by checklist verify commands):
  weight_dtype=fp8_e4m3fn
  output_shape=[...]
  nan_count=0
  inf_count=0
  cosine_similarity=<float>   (must be >= 0.98)
  STATUS=OK
"""
from __future__ import annotations

import sys
import time
import traceback

import torch
import torch.nn.functional as F

from tests.transformer.gpu_scripts._common import (
    CHECKPOINT_PATH,
    PLAN_LOGS,
    ensure_logs_dir,
    generate_or_load_input_seed,
    count_nan_inf,
    shape_str,
    _log_line,
    _checkpoint_sig,
    _input_seed_sig,
    load_transformer_fp8,
    assert_attn_weights_loaded,
)

BF16_BASELINE_PT = PLAN_LOGS / "p5_1_bf16_output.pt"


def main() -> int:
    logs = ensure_logs_dir()
    log_path = logs / "p5_4_fp8_smoke.log"
    out_path = logs / "p5_4_fp8_output.pt"

    with open(log_path, "w") as f:
        try:
            _log_line(f, "script", "p5_4_fp8_smoke")
            _log_line(f, "weight_dtype", "fp8_e4m3fn")
            _log_line(f, "checkpoint_path", str(CHECKPOINT_PATH))
            _log_line(f, "cuda_available", torch.cuda.is_available())

            assert torch.cuda.is_available(), "GPU required for P5.4"
            _log_line(f, "device", torch.cuda.get_device_name(0))

            assert BF16_BASELINE_PT.exists(), (
                f"bf16 baseline not found at {BF16_BASELINE_PT}. "
                "Run P5.1 first."
            )

            transformer, sd = load_transformer_fp8(CHECKPOINT_PATH, device="cuda")
            _log_line(f, "model_loaded", True)

            # Runtime dtype verification: at least one parameter must be fp8.
            _log_line(f, "weight_dtype_requested", "fp8_e4m3fn")
            fp8_param_count = 0
            total_param_count = 0
            sample_dtypes: dict = {}
            for name, p in transformer.named_parameters():
                total_param_count += 1
                if ".to_q." in name or ".to_k." in name or ".to_v." in name:
                    if name not in sample_dtypes and len(sample_dtypes) < 5:
                        sample_dtypes[name] = str(p.dtype)
                if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    fp8_param_count += 1
            _log_line(f, "total_param_count", total_param_count)
            _log_line(f, "fp8_param_count", fp8_param_count)
            _log_line(f, "attn_sample_dtypes", sample_dtypes)
            _log_line(f, "weight_dtype_actual", "fp8_e4m3fn" if fp8_param_count > 0 else "NOT_FP8")
            assert fp8_param_count > 0, (
                f"P5.4 loader regression: 0 fp8 parameters found across {total_param_count} total. "
                f"Sample attn dtypes: {sample_dtypes}. "
                "The production fp8 path (config.optimizations={'fp8': True}) is broken."
            )

            # Sanity-check: no attn.* missing keys across all blocks.
            if sd is not None:
                model_keys = set(transformer.state_dict().keys())
                ckpt_keys = set(sd.keys())
                assert_attn_weights_loaded(model_keys, ckpt_keys, f)
            else:
                _log_line(f, "attn_missing_keys_count", "skipped")

            # Use the SAME input as P5.1 (loaded from p5_input_seed.pt).
            hs, eh, seed = generate_or_load_input_seed()
            # fp8 forward: cast inputs to bfloat16 (fp8 ops accept bf16 activation).
            hs = hs.to("cuda", dtype=torch.bfloat16)
            eh = eh.to("cuda", dtype=torch.bfloat16)
            timestep = torch.tensor([500], device="cuda", dtype=torch.float32)

            _log_line(f, "seed", seed)
            _log_line(f, "input_hidden_states_shape", shape_str(hs))
            _log_line(f, "input_encoder_shape", shape_str(eh))

            transformer.eval()
            t0 = time.time()
            with torch.no_grad():
                out = transformer(hs, timestep=timestep, encoder_hidden_states=eh)
            elapsed = time.time() - t0
            _log_line(f, "forward_seconds", f"{elapsed:.3f}")

            nan, inf = count_nan_inf(out)
            _log_line(f, "output_shape", shape_str(out))
            _log_line(f, "output_dtype", str(out.dtype))
            _log_line(f, "nan_count", nan)
            _log_line(f, "inf_count", inf)
            assert nan == 0 and inf == 0, f"NaN/Inf in output: nan={nan}, inf={inf}"

            torch.save(out.detach().cpu(), out_path)
            _log_line(f, "fp8_output_saved", "p5_4_fp8_output.pt")

            # Cosine similarity vs bf16 baseline (P5.4 quality gate).
            # Requires dict-with-metadata format produced by the current P5.1 script.
            blob = torch.load(BF16_BASELINE_PT, map_location="cpu", weights_only=False)
            if not isinstance(blob, dict):
                _log_line(f, "ERROR", "baseline 포맷이 legacy tensor. P5.1 을 현재 checkpoint 로 재실행하세요.")
                raise AssertionError(
                    f"{BF16_BASELINE_PT} is a legacy raw tensor without metadata. "
                    "Delete it and re-run run_p5_1_bf16_smoke.sh on the current checkpoint."
                )
            bf16_out = blob["output"]
            ckpt_sig_now = _checkpoint_sig()
            if blob.get("checkpoint_sig") != ckpt_sig_now:
                _log_line(f, "ERROR", (
                    f"checkpoint_sig mismatch: "
                    f"baseline={blob.get('checkpoint_sig')} vs current={ckpt_sig_now}"
                ))
                raise AssertionError(
                    "P5.4 baseline stale — re-run P5.1 on current checkpoint."
                )
            seed_sig_now = _input_seed_sig()
            if blob.get("input_seed_sig") != seed_sig_now:
                _log_line(f, "ERROR", (
                    f"input_seed_sig mismatch: "
                    f"baseline={blob.get('input_seed_sig')} vs current={seed_sig_now}"
                ))
                raise AssertionError(
                    "P5.4 input seed drift — "
                    "remove logs/p5_input_seed.pt and re-run P5.1."
                )
            _log_line(f, "baseline_format", "dict-with-metadata")
            _log_line(f, "baseline_checkpoint_sig", blob["checkpoint_sig"])
            _log_line(f, "baseline_input_seed_sig", blob["input_seed_sig"])

            fp8_flat = out.detach().cpu().float().flatten()
            bf16_flat = bf16_out.float().flatten()
            cos = F.cosine_similarity(bf16_flat.unsqueeze(0), fp8_flat.unsqueeze(0), dim=1).item()
            _log_line(f, "cosine_similarity", f"{cos:.4f}")
            assert cos >= 0.98, f"fp8 quality regression vs bf16 baseline: cos={cos:.4f} < 0.98"

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
