"""P5.1 bf16 smoke entrypoint. Runs on a GPU pod.

Loads the MotifVideo transformer with weight_dtype=bfloat16, runs one forward
pass with the shared seeded input, and writes:
  - logs/p5_1_bf16_smoke.log   — required verify fields
  - logs/p5_1_bf16_output.pt   — bf16 output tensor (P5.4 baseline)
  - logs/p5_input_seed.pt      — shared hidden_states + encoder_hidden_states + seed

Required log fields (checked by checklist verify commands):
  weight_dtype=bfloat16
  output_shape=[...]
  nan_count=0
  inf_count=0
  baseline_saved=p5_1_bf16_output.pt
  STATUS=OK
"""
from __future__ import annotations

import sys
import time
import traceback

import torch

# Resolve via PYTHONPATH=<ComfyUI> so that both comfy.* and tests.* are importable.
from tests.transformer.gpu_scripts._common import (
    CHECKPOINT_PATH,
    ensure_logs_dir,
    generate_or_load_input_seed,
    count_nan_inf,
    shape_str,
    _log_line,
    _checkpoint_sig,
    _input_seed_sig,
    load_transformer_bf16,
    assert_attn_weights_loaded,
)


def main() -> int:
    logs = ensure_logs_dir()
    log_path = logs / "p5_1_bf16_smoke.log"
    out_path = logs / "p5_1_bf16_output.pt"

    with open(log_path, "w") as f:
        try:
            _log_line(f, "script", "p5_1_bf16_smoke")
            _log_line(f, "weight_dtype", "bfloat16")
            _log_line(f, "checkpoint_path", str(CHECKPOINT_PATH))
            _log_line(f, "cuda_available", torch.cuda.is_available())

            assert torch.cuda.is_available(), "GPU required for P5.1"
            _log_line(f, "device", torch.cuda.get_device_name(0))

            transformer, sd = load_transformer_bf16(CHECKPOINT_PATH, device="cuda")
            _log_line(f, "model_loaded", True)

            # Runtime dtype verification: at least one parameter must be bfloat16.
            _log_line(f, "weight_dtype_requested", "bfloat16")
            bf16_count = sum(1 for _, p in transformer.named_parameters() if p.dtype == torch.bfloat16)
            _log_line(f, "bf16_param_count", bf16_count)
            _log_line(f, "weight_dtype_actual", "bfloat16" if bf16_count > 0 else "NOT_BFLOAT16")
            assert bf16_count > 0, (
                f"P5.1 loader regression: 0 bfloat16 parameters found. "
                "The bf16 load path is broken."
            )

            # Sanity-check: no attn.* missing keys across all blocks (plan requirement R1).
            if sd is not None:
                model_keys = set(transformer.state_dict().keys())
                ckpt_keys = set(sd.keys())
                assert_attn_weights_loaded(model_keys, ckpt_keys, f)
            else:
                _log_line(f, "attn_missing_keys_count", "skipped")

            hs, eh, seed = generate_or_load_input_seed()
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

            ckpt_sig = _checkpoint_sig()
            seed_sig = _input_seed_sig()
            torch.save(
                {
                    "output": out.detach().cpu(),
                    "checkpoint_sig": ckpt_sig,
                    "weight_dtype": "bfloat16",
                    "input_seed_sig": seed_sig,
                },
                out_path,
            )
            _log_line(f, "baseline_saved", "p5_1_bf16_output.pt")
            _log_line(f, "checkpoint_sig", ckpt_sig)
            _log_line(f, "input_seed_sig", seed_sig)
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
