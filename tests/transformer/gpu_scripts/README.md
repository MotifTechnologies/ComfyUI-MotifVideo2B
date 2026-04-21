# P5 GPU Smoke Scripts (fp8 Phase 2, #18)

## Prereq (GPU pod)
- CUDA available
- ComfyUI checked out at `../../..` relative to this repo (standard `ComfyUI/custom_nodes/<repo>` layout)
- `MOTIF_CHECKPOINT_PATH` env var optional (default:
  `/lustrefs/team-multimodal/checkpoints/base_checkpoint/model_cross_attn_18_550/transformer/diffusion_pytorch_model.safetensors`)

## Run order

1. **P5.1** — generates shared input seed + bf16 baseline output:
   ```
   bash tests/transformer/gpu_scripts/run_p5_1_bf16_smoke.sh
   ```
2. **P5.2** — fp16 smoke (uses same input seed written by P5.1):
   ```
   bash tests/transformer/gpu_scripts/run_p5_2_fp16_smoke.sh
   ```
3. **P5.3** — main kernel flips `config.py optimizations = {"fp8": True}` (no script; done on CPU pod)

4. **P5.4** — fp8 smoke + cosine similarity vs P5.1 baseline (requires P5.1 + P5.3 done):
   ```
   bash tests/transformer/gpu_scripts/run_p5_4_fp8_smoke.sh
   ```
5. **P5.5** — sage latency guard (provide baseline for delta reporting):
   ```
   MOTIF_BASELINE_SECONDS=1200 bash tests/transformer/gpu_scripts/run_p5_5_sage_latency.sh
   ```

## Outputs

All logs and tensors go to `.plans/20260421-fp8-phase2-attention-replace/logs/`:

| File | Written by | Purpose |
|------|-----------|---------|
| `p5_input_seed.pt` | P5.1 (first run) | Shared seeded input reused by P5.2/P5.4/P5.5 |
| `p5_1_bf16_smoke.log` | P5.1 | Verify fields: weight_dtype, output_shape, nan_count, STATUS |
| `p5_1_bf16_output.pt` | P5.1 | bf16 baseline tensor for cosine similarity in P5.4 |
| `p5_2_fp16_smoke.log` | P5.2 | Verify fields: weight_dtype=float16, nan_count, STATUS |
| `p5_4_fp8_smoke.log` | P5.4 | Verify fields: cosine_similarity, weight_dtype=fp8_e4m3fn, STATUS |
| `p5_4_fp8_output.pt` | P5.4 | fp8 output tensor |
| `p5_5_sage_latency.log` | P5.5 | Verify fields: elapsed_seconds, use_sage_all_true, STATUS |

## Acceptance criteria

Each script writes `STATUS=OK` on success, `STATUS=FAIL` + traceback on failure.

Key thresholds:
- P5.4: `cosine_similarity >= 0.98` (fp8 vs bf16 quality)
- P5.5: `elapsed_seconds <= 1320` (20 min + 10%; regression from 2-hour baseline)

## COMFYUI_ROOT override

If ComfyUI is not at the default location (two levels above this repo), set:
```
COMFYUI_ROOT=/path/to/ComfyUI bash tests/transformer/gpu_scripts/run_p5_1_bf16_smoke.sh
```

## Step count override (P5.5)

Default is 50 forward steps (flow-matching Euler default):
```
MOTIF_STEP_COUNT=20 bash tests/transformer/gpu_scripts/run_p5_5_sage_latency.sh
```
