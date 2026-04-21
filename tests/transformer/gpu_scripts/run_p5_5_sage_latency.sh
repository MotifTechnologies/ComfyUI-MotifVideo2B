#!/usr/bin/env bash
# P5.5: sage attention latency smoke — 2-hour regression guard (R2).
#
# Applies apply_sage_attention to bf16-loaded transformer, then measures
# wall-clock time for forward × STEP_COUNT iterations (default 50 steps).
# This approximates end-to-end sampling latency without the full ComfyUI
# sampler loop.
#
# Outputs (written to .plans/20260421-fp8-phase2-attention-replace/logs/):
#   p5_5_sage_latency.log — required verify fields (elapsed_seconds, use_sage=True, ...)
#
# Usage (from ComfyUI-MotifVideo1.9B dir on a GPU pod):
#   bash tests/transformer/gpu_scripts/run_p5_5_sage_latency.sh
#
# Provide baseline for relative comparison (optional but recommended):
#   MOTIF_BASELINE_SECONDS=1200 bash tests/transformer/gpu_scripts/run_p5_5_sage_latency.sh
#
# Optional env overrides:
#   MOTIF_CHECKPOINT_PATH    — override default checkpoint path
#   MOTIF_BASELINE_SECONDS   — baseline elapsed seconds from main branch (used for delta calc)
#   MOTIF_STEP_COUNT         — number of forward steps to simulate (default: 50)
#   COMFYUI_ROOT             — override ComfyUI root (default: ../../..)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"

export PYTHONPATH="${PYTHONPATH:-}:${COMFYUI_ROOT}"

echo "[P5.5] repo_root=${REPO_ROOT}"
echo "[P5.5] comfyui_root=${COMFYUI_ROOT}"
echo "[P5.5] checkpoint=${MOTIF_CHECKPOINT_PATH:-<default>}"
echo "[P5.5] baseline_seconds=${MOTIF_BASELINE_SECONDS:-<not provided>}"
echo "[P5.5] step_count=${MOTIF_STEP_COUNT:-50}"

cd "$REPO_ROOT"
exec "${COMFYUI_ROOT}/.venv/bin/python" -u \
    tests/transformer/gpu_scripts/run_p5_5_sage_latency.py
