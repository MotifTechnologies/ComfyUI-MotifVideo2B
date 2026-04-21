#!/usr/bin/env bash
# P5.2: fp16 checkpoint load + 1-step forward smoke.
#
# Outputs (written to .plans/20260421-fp8-phase2-attention-replace/logs/):
#   p5_2_fp16_smoke.log   — required verify fields (STATUS=OK, nan_count=0, ...)
#
# Usage (from ComfyUI-MotifVideo1.9B dir on a GPU pod):
#   bash tests/transformer/gpu_scripts/run_p5_2_fp16_smoke.sh
#
# Optional env overrides:
#   MOTIF_CHECKPOINT_PATH   — override default checkpoint path
#   COMFYUI_ROOT            — override ComfyUI root (default: ../../..)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"

export PYTHONPATH="${PYTHONPATH:-}:${COMFYUI_ROOT}"

echo "[P5.2] repo_root=${REPO_ROOT}"
echo "[P5.2] comfyui_root=${COMFYUI_ROOT}"
echo "[P5.2] checkpoint=${MOTIF_CHECKPOINT_PATH:-<default>}"

cd "$REPO_ROOT"
exec "${COMFYUI_ROOT}/.venv/bin/python" -u \
    tests/transformer/gpu_scripts/run_p5_2_fp16_smoke.py
