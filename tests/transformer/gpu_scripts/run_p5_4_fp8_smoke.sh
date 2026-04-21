#!/usr/bin/env bash
# P5.4: fp8 checkpoint load + 1-step smoke + cosine similarity vs bf16 baseline.
#
# PREREQUISITE: P5.1 must have run first (generates p5_input_seed.pt and p5_1_bf16_output.pt).
# PREREQUISITE: P5.3 must have flipped config.py optimizations = {"fp8": True}.
#
# Outputs (written to .plans/20260421-fp8-phase2-attention-replace/logs/):
#   p5_4_fp8_smoke.log    — required verify fields (cosine_similarity, STATUS=OK, ...)
#   p5_4_fp8_output.pt    — fp8 output tensor
#
# Usage (from ComfyUI-MotifVideo1.9B dir on a GPU pod):
#   bash tests/transformer/gpu_scripts/run_p5_4_fp8_smoke.sh
#
# Optional env overrides:
#   MOTIF_CHECKPOINT_PATH   — override default checkpoint path
#   COMFYUI_ROOT            — override ComfyUI root (default: ../../..)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"

export PYTHONPATH="${PYTHONPATH:-}:${COMFYUI_ROOT}"

echo "[P5.4] repo_root=${REPO_ROOT}"
echo "[P5.4] comfyui_root=${COMFYUI_ROOT}"
echo "[P5.4] checkpoint=${MOTIF_CHECKPOINT_PATH:-<default>}"

cd "$REPO_ROOT"
exec "${COMFYUI_ROOT}/.venv/bin/python" -u \
    tests/transformer/gpu_scripts/run_p5_4_fp8_smoke.py
