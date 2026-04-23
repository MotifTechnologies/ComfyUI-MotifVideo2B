#!/bin/bash
# run_measure.sh — one-shot fp8 workflow measurement wrapper.
#
# Usage:
#   bash scripts/run_measure.sh <CONDITION>
#     CONDITION: B1 | B2 | F2 | F3
#     (F1 requires manual pre-fix file swap — see measure_README.md)
#
# What this does:
#   1. Starts nvidia-smi polling in background (scoped to the GPU you run on).
#   2. Launches ComfyUI in foreground with the right --highvram flag.
#   3. You run the workflow in the ComfyUI UI (browser).
#   4. When the workflow is done, press Ctrl+C here in this terminal.
#      The script then extracts VRAM peak, parses the log, and prints a summary.
#
# Core check (most important):
#   bash scripts/run_measure.sh F2
#   → look for: fallback_log_count=0 and vram_peak_gb <= 40

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFY_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

CONDITION="${1:-}"
if [[ -z "$CONDITION" ]]; then
    cat <<'EOF'
Usage: bash scripts/run_measure.sh <CONDITION>
  CONDITION: B1 | B2 | F2 | F3

Quick check (recommended):
  bash scripts/run_measure.sh F2
    → This is the fp8 fix validation. Success when:
      - fallback_log_count = 0
      - vram_peak_gb <= 40

Condition map:
  B1 = bf16 + --highvram     (speed ceiling, bf16 regression guard)
  B2 = bf16 + NORMAL_VRAM    (prior plan staged baseline)
  F2 = fp8_e4m3fn + NORMAL_VRAM  (CORE pass/fail)
  F3 = fp8_e4m3fn + --highvram   (speed target)

Before running:
  In the ComfyUI UI, set Load Diffusion Model weight_dtype to
  match the condition (bfloat16 or fp8_e4m3fn).

Flow:
  1. Script starts nvidia-smi polling + ComfyUI.
  2. You run the workflow in the browser.
  3. Ctrl+C this terminal when done. Script auto-parses.
EOF
    exit 1
fi

case "$CONDITION" in
    B1|F3) HIGHVRAM=1 ;;
    B2|F2) HIGHVRAM=0 ;;
    F1)
        echo "ERROR: F1 requires pre-fix file swap. See:"
        echo "  .plans/20260423-fp8-bias-fallback-fix/logs/measure_README.md"
        echo "  section 'F1 (fp8 현재, 수정 전)'"
        exit 1
        ;;
    *)
        echo "ERROR: invalid CONDITION '$CONDITION'. Expected B1|B2|F2|F3."
        exit 1
        ;;
esac

LOG_DIR="$REPO_ROOT/.plans/20260423-fp8-bias-fallback-fix/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${CONDITION}.log"
POLL_FILE="/tmp/${CONDITION}_vram_poll.log"

# Scope nvidia-smi to the GPU ComfyUI actually runs on.
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

POLL_PID=""

cleanup() {
    trap '' EXIT INT TERM
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ Measurement finished: ${CONDITION}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -n "$POLL_PID" ]] && kill -0 "$POLL_PID" 2>/dev/null; then
        kill "$POLL_PID" 2>/dev/null || true
        wait "$POLL_PID" 2>/dev/null || true
    fi

    if [[ -s "$POLL_FILE" ]]; then
        max_mib="$(sort -n "$POLL_FILE" | tail -1)"
        if [[ -n "$max_mib" ]]; then
            max_gb="$(awk "BEGIN{printf \"%.2f\", ${max_mib}/1024}")"
            echo "VRAM_PEAK_GB=${max_gb}" >> "$LOG_FILE"
            echo "▶ VRAM peak: ${max_gb} GB (raw max MiB = ${max_mib})"
        fi
    else
        echo "▶ WARN: VRAM poll file empty — vram_peak_gb will be 'unknown'"
    fi

    echo ""
    echo "--- Parsed key=value ---"
    bash "$REPO_ROOT/scripts/measure_fp8_workflow.sh" "$CONDITION" "$LOG_FILE" || true
    echo ""
    echo "▶ Full log: $LOG_FILE"
    echo "▶ Poll log: $POLL_FILE"
    echo ""
    if [[ "$CONDITION" == "F2" ]]; then
        echo "Core pass/fail check:"
        echo "  fallback_log_count=0  AND  vram_peak_gb <= 40  →  PASS"
    fi
}
trap cleanup EXIT INT TERM

echo "▶ Condition   : ${CONDITION}"
echo "▶ GPU_ID      : ${GPU_ID}  (from CUDA_VISIBLE_DEVICES or default 0)"
echo "▶ Launch flag : $([[ $HIGHVRAM -eq 1 ]] && echo '--highvram' || echo '(NORMAL_VRAM / default)')"
echo "▶ Log file    : ${LOG_FILE}"
echo "▶ Poll file   : ${POLL_FILE}"
echo ""
echo "Next steps (do NOT close this terminal):"
echo "  1) Open ComfyUI browser UI (already running after launch below)."
echo "  2) In Load Diffusion Model node: set weight_dtype to"
echo "     $([[ ${CONDITION:0:1} == 'F' ]] && echo 'fp8_e4m3fn' || echo 'bfloat16')."
echo "  3) Queue the workflow (workflows/i2v_example.json — 50 steps)."
echo "  4) When the workflow finishes, press Ctrl+C here."
echo "     The script will auto-extract VRAM peak and print a summary."
echo ""
echo "Launching ComfyUI in 3 seconds..."
sleep 3

# Start VRAM polling (background).
nvidia-smi \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits \
    --id="$GPU_ID" \
    -l 1 \
    > "$POLL_FILE" &
POLL_PID=$!

# Launch ComfyUI in foreground. User presses Ctrl+C when done.
cd "$COMFY_ROOT"
if [[ $HIGHVRAM -eq 1 ]]; then
    python main.py --highvram 2>&1 | tee "$LOG_FILE"
else
    python main.py 2>&1 | tee "$LOG_FILE"
fi
