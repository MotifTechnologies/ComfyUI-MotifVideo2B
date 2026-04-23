#!/usr/bin/env bash
# measure_fp8_workflow.sh — Parse a ComfyUI run log and emit key=value metrics
#
# Usage:
#   bash scripts/measure_fp8_workflow.sh <condition_tag> <log_file_path>
#
# Arguments:
#   condition_tag   : B1 | B2 | F1 | F2 | F3
#   log_file_path   : Path to a captured ComfyUI stdout/stderr log
#
# Output (stdout, one key=value per line):
#   vram_peak_gb              float   — peak GPU VRAM in GB
#   wallclock_seconds_per_step float  — s/it from the last tqdm progress line
#   fallback_log_count        int     — number of "Exception during fp8 op" lines
#   staged_mb                 int     — MB offloaded for the transformer model;
#                                       0 when loaded completely; "unknown" when
#                                       anchor not found (fail-closed)
#   loaded_completely         true|false|unknown
#   commit_sha                string  — git HEAD SHA of the repo
#   dtype                     bf16 | fp8_e4m3fn
#                             (only e4m3fn is in the current P3 matrix; e5m2 is
#                             not mapped by any B*/F* condition — add a new
#                             condition and extend map_dtype_vram to support it)
#   vram_mode                 highvram | normalvram
#   condition                 echoed condition_tag
#
# VRAM peak detection priority:
#   1. User-appended line in the log: VRAM_PEAK_GB=<float>
#   2. User-appended line in the log: MEMORY_USED_MIB=<int>
#      (from: nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
#   3. nvidia-smi inline line with pattern "<used> MiB / <total> MiB" —
#      the *used* (first) value is extracted; total is ignored.
#   4. Falls back to vram_peak_gb=unknown (exit 0 preserved)
#
# Staged MB detection:
#   Looks for the transformer model log line immediately after
#   "Requested to load MotifVideoModel" (anchor strategy).
#   If the anchor is absent, staged_mb=unknown is returned (fail-closed) and a
#   WARNING is written to stderr.  Silent fallback to other models is avoided.
#
# loaded_completely detection:
#   Same anchor strategy. If anchor is absent, loaded_completely=unknown is
#   returned (fail-closed) and a WARNING is written to stderr.
#
# Note: Wallclock is parsed from tqdm lines formatted as "29.36s/it".
#   If multiple values exist, the last one is used (most representative).

set -euo pipefail

# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $(basename "$0") <condition_tag> <log_file_path>" >&2
    echo "  condition_tag : B1 | B2 | F1 | F2 | F3" >&2
    echo "  log_file_path : path to ComfyUI run log" >&2
    exit 1
}

[[ $# -lt 2 ]] && usage

CONDITION_TAG="$1"
LOG_FILE="$2"

# ---------------------------------------------------------------------------
# validate inputs
# ---------------------------------------------------------------------------
validate_condition() {
    local tag="$1"
    case "$tag" in
        B1|B2|F1|F2|F3) ;;
        *)
            echo "ERROR: invalid condition '$tag'. Must be one of: B1 B2 F1 F2 F3" >&2
            exit 1
            ;;
    esac
}

validate_log_file() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "ERROR: log file not found: '$path'" >&2
        exit 1
    fi
    if [[ ! -r "$path" ]]; then
        echo "ERROR: log file not readable: '$path'" >&2
        exit 1
    fi
}

validate_condition "$CONDITION_TAG"
validate_log_file  "$LOG_FILE"

# ---------------------------------------------------------------------------
# map condition tag → dtype + vram_mode
# ---------------------------------------------------------------------------
map_dtype_vram() {
    local tag="$1"
    case "$tag" in
        B1) DTYPE="bf16";        VRAM_MODE="highvram"   ;;
        B2) DTYPE="bf16";        VRAM_MODE="normalvram" ;;
        F1) DTYPE="fp8_e4m3fn";  VRAM_MODE="normalvram" ;;
        F2) DTYPE="fp8_e4m3fn";  VRAM_MODE="normalvram" ;;
        F3) DTYPE="fp8_e4m3fn";  VRAM_MODE="highvram"   ;;
    esac
}

map_dtype_vram "$CONDITION_TAG"

# ---------------------------------------------------------------------------
# get_commit_sha — detect repo root via dirname of this script, then query git
# ---------------------------------------------------------------------------
get_commit_sha() {
    # This script lives in <repo_root>/scripts/, so repo root = one level up.
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local repo_root
    repo_root="$(dirname "$script_dir")"

    local sha
    if sha="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null)"; then
        echo "$sha"
    else
        echo "unknown"
    fi
}

# ---------------------------------------------------------------------------
# parse_wallclock — extract the last s/it value from tqdm progress lines
#
# ComfyUI tqdm format (observed):
#   " 16%|███...| 8/50 [03:54<20:33, 29.36s/it]"
# We want the float before "s/it".
# ---------------------------------------------------------------------------
parse_wallclock() {
    local log="$1"
    # Match "<float>s/it" — the last occurrence is the most representative.
    # grep -oE extracts all matches; tail -1 takes the last.
    local val
    val="$(grep -oE '[0-9]+\.[0-9]+s/it' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')" || true
    if [[ -n "$val" ]]; then
        echo "$val"
    else
        echo "unknown"
    fi
}

# ---------------------------------------------------------------------------
# parse_fallback_count — count "Exception during fp8 op" lines
# ---------------------------------------------------------------------------
parse_fallback_count() {
    local log="$1"
    local count
    count="$(grep -c 'Exception during fp8 op' "$log" 2>/dev/null)" || count=0
    echo "$count"
}

# ---------------------------------------------------------------------------
# parse_staged_mb — extract MB offloaded for the MotifVideoModel transformer
#
# Strategy (anchor-based, fail-closed):
#   1. Find the line immediately after "Requested to load MotifVideoModel"
#      (this is the model_patcher load log for the transformer).
#   2. If that line is "loaded partially; ...", extract the MB offloaded value.
#      MB values may be integer or float (e.g., "3742 MB offloaded" or
#      "3742.5 MB offloaded"); both are matched.
#   3. If that line is "loaded completely; ...", staged_mb = 0.
#   4. If anchor is absent: return "unknown" and emit a WARNING to stderr.
#      Silent fallback to other models' offloaded values is avoided.
#
# ComfyUI model_patcher.py log formats (confirmed from source):
#   loaded partially; <stat> X MB loaded, Y MB offloaded, ...
#   loaded completely; <stat> X MB loaded, full load: True
# ---------------------------------------------------------------------------
parse_staged_mb() {
    local log="$1"

    # Anchor on "Requested to load MotifVideoModel"
    # shellcheck disable=SC2016
    # Rationale: awk uses single-quoted program strings; $ must not be
    # expanded by the shell.
    local anchor_line
    anchor_line="$(awk '
        /Requested to load MotifVideoModel/ { found=1; next }
        found { print; exit }
    ' "$log" 2>/dev/null)" || anchor_line=""

    if [[ -n "$anchor_line" ]]; then
        # "loaded completely" line → staged_mb = 0
        if echo "$anchor_line" | grep -q 'loaded completely'; then
            echo "0"
            return
        fi
        # "loaded partially" line → extract MB offloaded (int or float)
        if echo "$anchor_line" | grep -q 'loaded partially'; then
            local offloaded
            # Pattern accepts both integer and float: "3742 MB offloaded" and
            # "3742.5 MB offloaded"; extract the integer part before the decimal.
            offloaded="$(echo "$anchor_line" | grep -oE '[0-9]+(\.[0-9]+)? MB offloaded' | grep -oE '^[0-9]+')" || offloaded=""
            if [[ -n "$offloaded" ]]; then
                echo "$offloaded"
                return
            fi
        fi
    fi

    # Anchor absent or line did not match expected patterns — fail-closed.
    # Do NOT fall back to other models' offloaded values.
    echo "WARNING: anchor 'Requested to load MotifVideoModel' not found;" \
         "staged_mb defaulting to unknown" >&2
    echo "unknown"
}

# ---------------------------------------------------------------------------
# parse_loaded_completely — true|false|unknown for the transformer load state
#
# Same anchor strategy as parse_staged_mb (fail-closed).
# If the anchor "Requested to load MotifVideoModel" is absent, returns
# "unknown" and emits a WARNING to stderr.  Falling back to other models'
# "loaded completely" lines is avoided to prevent misclassification.
# ---------------------------------------------------------------------------
parse_loaded_completely() {
    local log="$1"

    # Anchor on "Requested to load MotifVideoModel"
    # shellcheck disable=SC2016
    # Rationale: awk single-quoted program; $ must not expand in shell.
    local anchor_line
    anchor_line="$(awk '
        /Requested to load MotifVideoModel/ { found=1; next }
        found { print; exit }
    ' "$log" 2>/dev/null)" || anchor_line=""

    if [[ -n "$anchor_line" ]]; then
        if echo "$anchor_line" | grep -q 'loaded completely'; then
            echo "true"
            return
        fi
        if echo "$anchor_line" | grep -q 'loaded partially'; then
            echo "false"
            return
        fi
    fi

    # Anchor absent — fail-closed.  Do NOT scan other models' load lines.
    echo "WARNING: anchor 'Requested to load MotifVideoModel' not found;" \
         "loaded_completely defaulting to unknown" >&2
    echo "unknown"
}

# ---------------------------------------------------------------------------
# parse_vram_peak_gb — detect VRAM peak from log
#
# Priority 1: user-appended "VRAM_PEAK_GB=<float>" line
# Priority 2: user-appended "MEMORY_USED_MIB=<int>" line
#             Recommended source: nvidia-smi --query-gpu=memory.used \
#               --format=csv,noheader,nounits >> <log>; echo "MEMORY_USED_MIB=$(tail -1 <log_tmp>)" >> <log>
# Priority 3: inline nvidia-smi line matching "<used> MiB / <total> MiB"
#             Only the *used* (first) value is extracted; total is ignored to
#             avoid returning total VRAM (e.g., 81920 MiB on an A100) as peak.
# Fallback:   "unknown"
# ---------------------------------------------------------------------------
parse_vram_peak_gb() {
    local log="$1"

    # Priority 1: explicit VRAM_PEAK_GB=<float> tag
    local explicit
    explicit="$(grep -oE '^VRAM_PEAK_GB=[0-9]+(\.[0-9]+)?' "$log" 2>/dev/null | tail -1 | cut -d= -f2)" || explicit=""
    if [[ -n "$explicit" ]]; then
        echo "$explicit"
        return
    fi

    # Priority 2: MEMORY_USED_MIB=<int> tag (from nvidia-smi csv output)
    local used_mib_tag
    used_mib_tag="$(grep -oE '^MEMORY_USED_MIB=[0-9]+' "$log" 2>/dev/null | tail -1 | cut -d= -f2)" || used_mib_tag=""
    if [[ -n "$used_mib_tag" ]]; then
        awk -v mib="$used_mib_tag" 'BEGIN { printf "%.2f\n", mib / 1024 }'
        return
    fi

    # Priority 3: inline nvidia-smi line "<used> MiB / <total> MiB"
    # Extract only the used (first) MiB value; discard total.
    # Example: "47000 MiB / 81920 MiB" → used=47000, total=81920 (ignored).
    local used_mib=0
    local line used
    while IFS= read -r line; do
        # Extract the integer before the first " MiB" token (the used value).
        used="$(echo "$line" | grep -oE '^[0-9]+')" || used=""
        if [[ -n "$used" ]] && [[ "$used" -gt "$used_mib" ]]; then
            used_mib="$used"
        fi
    done < <(grep -oE '[0-9]+[[:space:]]*MiB[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*MiB' "$log" 2>/dev/null \
             | grep -oE '^[0-9]+' || true)

    if [[ "$used_mib" -gt 0 ]]; then
        # Convert MiB to GB (float, 2 decimal places) using awk
        awk -v mib="$used_mib" 'BEGIN { printf "%.2f\n", mib / 1024 }'
        return
    fi

    echo "unknown"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
COMMIT_SHA="$(get_commit_sha)"
WALLCLOCK="$(parse_wallclock      "$LOG_FILE")"
FALLBACK="$(parse_fallback_count  "$LOG_FILE")"
STAGED="$(parse_staged_mb         "$LOG_FILE")"
LOADED="$(parse_loaded_completely "$LOG_FILE")"
VRAM_GB="$(parse_vram_peak_gb     "$LOG_FILE")"

echo "vram_peak_gb=${VRAM_GB}"
echo "wallclock_seconds_per_step=${WALLCLOCK}"
echo "fallback_log_count=${FALLBACK}"
echo "staged_mb=${STAGED}"
echo "loaded_completely=${LOADED}"
echo "commit_sha=${COMMIT_SHA}"
echo "dtype=${DTYPE}"
echo "vram_mode=${VRAM_MODE}"
echo "condition=${CONDITION_TAG}"
