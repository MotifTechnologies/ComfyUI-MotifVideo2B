#!/usr/bin/env bash
# run_on_gpu_pod.sh
#
# Run ops_primitives test suite on a GPU pod in a single command.
# Executes:
#   1. pytest: tests/transformer/test_ops_primitives.py
#   2. Codex blind test (from resolved plan slug; skipped if .codex-test-latest.md not found)
#
# Plan slug resolution priority (4 levels):
#   1. --plan <slug>       explicit argument
#   2. $MOTIF_PLAN_SLUG    environment variable
#   3. current git branch  pattern feat/YYYYMMDD-* → YYYYMMDD-* portion extracted
#   4. fatal error         detached HEAD / no pattern match → exit 1 with usage
#
# Usage:
#   bash tests/transformer/run_on_gpu_pod.sh
#   bash tests/transformer/run_on_gpu_pod.sh --plan 20260420-fp8-phase1-ops-injection
#   bash tests/transformer/run_on_gpu_pod.sh --blind-test /path/to/custom-blind-test.md
#   MOTIF_PLAN_SLUG=20260420-fp8-phase1-ops-injection bash tests/transformer/run_on_gpu_pod.sh
#
# Options:
#   --plan <slug>           Explicit plan slug (overrides env var and branch detection)
#   --blind-test <path>     Direct path to a blind-test .md file (escape hatch; skips slug resolution)
#   -h, --help              Show this help message and exit
#
# Environment overrides:
#   COMFYUI_ROOT            Override ComfyUI root (default: two levels above repo root,
#                           assuming ComfyUI/custom_nodes/<repo> checkout layout)
#   MOTIF_PLAN_SLUG         Plan slug (level 2 in resolution chain)
#
# Exit code: 0 if all executed tests pass, non-zero otherwise.

set -euo pipefail

# ---------------------------------------------------------------------------
# Path resolution — no hardcoded absolute paths
# ---------------------------------------------------------------------------
SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
REPO_ROOT=$( cd "$SCRIPT_DIR/../.." && pwd )
# Default ComfyUI root: repo is at ComfyUI/custom_nodes/<repo>, so go up 2 levels.
DEFAULT_COMFYUI_ROOT=$( cd "$REPO_ROOT/../.." && pwd )
COMFYUI_ROOT="${COMFYUI_ROOT:-$DEFAULT_COMFYUI_ROOT}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
EXPLICIT_PLAN_SLUG=""
EXPLICIT_BLIND_TEST=""

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan)
            if [[ -z "${2:-}" ]]; then
                echo "[run_on_gpu_pod.sh] ERROR: --plan requires a slug argument." >&2
                exit 1
            fi
            EXPLICIT_PLAN_SLUG="$2"
            shift 2
            ;;
        --blind-test)
            if [[ -z "${2:-}" ]]; then
                echo "[run_on_gpu_pod.sh] ERROR: --blind-test requires a file path argument." >&2
                exit 1
            fi
            EXPLICIT_BLIND_TEST="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "[run_on_gpu_pod.sh] ERROR: Unknown option: $1" >&2
            echo "  Use -h or --help for usage." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Plan slug resolution (4-level priority chain)
# Only needed when --blind-test is not provided.
# ---------------------------------------------------------------------------
CODEX_BLIND_MD=""
BLIND_TEST_SKIPPED=0

if [[ -n "$EXPLICIT_BLIND_TEST" ]]; then
    # Escape hatch: direct file path provided
    if [[ ! -f "$EXPLICIT_BLIND_TEST" ]]; then
        echo "[run_on_gpu_pod.sh] ERROR: --blind-test file not found: $EXPLICIT_BLIND_TEST" >&2
        exit 1
    fi
    CODEX_BLIND_MD="$EXPLICIT_BLIND_TEST"
else
    # Resolve plan slug via 4-level priority chain
    PLAN_SLUG=""

    # Level 1: explicit --plan argument
    if [[ -n "$EXPLICIT_PLAN_SLUG" ]]; then
        PLAN_SLUG="$EXPLICIT_PLAN_SLUG"
        echo "[info] Plan slug from --plan argument: $PLAN_SLUG"

    # Level 2: environment variable
    elif [[ -n "${MOTIF_PLAN_SLUG:-}" ]]; then
        PLAN_SLUG="$MOTIF_PLAN_SLUG"
        echo "[info] Plan slug from MOTIF_PLAN_SLUG env: $PLAN_SLUG"

    # Level 3: git branch name, pattern feat/YYYYMMDD-* → YYYYMMDD-* portion
    else
        CURRENT_BRANCH=""
        CURRENT_BRANCH=$( git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true )
        if [[ -n "$CURRENT_BRANCH" && "$CURRENT_BRANCH" != "HEAD" ]]; then
            # Extract slug from feat/YYYYMMDD-... pattern
            if [[ "$CURRENT_BRANCH" =~ ^feat/([0-9]{8}-.+)$ ]]; then
                PLAN_SLUG="${BASH_REMATCH[1]}"
                echo "[info] Plan slug extracted from branch '$CURRENT_BRANCH': $PLAN_SLUG"
            fi
        fi

        # Level 4: fatal error — could not determine slug
        if [[ -z "$PLAN_SLUG" ]]; then
            echo "[run_on_gpu_pod.sh] ERROR: Cannot determine plan slug." >&2
            echo "  Current branch: '${CURRENT_BRANCH:-<not available>}'" >&2
            echo "  Branch did not match pattern feat/YYYYMMDD-* (detached HEAD or different naming)." >&2
            echo "" >&2
            echo "  Provide slug explicitly via one of:" >&2
            echo "    bash tests/transformer/run_on_gpu_pod.sh --plan <slug>" >&2
            echo "    MOTIF_PLAN_SLUG=<slug> bash tests/transformer/run_on_gpu_pod.sh" >&2
            echo "    bash tests/transformer/run_on_gpu_pod.sh --blind-test <path-to.md>" >&2
            exit 1
        fi
    fi

    # Resolve .codex-test-latest.md from the determined slug
    CANDIDATE_MD="$REPO_ROOT/.plans/$PLAN_SLUG/.codex-test-latest.md"
    if [[ -f "$CANDIDATE_MD" ]]; then
        CODEX_BLIND_MD="$CANDIDATE_MD"
    else
        echo "[info] No .codex-test-latest.md found at $CANDIDATE_MD — blind test will be skipped."
        BLIND_TEST_SKIPPED=1
    fi
fi

# ---------------------------------------------------------------------------
# Python / pytest setup
# ---------------------------------------------------------------------------
PYTHON="$COMFYUI_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "[run_on_gpu_pod.sh] ERROR: Python not found at $PYTHON" >&2
    echo "  Set COMFYUI_ROOT to the correct ComfyUI installation directory." >&2
    exit 1
fi

PYTEST_TARGET="$REPO_ROOT/tests/transformer/test_ops_primitives.py"

PYTEST_EXIT=0
CODEX_EXIT=0

# ---------------------------------------------------------------------------
# Step 1: pytest
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[1/2] Running pytest: $PYTEST_TARGET"
echo "======================================================================"
PYTHONPATH="$COMFYUI_ROOT" \
    "$PYTHON" -m pytest "$PYTEST_TARGET" -v || PYTEST_EXIT=$?

# ---------------------------------------------------------------------------
# Step 2: Codex blind test (optional)
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[2/2] Codex blind test"
echo "======================================================================"
if [[ $BLIND_TEST_SKIPPED -eq 1 ]]; then
    echo "Codex blind test not found — skipped."
else
    # Guard: count bash fenced blocks in the markdown file.
    # Exactly 1 block must be present; 0 → skip, 2+ → error (ambiguous which to run).
    BASH_BLOCK_COUNT=$(grep -c '^```bash' "$CODEX_BLIND_MD" || true)

    if [[ "$BASH_BLOCK_COUNT" -eq 0 ]]; then
        echo "WARNING: No bash block found in $CODEX_BLIND_MD — Codex blind test skipped."
        BLIND_TEST_SKIPPED=1
    elif [[ "$BASH_BLOCK_COUNT" -ge 2 ]]; then
        echo "ERROR: $CODEX_BLIND_MD contains $BASH_BLOCK_COUNT bash fenced blocks (expected exactly 1)." >&2
        echo "  Cannot safely determine which block to execute." >&2
        echo "  Use --blind-test <path> to supply a single-block .md or a bare .sh file." >&2
        exit 1
    else
        # Exactly 1 bash block — extract and execute
        TMPSCRIPT=$(mktemp /tmp/codex_blind_XXXXXX.sh)
        trap 'rm -f "$TMPSCRIPT"' EXIT

        # Use awk to extract content between the single ```bash and closing ```
        awk '/^```bash/{found=1; next} found && /^```/{exit} found{print}' \
            "$CODEX_BLIND_MD" > "$TMPSCRIPT"

        if [[ ! -s "$TMPSCRIPT" ]]; then
            echo "WARNING: Extracted bash block from $CODEX_BLIND_MD is empty — skipped."
            BLIND_TEST_SKIPPED=1
        else
            chmod +x "$TMPSCRIPT"
            echo "Extracted bash block from $CODEX_BLIND_MD, executing..."
            REPO_ROOT="$REPO_ROOT" PYTHONPATH="$COMFYUI_ROOT" bash "$TMPSCRIPT" || CODEX_EXIT=$?
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
if [[ $PYTEST_EXIT -eq 0 ]]; then
    echo "  pytest         : PASS"
else
    echo "  pytest         : FAIL (exit $PYTEST_EXIT)"
fi

if [[ $BLIND_TEST_SKIPPED -eq 1 ]]; then
    echo "  codex blind    : SKIPPED (no .codex-test-latest.md for slug)"
elif [[ $CODEX_EXIT -eq 0 ]]; then
    echo "  codex blind    : PASS"
else
    echo "  codex blind    : FAIL (exit $CODEX_EXIT)"
fi

OVERALL=$((PYTEST_EXIT + CODEX_EXIT))
if [[ $OVERALL -eq 0 ]]; then
    echo ""
    echo "All tests passed."
    exit 0
else
    echo ""
    echo "One or more tests FAILED."
    exit 1
fi
