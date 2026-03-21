#!/bin/bash
# mirror-check.sh
# PostToolUse hook: src/에 새 모듈 폴더가 생겼을 때 미러링 폴더 존재 여부 경고
#
# 입력: stdin으로 JSON (tool_name, file_path 등)
# 출력: stderr로 경고 메시지 (있을 때만)

set -euo pipefail

# jq 없으면 skip
if ! command -v jq &>/dev/null; then
  exit 0
fi

# src/ 디렉토리가 없는 프로젝트에서는 미러링 체크 불필요
if [[ ! -d "$CLAUDE_PROJECT_DIR/src" ]]; then
  exit 0
fi

# Read hook input from stdin
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Strip absolute path prefix if present
FILE_PATH="${FILE_PATH#$CLAUDE_PROJECT_DIR/}"

# Only check if the modified file is under src/
if [[ "$FILE_PATH" != src/* ]]; then
  exit 0
fi

# Extract the module directory (first level under src/)
MODULE=$(echo "$FILE_PATH" | cut -d'/' -f2)

if [[ -z "$MODULE" || "$MODULE" == "__init__.py" || "$MODULE" == *.py ]]; then
  exit 0
fi

# Check mirroring directories
MISSING=()
for DIR in docs scripts tests results; do
  if [[ ! -d "$CLAUDE_PROJECT_DIR/$DIR/$MODULE" ]]; then
    MISSING+=("$DIR/$MODULE/")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "⚠️ 미러링 누락: src/$MODULE/ 에 대응하는 폴더가 없습니다:" >&2
  for M in "${MISSING[@]}"; do
    echo "   → $M" >&2
  done
fi
