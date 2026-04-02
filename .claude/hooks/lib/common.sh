#!/usr/bin/env bash
# lib/common.sh — hooks 공통 유틸리티
# source로 로드하여 사용: source "$(dirname "$0")/lib/common.sh"

# source guard: 중복 로드 방지
[[ -n "${_COMMON_SH_LOADED:-}" ]] && return 0
_COMMON_SH_LOADED=1

# === lock 파일 기본값 (hook에서 LOCK_FILE을 먼저 정의하면 그 값을 사용) ===
STATE_DIR="${CLAUDE_PROJECT_DIR:-.}"
: "${LOCK_FILE:=${STATE_DIR}/.hook-state.lock}"

# === flock 가용성 감지 ===
if command -v flock >/dev/null 2>&1; then
  HAS_FLOCK=1
else
  HAS_FLOCK=0
fi

# === lock 획득 (flock 사용 가능 시) ===
# 인자: lock mode (-x 배타, -s 공유)
acquire_lock() {
  if [ "$HAS_FLOCK" = "1" ]; then
    exec 9>"$LOCK_FILE"
    if ! flock "$1" -w 3 9; then
      exec 9>&-
      return 1
    fi
  fi
  return 0
}

# === lock 해제 ===
release_lock() {
  if [ "$HAS_FLOCK" = "1" ]; then
    exec 9>&-
  fi
}

# === JSON 필드 추출 (jq 우선, python3 fallback) ===
extract_field() {
  local json="$1"
  local field="$2"

  if command -v jq &>/dev/null; then
    echo "$json" | jq -r "$field" 2>/dev/null || echo ""
  elif command -v python3 &>/dev/null; then
    echo "$json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    keys = '${field}'.strip('.').split('.')
    val = data
    for k in keys:
        val = val[k]
    print(val)
except:
    print('')
" 2>/dev/null || echo ""
  else
    echo ""
  fi
}
