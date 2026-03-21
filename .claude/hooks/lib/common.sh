#!/usr/bin/env bash
# lib/common.sh — hooks 공통 유틸리티
# source로 로드하여 사용: source "$(dirname "$0")/lib/common.sh"

# source guard: 중복 로드 방지
[[ -n "${_COMMON_SH_LOADED:-}" ]] && return 0
_COMMON_SH_LOADED=1

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
