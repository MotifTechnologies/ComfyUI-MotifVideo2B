#!/bin/bash
# plan-gate.sh — PreToolUse hook
# Write|Edit 도구 호출 전 활성 플랜 존재 여부를 확인하여 차단
#
# 입력: stdin으로 JSON (tool_name, tool_input.file_path 등)
# 출력: stderr로 경고 (플랜 없을 때)
# 종료: exit 2 (차단 모드)

set -eo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"

# jq 없으면 skip
if ! command -v jq &>/dev/null; then
  exit 0
fi

# stdin에서 file_path 추출
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

# 절대경로 → 상대경로 변환
FILE_PATH="${FILE_PATH#$PROJECT_DIR/}"

# "시스템 동작 영향" 경로 — 예외에서 제외하고 plan gate 체크 적용
# CLAUDE.md 분류표: agents/*.md, settings.json은 tester 호출 필수
SYSTEM_IMPACT_PATHS=(".claude/agents/" ".claude/settings.json" "settings.json")
IS_SYSTEM_IMPACT=false
for sip in "${SYSTEM_IMPACT_PATHS[@]}"; do
  if [[ "$FILE_PATH" == ${sip}* ]] || [[ "$FILE_PATH" == "$sip" ]]; then
    IS_SYSTEM_IMPACT=true
    break
  fi
done

if [[ "$IS_SYSTEM_IMPACT" == false ]]; then
  # 예외 경로: 시스템/도구 파일은 플랜 없이도 수정 허용
  EXCEPTIONS=(".claude/" ".plans/" ".manuals/" ".analyze/" "tests/" "experiments/" ".hook-state" ".gitignore")
  for ex in "${EXCEPTIONS[@]}"; do
    if [[ "$FILE_PATH" == ${ex}* ]]; then
      exit 0
    fi
  done
fi

# .plans/ 최신 폴더에서 활성 체크리스트 확인
PLANS_DIR="$PROJECT_DIR/.plans"
if [[ ! -d "$PLANS_DIR" ]]; then
  echo "🚫 [plan-gate] 차단: 플랜 없이 파일 수정 불가: $FILE_PATH" >&2
  echo "   → /task-plan으로 작업 계획을 먼저 생성하세요." >&2
  exit 2
fi

# 최신 플랜 폴더 찾기 (체크리스트 있는 것만)
LATEST_PLAN=""
for dir in $(ls -td "$PLANS_DIR"/*/ 2>/dev/null); do
  if [[ -f "$dir/03_checklist.md" ]]; then
    LATEST_PLAN="$dir"
    break
  fi
done

if [[ -z "$LATEST_PLAN" ]]; then
  echo "🚫 [plan-gate] 차단: 활성 플랜이 없습니다. 파일 수정 불가: $FILE_PATH" >&2
  echo "   → /task-plan으로 작업 계획을 먼저 생성하세요." >&2
  exit 2
fi

# 활성 체크리스트 확인 ([ ] 항목이 1개 이상)
UNCHECKED=$(grep -c '^\- \[ \]' "$LATEST_PLAN/03_checklist.md" || true)
if [[ "${UNCHECKED:-0}" -eq 0 ]]; then
  echo "🚫 [plan-gate] 차단: 체크리스트가 모두 완료된 상태입니다. 파일 수정 불가: $FILE_PATH" >&2
  echo "   → 새 플랜을 생성하거나 체크리스트를 갱신하세요." >&2
  exit 2
fi

# 체인 상태 확인 (.hook-state에서 chain_state 읽기)
STATE_FILE="$PROJECT_DIR/.hook-state"
LOCK_FILE="${STATE_FILE}.lock"
source "$(dirname "$0")/lib/common.sh"

CHAIN_STATE=""
if [[ -f "$STATE_FILE" ]]; then
  # === 공유 잠금 획득 (읽기 구간) ===
  acquire_lock -s || true

  # { grep ... || true; } 패턴: key 미존재 시 grep exit 1 → pipefail 방어
  CHAIN_STATE=$({ grep -m1 '^chain_state=' "$STATE_FILE" || true; } | cut -d= -f2-)

  # === 읽기 잠금 해제 ===
  release_lock
fi

# chain_state별 차단
case "$CHAIN_STATE" in
  developer_done)
    echo "🚫 [plan-gate] 차단: developer 완료 후 tester가 아직 호출되지 않았습니다. tester를 먼저 호출하세요." >&2
    exit 2
    ;;
  tester_done)
    echo "🚫 [plan-gate] 차단: tester 완료 후 reviewer가 아직 호출되지 않았습니다. reviewer를 먼저 호출하세요." >&2
    exit 2
    ;;
  planner_done)
    echo "🚫 [plan-gate] 차단: planner 완료 후 reviewer가 아직 호출되지 않았습니다. reviewer를 먼저 호출하세요." >&2
    exit 2
    ;;
  reviewer_revise)
    echo "🚫 [plan-gate] 차단: reviewer가 REVISE 판정했습니다. 수정 후 reviewer를 재호출하세요." >&2
    exit 2
    ;;
esac

# 활성 플랜 있음 → 통과
exit 0
