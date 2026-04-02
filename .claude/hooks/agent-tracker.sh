#!/usr/bin/env bash
# agent-tracker.sh — PostToolUse:Agent hook
# Agent 도구 호출 완료 시 에이전트 타입을 추적하여 .hook-state에 기록
# 추적 대상: planner, developer, tester, reviewer, recommender (5개)
# 체인 상태 전이 + stderr 리마인더 출력

set -eo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
STATE_FILE="$PROJECT_DIR/.hook-state"
LOCK_FILE="${STATE_FILE}.lock"
source "$(dirname "$0")/lib/common.sh"

# === stdin에서 JSON 읽기 ===
INPUT=$(cat)

SUBAGENT_TYPE=$(extract_field "$INPUT" '.tool_input.subagent_type')

# === 추적 대상 필터 ===
case "$SUBAGENT_TYPE" in
  planner|developer|tester|reviewer|recommender)
    ;;
  *)
    # 추적 대상이 아닌 에이전트 → 무시
    exit 0
    ;;
esac

# === 배타 잠금 획득 (읽기+쓰기 구간 보호) ===
acquire_lock -x || true

# === .hook-state 읽기 (grep+cut으로 안전하게 파싱) ===
if [[ -f "$STATE_FILE" ]]; then
  # { grep ... || true; } 패턴: key 미존재 시 grep exit 1 → pipefail 방어
  checked_count=$({ grep -m1 "^checked_count=" "$STATE_FILE" || true; } | cut -d= -f2-)
  total_count=$({ grep -m1 "^total_count=" "$STATE_FILE" || true; } | cut -d= -f2-)
  jira_asked=$({ grep -m1 "^jira_asked=" "$STATE_FILE" || true; } | cut -d= -f2-)
  context_noted=$({ grep -m1 "^context_noted=" "$STATE_FILE" || true; } | cut -d= -f2-)
  plan_dir=$({ grep -m1 "^plan_dir=" "$STATE_FILE" || true; } | cut -d= -f2-)
  plan_mtime=$({ grep -m1 "^plan_mtime=" "$STATE_FILE" || true; } | cut -d= -f2-)
  last_agent=$({ grep -m1 "^last_agent=" "$STATE_FILE" || true; } | cut -d= -f2-)
  chain_state=$({ grep -m1 "^chain_state=" "$STATE_FILE" || true; } | cut -d= -f2-)
fi

# === 체인 상태 결정 ===
CHAIN_STATE=""
case "$SUBAGENT_TYPE" in
  planner)
    CHAIN_STATE="planner_done"
    ;;
  developer)
    CHAIN_STATE="developer_done"
    ;;
  tester)
    CHAIN_STATE="tester_done"
    ;;
  reviewer)
    # reviewer 응답에서 PASS/REVISE 판별
    RESPONSE_TEXT=$(extract_field "$INPUT" '.tool_response.content[0].text')
    if echo "$RESPONSE_TEXT" | grep -q "PASS"; then
      CHAIN_STATE="reviewer_passed"
    elif echo "$RESPONSE_TEXT" | grep -q "REVISE"; then
      CHAIN_STATE="reviewer_revise"
    else
      # PASS/REVISE 판별 불가 시 기본값
      CHAIN_STATE="reviewer_done"
    fi
    ;;
  recommender)
    CHAIN_STATE="chain_complete"
    ;;
esac

# === .hook-state 갱신 (cat > 전체 덮어쓰기, detect-state.sh와 통일) ===
# 55~63행에서 읽은 기존 필드를 그대로 재작성하고, last_agent/chain_state만 새 값으로 갱신
cat > "$STATE_FILE" <<EOF
checked_count=${checked_count:-}
total_count=${total_count:-}
jira_asked=${jira_asked:-}
context_noted=${context_noted:-}
plan_dir=${plan_dir:-}
plan_mtime=${plan_mtime:-}
last_agent=$SUBAGENT_TYPE
chain_state=$CHAIN_STATE
EOF

# === 잠금 해제 ===
release_lock

# === stderr 리마인더 출력 ===
case "$CHAIN_STATE" in
  planner_done)
    echo "⚠️ [CHAIN] planner 완료. reviewer 호출이 필요합니다." >&2
    ;;
  developer_done)
    echo "⚠️ [CHAIN] developer 완료. tester 호출이 필요합니다." >&2
    ;;
  tester_done)
    echo "⚠️ [CHAIN] tester 완료. reviewer 호출이 필요합니다." >&2
    ;;
  reviewer_passed)
    echo "✅ [CHAIN] reviewer PASS. recommender 호출 후 사용자 보고하세요." >&2
    ;;
  reviewer_revise)
    echo "🔄 [CHAIN] reviewer REVISE. 수정 후 reviewer 재호출하세요." >&2
    ;;
  chain_complete)
    echo "✅ [CHAIN] 체인 완료. 사용자에게 보고하세요." >&2
    ;;
esac

exit 0
