#!/bin/bash
# detect-state.sh — Stop hook
# .plans/ 최신 폴더의 체크리스트 상태를 감지하고 태그 메시지를 출력
# Phase 2-2: .hook-state 관리 기반 + 진행률 출력
# Phase 2-3: 태그 메시지 분기 (CONTEXT_DONE, PLAN_CREATED, CHECKLIST_DONE, PLAN_COMPLETE, PLAN_MODIFIED)

set -eo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
PLANS_DIR="$PROJECT_DIR/.plans"
STATE_FILE="$PROJECT_DIR/.hook-state"
LOCK_FILE="${STATE_FILE}.lock"
source "$(dirname "$0")/lib/common.sh"

# === .plans/ 최신 폴더 찾기 (체크리스트 있는 것만) ===
if [[ ! -d "$PLANS_DIR" ]]; then
  exit 0
fi

LATEST_PLAN=""
for dir in $(ls -td "$PLANS_DIR"/*/ 2>/dev/null); do
  if [[ -f "$dir/03_checklist.md" ]]; then
    LATEST_PLAN="$dir"
    break
  fi
done

if [[ -z "$LATEST_PLAN" ]]; then
  exit 0
fi

PLAN_DIR_NAME=$(basename "$LATEST_PLAN")
CHECKLIST="$LATEST_PLAN/03_checklist.md"
CONTEXT="$LATEST_PLAN/02_context.md"

# === .hook-state 읽기/초기화 ===
init_state() {
  acquire_lock -x || true
  cat > "$STATE_FILE" <<EOF
checked_count=0
total_count=0
jira_asked=false
context_noted=false
plan_dir=
plan_mtime=
chain_state=
last_agent=
EOF
  release_lock
}

if [[ ! -f "$STATE_FILE" ]]; then
  init_state
fi

# .hook-state에서 각 변수를 grep+cut으로 안전하게 읽기
# { grep ... || true; } 패턴: key 미존재 시 grep exit 1 → pipefail 방어
# === 공유 잠금 획득 (읽기 구간) ===
acquire_lock -s || true

checked_count=$({ grep -m1 "^checked_count=" "$STATE_FILE" || true; } | cut -d= -f2-)
total_count=$({ grep -m1 "^total_count=" "$STATE_FILE" || true; } | cut -d= -f2-)
jira_asked=$({ grep -m1 "^jira_asked=" "$STATE_FILE" || true; } | cut -d= -f2-)
context_noted=$({ grep -m1 "^context_noted=" "$STATE_FILE" || true; } | cut -d= -f2-)
plan_dir=$({ grep -m1 "^plan_dir=" "$STATE_FILE" || true; } | cut -d= -f2-)
plan_mtime=$({ grep -m1 "^plan_mtime=" "$STATE_FILE" || true; } | cut -d= -f2-)
chain_state=$({ grep -m1 "^chain_state=" "$STATE_FILE" || true; } | cut -d= -f2-)
last_agent=$({ grep -m1 "^last_agent=" "$STATE_FILE" || true; } | cut -d= -f2-)

# === 읽기 잠금 해제 ===
release_lock

# 플랜 디렉토리가 바뀌었으면 state 리셋
if [[ "${plan_dir:-}" != "$PLAN_DIR_NAME" ]]; then
  init_state
  checked_count=0
  total_count=0
  jira_asked=false
  context_noted=false
  plan_dir=""
  plan_mtime=""
  chain_state=""
  last_agent=""
  plan_dir="$PLAN_DIR_NAME"
  # 이미 context가 존재하면 CONTEXT_DONE 오발동 방지
  if [[ -f "$LATEST_PLAN/02_context.md" ]]; then
    context_noted=true
  fi
fi

# === 현재 상태 수집 ===
CURRENT_CHECKED=0
CURRENT_TOTAL=0
CURRENT_UNCHECKED=0

if [[ -f "$CHECKLIST" ]]; then
  CURRENT_CHECKED=$(grep -c '^\- \[x\]' "$CHECKLIST" || true)
  CURRENT_CHECKED=${CURRENT_CHECKED:-0}
  CURRENT_UNCHECKED=$(grep -c '^\- \[ \]' "$CHECKLIST" || true)
  CURRENT_UNCHECKED=${CURRENT_UNCHECKED:-0}
  CURRENT_TOTAL=$((CURRENT_CHECKED + CURRENT_UNCHECKED))
fi

CONTEXT_EXISTS=false
if [[ -f "$CONTEXT" ]]; then
  CONTEXT_EXISTS=true
fi

CHECKLIST_EXISTS=false
if [[ -f "$CHECKLIST" ]]; then
  CHECKLIST_EXISTS=true
fi

# === 플랜 3대 문서 mtime 수집 ===
PLAN_DOC="$LATEST_PLAN/01_plan.md"
CURRENT_PLAN_MTIME=""
for f in "$PLAN_DOC" "$CONTEXT" "$CHECKLIST"; do
  if [[ -f "$f" ]]; then
    mtime=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo "0")
    CURRENT_PLAN_MTIME="${CURRENT_PLAN_MTIME}${mtime}:"
  fi
done

# === 태그 메시지 분기 (2-3) ===

# a. 02_context.md 신규/변경 + context_noted=false
if [[ "$CONTEXT_EXISTS" == "true" && "${context_noted:-false}" == "false" ]]; then
  echo "[HOOK:CONTEXT_DONE]" >&2
  context_noted=true
fi

# b. 03_checklist.md [x] 증가
skip_plan_modified=false
if [[ "$CURRENT_CHECKED" -gt "${checked_count:-0}" ]]; then
  # c. [ ] == 0 이면 PLAN_COMPLETE
  if [[ "$CURRENT_UNCHECKED" -eq 0 && "$CURRENT_TOTAL" -gt 0 ]]; then
    echo "[HOOK:PLAN_COMPLETE]" >&2
  else
    echo "[HOOK:CHECKLIST_DONE]" >&2
  fi

  # Phase 2-3: 체인 미완료 감지
  case "${chain_state:-}" in
    reviewer_passed|chain_complete)
      # 체인 완료 — 추가 경고 없음
      ;;
    planner_done)
      echo "[HOOK:CHAIN_INCOMPLETE] planner 완료 후 reviewer 미호출. [x] 마킹 전 reviewer를 호출하세요." >&2
      ;;
    developer_done)
      echo "[HOOK:CHAIN_INCOMPLETE] tester→reviewer 체인 미완료. [x] 마킹 전 tester, reviewer를 호출하세요." >&2
      ;;
    tester_done)
      echo "[HOOK:CHAIN_INCOMPLETE] reviewer 미호출. [x] 마킹 전 reviewer를 호출하세요." >&2
      ;;
    reviewer_revise)
      echo "[HOOK:CHAIN_INCOMPLETE] reviewer REVISE 상태. 수정 후 reviewer 재호출하세요." >&2
      ;;
    *)
      echo "[HOOK:CHAIN_INCOMPLETE] 에이전트 체인 기록 없음. developer→tester→reviewer 순서를 확인하세요." >&2
      ;;
  esac

  # 다음 체크리스트 항목을 위해 chain_state 리셋
  chain_state=""

  skip_plan_modified=true
fi

# d. 03_checklist.md 신규 + jira_asked=false
if [[ "$CHECKLIST_EXISTS" == "true" && "${checked_count:-0}" -eq 0 && "$CURRENT_TOTAL" -gt 0 && "${total_count:-0}" -eq 0 && "${jira_asked:-false}" == "false" ]]; then
  echo "[HOOK:PLAN_CREATED]" >&2
  jira_asked=true
  skip_plan_modified=true
fi

# f. planner_done 상태에서 reviewer 미호출 감지
if [[ "$skip_plan_modified" == "false" && "${chain_state:-}" == "planner_done" ]]; then
  echo "[HOOK:CHAIN_INCOMPLETE] reviewer 미호출. planner 완료 후 reviewer를 호출하세요." >&2
  skip_plan_modified=true
fi

# e. 플랜 3대 문서 mtime 변경 감지 (CHECKLIST_DONE/PLAN_COMPLETE/PLAN_CREATED 직후는 제외)
if [[ "$skip_plan_modified" == "false" && -n "${plan_mtime:-}" && -n "$CURRENT_PLAN_MTIME" && "$CURRENT_PLAN_MTIME" != "${plan_mtime:-}" ]]; then
  echo "[HOOK:PLAN_MODIFIED]" >&2
fi

# === 진행률 출력 ===
if [[ "$CURRENT_TOTAL" -gt 0 ]]; then
  echo "📊 진행률: ${CURRENT_CHECKED}/${CURRENT_TOTAL} ($(( CURRENT_CHECKED * 100 / CURRENT_TOTAL ))%)" >&2
fi

# === 배타 잠금 획득 (쓰기 구간) ===
acquire_lock -x || true

# === .hook-state 업데이트 (agent-tracker 필드 보존) ===
cat > "$STATE_FILE" <<EOF
checked_count=$CURRENT_CHECKED
total_count=$CURRENT_TOTAL
jira_asked=${jira_asked:-false}
context_noted=${context_noted:-false}
plan_dir=$PLAN_DIR_NAME
plan_mtime=$CURRENT_PLAN_MTIME
last_agent=${last_agent:-}
chain_state=${chain_state:-}
EOF

# === 쓰기 잠금 해제 ===
release_lock

exit 0
