#!/usr/bin/env bash
# commit-guard.sh — PostToolUse:Bash hook
# git commit 감지 → 체크리스트 대비 커밋 수 경고 (번들 커밋 방지)
# PostToolUse이므로 차단 불가, exit 0으로만 종료

set -eo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
STATE_FILE="$PROJECT_DIR/.hook-state"
LOCK_FILE="${STATE_FILE}.lock"

# === flock 가용성 체크 ===
if command -v flock >/dev/null 2>&1; then
  HAS_FLOCK=1
else
  HAS_FLOCK=0
fi

acquire_lock() {
  # $1: lock mode (-x exclusive, -s shared)
  if [ "$HAS_FLOCK" = "1" ]; then
    exec 9>"$LOCK_FILE"
    if ! flock "$1" -w 3 9; then
      exec 9>&-
      return 1
    fi
  fi
  return 0
}

release_lock() {
  if [ "$HAS_FLOCK" = "1" ]; then
    exec 9>&-
  fi
}

# === stdin에서 JSON 읽기 ===
INPUT=$(cat)

# === JSON 필드 추출 유틸 로드 ===
source "$(dirname "$0")/lib/common.sh"

# === git commit인지 확인 (early return — 성능 중요) ===
COMMAND=$(extract_field "$INPUT" '.tool_input.command')

if [[ -z "$COMMAND" ]] || ! echo "$COMMAND" | grep -q 'git commit'; then
  exit 0
fi

# === .hook-state에서 checked_count 읽기 ===
checked_count=0
if [[ -f "$STATE_FILE" ]]; then
  # === 공유 잠금 획득 (읽기 구간) ===
  acquire_lock -s || true

  checked_count=$({ grep -m1 "^checked_count=" "$STATE_FILE" || true; } | cut -d= -f2-)
  checked_count=${checked_count:-0}

  # === 읽기 잠금 해제 ===
  release_lock
fi

# === main 브랜치와의 분기점 기준 커밋 수 계산 ===
CURRENT_BRANCH=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# main 브랜치에서 직접 작업 시 경고 스킵
if [[ -z "$CURRENT_BRANCH" ]] || [[ "$CURRENT_BRANCH" == "main" ]] || [[ "$CURRENT_BRANCH" == "master" ]]; then
  exit 0
fi

# main 또는 master 기준 분기점 찾기
MAIN_BRANCH=""
if git -C "$PROJECT_DIR" rev-parse --verify main &>/dev/null; then
  MAIN_BRANCH="main"
elif git -C "$PROJECT_DIR" rev-parse --verify master &>/dev/null; then
  MAIN_BRANCH="master"
else
  # main/master 브랜치 없음 → 경고 스킵
  exit 0
fi

MERGE_BASE=$(git -C "$PROJECT_DIR" merge-base "$MAIN_BRANCH" HEAD 2>/dev/null || echo "")

# merge-base 실패 또는 자기 자신 → 경고 스킵
if [[ -z "$MERGE_BASE" ]]; then
  exit 0
fi

CURRENT_HEAD=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "")
if [[ "$MERGE_BASE" == "$CURRENT_HEAD" ]]; then
  exit 0
fi

# 분기점 이후 전체 커밋 수
ALL_COMMITS=$(git -C "$PROJECT_DIR" rev-list --count "${MERGE_BASE}..HEAD" 2>/dev/null || echo "0")

# fixup!/squash! prefix 커밋 수
FIXUP_SQUASH_COUNT=$(git -C "$PROJECT_DIR" log --oneline "${MERGE_BASE}..HEAD" 2>/dev/null | grep -cE '^[0-9a-f]+ (fixup!|squash!) ' || true)

# cherry-pick 커밋 수 (body에 "cherry picked from commit" 포함)
CHERRY_PICK_COUNT=$(git -C "$PROJECT_DIR" log --format='%b' "${MERGE_BASE}..HEAD" 2>/dev/null | grep -c 'cherry picked from commit' || true)

# 실질 커밋 수 = 전체 - fixup - squash - cherry-pick
BRANCH_COMMITS=$((ALL_COMMITS - FIXUP_SQUASH_COUNT - CHERRY_PICK_COUNT))
if [[ "$BRANCH_COMMITS" -lt 0 ]]; then
  BRANCH_COMMITS=0
fi

# === 비교: 커밋 수 > checked_count + 1 이면 경고 ===
THRESHOLD=$((checked_count + 1))

if [[ "$BRANCH_COMMITS" -gt "$THRESHOLD" ]]; then
  echo "⚠️ [COMMIT-GUARD] 번들 커밋 의심: 브랜치 커밋 ${BRANCH_COMMITS}개 > 완료 항목 ${checked_count}개. 체크리스트 항목별 분리 커밋을 확인하세요." >&2
fi

exit 0
