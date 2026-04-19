#!/bin/bash
# codex-analyze.sh — Codex 딥 분석 래퍼
#
# 사용법:
#   bash codex-analyze.sh --topic "주제" file1 dir1 ...
#   bash codex-analyze.sh --topic "주제" --tracks "track1:설명1,track2:설명2" file1 dir1 ...
#
# 인자:
#   --topic "subject"                          필수. 분석 주제/질문.
#   --tracks "name1:desc1,name2:desc2"         선택. 트랙 정의 (쉼표 구분).
#                                              생략 시 단일 종합 분석 수행.
#   나머지 인자: 분석 대상 파일/디렉토리 경로 (1개 이상 필수)
#
# 환경 변수:
#   CLAUDE_PROJECT_DIR — 프로젝트 루트 (미설정 시 현재 디렉토리)
#
# 인증: codex login --device-auth (ChatGPT 계정, ~/.codex/auth.json에 캐시)
#
# 출력: 분석 결과 (stdout 요약 + 세션 폴더에 상세 파일)
# 종료 코드: 0=성공, 1=실패/미설치
#
# .hook-state 기록: chain_state=analyzer_done

set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
HOOK_STATE="$PROJECT_DIR/.hook-state"

source "$(dirname "$0")/codex-common.sh"

# 전제 조건 체크 (codex CLI 설치 여부)
_check_codex_prerequisites

# 인자 파싱
TOPIC=""
TRACKS=""
declare -a TARGET_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --topic 다음에 분석 주제 값이 필요합니다."
        exit 1
      fi
      TOPIC="$2"
      shift 2
      ;;
    --tracks)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --tracks 다음에 트랙 정의 값이 필요합니다."
        exit 1
      fi
      TRACKS="$2"
      shift 2
      ;;
    --*)
      echo "ERROR: 알 수 없는 옵션 '$1'. 사용법을 확인하세요."
      exit 1
      ;;
    *)
      TARGET_PATHS+=("$1")
      shift
      ;;
  esac
done

# 필수 인자 검증
if [[ -z "$TOPIC" ]]; then
  echo "ERROR: --topic 이 필수입니다."
  echo "사용법: bash codex-analyze.sh --topic \"주제\" [--tracks \"name1:desc1,name2:desc2\"] file1 dir1 ..."
  exit 1
fi

if [[ ${#TARGET_PATHS[@]} -eq 0 ]]; then
  echo "ERROR: 분석 대상 파일 또는 디렉토리를 1개 이상 지정하세요."
  echo "사용법: bash codex-analyze.sh --topic \"주제\" [--tracks \"name1:desc1,name2:desc2\"] file1 dir1 ..."
  exit 1
fi

# 세션 폴더명 생성: YYYYMMDD-HHMMSS-{sanitized_topic}
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
SANITIZED_TOPIC=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-\|-$//g' | cut -c1-30)
SESSION_DIR="$PROJECT_DIR/.analyze/${TIMESTAMP}-${SANITIZED_TOPIC}"
mkdir -p "$SESSION_DIR"

# 파일/디렉토리 목록 문자열 구성
FILE_LIST=""
for path in "${TARGET_PATHS[@]}"; do
  FILE_LIST+="- ${path}"$'\n'
done

# 트랙 파싱: "name1:desc1,name2:desc2" → 배열
declare -a TRACK_NAMES=()
declare -a TRACK_DESCS=()

if [[ -n "$TRACKS" ]]; then
  IFS=',' read -ra TRACK_ENTRIES <<< "$TRACKS"
  for entry in "${TRACK_ENTRIES[@]}"; do
    track_name="${entry%%:*}"
    track_desc="${entry#*:}"
    TRACK_NAMES+=("$track_name")
    TRACK_DESCS+=("$track_desc")
  done
fi

# 00_overview.md 작성
{
  echo "# 분석 세션: ${TOPIC}"
  echo ""
  echo "생성 시각: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  echo "## 분석 대상 파일/디렉토리"
  echo ""
  echo "$FILE_LIST"
  if [[ ${#TRACK_NAMES[@]} -gt 0 ]]; then
    echo "## 트랙 목록"
    echo ""
    echo "| 번호 | 트랙명 | 설명 |"
    echo "|------|--------|------|"
    for i in "${!TRACK_NAMES[@]}"; do
      printf "| %02d | %s | %s |\n" "$((i+1))" "${TRACK_NAMES[$i]}" "${TRACK_DESCS[$i]}"
    done
  else
    echo "## 분석 방식"
    echo ""
    echo "단일 종합 분석 (트랙 미지정)"
  fi
} > "$SESSION_DIR/00_overview.md"

echo "분석 세션 시작: ${SESSION_DIR#$PROJECT_DIR/}"
echo ""

# 임시 파일 — EXIT 시 자동 정리
RESULT_FILE=$(_portable_mktemp codex-analyze md)
trap 'rm -f "$RESULT_FILE"' EXIT

# 분석 실행 함수
_run_analysis() {
  local track_num="$1"     # e.g. "01" or "00"
  local track_name="$2"    # e.g. "architecture" or "comprehensive"
  local scope_desc="$3"    # e.g. "아키텍처 분석" or "Comprehensive analysis"
  local out_file="$4"      # 저장 경로

  ANALYZE_PROMPT="You are a deep code analyzer. Your task is to thoroughly investigate the following codebase area and report findings.

## Analysis Topic
${TOPIC}

## Investigation Scope
${scope_desc}

## Files/Directories to Investigate
${FILE_LIST}

## Output Format
Report your findings in this EXACT format:

## Key Findings
- [CRITICAL] finding description — file:line evidence
- [BUG] finding description — file:line evidence
- [ROOT_CAUSE] finding description — evidence
- [INFO] finding description — context

## Root Cause Analysis
(If applicable — explain the underlying cause of any issues found)

## Recommendations
1. recommendation — priority: 높음/중간/낮음
2. ...

## Additional Investigation Needed
(Areas that need further analysis, or \"없음\" if complete)

Rules:
- Read the actual files in the codebase to gather evidence
- Cite specific file paths and line numbers
- Tag each finding with severity: CRITICAL, BUG, ROOT_CAUSE, or INFO
- Focus on: logic errors, design flaws, hidden dependencies, race conditions
- Do NOT suggest code changes — analysis only
- For each analysis step, pair it with a Verify line specifying how the finding can be objectively confirmed (command:, file:, test: prefix). If no objective verification is possible, mark it as a blocker. Do not rely on self-report.
- Write descriptions in Korean"

  echo "트랙 ${track_num} [${track_name}] 분석 중..."

  if (cd "$PROJECT_DIR" && echo "$ANALYZE_PROMPT" | _codex_run_readonly 600 "$RESULT_FILE"); then

    RESULT=$(cat "$RESULT_FILE")
    cp "$RESULT_FILE" "$out_file"
    _extract_summary "$RESULT"
    echo "상세 결과: ${out_file#$PROJECT_DIR/}"
    echo ""
  else
    echo "ERROR: Codex 분석 타임아웃 또는 실패 (트랙: ${track_name})"
    exit 1
  fi
}

# 트랙별 또는 단일 분석 실행
if [[ ${#TRACK_NAMES[@]} -gt 0 ]]; then
  for i in "${!TRACK_NAMES[@]}"; do
    track_num=$(printf "%02d" "$((i+1))")
    track_name="${TRACK_NAMES[$i]}"
    track_desc="${TRACK_DESCS[$i]}"
    safe_name=$(echo "$track_name" | sed 's/[^a-zA-Z0-9_-]/-/g' | sed 's/-\+/-/g' | sed 's/^-\|-$//g' | cut -c1-30)
    out_file="${SESSION_DIR}/track_${track_num}_${safe_name}.md"
    _run_analysis "$track_num" "$track_name" "$track_desc" "$out_file"
  done
else
  out_file="${SESSION_DIR}/track_00_comprehensive.md"
  _run_analysis "00" "comprehensive" "Comprehensive analysis" "$out_file"
fi

# 통합 결론 생성 (트랙이 2개 이상인 경우)
TRACK_FILES=$(find "$SESSION_DIR" -name 'track_*.md' -type f | sort)
TRACK_COUNT=$(echo "$TRACK_FILES" | wc -l)

if [[ "$TRACK_COUNT" -ge 2 ]]; then
  echo "트랙 결과 통합 중..."

  TRACK_CONTENTS=""
  while IFS= read -r tf; do
    TRACK_CONTENTS+="=== $(basename "$tf") ==="$'\n'
    TRACK_CONTENTS+="$(cat "$tf")"$'\n'$'\n'
  done <<< "$TRACK_FILES"

  CONSOLIDATION_PROMPT="You are a senior code analyst. Consolidate the following investigation track results into a single conclusion document.

## Analysis Topic
${TOPIC}

## Track Results
${TRACK_CONTENTS}

## Output Format
Write the conclusion in this EXACT format:

# 분석 결론: ${TOPIC}

## 요약
(1-paragraph summary of all findings across tracks)

## 핵심 발견
1. [CRITICAL] finding — source: track_XX
2. [BUG] finding — source: track_XX
3. [ROOT_CAUSE] finding — source: track_XX
4. [INFO] finding — source: track_XX

## Root Cause 분석
(Synthesized root cause explanation drawing from all tracks)

## 권장 조치
1. action — priority: 높음/중간/낮음
2. ...

## 추가 조사 필요 여부
(Areas needing further analysis, or \"없음\")

Rules:
- Deduplicate findings across tracks
- Preserve the most severe assessment when tracks disagree
- Cite source track for each finding
- For each action item, include a Verify line with objective confirmation means (command:, file:, test: prefix). Self-report-only verification is insufficient.
- Write in Korean"

  if (cd "$PROJECT_DIR" && echo "$CONSOLIDATION_PROMPT" | _codex_run_readonly 600 "$RESULT_FILE"); then

    cp "$RESULT_FILE" "$SESSION_DIR/99_conclusion.md"
    echo ""
    echo "━━━ 통합 결론 ━━━"
    cat "$SESSION_DIR/99_conclusion.md"
    echo ""
  else
    echo "WARN: 통합 결론 생성 실패. 트랙별 결과를 직접 확인하세요."
  fi
fi

# 최신 결과를 출력 디렉토리에 저장
_OUTPUT_DIR=$(_get_output_dir)
if [[ -f "$SESSION_DIR/99_conclusion.md" ]]; then
  cp "$SESSION_DIR/99_conclusion.md" "$_OUTPUT_DIR/.codex-analyze-latest.md"
else
  cp "$RESULT_FILE" "$_OUTPUT_DIR/.codex-analyze-latest.md"
fi

# .hook-state 기록
_update_hook_state "chain_state" "analyzer_done"

echo "━━━ Codex Analyze: DONE ━━━"
echo "세션 폴더: ${SESSION_DIR#$PROJECT_DIR/}"
