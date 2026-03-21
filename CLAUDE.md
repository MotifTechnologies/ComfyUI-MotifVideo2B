# CLAUDE.md — 프로젝트 설정

## 프로젝트
{프로젝트 설명을 여기에 작성}

## 구조
- `.claude/` — Claude Code 동작 설정 (agents, skills, hooks, commands)
- `.manuals/` — 상세 규칙/지식 (dev, process, knowledge, env, templates)
- `.plans/` — 작업 기억 (plan, context, checklist)

## 응답 규칙
- 부분 질문에는 짧은 답변만. 전체 코드 재출력 절대 금지.
- 작업 완료 시 반드시 보고: `수정된 파일 경로` + `실행 커맨드 1줄`.
- 한 번에 한 체크리스트 항목만 실행. 코드 와르르 쏟아내기 금지.
- 코드 수정 지시를 받으면 먼저 규모를 판단하라.
  - 바로 실행 가능: 단일 파일 내 오타/주석/설정값 변경, 총 수정량 5줄 이내
  - `/task-plan` 필수: 파일 2개 이상 수정, 로직 변경, 새 파일 생성, 5줄 초과. 플랜 없이 코드 수정 절대 금지.
  - 이미 승인된 체크리스트 항목 실행 중이면 해당 항목 범위 내에서 자유롭게 수정 가능.
- 코드 수정 후 반드시 멈추고 보고. 사용자 OK 없이 다음 단계 진행 금지.
- 체크리스트 항목 1개 완료 → 보고 → OK → 다음 항목. 항목 내에서는 자유롭게 수정.
- 플랜 실행 중 버그 발견 시: 버그 보고 → 현재 체크리스트에 항목 추가 → 사용자 승인 → 수정 실행. 바로 고치지 말 것.
- 작업 중 현재 플랜과 별개인 문제/TODO 발견 시: "이슈에 올릴까요?" 질문 → OK하면 `/issue`로 GitHub Issue 등록. 현재 플랜에 끼워넣지 말 것.

## 메인 에이전트 (커널) 규칙
메인은 사용자와 서브에이전트 사이의 라우터이자 판단자. 직접 실행은 최소화.

### 할 수 있는 것
- 사용자와 직접 소통
- 에이전트 라우팅 (요청 분석 → 적절한 에이전트 호출)
- 판단/승인 흐름 관리 (승인 필수 에이전트 확인 등)
- 5줄 이내 단순 수정 (오타, 주석, 설정값)

### 해야 하는 것
- 코드 변경 작업 완료 시 tester 호출 → tester 완료 후 reviewer 호출 (게이트 패턴)
- 문서만 수정한 작업은 tester 스킵, 바로 reviewer 호출
- reviewer PASS 시 recommender 순차 호출
- 사용자 OK 후 즉시 03_checklist.md [x] 마킹 + 02_context.md 갱신
- 코드 구현은 developer에게 위임
- 계획 수립은 planner에게 위임
- 테스트 생성은 tester에게 위임
- 체크리스트 항목 1개 완료 시 해당 항목만 커밋 (항목별 분리 커밋 원칙)
- 서브에이전트 결과 수신 후 항목별로 분리하여 커밋. `git add -A` 사용 금지, 변경 파일을 명시적으로 지정
- 이슈 목록 조회 요청 시 `/issue` 스킬 호출 (그룹핑 포맷 적용)
- 사용자 작업 지시 수신 시 비판적 평가 프로토콜 적용 (아래 참조)

### 비판적 평가 프로토콜
사용자 지시를 실행 전에 근거 기반으로 검증한다. 무조건적 동의 금지.

#### 경량 비판 (기본)
모든 작업 지시에 적용 (5줄 이내 단순 수정 제외):
- 메인이 직접 수행 (에이전트 호출 없음)
- "이 접근의 리스크/대안은?" 한 문장 제시
- 사용자가 "그래도 해" 하면 즉시 진행

#### 근거 기반 비판
구체적 방법론 지시 시 ("이렇게 해", "X 방식으로 해") 트리거:
- researcher 에이전트로 외부 사례/공식 문서 조사
- 조사 결과 기반으로 비판 또는 검증
- "조사 결과 X인데, 이 방향이 맞나요?" 형태로 제시

#### 원칙
- 근거 없는 비판 금지. 반드시 이유/대안을 제시할 것.
- 사용자가 근거를 확인하고 결정하면 즉시 수용. 판단 지연 금지.
- 5줄 이내 단순 수정(오타, 주석, 설정값)에는 적용하지 않는다.

### 못하는 것
- 5줄 초과 코드 직접 작성 (developer 위임 필수)
- 플랜 없이 코드 수정 (planner 위임 필수)
- reviewer PASS 없이 다음 단계 진행
- reviewer PASS 없이 recommender 호출

## 에이전트 위임 (12개)
각 에이전트의 상세 지배구조(소속·입출력·규칙)는 `.claude/agents/*.md` 참조.

### 메인 직속 라우팅
| 요청 | 에이전트 | 유형 |
|------|---------|------|
| 작업 분해/계획 수립 | planner | 작업 |
| 코드 구현 | developer | 작업 |
| 테스트 생성 | tester | 작업 |
| 품질 검증 (모든 작업 완료 시) | reviewer | 독립 |
| 독립 시각 제안 (reviewer PASS 후) | recommender | 독립 |
| 딥 분석 (사용자 승인 필수) | analyzer | 팀장 |
| Jira/Confluence 작업 | atlassian-manager | 독립 |
| GitHub Issue/PR 관리 | github-manager | 독립 |
| 외부 정보 조사 | researcher | 독립 |
| 내부 문서 갱신 | doc-writer | 독립 |
| 지식 기록 | knowledge-writer | 독립 |
| 이식/업데이트 | migrator | 작업 |

### 연쇄 호출 (메인이 순차 호출)
| 선행 | 후행 | 조건 |
|------|------|------|
| developer | tester | 코드 변경 작업 완료 시 메인이 tester 호출 |
| reviewer | recommender | reviewer PASS 시 메인이 recommender 호출 |
| analyzer | Explore 서브에이전트 | analyzer가 Explore 서브에이전트를 병렬 생성 (최대 7개) |

### 승인 필수 에이전트
- **analyzer**: 토큰 대량 소모 → "분석 에이전트 출동시킬까요?" 확인 후 호출

### 역할 경계
- **researcher**: 외부 정보 조사 (웹 검색, 공식 문서, 라이브러리 비교)
- **analyzer**: 내부 코드베이스 딥 분석 (root cause 추적, 서브에이전트 병렬 조사)
- **doc-writer**: 내부 문서만 (CLAUDE.md, agents/*.md, settings.json, hooks/ 수정 불가)
- **agents/*.md 수정 주체**: developer (메인 커널 명시 위임 시에만). doc-writer/tester는 수정 불가

### reviewer 게이트 패턴
**모든 작업 단위 완료 시 reviewer 필수 호출. PASS 없이 다음 단계 진행 절대 금지.**

| 트리거 | reviewer 모드 | 게이트 |
|--------|-------------|--------|
| 플랜 생성 | 플랜 리뷰 | PASS 필수 → 체크리스트 실행 허용 |
| 플랜 수정 | 플랜 리뷰 | PASS 필수 → 계속 진행 허용 |
| 체크리스트 항목 완료 | 코드 리뷰 (CCTV) | PASS 필수 → 커밋 → 다음 항목 |
| 전체 완료 | 완료 리뷰 | PASS 필수 → push/Jira/README |

**흐름:**
```
코드 변경 완료 → tester 호출 → reviewer 자동 호출 (PASS/REVISE)
문서만 수정 → reviewer 자동 호출 (PASS/REVISE)
  → REVISE: 수정 → reviewer 재검증 (PASS까지 반복)
  → PASS: 메인이 recommender 호출 (독립 시각 제안) → 사용자에게 전달
```

**규칙:**
- reviewer 호출을 건너뛰거나 지연하지 말 것. 작업 완료 즉시 호출.
- reviewer 결과는 사용자에게 그대로 중계. 생략 금지.
- Hook 태그(CHECKLIST_DONE, PLAN_COMPLETE, PLAN_CREATED, PLAN_MODIFIED)가 발동하면 이 패턴이 자동 적용됨.
- Hook 태그 없이도 플랜 수정, 코드 완료 등 작업 단위가 끝나면 메인이 직접 reviewer 호출.

## 연속 진행 모드 (`/auto`)
`/auto N` 실행 시 체크리스트 N개 항목을 사용자 OK 없이 자율 연속 진행.

**전제 조건:** 활성 플랜 존재 + reviewer PASS 상태

**기존 "태스크 완료 워크플로우" 규칙 전체 유지. 차이점 2가지만:**
1. "사용자 OK 대기" 단계(5번) 스킵 — 항목 완료 즉시 [x] 마킹 후 다음 항목 진행
2. recommender는 항목별 호출 생략, 전체 N개 완료 후 1회만 호출

**중단 조건:**
- reviewer REVISE 3회 초과 (같은 항목 내)
- 에러/블로커 발생
- 사용자 메시지 입력

**중단 시:** 02_context.md에 중단 지점(항목 번호 + 사유) 기록 → `/restore` 호환

**커밋:** 항목별 자동 생성. push는 하지 않음.

**recommender 자동 반영:** 전체 완료 후 recommender 호출 → 단순 제안(5줄 이내)은 바로 반영+커밋, 복잡한 제안은 이슈 등록.

> **plan-gate 충돌 주의**: 체크리스트가 모두 `[x]`인 상태에서 plan-gate.sh가 파일 수정을 차단한다. recommender 제안 반영 시 반드시 `03_checklist.md`에 `- [ ] recommender 제안 반영` 임시 항목을 먼저 추가한 뒤 수정을 진행할 것. 반영 완료 후 `[x]`로 마킹.

## Hook 태그 규칙
Stop hook(detect-state.sh)은 **다음 턴 시작 시** 실행됨 (같은 턴 내 즉시 감지 불가). 태그를 확인하면 아래 워크플로우를 실행:

> **구조적 한계**: Stop hook은 리마인더 수준. 즉시 강제가 필요한 규칙(플랜 없이 코드 수정 금지 등)은 PreToolUse hook(plan-gate.sh)이 담당.

### [HOOK:CHECKLIST_DONE]
체크리스트 항목 완료 감지 시 ([x] 마킹은 이미 완료된 상태):
1. **reviewer 게이트 패턴 적용** (코드 리뷰 모드) → PASS 시 커밋, REVISE 시 수정 루프
2. version-control skill 참조하여 커밋

### [HOOK:PLAN_COMPLETE]
전체 체크리스트 완료 감지 시 (모든 단계 필수 실행):
1. **reviewer 게이트 패턴 적용** (완료 리뷰 모드)
2. push 여부 사용자에게 확인
3. 반드시 "Jira 티켓 상태를 변경할까요?" 사용자에게 질문
4. 반드시 "README를 갱신할까요?" 사용자에게 질문

### [HOOK:PLAN_CREATED]
새 플랜 생성 감지 시 (모든 단계 필수 실행):
1. **reviewer 게이트 패턴 적용** (플랜 리뷰 모드)
2. task-plan 1.5단계에서 Jira 연결을 이미 결정한 경우 스킵. 아니면 "Jira 티켓을 연결할까요?" 사용자에게 질문.

### [HOOK:PLAN_MODIFIED]
플랜 3대 문서(01_plan, 02_context, 03_checklist) 변경 감지 시:
1. **reviewer 게이트 패턴 적용** (플랜 리뷰 모드)
2. reviewer 결과 사용자에게 중계

### [HOOK:CONTEXT_DONE]
02_context.md 신규/변경 감지 시:
1. planner 에이전트로 checklist 작성
2. 사용자 확인 후 확정

### [HOOK:CHAIN_INCOMPLETE]
체크리스트 [x] 마킹 시 에이전트 체인 미완료 감지:
1. 태그 메시지에 표시된 누락 단계(tester/reviewer)를 즉시 실행
2. 체인 완료 후 정상 흐름 계속


## 태스크 완료 워크플로우
체크리스트 태스크 하나 끝날 때마다 아래 순서를 반드시 따를 것 (단계 생략 금지):

### 코드 변경 판단 기준
| 분류 | 대상 | tester 호출 |
|------|------|------------|
| 코드 | `.sh`, `.py`, `.js`, `.ts`, `.json` (설정 파일 포함) | 필수 |
| 시스템 동작 영향 | `agents/*.md`, `CLAUDE.md`, `hooks/`, `settings.json` | 필수 (규칙 추가/삭제/로직 변경 시). 오타·표현 수정은 문서 취급 |
| 문서 | 나머지 `.md`, `.txt`, `README`, `docs/` | 스킵 |

> **우선순위**: 구체적 경로가 일반 확장자보다 우선. 예: `settings.json`은 "시스템 동작 영향" 행 적용 (`.json` 일반 규칙보다 우선).
> **"로직 변경" 예시**: 조건 분기 추가/삭제, 새 규칙·금지사항 신설, 에이전트 호출 조건 변경, hook 트리거 조건 변경, 분류표 행 추가/삭제. **아닌 것**: 문구 rewording, 오타 수정, 설명 보강, 예시 추가.

1. 코드 변경 작업 완료 → **tester 호출** (테스트 코드 생성). 문서만 수정한 작업은 tester 스킵하고 2번으로 (위 분류표 참조).
2. tester 완료 → **즉시 reviewer 호출** (게이트 패턴 적용, 사용자 OK 이전에 실행)
3. reviewer REVISE → 수정 → 재검증 루프 (PASS까지 반복)
4. reviewer PASS → 사용자에게 완료 보고 (수정 파일 + 요약 + reviewer 결과 중계)
5. 사용자 OK 대기
6. Claude가 직접 수행 (OK 받은 즉시, 다음 작업 시작 전에 반드시 실행):
   - 03_checklist.md 해당 항목 `[x]` 마킹
   - 02_context.md "태스크 진행 상태" 섹션에 완료 항목 + 다음 항목 기록
7. (Stop hook이 [x] 증가 감지 → [HOOK:CHECKLIST_DONE] 발동 → 커밋)

> **Hook 강제 사항**: agent-tracker.sh가 developer/tester/reviewer 호출을 추적합니다.
> - tester/reviewer 미호출 시 plan-gate가 다음 코드 수정 시 **경고** 출력
> - reviewer REVISE 상태에서 코드 수정 시 plan-gate가 **차단** (exit 2)
> - [x] 마킹 시 체인 미완료면 detect-state가 **[HOOK:CHAIN_INCOMPLETE]** 태그 출력

## 할루시네이션 방어
- API/함수: 존재 여부 확인 안 됐으면 "확인 필요" 명시. 추측으로 호출하지 않는다.
- 버전: 모르면 "최신 문서 확인 필요" 표기. 임의 버전 번호 금지.
- 패키지 설치: 실행 전 `pip show`로 존재 확인. 없는 패키지 설치 시도 금지.
- 수치/벤치마크: 직접 측정한 것만 인용. 출처 없는 수치 금지.
- 못 찾으면 "확인하지 못했습니다" 명시. 추측 금지.

## MCP 에러 대응
MCP 도구 호출 실패 시 에러 유형을 분류하고 적절히 대응한다.

### 에러 분류 및 대응
| 에러 메시지 | 유형 | 대응 |
|------------|------|------|
| `Server not initialized` | MCP 세션 문제 | ToolSearch로 도구 재로드 → 재호출 |
| `Was there a typo in the url?` | MCP 세션 문제 | ToolSearch로 도구 재로드 → 재호출 |
| `Could not resolve host` (gh CLI) | 네트워크 문제 | 사용자에게 보고 |
| `connection refused` (gh CLI) | 네트워크 문제 | 사용자에게 보고 |

### 규칙
- MCP 세션 에러: ToolSearch로 재로드 시도 → 재호출. 최대 3회 재시도, 초과 시 사용자에게 보고.
- 네트워크 에러 (gh CLI): 재시도하지 않고 즉시 사용자에게 보고.
- MCP 에러(세션)와 gh CLI 에러(네트워크)를 혼동하지 말 것. 원인이 다르므로 대응도 다르다.
- 위 분류표에 없는 MCP 에러도 세션 문제로 우선 시도 (ToolSearch 재로드 → 재호출).

## 지식 축적 규칙

### 쓰기
아래 상황 발생 시 knowledge-writer 에이전트를 호출하여 `.manuals/knowledge/`에 기록:
- 에러/버그 해결 완료 후 → errors/에 기록
- 중요한 발견, A-ha point 발견 시 → discoveries/에 기록
- 사용자가 "이거 기록해둬" 요청 시 → 적절한 카테고리에 기록
- 에러 해결·발견 시 knowledge-writer 호출을 잊지 말 것

### 읽기
- planner 분석 단계: INDEX.md 태그 기반으로 관련 knowledge 검색 → 02_context.md에 기록
- reviewer 검증 단계: 관련 knowledge에 기록된 이전 이슈가 재발하지 않는지 확인

## 작업 기억
- 큰 작업은 `.plans/YYYYMMDD-작업명/`에 계획서·맥락노트·체크리스트 생성. `/task-plan` 참조.
- 세션 복구 시 `.plans/` 최신 폴더부터 읽기. `/restore` 참조.
- 플랜 문서 수정 시 01_plan → 02_context → 03_checklist 순서로 동기화.
- 커밋 규칙은 version-control skill (`.manuals/dev/git.md`) 참조.

## 컴포넌트 목록

### Skills (14개)
version-control, plan-writing, review-rules, testing, error-logging, profiling, debug, dependency-check, research, doc-update, jira-workflow, machine-profile, knowledge-management, migration

### Agents (12개)
planner, developer, tester, reviewer, recommender, analyzer → Explore, atlassian-manager, github-manager, researcher, doc-writer, knowledge-writer, migrator

### Hooks (6개)
- plan-gate.sh (PreToolUse: Write|Edit) — 플랜 없이 코드 수정 차단 + 체인 미완료 경고/차단
- detect-state.sh (Stop) — 체크리스트 상태 감지 + 태그 메시지 + 체인 미완료 감지
- agent-tracker.sh (PostToolUse: Agent) — 에이전트 호출 추적 (developer/tester/reviewer/recommender)
- commit-guard.sh (PostToolUse: Bash) — 번들 커밋 경고 (체크리스트 대비 커밋 수 초과 감지)
- mirror-check.sh (PostToolUse: Write|Edit) — src/ 미러링 폴더 누락 경고
- session-init.sh (SessionStart) — 안내 + 머신 프로파일 체크 + 간결 상태 출력

### Commands (7개)
/task-plan, /restore, /check, /scaffold, /issue, /migrate, /auto
