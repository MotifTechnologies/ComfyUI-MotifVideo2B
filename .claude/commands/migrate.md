# /migrate -- claude-code-kit 이식/업데이트

기존 프로젝트에 claude-code-kit를 이식하거나, 이미 설치된 template을 최신 버전으로 업데이트합니다.

migrator 에이전트를 호출하여 작업을 수행하세요.

## 모드 판별

`$ARGUMENTS`를 확인합니다:

- **비어있음 또는 `--update` 외 값**: 이식 모드 (기본)
- **`--update`**: 업데이트 모드

## 이식 모드 (기본)

migrator 에이전트를 **이식 모드**로 호출합니다.

migrator가 아래를 순서대로 수행합니다:
1. GitHub 접근성 체크 (gh/ssh)
2. claude-code-kit repo clone
3. 기존 파일 백업 (`_old_claude_files/`)
4. setup.sh 실행 (새 구조 설치)
5. 기존 파일 분석 + 분류 제안 + 사용자 확인 후 적용

## 업데이트 모드 (`--update`)

### 전제 조건
- `.claude/.template-manifest` 파일이 존재해야 합니다
- manifest가 없으면 아래 안내 메시지를 출력하고 중단합니다:
  ```
  .claude/.template-manifest가 없습니다.
  먼저 setup.sh를 실행하거나, /migrate로 이식을 완료한 뒤 다시 시도하세요.
  ```

### 프로세스

migrator 에이전트를 **업데이트 모드**로 호출합니다.

migrator가 아래를 순서대로 수행합니다:

1. **Repo Fetch**: claude-code-kit 최신 template을 /tmp에 clone 또는 pull
2. **Manifest 비교**: template 파일 목록과 `.claude/.template-manifest`를 대조하여 변경분을 4가지 카테고리(새 파일, 삭제된 파일, 수정된 파일, 미변경 파일)로 분류
3. **계층별 처리**:
   - A계층 (PROTECTED_PATHS): 완전 스킵 -- 존재 여부조차 확인하지 않음
   - C계층 미수정 (현재 해시 == manifest 해시): 자동 교체
   - C계층 수정됨 (현재 해시 != manifest 해시): `.new` 파일 생성 + diff + 사용자 선택 ([교체]/[유지]/[수동 머지])
   - B계층 (CLAUDE.md, settings.json, INDEX.md 등): 파일별 LLM 머지 또는 JSON 머지 제안
   - manifest에 없는 파일 (사용자 추가): 절대 건드리지 않음
4. **Manifest 갱신 + 보고**: 교체/머지된 파일의 해시 재계산, VERSION/INSTALLED 갱신, 변경 로그 출력
