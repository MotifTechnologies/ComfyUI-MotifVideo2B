---
name: migration
description: 이식/업데이트 분류 규칙. 기존 프로젝트에 template 적용하거나 업데이트할 때 적용.
---

# 분류 규칙 (Migration)

이식/업데이트 시 `.manuals/process/migration.md`를 읽고 따를 것.

## 3계층 보존 전략
- **A계층 (사용자 전용)**: 절대 보호. knowledge/, .plans/, .manuals/env/, experiments/, settings.local.json
- **B계층 (하이브리드)**: 머지 필요. CLAUDE.md, settings.json, INDEX.md, .manuals/dev,process/ (수정됨)
- **C계층 (Template 전용)**: manifest 해시 비교. 미수정=교체, 수정됨=.new+확인

## PROTECTED_PATHS (A계층)
```
.manuals/knowledge/errors/
.manuals/knowledge/discoveries/
.manuals/env/
.plans/
.analyze/
experiments/          # TEMPLATE.md, .gitkeep 제외
.claude/settings.local.json
```

## 충돌 해결 우선순위
1. **Template 우선**: 시스템 동작 영역 (hook 태그, 게이트 패턴, 에이전트 체인, settings.json hook 등록)
2. **기존 우선**: 프로젝트 특화 영역 (코딩 컨벤션, 테스트 방식, 브랜치 전략)
3. **머지**: 양쪽 유효 (응답 규칙, permissions, .gitignore)
4. **판단 불가**: 사용자에게 선택지 제시

## Manifest 비교 방법
- 포맷: key=value 헤더 + tab-separated 파일 목록 (`.claude/.template-manifest`)
- 비교: `sha256sum <파일>` vs manifest 기록 해시
- 해시 일치 = 미수정 -> 자동 교체 가능
- 해시 불일치 = 수정됨 -> .new 생성 + 사용자 확인
- manifest에 없는 파일 = 사용자 추가 -> 절대 건드리지 않음

## 상세 규칙
- 파일별 분류 매핑 테이블: `.manuals/process/migration.md` 3절
- CLAUDE.md 섹션 분류: `.manuals/process/migration.md` 4절
- 에이전트/스킬 머지: `.manuals/process/migration.md` 5-6절
- 이식/업데이트 처리 매트릭스: `.manuals/process/migration.md` 10절
