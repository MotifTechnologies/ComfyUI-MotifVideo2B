---
name: error-logging
description: 에러/버그 해결 기록 유도. 에러 수정, 디버깅 완료, 트러블슈팅 후 적용.
---

# 에러 기록 유도

에러/버그를 해결했으면 knowledge-writer agent를 호출하여 기록할 것.

## 기록 트리거
- 에러/버그 해결 완료 시
- 환경 이슈 해결 시 (CUDA, 의존성 충돌 등)
- 사용자가 "이거 기록해둬" 요청 시

## 기록 대상
- 재현 조건
- 원인
- 해결법
- 관련 파일/명령어

## 기록 방법
knowledge-writer agent 호출. 실제 분류/저장은 knowledge-writer가 담당.
카테고리/포맷은 `.manuals/knowledge/INDEX.md` 참고.
