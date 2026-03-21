---
name: testing
description: 테스트 작성 규칙. 테스트 코드 생성, 테스트 실행, 커버리지 확인 시 적용.
---

# 테스트 규칙

테스트 작성 시 `.manuals/process/testing.md`를 읽고 따를 것.

## 핵심 (목차만)
- 네이밍: test_<기능>_<시나리오>_<기대결과>
- 구조: src/ 미러링 (tests/ 하위에 동일 경로)
- 커버리지: 핵심 로직 필수, 유틸/설정은 선택
- 상세 규칙은 `.manuals/process/testing.md` 참고
