---
name: review-rules
description: 리뷰 규칙 (코드/플랜/완료). 리뷰 요청, 체크리스트 완료, 플랜 생성, 전체 작업 완료 시 적용.
---

# 리뷰 규칙

리뷰 수행 시 `.manuals/process/review.md`를 읽고 따를 것.

## 모드 판단
- **코드 리뷰**: 체크리스트 항목 완료 후 → `.manuals/process/review.md` 코드 리뷰 섹션
- **플랜 리뷰**: 플랜/체크리스트 신규 생성 후 → `.manuals/process/review.md` 플랜 리뷰 섹션
- **완료 리뷰**: 전체 체크리스트 [ ] == 0 → `.manuals/process/review.md` 완료 리뷰 섹션

## 핵심 (목차만)
- 코드: 버그, 성능, 보안, 네이밍, 일관성
- 플랜: 커버리지, 실행 가능성, 빠진 것, 리스크
- 완료: 목표 달성, 전체 일관성, 문서 상태
- 상세 체크리스트는 `.manuals/process/review.md` 참고
