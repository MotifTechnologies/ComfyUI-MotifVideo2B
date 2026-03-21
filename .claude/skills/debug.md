---
name: debug
description: 디버깅 절차. 에러 발생, 버그 조사, 트러블슈팅 시 적용.
---

# 디버깅 규칙

## 첫 번째 단계: 기존 지식 검색
디버깅 시작 전 반드시 `.manuals/knowledge/errors/`를 Grep으로 검색.
이미 해결된 문제일 수 있음.

## 디버깅 절차
1. **재현**: 에러를 안정적으로 재현할 수 있는 조건 확인
2. **기존 지식 검색**: `.manuals/knowledge/errors/` + `.manuals/knowledge/discoveries/`
3. **범위 좁히기**: 이분법으로 원인 범위 축소
4. **원인 확인**: 로그, 스택트레이스, 디버거 활용
5. **수정 + 검증**: 수정 후 재현 조건에서 재확인

## 해결 후
에러 해결 완료 시 error-logging skill에 따라 knowledge-writer agent로 기록.
(debug = 검색, error-logging = 기록 유도)
