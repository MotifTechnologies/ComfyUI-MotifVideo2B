---
name: profiling
description: 프로파일링/성능 측정 규칙. 최적화, 성능 비교, 벤치마크 시 적용.
---

# 프로파일링 규칙

성능 측정/최적화 시 `.manuals/dev/profiling.md`를 읽고 따를 것.

## 핵심 (목차만)
- 최적화 전 반드시 측정 먼저
- 옵션 2~3개 비교 → 수치 기반 선택
- before/after 로깅 필수
- 도구: GPU(torch.profiler, nsys), CPU(cProfile), 메모리(tracemalloc), I/O(iostat)
- 상세 절차는 `.manuals/dev/profiling.md` 참고
