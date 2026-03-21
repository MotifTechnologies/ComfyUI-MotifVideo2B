---
name: dependency-check
description: 패키지/의존성 설치 규칙. pip install, 패키지 추가, CUDA/PyTorch 호환 확인 시 적용.
---

# 의존성 확인 규칙

패키지 설치/변경 시 `.manuals/dev/dependency.md`를 읽고 따를 것.

## 핵심 (목차만)
- 설치 전 존재 확인 (pip show, conda list)
- CUDA/PyTorch/Python 버전 호환 확인
- requirements.txt 또는 pyproject.toml 반영
- 상세 규칙은 `.manuals/dev/dependency.md` 참고
