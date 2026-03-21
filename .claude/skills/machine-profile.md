---
name: machine-profile
description: 머신 환경 프로파일링 규칙. 머신 스펙 확인, GPU/CPU/메모리/디스크 정보 수집 시 적용.
---

# 머신 프로파일 규칙

머신 환경 확인 시 아래 절차를 따를 것.

## 핵심
- 결과는 `.manuals/env/machine-profile.md`에 캐시
- 이미 존재하면 재수집 불필요 (사용자 요청 시만 갱신)
- SessionStart에서 없으면 `[HOOK:NEED_MACHINE_PROFILE]` 발생

## 출력 포맷
```
# Machine Profile
- Host: {hostname}
- OS: {distro + kernel}
- CPU: {model} x {cores}
- RAM: {total}
- GPU: {model} x {count}, VRAM {per card}
- CUDA: {version}
- Disk: {mount} {total}/{available}
- Python: {version}
- PyTorch: {version} (CUDA {torch.cuda})
- Conda env: {name}
```
