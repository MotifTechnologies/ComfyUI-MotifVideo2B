#!/usr/bin/env python3
"""ComfyUI-MotifVideo1.9B install script.

ComfyUI Manager 는 custom node 설치 시 이 파일이 존재하면 자동 실행한다.
동작:
  1) requirements.txt 의 pip install-able 의존성 설치 (Manager 도 별도로 호출
     하지만, 수동 git clone 경로 호환을 위해 여기서도 실행).
  2) CUDA 가용 여부 탐지.
  3) 현재 GPU 의 compute capability (CUDA arch) 탐지.
  4) 이미 sageattention 이 import 가능하면 skip.
  5) `TORCH_CUDA_ARCH_LIST` 환경변수를 해당 arch 로 세팅한 뒤
     `pip install sageattention --no-build-isolation` 로 source build.
     - 5~15 분 소요. 완료되면 MotifVideoLoader 가 자동 활성화.
  6) 빌드 실패해도 exit 0 — SageAttention 없이도 기본 attention 으로 동작.

수동 설치 fallback:
    TORCH_CUDA_ARCH_LIST=9.0 \
    <comfyui_python> -m pip install sageattention --no-build-isolation

비-CUDA 환경 (CPU only) 은 SageAttention 자체가 의미 없으므로 skip.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQ = HERE / "requirements.txt"

PY = sys.executable


def _run(args: list[str], env: dict | None = None, check: bool = True) -> int:
    print(f"[MotifVideo install] $ {' '.join(args)}")
    return subprocess.call(args, env=env) if not check else subprocess.check_call(args, env=env)


def _log(msg: str) -> None:
    print(f"[MotifVideo install] {msg}", flush=True)


def install_requirements() -> None:
    if not REQ.exists():
        return
    try:
        _run([PY, "-m", "pip", "install", "-r", str(REQ)])
    except subprocess.CalledProcessError as e:
        _log(f"WARN: pip install -r requirements.txt failed: {e}. Continuing.")


def detect_cuda_arch() -> str | None:
    """현재 환경의 CUDA arch ("9.0", "8.9" 등) 반환. 미가용 시 None."""
    try:
        import torch
    except ImportError:
        _log("torch not importable — deferring to later import. Skipping SageAttention.")
        return None
    if not torch.cuda.is_available():
        _log("CUDA not available — skipping SageAttention (GPU-only).")
        return None
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"{major}.{minor}"
        _log(f"detected CUDA arch: {arch} (device: {torch.cuda.get_device_name(0)})")
        return arch
    except Exception as e:
        _log(f"WARN: failed to detect CUDA arch ({e}). Defaulting to 8.0.")
        return "8.0"


def sageattention_already_installed() -> bool:
    try:
        import sageattention  # noqa: F401
        _log("sageattention already importable — skip.")
        return True
    except ImportError:
        return False


def install_sageattention(arch: str) -> bool:
    """sageattention source build. 성공 True / 실패 False. 예외는 삼킨다."""
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = arch
    _log(f"Building sageattention for CUDA arch {arch}. Source build — 5~15 minutes expected.")
    _log("If this hangs, check `nvcc --version` is installed and matches your PyTorch CUDA build.")
    try:
        _run(
            [PY, "-m", "pip", "install", "sageattention", "--no-build-isolation", "-v"],
            env=env,
        )
        _log("sageattention installed successfully — MotifVideo will use it automatically.")
        return True
    except subprocess.CalledProcessError as e:
        _log(f"WARN: sageattention build failed (exit {e.returncode}).")
        _log("MotifVideo falls back to default attention (slower). Retry manually with:")
        _log(f"  TORCH_CUDA_ARCH_LIST={arch} {PY} -m pip install sageattention --no-build-isolation")
        return False


def main() -> int:
    _log(f"python: {PY}")
    install_requirements()

    arch = detect_cuda_arch()
    if arch is None:
        return 0  # CPU-only or torch not ready — skip cleanly.

    if sageattention_already_installed():
        return 0

    install_sageattention(arch)
    return 0  # Always succeed so ComfyUI Manager does not mark the node as broken.


if __name__ == "__main__":
    raise SystemExit(main())
