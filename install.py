#!/usr/bin/env python3
"""ComfyUI Manager custom node install script — delegates to pip install -r requirements.txt.

ComfyUI Manager 는 custom node 설치 시 이 파일이 존재하면 자동 실행한다.
requirements.txt 의 pip install-able 의존성을 설치한다 (Manager 도 별도로 호출하지만
수동 git clone 경로 호환을 위해 여기서도 실행).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQ = HERE / "requirements.txt"

PY = sys.executable


def _run(args: list[str], check: bool = True) -> int:
    print(f"[MotifVideo install] $ {' '.join(args)}")
    return subprocess.call(args) if not check else subprocess.check_call(args)


def _log(msg: str) -> None:
    print(f"[MotifVideo install] {msg}", flush=True)


def install_requirements() -> None:
    if not REQ.exists():
        return
    try:
        _run([PY, "-m", "pip", "install", "-r", str(REQ)])
    except subprocess.CalledProcessError as e:
        _log(f"WARN: pip install -r requirements.txt failed: {e}. Continuing.")


def main() -> int:
    _log(f"python: {PY}")
    install_requirements()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
