#!/usr/bin/env python3
"""ComfyUI Manager custom node install script — delegates to pip install -r requirements.txt.

ComfyUI Manager auto-runs this file when it is present during custom-node
installation. It installs the pip-installable dependencies from
requirements.txt (Manager also invokes pip itself, but we run it here too
so the manual `git clone` path works the same way).
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
