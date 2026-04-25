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


def install_requirements() -> int:
    if not REQ.exists():
        print(
            f"[MotifVideo install] ERROR: requirements.txt not found at {REQ}.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        _run([PY, "-m", "pip", "install", "-r", str(REQ)])
    except subprocess.CalledProcessError as e:
        code = e.returncode
        print(
            f"[MotifVideo install] ERROR: pip install -r requirements.txt failed"
            f" (exit {code}). See stderr above.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


def main() -> int:
    _log(f"python: {PY}")
    return install_requirements()


if __name__ == "__main__":
    raise SystemExit(main())
