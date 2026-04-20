# tests/transformer/test_motif_video_model_no_cast.py
#
# Verifies P4.1 static criterion:
#   models/__init__.py must NOT contain .to(dtype=torch.bfloat16).
#
# No GPU required — pure file inspection.

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_motif_video_model_no_forced_bfloat16_cast():
    """models/__init__.py must NOT contain .to(dtype=torch.bfloat16).

    Forced cast was removed in P4.1: comfy.ops handles weight dtype at load
    time; a hard cast would bypass the manual_cast circuit and corrupt
    fp8/quantized weights.
    """
    source_path = os.path.join(_REPO_ROOT, "models", "__init__.py")
    with open(source_path, encoding="utf-8") as f:
        source = f.read()

    assert ".to(dtype=torch.bfloat16)" not in source, (
        "models/__init__.py must not contain '.to(dtype=torch.bfloat16)' — "
        "forced cast was removed in P4.1 (comfy.ops handles dtype at load time)."
    )
