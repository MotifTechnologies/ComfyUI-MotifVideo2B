# ARCHIVED: P4.2 완료 후 MotifVideoAttnProcessor2_0 삭제로 이 스크립트는 재실행 불가.
# 체크인된 tests/transformer/fixtures/attention_golden_*.pt 파일이 단일 진실.
"""P2.2 golden fixture 재현 스크립트.

P2.2 구현 시점에 살아있는 MotifVideoAttnProcessor2_0 의 __call__ 로직으로
같은 seed + 같은 input 으로 reference output 을 생성. MotifVideoAttention.forward
이 처리기와 수치적으로 일치해야 test_attention_forward_sdpa.py 통과.

re-run 조건: P4.2 에서 MotifVideoAttnProcessor2_0 가 삭제되기 전까지 재현 가능.
그 이후에는 tests/transformer/fixtures/attention_golden_{single,dual}_{out,ctx}.pt 가
단일 진실로 고정 (recommender 제안, reviewer 지적 반영).

weights 는 저장하지 않음. `torch.manual_seed(0)` + 동일 signature 로 `MotifVideoAttention` 을
다시 만들면 bit-for-bit 동일 파라미터 재현 (deterministic init).

실행: PYTHONPATH=<ComfyUI-root>/.venv/lib/python3.12/site-packages:<ComfyUI-root> \\
       python tests/transformer/fixtures/generate_attention_golden.py
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import pathlib
import sys
import types

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FIXTURES_DIR = pathlib.Path(__file__).resolve().parent
_REPO = _FIXTURES_DIR.parents[2]  # ComfyUI-MotifVideo1.9B/
_BASE = _REPO / "models" / "transformer"

# ---------------------------------------------------------------------------
# comfy.ops mock (CUDA-free environment)
# ---------------------------------------------------------------------------

def _make_comfy_ops_mock():
    class _MockOps:
        class Linear(nn.Linear):
            def __init__(self, *args, dtype=None, device=None, **kwargs):
                super().__init__(*args, **kwargs)

        class RMSNorm(nn.RMSNorm):
            def __init__(self, normalized_shape, eps=None, dtype=None, device=None, **kwargs):
                super().__init__(normalized_shape, eps=eps or 1e-6)

    mock_comfy = types.ModuleType("comfy")
    mock_comfy.__path__ = []
    mock_comfy.__package__ = "comfy"
    mock_ops = types.ModuleType("comfy.ops")
    mock_ops.disable_weight_init = _MockOps
    mock_comfy.ops = mock_ops
    return mock_comfy, mock_ops


def _should_use_real_comfy() -> bool:
    if os.environ.get("MOTIF_FORCE_MOCK_COMFY") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import comfy.ops  # noqa: F401
        return True
    except Exception:
        return False


def _install_mock() -> None:
    mock_comfy, mock_ops = _make_comfy_ops_mock()
    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.ops"] = mock_ops


# Install mock before any models import
if not _should_use_real_comfy():
    _install_mock()

# ---------------------------------------------------------------------------
# Load modules via importlib (bypasses models/__init__.py CUDA init)
# Register fake package context so relative imports work
# ---------------------------------------------------------------------------

def _register_package(name: str, path: pathlib.Path) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_mod(name: str, file_path: pathlib.Path) -> types.ModuleType:
    spec = _ilu.spec_from_file_location(
        name, file_path,
        submodule_search_locations=[str(file_path.parent)],
    )
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = "models.transformer"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_register_package("models", _REPO / "models")
_register_package("models.transformer", _BASE)

_load_mod("models.transformer.ops_primitives", _BASE / "ops_primitives.py")
_load_mod("models.transformer.tread_mixin", _BASE / "tread_mixin.py")
_tmv = _load_mod("models.transformer.transformer_motif_video", _BASE / "transformer_motif_video.py")
MotifVideoAttnProcessor2_0 = _tmv.MotifVideoAttnProcessor2_0

_attn_spec = _ilu.spec_from_file_location(
    "models.transformer.attention", _BASE / "attention.py",
    submodule_search_locations=[str(_BASE)],
)
_attn_mod = _ilu.module_from_spec(_attn_spec)
_attn_mod.__package__ = "models.transformer"
_attn_mod._DEFAULT_OPS = None
sys.modules["models.transformer.attention"] = _attn_mod
_attn_spec.loader.exec_module(_attn_mod)
MotifVideoAttention = _attn_mod.MotifVideoAttention

# ---------------------------------------------------------------------------
# Dims from expected_attn_keys.json
# ---------------------------------------------------------------------------
EXPECTED = json.loads((_REPO / "tests/transformer/expected_attn_keys.json").read_text())


def _dims():
    hidden = EXPECTED["single_block"]["to_q.weight"]["shape"][0]
    head_dim = EXPECTED["single_block"]["norm_q.weight"]["shape"][0]
    return hidden // head_dim, head_dim, hidden


def _inputs(hidden, seed=42):
    g = torch.Generator().manual_seed(seed)
    B, L, E = 2, 16, 8
    hs = torch.randn(B, L, hidden, generator=g)
    eh = torch.randn(B, E, hidden, generator=g)
    return hs, eh


def _save(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    print(f"  -> {path.relative_to(_REPO)} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def main():
    num_heads, head_dim, hidden = _dims()
    fixtures_dir = _REPO / "tests/transformer/fixtures"

    torch.manual_seed(0)  # for weight init

    # --- Single block (pre_only=True) ---
    print("[single] generating...")
    attn_s = MotifVideoAttention(num_heads, head_dim, pre_only=True, added_kv=False, qk_norm="rms_norm")
    processor = MotifVideoAttnProcessor2_0()
    hs, eh = _inputs(hidden, seed=42)
    with torch.no_grad():
        out_s, ctx_s = processor(
            attn_s, hidden_states=hs, encoder_hidden_states=eh,
            attention_mask=None, image_rotary_emb=None,
        )
    _save(out_s, fixtures_dir / "attention_golden_single_out.pt")
    if ctx_s is not None:
        _save(ctx_s, fixtures_dir / "attention_golden_single_ctx.pt")

    # --- Dual block (pre_only=False, added_kv=True) ---
    print("[dual] generating...")
    torch.manual_seed(0)
    attn_d = MotifVideoAttention(num_heads, head_dim, pre_only=False, added_kv=True, qk_norm="rms_norm")
    hs, eh = _inputs(hidden, seed=42)
    with torch.no_grad():
        out_d, ctx_d = processor(
            attn_d, hidden_states=hs, encoder_hidden_states=eh,
            attention_mask=None, image_rotary_emb=None,
        )
    _save(out_d, fixtures_dir / "attention_golden_dual_out.pt")
    _save(ctx_d, fixtures_dir / "attention_golden_dual_ctx.pt")
    print("done.")


if __name__ == "__main__":
    main()
