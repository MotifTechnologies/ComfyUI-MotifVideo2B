"""
tests/nodes/test_empty_motif_latent.py

Item 3: EmptyMotifLatent.generate() — 16-픽셀 floor snap + latent shape 검증.

환경 제약: CPU-only → comfy.model_management 를 mock 처리한 뒤
importlib.util.spec_from_file_location 으로 nodes/latent.py 직접 로드.
"""

import importlib.util
import sys
import types
import pathlib
import pytest


# ---------------------------------------------------------------------------
# comfy mock — GPU 없는 환경에서 import 실패 방지
# ---------------------------------------------------------------------------
def _install_comfy_mock():
    """comfy.model_management 를 CPU stub 으로 등록한다.

    nodes/latent.py 는 모듈 최상단에서
        import comfy.model_management
        ...comfy.model_management.intermediate_device()...
    형태로 사용한다. sys.modules 에 등록할 때 comfy 패키지 오브젝트에도
    model_management 속성을 붙여야 AttributeError 를 막을 수 있다.
    """
    mm = types.ModuleType("comfy.model_management")
    mm.intermediate_device = lambda: "cpu"

    # comfy 패키지 오브젝트가 이미 있으면 재사용, 없으면 새로 생성
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    comfy.model_management = mm

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm


_install_comfy_mock()

# ---------------------------------------------------------------------------
# nodes/latent.py 직접 로드 (source drift 방어)
# ---------------------------------------------------------------------------
_LATENT_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "nodes" / "latent.py"
)


def _load_latent_module():
    spec = importlib.util.spec_from_file_location("nodes.latent", _LATENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_latent_mod = _load_latent_module()
EmptyMotifLatent = _latent_mod.EmptyMotifLatent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _generate(width, height, num_frames=121, batch_size=1):
    """EmptyMotifLatent().generate() 결과 tuple 의 첫 번째 dict 반환."""
    node = EmptyMotifLatent()
    result = node.generate(width=width, height=height,
                           num_frames=num_frames, batch_size=batch_size)
    # ComfyUI 노드 규약: ({"samples": tensor},)
    return result[0]["samples"]


# ---------------------------------------------------------------------------
# Test 1 — 비정렬 입력: 1281×737 → snap → 1280×736
# ---------------------------------------------------------------------------
def test_generate_snaps_non_multiple_of_16():
    """width/height 가 16 비배수이면 floor snap 후 올바른 shape 반환."""
    tensor = _generate(width=1281, height=737, num_frames=121, batch_size=1)
    # T//4 + 1 = 121//4 + 1 = 31
    # H_snapped // 8 = 736 // 8 = 92
    # W_snapped // 8 = 1280 // 8 = 160
    assert tensor.shape == (1, 16, 31, 92, 160), (
        f"Expected (1,16,31,92,160), got {tuple(tensor.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — 정렬 입력 no-op: 1280×736 → snap 없이 동일 shape
# ---------------------------------------------------------------------------
def test_generate_aligned_input_no_op():
    """기본값 1280×736 은 이미 16-배수 → snap no-op, shape 동일."""
    tensor = _generate(width=1280, height=736, num_frames=121, batch_size=1)
    assert tensor.shape == (1, 16, 31, 92, 160), (
        f"Expected (1,16,31,92,160), got {tuple(tensor.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — 정렬된 다른 해상도: 1296×752 → snap no-op
# ---------------------------------------------------------------------------
def test_generate_aligned_larger_resolution():
    """1296×752 (둘 다 16-배수) → snap no-op, shape 반영."""
    tensor = _generate(width=1296, height=752, num_frames=121, batch_size=1)
    # H_snapped // 8 = 752 // 8 = 94
    # W_snapped // 8 = 1296 // 8 = 162
    assert tensor.shape == (1, 16, 31, 94, 162), (
        f"Expected (1,16,31,94,162), got {tuple(tensor.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — INPUT_TYPES 에 width/height step=16 이 선언되어 있는지 확인
# ---------------------------------------------------------------------------
def test_input_types_step_is_16():
    """INPUT_TYPES 의 width, height 에 step=16 이 선언되어 있어야 한다."""
    input_types = EmptyMotifLatent.INPUT_TYPES()
    required = input_types.get("required", {})

    width_spec = required.get("width", (None, {}))
    height_spec = required.get("height", (None, {}))

    # width_spec 은 (type_name, options_dict) 형태
    width_opts = width_spec[1] if len(width_spec) > 1 else {}
    height_opts = height_spec[1] if len(height_spec) > 1 else {}

    assert width_opts.get("step") == 16, (
        f"width step 이 16 이어야 하지만 실제: {width_opts.get('step')}"
    )
    assert height_opts.get("step") == 16, (
        f"height step 이 16 이어야 하지만 실제: {height_opts.get('step')}"
    )


# ---------------------------------------------------------------------------
# Test 5 — 경계값: width/height 가 15 (16 미만) → 0으로 snap
# ---------------------------------------------------------------------------
def test_generate_width_height_below_16_snaps_to_zero():
    """15×15 입력은 0×0 으로 snap. 예외 없이 처리되어야 한다 (shape 검증)."""
    try:
        tensor = _generate(width=15, height=15, num_frames=1, batch_size=1)
        # 0×0 snap → dim = 0 // 8 = 0 → tensor dim 이 0
        # 구현이 어떻게 처리하든 예외 없이 반환해야 한다.
        # shape 중 H/W 차원이 0 이면 snap 이 적용된 것.
        h_dim = tensor.shape[3]
        w_dim = tensor.shape[4]
        assert h_dim == 0 and w_dim == 0, (
            f"15×15 입력은 0×0 snap 이어야 하지만 shape={tuple(tensor.shape)}"
        )
    except Exception as exc:
        pytest.fail(f"15×15 입력에서 예기치 않은 예외: {exc}")


# ---------------------------------------------------------------------------
# Test 6 — batch_size=2 → batch dim 반영 확인
# ---------------------------------------------------------------------------
def test_generate_batch_size_reflected_in_shape():
    """batch_size=2 이면 latent 의 첫 번째 dim 이 2."""
    tensor = _generate(width=1280, height=736, num_frames=121, batch_size=2)
    assert tensor.shape[0] == 2, (
        f"batch dim 이 2 이어야 하지만 실제: {tensor.shape[0]}"
    )
