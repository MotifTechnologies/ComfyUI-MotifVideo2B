"""MotifVideo transformer compile + inductor/SDP 설정 헬퍼.

설계 원칙:
1. 모듈 import 만으로는 어떤 전역 상태도 바꾸지 않는다. 전역
   `torch._dynamo`/`torch._inductor`/SDP 설정은 `apply_compile()` 이 처음
   호출되는 시점(사용자가 실제로 MotifVideoModel 노드를 실행하는 시점)에만
   적용된다. ComfyUI 는 모든 커스텀 노드를 시작 시 import 하므로 import-time
   side-effect 는 같은 프로세스의 다른 모델에 영향을 줄 수 있다.
2. 본 모듈은 PyTorch private API 일부 (`torch._dynamo.config`,
   `torch._inductor.config`, `torch._inductor.runtime.triton_heuristics`,
   `torch.backends.cuda.enable_*_sdp`) 를 사용한다. PyTorch 버전에 따라 필드
   이름이 바뀌거나 사라질 수 있으므로, import 와 각 필드 접근은 전부 try
   로 감싸 실패 시 해당 항목만 건너뛴다. 모든 가드가 깨져도
   `torch.compile` 까지 도달 전에 `apply_compile` 은 원본 transformer 를
   반환하도록 한다.
3. `torch.compile` wrap 자체는 try 로 감싸지만, `torch.compile` 의 특성상
   실제 그래프 컴파일은 첫 forward 에서 lazy 로 일어나므로 런타임 실패는
   여기서 잡히지 않는다. 이 경우 사용자는 `MOTIFVIDEO_DISABLE_COMPILE=1`
   환경 변수로 compile 자체를 건너뛰도록 지시할 수 있다 (본 플랜 R3 명시
   fallback). 자동 first-forward fallback 은 plan 스코프 외.

포트 출처: motif-models `compile_configs.py`. SageAttention/channels_last
관련 블록은 P3.2 이후 별도 항목에서 추가 예정.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


# Guarded imports — PyTorch 버전에 따라 존재하지 않을 수 있다.
try:
    import torch._dynamo.config as _dynamo_config
except ImportError as exc:
    logger.warning("[MotifVideo] torch._dynamo.config unavailable (%s) — compile will be disabled", exc)
    _dynamo_config = None

try:
    import torch._inductor.config as _inductor_config
except ImportError as exc:
    logger.warning("[MotifVideo] torch._inductor.config unavailable (%s) — compile will be disabled", exc)
    _inductor_config = None

try:
    from torch._inductor.runtime.triton_heuristics import TRITON_MAX_BLOCK as _TRITON_MAX_BLOCK
except ImportError as exc:
    logger.warning("[MotifVideo] TRITON_MAX_BLOCK unavailable (%s) — skipping triton block override", exc)
    _TRITON_MAX_BLOCK = None


_GLOBAL_CONFIG_APPLIED = False


def _set_attr_guarded(obj, path: str, value) -> None:
    """`obj.a.b.c = value` 를 try 로 감싸 실패 시 warning + skip."""
    if obj is None:
        return
    try:
        parts = path.split(".")
        target = obj
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], value)
    except Exception as exc:  # noqa: BLE001 — private API tolerance
        logger.warning("[MotifVideo] skip config %s=%r (%s)", path, value, exc)


def _call_guarded(fn, *args, label: str = "") -> None:
    """callable 을 try 로 감싸 실패 시 warning + skip."""
    if fn is None:
        return
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MotifVideo] skip call %s (%s)", label or getattr(fn, "__name__", "<fn>"), exc)


def _apply_global_config() -> None:
    """첫 호출에서만 전역 inductor/dynamo/SDP 설정을 적용 (이후 호출은 no-op).

    각 설정 할당은 개별 try 로 가드. 일부 필드가 PyTorch 버전 차이로 존재하지
    않아도 나머지는 적용된다. 모든 가드가 깨져도 예외는 상위로 전파하지 않는다.
    """
    global _GLOBAL_CONFIG_APPLIED
    if _GLOBAL_CONFIG_APPLIED:
        return

    _set_attr_guarded(_dynamo_config, "capture_scalar_outputs", True)

    if _TRITON_MAX_BLOCK is not None:
        try:
            _TRITON_MAX_BLOCK["X"] = 8192
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MotifVideo] skip TRITON_MAX_BLOCK[X]=8192 (%s)", exc)

    # ComfyUI 환경 분기: 원본 compile_configs.py 와 3종 설정만 다르다.
    # - freezing=True → False : weight inline 로 인한 GPU 상주가 ComfyUI 의
    #   dynamic VRAM offload 와 충돌하여 120GB 폭증을 유발함. offload 가능 상태 유지.
    # - max_autotune_pointwise=True → False : autotune kernel variant 다수를 동시
    #   유지해 VRAM 누적. 원본은 전용 프로세스라 견디지만 ComfyUI 공존 환경엔 부적합.
    # - coordinate_descent_tuning=True → False : autotune 경로 추가 확장. 동일 사유.
    # 04_log.md 'P3.1.1 VRAM 폭증 fix' 항목 참조.
    for path, value in (
        ("compile_threads", 1),
        ("freezing", False),
        ("permute_fusion", True),
        ("split_reductions", True),
        ("layout_optimization", False),
        ("shape_padding", True),
        ("force_shape_pad", False),
        ("max_fusion_size", 4),
        ("triton.autotune_pointwise", False),
        ("triton.cudagraphs", False),
        ("aggressive_fusion", False),
        ("max_autotune_gemm", False),
        ("max_autotune_pointwise", False),
        ("epilogue_fusion", True),
        ("coordinate_descent_tuning", False),
        ("cpp_wrapper", True),
        ("cpp.enable_kernel_profile", False),
    ):
        _set_attr_guarded(_inductor_config, path, value)

    _call_guarded(getattr(torch.backends.cuda, "enable_math_sdp", None), True, label="enable_math_sdp")
    _call_guarded(getattr(torch.backends.cuda, "enable_mem_efficient_sdp", None), True, label="enable_mem_efficient_sdp")
    _call_guarded(getattr(torch.backends.cuda, "enable_flash_sdp", None), True, label="enable_flash_sdp")
    _call_guarded(getattr(torch.backends.cuda, "enable_cudnn_sdp", None), False, label="enable_cudnn_sdp")

    _GLOBAL_CONFIG_APPLIED = True
    logger.info("[MotifVideo] inductor/SDP global config applied (lazy, first apply_compile call)")


def apply_channels_last_3d(transformer):
    """transformer 를 channels_last_3d 메모리 포맷으로 변환한다 (in-place + return).

    원본 `pipeline_setup_common.py:setup_sage_attention` 이 SageAttention 적용과
    함께 호출하는 설정이지만, 메모리 레이아웃 자체는 SageAttention 과 독립이므로
    별도 함수로 분리한다. 실패 시 skip (warning) — 다른 메모리 포맷이 이미 설정된
    weight 에서 호출되는 등 엣지 케이스를 허용.
    """
    try:
        transformer.to(memory_format=torch.channels_last_3d)
        logger.info("[MotifVideo] channels_last_3d memory format applied")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MotifVideo] channels_last_3d apply failed (%s) — skipping", exc)
    return transformer


def apply_compile(transformer):
    """transformer 를 torch.compile 로 감싸 반환.

    Fallback 경로:
    - `MOTIFVIDEO_DISABLE_COMPILE=1` 환경 변수: 전역 설정 skip + 원본 반환.
    - CUDA 미가용 (CPU-only 환경): 전역 설정 skip + 원본 반환. 본 설정은 Inductor
      CUDA 경로 최적화용이라 CPU-only 환경에서는 의미가 없고 SDP cuDNN 토글이
      오류를 유발할 수 있다.
    - 전역 설정에 필요한 PyTorch private 모듈 중 하나라도 import 실패:
      전역 설정 skip + 원본 반환 (위 try/except 결과 기반).
    - `torch.compile(...)` wrap 호출 자체가 예외: 원본 반환.

    ⚠️ `torch.compile` 은 보통 즉시 성공하고 실제 그래프 컴파일은 첫 forward
    에서 lazy 로 일어난다. 첫 forward 에서 Inductor 실패가 RuntimeError 등으
    로 나오는 경우, 본 함수의 try/except 는 그 시점을 잡지 않는다. 해당 상황
    에서는 사용자가 `MOTIFVIDEO_DISABLE_COMPILE=1` 로 재실행하면 즉시 eager
    경로로 돌아갈 수 있다 (plan R3 명시 fallback).
    """
    if os.environ.get("MOTIFVIDEO_DISABLE_COMPILE") == "1":
        logger.warning("[MotifVideo] MOTIFVIDEO_DISABLE_COMPILE=1 — skipping torch.compile and global config")
        return transformer

    if not torch.cuda.is_available():
        logger.warning("[MotifVideo] CUDA not available — skipping torch.compile and global config (inductor targets CUDA)")
        return transformer

    if _dynamo_config is None or _inductor_config is None:
        logger.warning("[MotifVideo] torch._dynamo/_inductor unavailable — skipping torch.compile")
        return transformer

    _apply_global_config()

    try:
        # mode 분기: 원본은 "reduce-overhead" 이나 해당 모드는 CUDA graph 기반
        # buffer pre-allocation 을 시도 (cpp_wrapper=True 로 결국 skip 되지만
        # 여전히 intermediate buffer pool 유지). ComfyUI 환경에서 VRAM 증가 원인
        # 중 하나였음. "default" 로 전환하여 buffer 관리를 런타임 기본에 맡김.
        return torch.compile(transformer, mode="default", fullgraph=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[MotifVideo] torch.compile wrap failed (%s) — returning uncompiled transformer. "
            "NOTE: inductor/SDP global config is already applied to this process; "
            "restart Python to restore defaults, or set MOTIFVIDEO_DISABLE_COMPILE=1 before retrying.",
            exc,
        )
        return transformer
