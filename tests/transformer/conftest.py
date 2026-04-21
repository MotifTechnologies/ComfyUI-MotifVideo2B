"""CPU-pod 환경에서 transformer_motif_video 를 collect 가능하게 하는 conftest.

transformer_motif_video.py 가 diffusers 를 module-level 로 import 하므로
diffusers 미설치 CPU pod 에선 collection error. 본 conftest 는 diffusers 가
"실제로 import 안 되는" 경우만 stub 을 설치하고, session 종료 시 복원한다.

GPU pod / diffusers 설치 환경에선 stub 을 설치하지 않아 real diffusers 가
그대로 사용된다.

또한 models / models.transformer namespace 를 sys.modules 에 사전 주입하여
_load_module("models.transformer.*") 패턴의 relative import 가 정상 동작하도록 한다.
주입된 stub 은 session 종료 시 복원된다.

설계 노트:
  - stub 설치는 module-level 에서 즉시 실행한다. pytest collection 단계에서
    test 파일의 module-level 코드(_load_module 호출)가 실행되므로, session-scoped
    fixture 로는 타이밍을 보장할 수 없다. 대신 conftest 자체가 import 되는 시점에
    stub 을 주입하고, pytest session 종료 시 복원 fixture 로 teardown 한다.
"""
from __future__ import annotations

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# diffusers availability check
# ---------------------------------------------------------------------------

def _diffusers_available() -> bool:
    if "diffusers" in sys.modules:
        return True
    try:
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def _build_diffusers_stub() -> dict:
    """최소 stub 을 설치하고 복원용 snapshot(dict) 를 반환한다."""
    saved: dict = {}

    stub_names = (
        "diffusers",
        "diffusers.configuration_utils",
        "diffusers.hooks",
        "diffusers.hooks._helpers",
        "diffusers.loaders",
        "diffusers.loaders.peft",
        "diffusers.loaders.single_file_model",
        "diffusers.models",
        "diffusers.models.attention",
        "diffusers.models.attention_processor",
        "diffusers.models.cache_utils",
        "diffusers.models.embeddings",
        "diffusers.models.modeling_outputs",
        "diffusers.models.modeling_utils",
        "diffusers.models.normalization",
        "diffusers.models.transformers",
        "diffusers.models.transformers.transformer_2d",
        "diffusers.utils",
    )
    for name in stub_names:
        saved[name] = sys.modules.get(name)

    def _stub(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    import torch.nn as nn

    root = _stub("diffusers")

    # configuration_utils
    cfg = _stub("diffusers.configuration_utils")
    class _ConfigMixin:
        _internal_dict: dict = {}
        @classmethod
        def register_to_config(cls, *a, **kw): pass
    def _register_to_config(fn):
        return fn
    cfg.ConfigMixin = _ConfigMixin
    cfg.register_to_config = _register_to_config
    root.ConfigMixin = _ConfigMixin
    root.register_to_config = _register_to_config

    # hooks / hooks._helpers
    hooks = _stub("diffusers.hooks")
    helpers = _stub("diffusers.hooks._helpers")
    helpers.TransformerBlockRegistry = None
    helpers.TransformerBlockMetadata = None
    hooks._helpers = helpers

    # loaders
    loaders = _stub("diffusers.loaders")
    class _FromOriginalModelMixin: pass
    class _PeftAdapterMixin: pass
    loaders.FromOriginalModelMixin = _FromOriginalModelMixin
    loaders.PeftAdapterMixin = _PeftAdapterMixin
    peft = _stub("diffusers.loaders.peft")
    peft.PeftAdapterMixin = _PeftAdapterMixin
    sfm = _stub("diffusers.loaders.single_file_model")
    sfm.FromOriginalModelMixin = _FromOriginalModelMixin
    loaders.peft = peft
    loaders.single_file_model = sfm

    # models namespace
    models_mod = _stub("diffusers.models")

    # models.attention_processor
    attn_proc = _stub("diffusers.models.attention_processor")
    class _DiffusersAttention(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
    class _AttentionProcessor: pass
    attn_proc.Attention = _DiffusersAttention
    attn_proc.AttentionProcessor = _AttentionProcessor
    models_mod.attention_processor = attn_proc

    # models.attention
    attn = _stub("diffusers.models.attention")
    attn.FeedForward = type("FeedForward", (), {})
    models_mod.attention = attn

    # models.cache_utils
    cache = _stub("diffusers.models.cache_utils")
    class _CacheMixin: pass
    cache.CacheMixin = _CacheMixin
    models_mod.cache_utils = cache

    # models.embeddings
    emb = _stub("diffusers.models.embeddings")
    class _Timesteps(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
        def forward(self, x):
            return x
    emb.Timesteps = _Timesteps
    emb.TimestepEmbedding = type("TimestepEmbedding", (), {})
    emb.PixArtAlphaTextProjection = type("PixArtAlphaTextProjection", (), {})
    emb.apply_rotary_emb = lambda *a, **kw: None
    models_mod.embeddings = emb

    # models.modeling_outputs
    mod_out = _stub("diffusers.models.modeling_outputs")
    class _Transformer2DModelOutput:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    mod_out.Transformer2DModelOutput = _Transformer2DModelOutput
    models_mod.modeling_outputs = mod_out

    # models.modeling_utils
    mod_utils = _stub("diffusers.models.modeling_utils")
    class _ModelMixin(nn.Module):
        @classmethod
        def from_pretrained(cls, *a, **kw): ...
        def save_pretrained(self, *a, **kw): ...
    mod_utils.ModelMixin = _ModelMixin
    models_mod.modeling_utils = mod_utils

    # models.normalization
    norm = _stub("diffusers.models.normalization")
    norm.AdaLayerNormContinuous = type("AdaLayerNormContinuous", (), {})
    norm.AdaLayerNormZero = type("AdaLayerNormZero", (), {})
    norm.AdaLayerNormZeroSingle = type("AdaLayerNormZeroSingle", (), {})
    models_mod.normalization = norm

    # models.transformers / transformer_2d
    tf = _stub("diffusers.models.transformers")
    tf2d = _stub("diffusers.models.transformers.transformer_2d")
    tf2d.TransformerBlockMetadata = type("TransformerBlockMetadata", (), {})
    tf2d.TransformerBlockRegistry = None
    tf.transformer_2d = tf2d
    models_mod.transformers = tf

    # utils
    utils = _stub("diffusers.utils")
    utils.USE_PEFT_BACKEND = False
    utils.is_torch_version = lambda *a, **kw: True
    utils.logging = types.SimpleNamespace(
        get_logger=lambda name: types.SimpleNamespace(
            warning=lambda *a, **kw: None,
            info=lambda *a, **kw: None,
            debug=lambda *a, **kw: None,
        )
    )
    utils.scale_lora_layers = lambda model, weight: None
    utils.unscale_lora_layers = lambda model, weight: None
    root.utils = utils

    return saved


def _build_models_namespace() -> dict:
    """models / models.transformer namespace 를 sys.modules 에 주입.
    저장된 원본을 dict 로 반환 (복원용).
    """
    saved: dict = {}
    for ns in ("models", "models.transformer"):
        saved[ns] = sys.modules.get(ns)
        if ns not in sys.modules:
            sys.modules[ns] = types.ModuleType(ns)
    return saved


def _restore_modules(saved: dict) -> None:
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Module-level installation (collection-time safe)
#
# pytest 는 conftest.py 를 test 파일보다 먼저 import 하므로,
# 여기서 module-level 로 stub 을 설치하면 test 파일의 _load_module 호출
# (collection 중 실행)보다 타이밍이 보장된다.
# ---------------------------------------------------------------------------

_diffusers_saved: dict | None = None
_models_saved: dict | None = None

# models namespace 는 diffusers 설치 여부와 무관하게 항상 사전 주입
_models_saved = _build_models_namespace()

if not _diffusers_available():
    _diffusers_saved = _build_diffusers_stub()


# ---------------------------------------------------------------------------
# Session-scoped fixture: teardown only
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _stub_teardown_session():
    """session 종료 시 module-level 에서 주입한 stub 을 복원한다."""
    yield
    if _diffusers_saved is not None:
        _restore_modules(_diffusers_saved)
    if _models_saved is not None:
        _restore_modules(_models_saved)
