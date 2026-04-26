"""
Unit tests for MotifVideoT5Gemma2Model wrapper (checklist item 8).

Scope:
- load_sd() two-format compatibility: HF re-serialized + motif-internal encoder. prefix.
- encode() output shape, dtype, pooled-None contract.
- encode_token_weights() uniform/non-uniform weight branches + empty-section edge case.
- set_clip_options / reset_clip_options / get_input_embeddings smoke.

Blind-test principle: item-8 diff (text_encoders/t5_gemma2.py implementation) was NOT
read before writing these tests. Tests are derived from the spec/checklist only.

Key discoveries (from probing load_sd() behaviour to build correct dummy state_dicts):
- load_state_dict(assign=False): shape mismatch raises RuntimeError, so dummy tensors
  must match the production shape. We use only small-shape keys (norm.weight [2560],
  embed_tokens.eoi_embedding [2560]) to stay within memory limits.
- vision_tower keys map to unexpected_keys (native encoder has no vision submodule).
- IncompatibleKeys type is torch.nn.modules.module._IncompatibleKeys.
- encode_token_weights([[]]): seq_len=0 causes reshape error in native encoder (known
  limitation, xfail-documented).
- encode_token_weights with mixed-length batch: IndexError in weight-indexing loop
  (known limitation, xfail-documented).

Environment: must be run with ComfyUI venv python:
  /lustrefs/team-multimodal/minsu/ComfyUI/.venv/bin/python \\
      -m pytest tests/text_encoders/test_t5_gemma2_wrapper.py -v

Layout: <comfy_root>/custom_nodes/<repo>/tests/text_encoders/<this_file>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ComfyUI environment bootstrap (must precede any comfy import)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_COMFYUI_ROOT_GUESS = _THIS_FILE.parents[4]
_COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", _COMFYUI_ROOT_GUESS)).resolve()

sys.argv = ["x", "--cpu"]
for _p in (str(_COMFYUI_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import pytest  # noqa: E402
import torch   # noqa: E402

from text_encoders.t5_gemma2 import MotifVideoT5Gemma2Model  # noqa: E402
from text_encoders.t5_gemma2_config import T5_GEMMA2_CONFIG   # noqa: E402


# ---------------------------------------------------------------------------
# Memory gate
#
# The wrapper hardcodes the production T5_GEMMA2_CONFIG (34 layers, hidden
# size 2560, vocab 262144), so this file cannot use a reduced fixture like
# test_t5_gemma2_native.py. A single bfloat16 instance is roughly 5 GB and
# parameter init via fill_() touches every weight before the first
# assertion. Skip the file on hosts without enough RAM so CI/dev machines
# don't OOM. Override with `RUN_HEAVY_WRAPPER_TESTS=1` to run anyway.
# ---------------------------------------------------------------------------
_HEAVY_REQUIRED_GB = 8
_FORCE_RUN = os.environ.get("RUN_HEAVY_WRAPPER_TESTS") == "1"

try:
    import psutil  # noqa: E402
    _AVAILABLE_GB = psutil.virtual_memory().available / (1024 ** 3)
    _ENOUGH_MEM = _AVAILABLE_GB >= _HEAVY_REQUIRED_GB
    _MEM_REASON = f"available={_AVAILABLE_GB:.1f}GB < required={_HEAVY_REQUIRED_GB}GB"
except ImportError:
    # psutil missing: fail open (run) so the file isn't silently skipped on hosts
    # without psutil but with plenty of RAM.
    _ENOUGH_MEM = True
    _MEM_REASON = "psutil unavailable, assuming enough memory"

pytestmark = pytest.mark.skipif(
    not (_ENOUGH_MEM or _FORCE_RUN),
    reason=(
        "Heavy wrapper test (production-cfg, ~5GB instance). "
        f"{_MEM_REASON}. Set RUN_HEAVY_WRAPPER_TESTS=1 to override."
    ),
)


# ---------------------------------------------------------------------------
# Production config constants (from T5_GEMMA2_CONFIG — no implementation read)
# ---------------------------------------------------------------------------
_HIDDEN_SIZE = T5_GEMMA2_CONFIG["text_config"]["hidden_size"]               # 2560
_NUM_HIDDEN_LAYERS = T5_GEMMA2_CONFIG["text_config"]["num_hidden_layers"]   # 34
_VOCAB_SIZE = T5_GEMMA2_CONFIG["text_config"]["vocab_size"]                 # 262144


# ---------------------------------------------------------------------------
# Module-scope fixture: single production-size wrapper instance (bfloat16/cpu)
#
# Production cfg (34 layers, hidden=2560) is instantiated once and shared
# across all tests. Weight init cost is paid once; each test reuses the same
# module-scope object (read-only for non-load_sd tests).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m() -> MotifVideoT5Gemma2Model:
    """MotifVideoT5Gemma2Model with production config on CPU, bfloat16."""
    model = MotifVideoT5Gemma2Model(device="cpu", dtype=torch.bfloat16, model_options={})
    # Fill parameters with small constant so forward passes don't produce NaN
    # (disable_weight_init leaves weights uninitialised by default).
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.fill_(0.01)
    return model


# ---------------------------------------------------------------------------
# Helpers: build dummy state_dicts with CORRECT production shapes
#
# load_state_dict(assign=False) raises RuntimeError on shape mismatch.
# We therefore use only the "small" keys whose shapes fit in memory:
#   - embed_tokens.eoi_embedding:  [2560]  (5 KB, bfloat16)
#   - norm.weight:                 [2560]  (5 KB, bfloat16)
# These are enough to verify key-mapping correctness (HF vs motif-prefix).
# ---------------------------------------------------------------------------

_H = _HIDDEN_SIZE  # 2560


def _small_sd_hf() -> dict[str, torch.Tensor]:
    """HF re-serialized format (no prefix): small shape-correct keys."""
    return {
        "embed_tokens.eoi_embedding": torch.zeros(_H, dtype=torch.bfloat16),
        "norm.weight": torch.zeros(_H, dtype=torch.bfloat16),
    }


def _small_sd_hf_with_vision() -> dict[str, torch.Tensor]:
    """HF re-serialized format including a vision_tower key."""
    sd = _small_sd_hf()
    sd["vision_tower.encoder.layers.0.weight"] = torch.zeros(4, dtype=torch.bfloat16)
    return sd


def _small_sd_motif_prefix() -> dict[str, torch.Tensor]:
    """Legacy motif-internal format: 'encoder.' prefix on every key."""
    return {f"encoder.{k}": v for k, v in _small_sd_hf().items()}


def _small_sd_motif_prefix_with_vision() -> dict[str, torch.Tensor]:
    """Legacy motif-internal format including an encoder.vision_tower key."""
    return {f"encoder.{k}": v for k, v in _small_sd_hf_with_vision().items()}


# ===========================================================================
# 1. load_sd() — HF re-serialized format
# ===========================================================================

class TestLoadSdHFFormat:
    """load_sd() with HF re-serialized (no-prefix) state_dict."""

    def test_load_sd_hf_format_supplied_keys_loaded(self, m):
        """Supplied text keys must NOT appear in missing_keys after HF-format load.

        The wrapper must prepend 'text_model.' to bare text keys so that
        norm.weight → text_model.norm.weight and
        embed_tokens.eoi_embedding → text_model.embed_tokens.eoi_embedding
        are successfully loaded (not missing).
        """
        sd = _small_sd_hf()
        result = m.load_sd(sd)
        missing = list(result.missing_keys)
        # The two keys we supplied must be consumed (not missing)
        assert "text_model.embed_tokens.eoi_embedding" not in missing, (
            "text_model.embed_tokens.eoi_embedding must not be in missing_keys "
            "(HF key 'embed_tokens.eoi_embedding' must map to text_model.* prefix)"
        )
        assert "text_model.norm.weight" not in missing, (
            "text_model.norm.weight must not be in missing_keys "
            "(HF key 'norm.weight' must map to text_model.* prefix)"
        )

    def test_load_sd_hf_format_no_unexpected_text_model_keys(self, m):
        """HF format: supplied text keys must not appear in unexpected_keys."""
        sd = _small_sd_hf()
        result = m.load_sd(sd)
        unexpected = list(result.unexpected_keys)
        text_unexpected = [k for k in unexpected if k.startswith("text_model.")]
        assert text_unexpected == [], (
            f"No text_model.* keys should be in unexpected_keys: {text_unexpected}"
        )

    def test_load_sd_hf_format_does_not_raise(self, m):
        """load_sd() with valid HF format must not raise any exception."""
        sd = _small_sd_hf()
        try:
            m.load_sd(sd)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"load_sd() raised {type(exc).__name__}: {exc}")


# ===========================================================================
# 2. load_sd() — motif-internal encoder. prefix format
# ===========================================================================

class TestLoadSdMotifEncoderPrefixFormat:
    """load_sd() with legacy motif-internal 'encoder.' prefix state_dict."""

    def test_load_sd_motif_encoder_prefix_supplied_keys_loaded(self, m):
        """encoder. prefix format: supplied keys must NOT appear in missing_keys.

        The wrapper must strip 'encoder.' then apply 'text_model.' mapping,
        producing the same final keys as the HF format path.
        """
        sd = _small_sd_motif_prefix()
        result = m.load_sd(sd)
        missing = list(result.missing_keys)
        assert "text_model.embed_tokens.eoi_embedding" not in missing, (
            "text_model.embed_tokens.eoi_embedding must not be in missing_keys "
            "(encoder.embed_tokens.eoi_embedding must strip 'encoder.' then prepend text_model.)"
        )
        assert "text_model.norm.weight" not in missing, (
            "text_model.norm.weight must not be in missing_keys "
            "(encoder.norm.weight must strip 'encoder.' then prepend text_model.)"
        )

    def test_load_sd_motif_encoder_prefix_does_not_raise(self, m):
        """load_sd() with encoder. prefix format must not raise any exception."""
        sd = _small_sd_motif_prefix()
        try:
            m.load_sd(sd)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"load_sd() raised {type(exc).__name__}: {exc}")

    def test_load_sd_both_formats_produce_identical_text_model_coverage(self, m):
        """HF and motif-encoder-prefix formats must map to identical final key consumption.

        If both formats cause the same text_model.* keys to be absent from
        missing_keys, the mapping is symmetric.
        """
        result_hf = m.load_sd(_small_sd_hf())
        result_motif = m.load_sd(_small_sd_motif_prefix())

        hf_missing_text = sorted(k for k in result_hf.missing_keys if k.startswith("text_model."))
        motif_missing_text = sorted(k for k in result_motif.missing_keys if k.startswith("text_model."))
        assert hf_missing_text == motif_missing_text, (
            f"HF and motif formats must produce identical text_model.* missing_keys coverage.\n"
            f"HF missing={hf_missing_text[:5]}...\n"
            f"Motif missing={motif_missing_text[:5]}..."
        )


# ===========================================================================
# 3. load_sd() — return type contract
# ===========================================================================

class TestLoadSdReturnType:
    """load_sd() must return an object with missing_keys and unexpected_keys."""

    def test_load_sd_returns_named_tuple_or_compat(self, m):
        """Result must expose .missing_keys and .unexpected_keys attributes."""
        result = m.load_sd(_small_sd_hf())
        assert hasattr(result, "missing_keys"), (
            "load_sd() result must have 'missing_keys' attribute"
        )
        assert hasattr(result, "unexpected_keys"), (
            "load_sd() result must have 'unexpected_keys' attribute"
        )

    def test_load_sd_missing_keys_is_iterable(self, m):
        """missing_keys must be iterable (list or sequence)."""
        result = m.load_sd(_small_sd_hf())
        try:
            _ = list(result.missing_keys)
        except TypeError:
            pytest.fail("result.missing_keys is not iterable")

    def test_load_sd_unexpected_keys_is_iterable(self, m):
        """unexpected_keys must be iterable (list or sequence)."""
        result = m.load_sd(_small_sd_hf())
        try:
            _ = list(result.unexpected_keys)
        except TypeError:
            pytest.fail("result.unexpected_keys is not iterable")


# ===========================================================================
# 4. load_sd() — vision_tower keys go to unexpected_keys
# ===========================================================================

class TestLoadSdVisionKeys:
    """Vision tower keys must be placed in unexpected_keys (native encoder omits vision)."""

    def test_load_sd_vision_keys_in_unexpected_hf_format(self, m):
        """HF format: vision_tower.* key must appear in unexpected_keys.

        The native T5Gemma2Encoder has no vision_tower submodule, so
        vision_tower keys (mapped directly without prefix) must end up
        as unexpected (not missing, not loaded).
        """
        sd = _small_sd_hf_with_vision()
        result = m.load_sd(sd)
        unexpected = list(result.unexpected_keys)
        missing = list(result.missing_keys)
        # Must not be in missing (wrapper doesn't add it as a required key)
        vision_missing = [k for k in missing if "vision_tower" in k]
        assert vision_missing == [], (
            f"vision_tower keys must not appear in missing_keys: {vision_missing}"
        )
        # Must be in unexpected (no submodule to receive it)
        vision_unexpected = [k for k in unexpected if "vision_tower" in k]
        assert len(vision_unexpected) > 0, (
            "vision_tower key must appear in unexpected_keys "
            "(native encoder has no vision_tower submodule)"
        )

    def test_load_sd_vision_keys_in_unexpected_motif_prefix_format(self, m):
        """Motif-encoder-prefix format: encoder.vision_tower.* must go to unexpected_keys."""
        sd = _small_sd_motif_prefix_with_vision()
        result = m.load_sd(sd)
        unexpected = list(result.unexpected_keys)
        missing = list(result.missing_keys)
        vision_missing = [k for k in missing if "vision_tower" in k]
        assert vision_missing == [], (
            f"vision_tower keys must not appear in missing_keys: {vision_missing}"
        )
        vision_unexpected = [k for k in unexpected if "vision_tower" in k]
        assert len(vision_unexpected) > 0, (
            "vision_tower key from motif-prefix format must appear in unexpected_keys"
        )


# ===========================================================================
# 5. encode() — output shape, dtype, and pooled-None contract
# ===========================================================================

class TestEncode:
    """encode() output contract."""

    def test_encode_basic_shape(self, m):
        """encode([[2, 100, 200, 1]]) → out shape (1, 4, HIDDEN_SIZE).

        All 4 tokens are valid (no padding), so trim produces seq_len=4.
        """
        tokens = [[2, 100, 200, 1]]
        out, pooled = m.encode(tokens)
        assert out.shape[0] == 1, f"batch dim must be 1, got {out.shape[0]}"
        assert out.shape[1] == 4, (
            f"seq_len must be 4 (all tokens valid), got {out.shape[1]}"
        )
        assert out.shape[2] == _HIDDEN_SIZE, (
            f"hidden_size must be {_HIDDEN_SIZE}, got {out.shape[2]}"
        )

    def test_encode_returns_pooled_none(self, m):
        """encode() second return value must be None."""
        tokens = [[2, 100, 1]]
        _, pooled = m.encode(tokens)
        assert pooled is None, f"pooled must be None, got {pooled!r}"

    def test_encode_dtype_float32(self, m):
        """encode() output tensor must be float32 (wrapper calls .float())."""
        tokens = [[2, 100, 1]]
        out, _ = m.encode(tokens)
        assert out.dtype == torch.float32, (
            f"encode() output must be float32, got {out.dtype}"
        )

    def test_encode_output_is_3d_tensor(self, m):
        """encode() first return value must be a 3-D tensor."""
        tokens = [[2, 50, 1]]
        out, _ = m.encode(tokens)
        assert isinstance(out, torch.Tensor), "encode() must return a tensor as first element"
        assert out.ndim == 3, f"output must be 3-D (B, S, H), got ndim={out.ndim}"

    def test_encode_with_padding_batch(self, m):
        """encode() with a batch of two sequences of different lengths.

        Per spec: when sequences have different lengths, wrapper pads the shorter
        one. max_len is used for output (no per-sequence trim for non-uniform).
        We verify shape and no exception.
        """
        tokens = [[2, 100, 200, 1], [2, 50, 1]]
        out, pooled = m.encode(tokens)
        assert out.shape[0] == 2, f"batch dim must be 2, got {out.shape[0]}"
        assert out.shape[2] == _HIDDEN_SIZE, (
            f"hidden_size must be {_HIDDEN_SIZE}, got {out.shape[2]}"
        )
        assert pooled is None

    def test_encode_single_token_sequence(self, m):
        """encode() with a minimal single-token sequence must not raise."""
        tokens = [[1]]  # just EOS
        try:
            out, _ = m.encode(tokens)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"encode() raised {type(exc).__name__} on single token: {exc}")
        assert out.ndim == 3


# ===========================================================================
# 6. encode_token_weights() — weight branch coverage
# ===========================================================================

class TestEncodeTokenWeights:
    """encode_token_weights() branch coverage and output contract."""

    def test_encode_token_weights_uniform_weights(self, m):
        """Uniform weight=1.0 path: must return (out, None) with correct shape."""
        token_weight_pairs = [[(2, 1.0), (100, 1.0), (1, 1.0)]]
        out, pooled = m.encode_token_weights(token_weight_pairs)
        assert isinstance(out, torch.Tensor), "first return must be a tensor"
        assert out.ndim == 3, f"output must be 3-D, got ndim={out.ndim}"
        assert out.shape[0] == 1, f"batch dim must be 1, got {out.shape[0]}"
        assert out.shape[2] == _HIDDEN_SIZE
        assert pooled is None, f"pooled must be None, got {pooled!r}"

    def test_encode_token_weights_non_uniform_weights(self, m):
        """Non-uniform weights (!=1.0) path: must enter weight-apply branch without error."""
        token_weight_pairs = [[(2, 1.0), (100, 0.5), (200, 2.0), (1, 1.0)]]
        out, pooled = m.encode_token_weights(token_weight_pairs)
        assert isinstance(out, torch.Tensor), "first return must be a tensor"
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == _HIDDEN_SIZE
        assert pooled is None

    def test_encode_token_weights_non_uniform_returns_float32(self, m):
        """encode_token_weights() output must be float32 also for non-uniform weights."""
        token_weight_pairs = [[(2, 0.8), (100, 1.5), (1, 1.0)]]
        out, _ = m.encode_token_weights(token_weight_pairs)
        assert out.dtype == torch.float32, (
            f"encode_token_weights output must be float32, got {out.dtype}"
        )

    def test_encode_token_weights_uniform_returns_float32(self, m):
        """encode_token_weights() output must be float32 for uniform weights."""
        token_weight_pairs = [[(2, 1.0), (1, 1.0)]]
        out, _ = m.encode_token_weights(token_weight_pairs)
        assert out.dtype == torch.float32, (
            f"encode_token_weights output must be float32, got {out.dtype}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: encode_token_weights([[]]): seq_len=0 causes native encoder to attempt "
            "reshape of 0-element tensor into [-1, ...] shape — ambiguous dimension. "
            "The wrapper does not guard against empty token sections. "
            "Spec requires 'sections==0 분기' to return without raising."
        ),
    )
    def test_encode_token_weights_empty_section(self, m):
        """Empty token-weight list (sections==0) must return without raising.

        Current implementation raises RuntimeError on seq_len=0 forward pass.
        Marked xfail(strict=True): passes when bug is present (expected failure),
        fails test run if implementation is fixed without removing xfail.
        """
        token_weight_pairs = [[]]
        result = m.encode_token_weights(token_weight_pairs)
        assert result is not None

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: encode_token_weights with mixed-length batch raises IndexError "
            "in weight-indexing loop (token_weight_pairs[k][j][1] out of range). "
            "Wrapper assumes all sequences in the batch have equal length after "
            "encode() pads them, but the weight-apply loop uses the original "
            "unpadded lengths. Multi-sequence batches not supported."
        ),
    )
    def test_encode_token_weights_batch_two_sequences(self, m):
        """Batch of two token_weight_pair sequences — currently raises IndexError (known bug)."""
        token_weight_pairs = [
            [(2, 1.0), (100, 0.5), (1, 1.0)],
            [(2, 1.0), (200, 1.5), (300, 1.0), (1, 1.0)],
        ]
        out, pooled = m.encode_token_weights(token_weight_pairs)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == _HIDDEN_SIZE
        assert pooled is None

    def test_encode_token_weights_zero_weight(self, m):
        """Zero weight is a valid edge value; must not raise."""
        token_weight_pairs = [[(2, 1.0), (100, 0.0), (1, 1.0)]]
        try:
            out, _ = m.encode_token_weights(token_weight_pairs)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"encode_token_weights with weight=0.0 raised {type(exc).__name__}: {exc}"
            )
        assert out.ndim == 3


# ===========================================================================
# 7. set_clip_options / reset_clip_options
# ===========================================================================

class TestClipOptions:
    """set_clip_options and reset_clip_options contract."""

    def test_set_clip_options_execution_device(self, m):
        """set_clip_options({'execution_device': torch.device('cpu')}) must update attribute."""
        m.set_clip_options({"execution_device": torch.device("cpu")})
        assert m.execution_device == torch.device("cpu"), (
            f"execution_device must be torch.device('cpu'), got {m.execution_device!r}"
        )

    def test_reset_clip_options_clears_execution_device(self, m):
        """reset_clip_options() must set execution_device back to None."""
        m.set_clip_options({"execution_device": torch.device("cpu")})
        m.reset_clip_options()
        assert m.execution_device is None, (
            f"execution_device must be None after reset, got {m.execution_device!r}"
        )

    def test_set_clip_options_empty_dict_no_error(self, m):
        """set_clip_options({}) must not raise."""
        try:
            m.set_clip_options({})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"set_clip_options({{}}) raised {type(exc).__name__}: {exc}")


# ===========================================================================
# 8. get_input_embeddings
# ===========================================================================

class TestGetInputEmbeddings:
    """get_input_embeddings() must return the embed_tokens from text_model."""

    def test_get_input_embeddings_returns_text_model_embed_tokens(self, m):
        """m.get_input_embeddings() must be m.encoder.text_model.embed_tokens."""
        emb = m.get_input_embeddings()
        expected = m.encoder.text_model.embed_tokens
        assert emb is expected, (
            f"get_input_embeddings() must return encoder.text_model.embed_tokens, "
            f"got {type(emb).__name__}"
        )

    def test_get_input_embeddings_is_nn_module(self, m):
        """embed_tokens must be an nn.Module."""
        import torch.nn as nn
        emb = m.get_input_embeddings()
        assert isinstance(emb, nn.Module), (
            f"embed_tokens must be nn.Module, got {type(emb).__name__}"
        )
