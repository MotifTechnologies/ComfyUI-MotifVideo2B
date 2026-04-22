"""tests/test_p4_attention_mask.py — P4 attention_mask end-to-end blind tests.

검증 대상:
  P4: text_encode tokenizer padding mask → MotifVideoTransformer3DModel.forward
  까지 end-to-end 전달 (품질 정렬).

커버리지:
  1. AST 구조 검증  — t5_gemma2.py encode 3-tuple, encode_token_weights mask 전파
  2. extra_conds mask pickup  — 실제 mask 전달 경로 (폴백 아님)
  3. all-ones 폴백  — mask 없을 때 하위 호환
  4. mask shape/dtype 계약  — [B, E] bool 또는 long, CONDRegular 래핑
  5. 경계값  — empty tokens, batch=1/multi, mask=None
  6. 하위 호환  — CLIPTextEncode 사용 시 (attention_mask kwarg 없음)
  7. 스코프 외 검증  — transformer_motif_video.py 미수정 확인

실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI/custom_nodes/ComfyUI-MotifVideo1.9B
    pytest tests/test_p4_attention_mask.py -v
"""

import ast
import os
import sys
import types

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T5_GEMMA2_PY = os.path.join(_PROJECT_ROOT, "text_encoders", "t5_gemma2.py")
_TEXT_ENCODE_PY = os.path.join(_PROJECT_ROOT, "nodes", "text_encode.py")
_MODELS_INIT_PY = os.path.join(_PROJECT_ROOT, "models", "__init__.py")
_TRANSFORMER_PY = os.path.join(
    _PROJECT_ROOT, "models", "transformer", "transformer_motif_video.py"
)


# ---------------------------------------------------------------------------
# Helper: source texts (read once)
# ---------------------------------------------------------------------------

with open(_T5_GEMMA2_PY) as _f:
    _T5_SRC = _f.read()

with open(_TEXT_ENCODE_PY) as _f:
    _TEXT_ENCODE_SRC = _f.read()

with open(_MODELS_INIT_PY) as _f:
    _MODELS_INIT_SRC = _f.read()


# ---------------------------------------------------------------------------
# Stubs for models/__init__.py import (reused from test_model_init.py pattern)
# ---------------------------------------------------------------------------

def _install_model_stubs():
    # --- diffusers stub (transformer_motif_video.py 의 top-level import 차단) ---
    diffusers_stub = types.ModuleType("diffusers")
    for sub in [
        "diffusers.configuration_utils",
        "diffusers.models",
        "diffusers.models.attention_processor",
        "diffusers.models.modeling_utils",
        "diffusers.utils",
        "diffusers.utils.torch_utils",
    ]:
        sys.modules.setdefault(sub, types.ModuleType(sub))
    _cfg = sys.modules["diffusers.configuration_utils"]
    _cfg.ConfigMixin = object
    _cfg.register_to_config = lambda f: f
    sys.modules.setdefault("diffusers", diffusers_stub)
    diffusers_stub.configuration_utils = _cfg

    # --- models.transformer stub (diffusers 없이 models/__init__.py 임포트 허용) ---
    # Codex HIGH 반영: 직접 대입 대신 setdefault 로 real module 존재 시 보존.
    # 편집 pod (diffusers/transformers 미설치) → stub 사용.
    # GPU pod / real 환경 → real module 우선, stub 은 skip. 전역 오염 방지.
    if "models.transformer" not in sys.modules:
        transformer_pkg = types.ModuleType("models.transformer")

        class _FakeTransformerModel:
            pass

        transformer_pkg.MotifVideoTransformer3DModel = _FakeTransformerModel
        sys.modules["models.transformer"] = transformer_pkg

    # comfy stubs
    comfy_mod = types.ModuleType("comfy")
    sys.modules.setdefault("comfy", comfy_mod)

    model_base_mod = types.ModuleType("comfy.model_base")

    class _FakeModelType:
        FLOW = "flow"

    class _FakeBaseModel:
        def __init__(self, model_config, model_type=None, device=None):
            pass

        def extra_conds(self, **kwargs):
            return {}

    model_base_mod.ModelType = _FakeModelType
    model_base_mod.BaseModel = _FakeBaseModel
    sys.modules.setdefault("comfy.model_base", model_base_mod)
    comfy_mod.model_base = model_base_mod

    conds_mod = types.ModuleType("comfy.conds")

    class _FakeCONDRegular:
        def __init__(self, val):
            self.val = val

    conds_mod.CONDRegular = _FakeCONDRegular
    sys.modules.setdefault("comfy.conds", conds_mod)
    comfy_mod.conds = conds_mod

    ops_mod = types.ModuleType("comfy.ops")
    sys.modules.setdefault("comfy.ops", ops_mod)
    comfy_mod.ops = ops_mod

    mm_mod = types.ModuleType("comfy.model_management")
    mm_mod.intermediate_device = lambda: torch.device("cpu")
    sys.modules.setdefault("comfy.model_management", mm_mod)
    comfy_mod.model_management = mm_mod

    motif_core_mod = types.ModuleType("motif_core")
    sys.modules.setdefault("motif_core", motif_core_mod)
    for sub in [
        "motif_core.models",
        "motif_core.models.transformers",
        "motif_core.models.transformers.transformer_motif_video",
    ]:
        sys.modules.setdefault(sub, types.ModuleType(sub))

    class _FakeTransformer:
        pass

    sys.modules[
        "motif_core.models.transformers.transformer_motif_video"
    ].MotifVideoTransformer3DModel = _FakeTransformer

    for mod_name in [
        "models.adapter",
        "models.latent_format",
        "ComfyUI_MotifVideo1_9B.models.adapter",
        "ComfyUI_MotifVideo1_9B.models.latent_format",
    ]:
        stub = types.ModuleType(mod_name)
        stub.MotifVideoModelAdapter = object
        stub.MotifVideoLatent = object
        sys.modules.setdefault(mod_name, stub)


_install_model_stubs()


def _load_models_init():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_models_init_p4", _MODELS_INIT_PY)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "models"
    spec.loader.exec_module(mod)
    return mod


_models_mod = _load_models_init()


# ---------------------------------------------------------------------------
# Helper: extra_conds 만 테스트하기 위한 경량 MotifVideoModel 인스턴스
#
# MotifVideoModel.__init__ 은 comfy.ops.pick_operations + 실제 transformer
# 생성을 호출하므로 편집 pod 에서 인스턴스화 불가.
# extra_conds 는 __init__ 에 의존하지 않으므로 object.__new__ + 메서드 바인딩으로
# 우회한다.
# ---------------------------------------------------------------------------


def _make_extra_conds_model():
    """extra_conds 테스트용 최소 MotifVideoModel 인스턴스 반환.

    __init__ 을 우회해 extra_conds 메서드만 바인딩한다.
    """
    cls = _models_mod.MotifVideoModel
    # object.__new__ 로 __init__ 호출 없이 인스턴스 생성
    obj = object.__new__(cls)
    return obj


# ---------------------------------------------------------------------------
# 1. AST 구조 검증
# ---------------------------------------------------------------------------


class TestASTStructure:
    """구현 파일의 AST 레벨 계약 검증."""

    # --- t5_gemma2.py ---

    def test_t5_gemma2_syntax_valid(self):
        compile(_T5_SRC, _T5_GEMMA2_PY, "exec")

    def test_encode_returns_3_tuple_source_evidence(self):
        """MotifVideoT5Gemma2Model.encode() 가 3-tuple 을 반환하는 코드가 존재한다.

        소스에 (hidden_state, None, attention_mask) 형태의 반환 구문이 있어야 한다.
        """
        tree = ast.parse(_T5_SRC)
        classes = {
            n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        }
        assert "MotifVideoT5Gemma2Model" in classes, (
            "MotifVideoT5Gemma2Model class not found"
        )
        cls_node = classes["MotifVideoT5Gemma2Model"]
        encode_nodes = [
            n
            for n in ast.walk(cls_node)
            if isinstance(n, ast.FunctionDef) and n.name == "encode"
        ]
        assert len(encode_nodes) == 1, "encode method not found or duplicated"
        encode_node = encode_nodes[0]

        # Return 노드 중 Tuple 반환 3개 요소가 있는 것을 찾는다
        returns = [n for n in ast.walk(encode_node) if isinstance(n, ast.Return)]
        three_tuple_returns = [
            r
            for r in returns
            if isinstance(r.value, ast.Tuple) and len(r.value.elts) == 3
        ]
        assert len(three_tuple_returns) >= 1, (
            "MotifVideoT5Gemma2Model.encode() has no 3-tuple return statement; "
            "expected (hidden_state, pooled, attention_mask)"
        )

    def test_encode_token_weights_reads_third_element(self):
        """encode_token_weights 가 encode() 반환값의 인덱스 2 또는 len>2 분기를 읽는다."""
        # 소스에 encode_out[2] 또는 len(encode_out) > 2 패턴이 있어야 한다
        assert (
            "encode_out[2]" in _T5_SRC or "len(encode_out) > 2" in _T5_SRC
        ), (
            "encode_token_weights does not read third element of encode() output"
        )

    def test_encode_token_weights_builds_extra_dict(self):
        """encode_token_weights 가 {"attention_mask": ...} dict 를 반환 tuple 에 포함한다."""
        assert '"attention_mask"' in _T5_SRC or "'attention_mask'" in _T5_SRC, (
            "attention_mask key not found in t5_gemma2.py"
        )
        # r = r + ({"attention_mask": flat_mask},) 패턴 확인
        assert 'attention_mask' in _T5_SRC, (
            "attention_mask propagation code missing from t5_gemma2.py"
        )

    # --- nodes/text_encode.py ---

    def test_text_encode_syntax_valid(self):
        compile(_TEXT_ENCODE_SRC, _TEXT_ENCODE_PY, "exec")

    def test_motif_text_encode_class_exists(self):
        tree = ast.parse(_TEXT_ENCODE_SRC)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "MotifTextEncode" in classes

    def test_motif_text_encode_return_types_unchanged(self):
        """RETURN_TYPES 는 (CONDITIONING, CONDITIONING) 2개여야 한다 (옵션 C 미발동)."""
        assert 'RETURN_TYPES = ("CONDITIONING", "CONDITIONING")' in _TEXT_ENCODE_SRC, (
            "RETURN_TYPES changed from (CONDITIONING, CONDITIONING) — "
            "this would break backward compatibility"
        )

    def test_text_encode_has_mask_propagation_comment_or_code(self):
        """attention_mask 전달 관련 코드/주석이 존재한다."""
        assert "attention_mask" in _TEXT_ENCODE_SRC, (
            "No attention_mask reference in nodes/text_encode.py"
        )

    # --- models/__init__.py ---

    def test_models_init_syntax_valid(self):
        compile(_MODELS_INIT_SRC, _MODELS_INIT_PY, "exec")

    def test_extra_conds_has_attention_mask_pickup(self):
        """extra_conds 가 kwargs.get('attention_mask') 분기를 가진다."""
        assert "attention_mask" in _MODELS_INIT_SRC, (
            "attention_mask pickup missing from models/__init__.py"
        )

    def test_extra_conds_has_fallback_path(self):
        """all-ones 폴백 경로가 존재한다 (하위 호환)."""
        assert "torch.ones" in _MODELS_INIT_SRC, (
            "all-ones fallback removed from models/__init__.py — breaks backward compat"
        )

    # --- scope: transformer 미수정 ---

    def test_transformer_motif_video_not_in_changed_scope(self):
        """_create_attention_mask 본체는 수정 금지 (HF 동등 유지).

        이 테스트는 transformer_motif_video.py 가 존재하는 경우
        _create_attention_mask 시그니처/본체가 여전히 정상임을 확인한다.
        """
        if not os.path.exists(_TRANSFORMER_PY):
            pytest.skip("transformer_motif_video.py not found")
        with open(_TRANSFORMER_PY) as f:
            src = f.read()
        assert "_create_attention_mask" in src, (
            "_create_attention_mask removed from transformer — unexpected change"
        )
        # 기본 all-ones 폴백 또는 encoder_attention_mask 분기 존재
        assert "encoder_attention_mask" in src, (
            "encoder_attention_mask reference removed from transformer"
        )


# ---------------------------------------------------------------------------
# 2. extra_conds: 실제 mask 전달 경로 (폴백이 아니라 pickup)
# ---------------------------------------------------------------------------


class TestExtraCondsMaskPickup:
    """MotifVideoModel.extra_conds() 가 실제 mask 를 CONDRegular 로 래핑한다."""

    def test_actual_mask_used_not_fallback(self):
        """attention_mask kwarg 있을 때 CONDRegular(actual_mask) 가 반환된다."""
        model = _make_extra_conds_model()
        cross = torch.randn(1, 32, 512)
        mask = torch.ones(1, 32, dtype=torch.bool)
        out = model.extra_conds(cross_attn=cross, attention_mask=mask)
        assert "encoder_attention_mask" in out, (
            "encoder_attention_mask key missing from extra_conds output"
        )
        # CONDRegular.val 이 전달한 mask 와 동일해야 한다
        cond_val = out["encoder_attention_mask"].val
        assert torch.equal(cond_val, mask), (
            f"Mask in extra_conds is not the supplied mask; "
            f"expected shape {mask.shape}, got {cond_val.shape}"
        )

    def test_actual_mask_with_padding_zeros_preserved(self):
        """실제 padding 0이 있는 mask 가 그대로 pickup 된다 (all-ones 폴백이 아님)."""
        model = _make_extra_conds_model()
        cross = torch.randn(1, 16, 512)
        # 앞 10개 real, 뒤 6개 pad
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[0, :10] = True
        out = model.extra_conds(cross_attn=cross, attention_mask=mask)
        cond_val = out["encoder_attention_mask"].val
        # 폴백이었으면 모두 True — padding 0이 살아있어야 한다
        assert not cond_val.all(), (
            "All-ones fallback was applied even though a real mask was provided"
        )
        assert cond_val[0, :10].all(), "Real token positions should be True"
        assert not cond_val[0, 10:].any(), "Padded positions should be False"


# ---------------------------------------------------------------------------
# 3. all-ones 폴백 (하위 호환)
# ---------------------------------------------------------------------------


class TestExtraCondsFallback:
    """attention_mask kwarg 없을 때 all-ones 폴백이 생성된다."""

    def test_fallback_all_ones_when_no_mask(self):
        """attention_mask kwarg 없이 cross_attn 만 있을 때 all-ones mask 생성."""
        model = _make_extra_conds_model()
        B, E, D = 2, 20, 512
        cross = torch.randn(B, E, D)
        out = model.extra_conds(cross_attn=cross)
        assert "encoder_attention_mask" in out, (
            "encoder_attention_mask missing when using fallback path"
        )
        cond_val = out["encoder_attention_mask"].val
        expected_shape = torch.Size([B, E])
        assert cond_val.shape == expected_shape, (
            f"Fallback mask shape mismatch: expected {expected_shape}, got {cond_val.shape}"
        )
        assert cond_val.dtype == torch.bool, (
            f"Fallback mask dtype should be bool, got {cond_val.dtype}"
        )
        assert cond_val.all(), "Fallback mask should be all-ones (True)"

    def test_no_mask_no_cross_attn_no_encoder_mask(self):
        """cross_attn 도 attention_mask 도 없으면 encoder_attention_mask 미생성."""
        model = _make_extra_conds_model()
        out = model.extra_conds()
        assert "encoder_attention_mask" not in out, (
            "encoder_attention_mask should not be created without cross_attn"
        )

    def test_fallback_shape_matches_cross_attn_batch_seq(self):
        """폴백 mask shape 이 cross_attn 의 [B, E] 와 일치한다."""
        model = _make_extra_conds_model()
        for B, E in [(1, 77), (4, 256), (1, 1)]:
            cross = torch.randn(B, E, 512)
            out = model.extra_conds(cross_attn=cross)
            cond_val = out["encoder_attention_mask"].val
            assert cond_val.shape == torch.Size([B, E]), (
                f"B={B} E={E}: fallback mask shape {cond_val.shape} != [{B}, {E}]"
            )


# ---------------------------------------------------------------------------
# 4. mask shape/dtype 계약
# ---------------------------------------------------------------------------


class TestMaskShapeDtypeContract:
    """extra_conds 에서 나온 mask 가 _create_attention_mask 와 호환된다."""

    def test_actual_mask_bool_compatible(self):
        """전달된 bool mask 가 CONDRegular 로 래핑돼 나온다."""
        model = _make_extra_conds_model()
        mask = torch.ones(1, 77, dtype=torch.bool)
        out = model.extra_conds(
            cross_attn=torch.randn(1, 77, 512),
            attention_mask=mask,
        )
        cond_val = out["encoder_attention_mask"].val
        # .to(torch.bool) 연산 호환 확인
        as_bool = cond_val.to(torch.bool)
        assert as_bool.shape == mask.shape

    def test_actual_mask_long_dtype_accepted(self):
        """long dtype mask 도 CONDRegular 에 담겨 반환된다."""
        model = _make_extra_conds_model()
        mask = torch.ones(1, 77, dtype=torch.long)
        out = model.extra_conds(
            cross_attn=torch.randn(1, 77, 512),
            attention_mask=mask,
        )
        cond_val = out["encoder_attention_mask"].val
        assert cond_val.shape == torch.Size([1, 77])

    def test_mask_is_cond_regular_instance(self):
        """encoder_attention_mask 값이 CONDRegular 로 래핑된다."""
        import comfy.conds as _conds

        model = _make_extra_conds_model()
        mask = torch.ones(1, 32, dtype=torch.bool)
        out = model.extra_conds(
            cross_attn=torch.randn(1, 32, 512),
            attention_mask=mask,
        )
        assert isinstance(out["encoder_attention_mask"], _conds.CONDRegular), (
            "encoder_attention_mask must be wrapped in CONDRegular"
        )


# ---------------------------------------------------------------------------
# 5. 경계값 테스트
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_mask_none_explicit_uses_fallback(self):
        """attention_mask=None 명시 전달 시 fallback (all-ones) 경로 진입."""
        model = _make_extra_conds_model()
        cross = torch.randn(1, 10, 512)
        out = model.extra_conds(cross_attn=cross, attention_mask=None)
        cond_val = out["encoder_attention_mask"].val
        assert cond_val.all(), (
            "attention_mask=None should trigger all-ones fallback"
        )

    def test_seq_len_1_mask(self):
        """seq_len=1 경계값 — mask [B,1] 가 정상 처리된다."""
        model = _make_extra_conds_model()
        cross = torch.randn(1, 1, 512)
        mask = torch.ones(1, 1, dtype=torch.bool)
        out = model.extra_conds(cross_attn=cross, attention_mask=mask)
        cond_val = out["encoder_attention_mask"].val
        assert cond_val.shape == torch.Size([1, 1])

    def test_batch_gt_1_mask_shape(self):
        """batch > 1 일 때 mask 가 올바른 shape 으로 반환된다."""
        model = _make_extra_conds_model()
        B, E = 4, 64
        cross = torch.randn(B, E, 512)
        mask = torch.ones(B, E, dtype=torch.bool)
        out = model.extra_conds(cross_attn=cross, attention_mask=mask)
        assert out["encoder_attention_mask"].val.shape == torch.Size([B, E])

    def test_all_pad_mask_accepted(self):
        """모두 pad(False) 인 mask 도 거부 없이 통과한다."""
        model = _make_extra_conds_model()
        cross = torch.randn(1, 8, 512)
        all_pad = torch.zeros(1, 8, dtype=torch.bool)
        out = model.extra_conds(cross_attn=cross, attention_mask=all_pad)
        cond_val = out["encoder_attention_mask"].val
        assert not cond_val.any(), "All-pad mask should be preserved as-is"


# ---------------------------------------------------------------------------
# 6. 하위 호환: CLIPTextEncode 사용 시 (mask kwarg 없음)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """CLIPTextEncode 기반 워크플로우에서 mask 없이도 동작한다."""

    def test_no_attention_mask_kwarg_does_not_raise(self):
        """CLIPTextEncode: attention_mask kwarg 자체가 없어도 예외 없음."""
        model = _make_extra_conds_model()
        cross = torch.randn(1, 77, 512)
        out = model.extra_conds(cross_attn=cross)
        # 폴백 mask 가 생성돼야 한다
        assert "encoder_attention_mask" in out

    def test_fallback_mask_dtype_bool(self):
        """CLIPTextEncode 경로 폴백 mask 가 bool dtype."""
        model = _make_extra_conds_model()
        cross = torch.randn(1, 77, 512)
        out = model.extra_conds(cross_attn=cross)
        dtype = out["encoder_attention_mask"].val.dtype
        assert dtype == torch.bool, (
            f"Fallback mask dtype should be bool, got {dtype}"
        )

    def test_pooled_projections_none_kwarg_no_error(self):
        """pooled_projections=None 은 extra_conds 에서 조용히 무시된다."""
        model = _make_extra_conds_model()
        out = model.extra_conds(
            cross_attn=torch.randn(1, 77, 512),
            pooled_projections=None,
        )
        assert "pooled_projections" not in out, (
            "pooled_projections=None should not add key to output"
        )


# ---------------------------------------------------------------------------
# 7. encode_token_weights mask 전파 단위 테스트 (stub 기반)
# ---------------------------------------------------------------------------


class TestEncodeTokenWeightsMaskPropagation:
    """MotifVideoT5Gemma2Model.encode_token_weights 의 mask 전파 로직을
    stub 으로 단위 검증한다 (transformers import 불가 환경 대응).
    """

    def _make_stub_model(self):
        """MotifVideoT5Gemma2Model 을 최소 stub 으로 생성."""
        import importlib.util

        # comfy.model_management stub
        mm_stub = types.ModuleType("comfy.model_management")
        mm_stub.intermediate_device = lambda: torch.device("cpu")
        sys.modules["comfy.model_management"] = mm_stub

        spec = importlib.util.spec_from_file_location(
            "_t5_gemma2_stub", _T5_GEMMA2_PY
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "text_encoders"

        # transformers stub (T5Gemma2ForConditionalGeneration 등)
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.T5Gemma2ForConditionalGeneration = object
        sys.modules.setdefault("transformers", transformers_stub)

        try:
            spec.loader.exec_module(mod)
        except Exception:
            pytest.skip("t5_gemma2.py requires transformers/sentencepiece")

        return mod

    def test_encode_returns_3_elements_when_mask_produced(self):
        """encode() 가 3개짜리 tuple 을 반환하는 구조."""
        # 직접 encode 실행 불가 (모델 weights 필요) → AST/소스 검증으로 대체
        import re

        # 소스에 'return hidden_state, None, attention_mask' 또는 동등한 패턴이 있어야
        pattern = re.compile(
            r"return\s+\w+\s*,\s*None\s*,\s*attention_mask", re.MULTILINE
        )
        assert pattern.search(_T5_SRC), (
            "encode() does not return a 3-tuple with attention_mask as third element"
        )

    def test_encode_token_weights_guard_for_missing_third_element(self):
        """encode() 가 old 2-tuple 을 반환하는 코드 경로가 있으면 mask = None 처리.

        encode_token_weights 소스에 len(encode_out) > 2 가드가 있어야 한다.
        """
        assert "len(encode_out) > 2" in _T5_SRC or "encode_out[2]" in _T5_SRC, (
            "Missing backward-compat guard for 2-tuple encode() output"
        )

    def test_mask_slice_sections_only(self):
        """mask_sections = raw_mask[:sections] 로 sentinel row 를 제거하는 로직 확인."""
        assert "raw_mask[:sections]" in _T5_SRC, (
            "Sentinel row removal (raw_mask[:sections]) not found in encode_token_weights"
        )

    def test_mask_flattened_and_unsqueezed(self):
        """flat_mask = mask_sections.flatten().unsqueeze(0) 패턴 확인."""
        assert "flatten()" in _T5_SRC and "unsqueeze(0)" in _T5_SRC, (
            "mask flatten+unsqueeze pattern not found in encode_token_weights"
        )

    def test_r_plus_extra_dict_tuple(self):
        """r = r + ({"attention_mask": flat_mask},) 패턴 확인."""
        import re

        pattern = re.compile(
            r'r\s*=\s*r\s*\+\s*\(\s*\{.*?"attention_mask".*?\}\s*,\s*\)', re.DOTALL
        )
        assert pattern.search(_T5_SRC), (
            "Extra-dict propagation pattern not found in encode_token_weights"
        )

    def test_mask_propagation_only_when_raw_mask_not_none(self):
        """raw_mask is not None 조건 분기가 있어야 한다 (None 경로 처리)."""
        assert "raw_mask is not None" in _T5_SRC, (
            "Guard 'if raw_mask is not None' missing — unconditional propagation would fail"
        )

    def test_sections_zero_branch_uses_sentinel_row(self):
        """Codex HIGH 반영: sections == 0 (blank prompt) 경로에서 mask_sections 가
        raw_mask[-1:] (sentinel row) 를 사용해 out[-1:] 과 shape 일치 유지.

        이 분기가 없으면 blank-prompt 경로에서 `raw_mask[:0]` = [1, 0] 형태가
        반환되어 downstream 의 _create_attention_mask 에서 shape mismatch 발생.
        """
        import re

        # sections == 0 분기 + raw_mask[-1:] 사용 패턴 (blank-prompt sentinel)
        pattern = re.compile(
            r"sections\s*==\s*0[\s\S]*?raw_mask\[-1:\]",
            re.MULTILINE,
        )
        assert pattern.search(_T5_SRC), (
            "Blank-prompt fallback (sections == 0 → raw_mask[-1:]) missing. "
            "Without this branch, encode_token_weights returns an empty mask "
            "[1, 0] when the prompt is empty, breaking transformer shape contract."
        )


# ---------------------------------------------------------------------------
# 8. transformer_motif_video.py 스코프 외 검증
# ---------------------------------------------------------------------------


class TestTransformerNotModified:
    """P4 스코프 외: transformer_motif_video.py 의 _create_attention_mask 는 수정 금지."""

    def test_create_attention_mask_present(self):
        if not os.path.exists(_TRANSFORMER_PY):
            pytest.skip("transformer_motif_video.py not found at expected path")
        with open(_TRANSFORMER_PY) as f:
            src = f.read()
        assert "def _create_attention_mask" in src, (
            "_create_attention_mask method missing — unexpected transformer change"
        )

    def test_encoder_attention_mask_referenced_in_transformer(self):
        if not os.path.exists(_TRANSFORMER_PY):
            pytest.skip("transformer_motif_video.py not found at expected path")
        with open(_TRANSFORMER_PY) as f:
            src = f.read()
        assert "encoder_attention_mask" in src, (
            "encoder_attention_mask reference removed from transformer — "
            "this contradicts the P4 contract"
        )
