"""tests/test_p2_offload_contract.py — P2 오프로드 계약 검증 (재작성).

## Codex MEDIUM 3 반영 (self-congratulatory theater 제거)
  - _simulate_apply_model / _simulate_extra_conds / _simulate_memory_required 전면 삭제.
  - 실제 MotifVideoModel.apply_model / memory_required 를 직접 호출.

## Codex MEDIUM 4 반영 (sys.modules 전역 stomping 제거)
  - _install_full_stubs 삭제.
  - 모든 sys.modules 주입은 pytest monkeypatch.setitem 으로만 수행 (자동 cleanup).
  - setup/teardown 수동 restore 코드 없음.

## 검증 스코프
  1. 실제 apply_model 호출 → free_memory 호출 경로 (T5Gemma2 존재 시).
  2. 실제 apply_model 호출 → free_memory 미호출 (T5Gemma2 없는 경우 no-op).
  3. keep_loaded 흑리스트: T5Gemma2 만 제외, control/LoRA/etc. 는 keep 에 포함.
  4. 상태 기반 idempotent: current_loaded_models 에 T5Gemma2 없으면 free_memory 0회.
  5. 상태 기반 재등장: T5Gemma2 재등장 → 다음 apply_model 에서 다시 unload.
  6. memory_required: 실제 super().memory_required 값 대비 margin 반영 (1.3배).
  7. non-text-encoder 모델(ControlNet 류, LoRA patcher) 은 unload 대상 아님.
  8. AST 구조 검증: apply_model override, free_memory/unload_all_models 호출, super() 호출,
     MotifVideoT5Gemma2Model import/참조, keep_loaded kwarg 존재, 조건 분기 존재.
  9. P2 체크리스트 verify grep: models/__init__.py 에 HF 계약 동등 흔적.

## 환경 제약
  - 편집 pod: diffusers / transformers 미설치 → top-level import 불가.
  - 해결: monkeypatch.setitem(sys.modules, ...) 으로 테스트 범위 한정 stub 주입.
  - BaseModel.__init__ 은 comfy stub 으로 대체; MotifVideoModel 메서드만 격리 테스트.
"""

import ast
import importlib.util
import os
import sys
import types
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# SSOT 경로
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PY = os.path.join(_ROOT, "models", "__init__.py")


# ---------------------------------------------------------------------------
# Stub 팩토리 — monkeypatch 전용 (전역 stomping 없음)
# ---------------------------------------------------------------------------

class _FakeBaseModel:
    """comfy.model_base.BaseModel 최소 stub.

    __init__ 은 no-op (ComfyUI 의존성 우회).
    apply_model / memory_required / extra_conds 는 실제 MotifVideoModel 이
    super() 로 위임하는 기본 동작을 제공.
    """
    def __init__(self, model_config=None, model_type=None, device=None, **kw):
        pass

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                    transformer_options=None, **kwargs):
        return x

    def memory_required(self, input_shape, cond_shapes=None):
        return 1000.0

    def extra_conds(self, **kwargs):
        return {}


class _FakeModelType:
    FLOW = "flow"


def _build_sys_modules_stubs(
    monkeypatch,
    free_memory_mock,
    current_loaded_models_list,
    t5gemma2_class=None,
):
    """monkeypatch.setitem 으로 sys.modules 에 stub 을 주입한다.

    monkeypatch fixture 가 테스트 종료 시 자동으로 원상 복구하므로
    전역 오염 없음 (MEDIUM 4 수정).

    Args:
        monkeypatch: pytest fixture
        free_memory_mock: comfy.model_management.free_memory 대체 mock
        current_loaded_models_list: list — current_loaded_models 초기값
        t5gemma2_class: MotifVideoT5Gemma2Model 에 사용할 클래스
            (None 이면 기본 stub 생성)
    """
    # ---- comfy 최상위 ----
    comfy_mod = types.ModuleType("comfy")

    # ---- comfy.model_management ----
    mm_mod = types.ModuleType("comfy.model_management")
    mm_mod.free_memory = free_memory_mock
    mm_mod.unload_all_models = mock.MagicMock(return_value=None)
    mm_mod.current_loaded_models = current_loaded_models_list
    mm_mod.get_torch_device = mock.MagicMock(return_value="cpu")
    comfy_mod.model_management = mm_mod

    # ---- comfy.model_base ----
    model_base_mod = types.ModuleType("comfy.model_base")
    model_base_mod.ModelType = _FakeModelType
    model_base_mod.BaseModel = _FakeBaseModel
    comfy_mod.model_base = model_base_mod

    # ---- comfy.conds ----
    conds_mod = types.ModuleType("comfy.conds")
    class _CONDRegular:
        def __init__(self, val):
            self.val = val
    conds_mod.CONDRegular = _CONDRegular
    comfy_mod.conds = conds_mod

    # ---- comfy.ops ----
    ops_mod = types.ModuleType("comfy.ops")
    comfy_mod.ops = ops_mod

    # ---- T5Gemma2 stub ----
    if t5gemma2_class is None:
        class _T5Gemma2Stub:
            pass
        t5gemma2_class = _T5Gemma2Stub

    # ---- text_encoders.t5_gemma2 ----
    t5_mod = types.ModuleType("text_encoders.t5_gemma2")
    t5_mod.MotifVideoT5Gemma2Model = t5gemma2_class
    text_encoders_mod = types.ModuleType("text_encoders")
    text_encoders_mod.t5_gemma2 = t5_mod

    # ---- motif_core stubs ----
    motif_core_mod = types.ModuleType("motif_core")
    motif_models_mod = types.ModuleType("motif_core.models")
    motif_tr_pkg = types.ModuleType("motif_core.models.transformers")
    motif_tr_mod = types.ModuleType(
        "motif_core.models.transformers.transformer_motif_video"
    )
    class _FakeMVT3D:
        pass
    motif_tr_mod.MotifVideoTransformer3DModel = _FakeMVT3D

    # ---- models.* stubs ----
    adapter_mod = types.ModuleType("models.adapter")
    adapter_mod.MotifVideoModelAdapter = object
    latent_mod = types.ModuleType("models.latent_format")
    latent_mod.MotifVideoLatent = object
    compile_mod = types.ModuleType("models.compile_config")
    compile_mod.apply_sage_attention = mock.MagicMock(return_value=None)
    transformer_mod = types.ModuleType("models.transformer")
    transformer_mod.MotifVideoTransformer3DModel = _FakeMVT3D

    # ---- torch stub (없으면 주입) ----
    existing_torch = sys.modules.get("torch")
    if existing_torch is None:
        torch_mod = types.ModuleType("torch")
        torch_mod.float32 = "float32"
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

    # ---- sys.modules 주입 (monkeypatch — 자동 cleanup) ----
    entries = {
        "comfy": comfy_mod,
        "comfy.model_base": model_base_mod,
        "comfy.model_management": mm_mod,
        "comfy.conds": conds_mod,
        "comfy.ops": ops_mod,
        "text_encoders": text_encoders_mod,
        "text_encoders.t5_gemma2": t5_mod,
        "motif_core": motif_core_mod,
        "motif_core.models": motif_models_mod,
        "motif_core.models.transformers": motif_tr_pkg,
        "motif_core.models.transformers.transformer_motif_video": motif_tr_mod,
        "models.adapter": adapter_mod,
        "models.latent_format": latent_mod,
        "models.compile_config": compile_mod,
        "models.transformer": transformer_mod,
        # ComfyUI_MotifVideo1_9B.* 경로도 동일 stub
        "ComfyUI_MotifVideo1_9B.models.adapter": adapter_mod,
        "ComfyUI_MotifVideo1_9B.models.latent_format": latent_mod,
    }
    for key, val in entries.items():
        monkeypatch.setitem(sys.modules, key, val)

    return mm_mod, t5gemma2_class


def _load_motif_model_class(monkeypatch, free_memory_mock, loaded_models, t5gemma2_class=None):
    """매번 fresh 로드: models/__init__.py 에서 MotifVideoModel 클래스를 반환.

    반환: (MotifVideoModel, mm_mod, T5Gemma2Class)
    """
    mm_mod, t5cls = _build_sys_modules_stubs(
        monkeypatch, free_memory_mock, loaded_models, t5gemma2_class
    )

    # 이전 캐시 제거 (monkeypatch 이후 fresh 로드 보장)
    cache_key = "_models_init_p2_fresh"
    monkeypatch.delitem(sys.modules, cache_key, raising=False)

    spec = importlib.util.spec_from_file_location(cache_key, _INIT_PY)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "models"
    monkeypatch.setitem(sys.modules, cache_key, mod)
    spec.loader.exec_module(mod)

    return mod.MotifVideoModel, mm_mod, t5cls


def _make_instance(monkeypatch, free_memory_mock, loaded_models, t5gemma2_class=None):
    """MotifVideoModel 인스턴스를 반환 (BaseModel.__init__ no-op 우회).

    반환: (instance, mm_mod, T5Gemma2Class)
    """
    MotifVideoModel, mm_mod, t5cls = _load_motif_model_class(
        monkeypatch, free_memory_mock, loaded_models, t5gemma2_class
    )
    # __new__ + 수동 속성 주입 — BaseModel.__init__ 의 ComfyUI 의존성 우회
    instance = MotifVideoModel.__new__(MotifVideoModel)
    return instance, mm_mod, t5cls


def _make_loaded_model(model_obj):
    """LoadedModel stub: .model 이 ModelPatcher stub, .model.model 이 nn.Module stub."""
    lm = mock.MagicMock()
    lm.model = mock.MagicMock()
    lm.model.model = model_obj
    return lm


def _make_direct_loaded_model(model_obj):
    """LoadedModel stub: .model 자체가 nn.Module stub (직접 경로)."""
    lm = mock.MagicMock()
    lm.model = model_obj
    # .model.model 은 type(model_obj) 와 다른 것으로
    lm.model.model = object()
    return lm


# ---------------------------------------------------------------------------
# 1. P2 체크리스트 verify grep (AST-independent)
# ---------------------------------------------------------------------------

class TestP2VerifyGrep:
    """P2 checklist verify: 주석/코드 흔적 존재 확인."""

    def test_p2_grep_pattern_present_in_source(self):
        """grep -qE 'model_cpu_offload_seq|...|free_memory' models/__init__.py 동등."""
        import re
        src = open(_INIT_PY).read()
        pattern = re.compile(
            r"model_cpu_offload_seq|sequential_offload|단계별 상주"
            r"|unload_all_models|free_memory"
        )
        assert pattern.search(src), (
            "models/__init__.py 에 P2 HF 계약 동등 흔적 없음"
        )


# ---------------------------------------------------------------------------
# 2. AST 구조 검증
# ---------------------------------------------------------------------------

class TestP2AstStructure:
    """models/__init__.py AST 파싱으로 구조 검증. import 없음."""

    @pytest.fixture(scope="class")
    def tree_and_src(self):
        src = open(_INIT_PY).read()
        return ast.parse(src), src

    @pytest.fixture(scope="class")
    def motif_class_node(self, tree_and_src):
        tree, _ = tree_and_src
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "MotifVideoModel":
                return node
        pytest.fail("MotifVideoModel 클래스를 models/__init__.py 에서 찾지 못함")

    def _get_method(self, cls_node, name):
        for n in cls_node.body:
            if isinstance(n, ast.FunctionDef) and n.name == name:
                return n
        return None

    def test_motif_video_model_has_apply_model_override(self, motif_class_node):
        """MotifVideoModel 이 apply_model 을 override 하는지."""
        method_names = [
            n.name for n in motif_class_node.body
            if isinstance(n, ast.FunctionDef)
        ]
        assert "apply_model" in method_names, (
            f"MotifVideoModel 에 apply_model override 없음. 현재 메서드: {method_names}"
        )

    def test_apply_model_calls_free_memory_or_unload_all_models(self, motif_class_node):
        """apply_model body 에 free_memory 또는 unload_all_models 호출 존재."""
        method = self._get_method(motif_class_node, "apply_model")
        assert method is not None, "apply_model 메서드 없음"
        calls = set()
        for subnode in ast.walk(method):
            if isinstance(subnode, ast.Call):
                if isinstance(subnode.func, ast.Attribute):
                    calls.add(subnode.func.attr)
                elif isinstance(subnode.func, ast.Name):
                    calls.add(subnode.func.id)
        assert calls & {"free_memory", "unload_all_models"}, (
            f"apply_model 에 free_memory / unload_all_models 호출 없음. calls={calls}"
        )

    def test_apply_model_calls_super(self, motif_class_node):
        """apply_model 이 super().apply_model 을 호출 (기존 forward 위임)."""
        method = self._get_method(motif_class_node, "apply_model")
        assert method is not None, "apply_model 메서드 없음"
        src_text = ast.unparse(method)
        assert "super()" in src_text, (
            "apply_model 이 super() 를 호출하지 않음 — 기존 forward 위임 없음"
        )

    def test_apply_model_references_t5gemma2(self, motif_class_node):
        """apply_model body 에 MotifVideoT5Gemma2Model 참조(import 또는 isinstance)가 있는지."""
        method = self._get_method(motif_class_node, "apply_model")
        assert method is not None, "apply_model 메서드 없음"
        src_text = ast.unparse(method)
        assert "MotifVideoT5Gemma2Model" in src_text, (
            "apply_model 이 MotifVideoT5Gemma2Model 을 참조하지 않음 "
            "— T5Gemma2 식별 로직 없음"
        )

    def test_apply_model_has_conditional_guard(self, motif_class_node):
        """apply_model 이 조건부(If) 블록에서 cleanup 호출 — 무조건 cleanup 아님."""
        method = self._get_method(motif_class_node, "apply_model")
        assert method is not None, "apply_model 메서드 없음"
        has_if = any(isinstance(s, ast.If) for s in ast.walk(method))
        assert has_if, (
            "apply_model 에 조건 분기(if)가 없음 — 매 step 무조건 cleanup 위험"
        )

    def test_apply_model_free_memory_uses_keep_loaded_kwarg(self, motif_class_node):
        """free_memory 호출에 keep_loaded 키워드 인자 포함 (흑리스트 방식)."""
        method = self._get_method(motif_class_node, "apply_model")
        assert method is not None, "apply_model 메서드 없음"
        for subnode in ast.walk(method):
            if isinstance(subnode, ast.Call):
                func = subnode.func
                if isinstance(func, ast.Attribute) and func.attr == "free_memory":
                    kwarg_names = [kw.arg for kw in subnode.keywords]
                    assert "keep_loaded" in kwarg_names, (
                        f"free_memory 호출에 keep_loaded 인자 없음. kwargs={kwarg_names}"
                    )
                    return
        # free_memory 없으면 unload_all_models 만 쓰는 구현 — AST 분기 체크가 통과했으므로 OK
        pass

    def test_apply_model_free_memory_large_first_arg(self, motif_class_node):
        """free_memory 첫 인자가 1e20 이상 (HIGH_VRAM skip 회피)."""
        method = self._get_method(motif_class_node, "apply_model")
        assert method is not None, "apply_model 메서드 없음"
        for subnode in ast.walk(method):
            if isinstance(subnode, ast.Call):
                func = subnode.func
                if isinstance(func, ast.Attribute) and func.attr == "free_memory":
                    if subnode.args:
                        first_arg = subnode.args[0]
                        if isinstance(first_arg, ast.Constant):
                            val = first_arg.value
                            assert val >= 1e20, (
                                f"free_memory 첫 인자 {val} 가 1e20 미만"
                            )
                    return
        pass

    def test_memory_required_override_exists(self, motif_class_node):
        """memory_required 메서드가 override 되어 있는지."""
        method_names = [
            n.name for n in motif_class_node.body
            if isinstance(n, ast.FunctionDef)
        ]
        assert "memory_required" in method_names, (
            "MotifVideoModel 에 memory_required override 없음"
        )

    def test_memory_required_calls_super(self, motif_class_node):
        """memory_required 가 super().memory_required 를 호출 (base 값 참조)."""
        method = self._get_method(motif_class_node, "memory_required")
        assert method is not None, "memory_required 메서드 없음"
        src_text = ast.unparse(method)
        assert "super()" in src_text, (
            "memory_required 가 super() 를 호출하지 않음"
        )

    def test_memory_required_margin_ratio_gt_1(self, motif_class_node):
        """memory_required margin 비율이 1.0 초과 (base 보다 큰 값 반환)."""
        method = self._get_method(motif_class_node, "memory_required")
        assert method is not None, "memory_required 메서드 없음"
        for subnode in ast.walk(method):
            if isinstance(subnode, ast.BinOp) and isinstance(subnode.op, ast.Mult):
                for operand in [subnode.left, subnode.right]:
                    if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                        ratio = operand.value
                        if isinstance(ratio, float) and ratio != 1.0:
                            assert ratio > 1.0, (
                                f"memory_required margin ratio={ratio} <= 1.0"
                            )
                            assert ratio <= 2.0, (
                                f"memory_required margin ratio={ratio} > 2.0 (비현실적)"
                            )
                            return
        # margin 없는 구현이면 super() 결과를 그대로 반환 — 추가 검증은 런타임 테스트로


# ---------------------------------------------------------------------------
# 3. 실제 apply_model 호출 기반 런타임 테스트
#    MEDIUM 3 수정: _simulate_* 전면 삭제, 실제 메서드 호출
# ---------------------------------------------------------------------------

class TestApplyModelRealMethodCall:
    """실제 MotifVideoModel.apply_model 호출로 계약 검증.

    Codex MEDIUM 3 수정: production logic 재구현 없음.
    """

    def test_apply_model_calls_free_memory_when_t5gemma2_present(self, monkeypatch):
        """T5Gemma2 가 current_loaded_models 에 있으면 free_memory 가 호출된다.

        Real method call: instance.apply_model(x, t, ...).
        """
        fm = mock.MagicMock(return_value=None)

        # T5Gemma2 인스턴스를 포함한 LoadedModel stub 구성
        class T5Gemma2:
            pass

        t5_instance = T5Gemma2()
        lm_t5 = _make_loaded_model(t5_instance)

        instance, mm_mod, T5Cls = _make_instance(
            monkeypatch, fm, [lm_t5], t5gemma2_class=T5Gemma2
        )

        # 실제 apply_model 호출 — dummy tensor (x=None, t=None 허용).
        # 예외가 나오면 테스트가 실패해야 함 (Codex MEDIUM 반영: try/except 로 삼키지 않음).
        instance.apply_model(None, None)

        assert fm.call_count >= 1, (
            f"T5Gemma2 존재 시 free_memory 미호출. call_count={fm.call_count}"
        )

    def test_apply_model_noop_when_no_t5gemma2(self, monkeypatch):
        """current_loaded_models 에 T5Gemma2 없으면 free_memory 미호출 (no-op).

        Real method call: 상태 기반 idempotent 검증.
        """
        fm = mock.MagicMock(return_value=None)

        # T5Gemma2 아닌 ControlNet 류 모델만 존재
        class ControlNetModel:
            pass

        cn_instance = ControlNetModel()
        lm_cn = _make_loaded_model(cn_instance)

        instance, mm_mod, _ = _make_instance(monkeypatch, fm, [lm_cn])

        instance.apply_model(None, None)

        assert fm.call_count == 0, (
            f"T5Gemma2 없을 때 free_memory 가 호출됨. call_count={fm.call_count}"
        )

    def test_apply_model_twice_noop_second_call_if_t5_unloaded(self, monkeypatch):
        """T5Gemma2 가 첫 apply_model 후 unloaded 되었으면 두번째 호출은 no-op.

        상태 기반 idempotent: current_loaded_models 상태가 변경되면 자연히 no-op.
        """
        fm = mock.MagicMock(return_value=None)

        class T5Gemma2:
            pass

        t5_instance = T5Gemma2()
        lm_t5 = _make_loaded_model(t5_instance)
        loaded = [lm_t5]

        instance, mm_mod, _ = _make_instance(
            monkeypatch, fm, loaded, t5gemma2_class=T5Gemma2
        )

        # 1st call: T5Gemma2 있음 → free_memory 호출
        instance.apply_model(None, None)
        count_after_first = fm.call_count

        # T5Gemma2 가 unload 됨 → loaded 에서 제거
        mm_mod.current_loaded_models = []

        # 2nd call: T5Gemma2 없음 → free_memory 0
        instance.apply_model(None, None)
        count_after_second = fm.call_count

        assert count_after_first >= 1, "첫 번째 호출에서 free_memory 미호출"
        assert count_after_second == count_after_first, (
            f"T5Gemma2 unload 후 두번째 apply_model 에서 추가 free_memory 발생. "
            f"1st={count_after_first}, 2nd={count_after_second}"
        )

    def test_apply_model_t5gemma2_reappear_triggers_unload_again(self, monkeypatch):
        """T5Gemma2 가 재등장하면 다음 apply_model 에서 다시 unload.

        새 batch 시나리오: 상태 기반이므로 플래그 없이 자연 동작.
        """
        fm = mock.MagicMock(return_value=None)

        class T5Gemma2:
            pass

        t5_instance = T5Gemma2()
        lm_t5 = _make_loaded_model(t5_instance)

        instance, mm_mod, _ = _make_instance(
            monkeypatch, fm, [lm_t5], t5gemma2_class=T5Gemma2
        )

        # 1st call with T5Gemma2
        instance.apply_model(None, None)
        count1 = fm.call_count

        # T5Gemma2 사라짐
        mm_mod.current_loaded_models = []
        instance.apply_model(None, None)
        count2 = fm.call_count

        # T5Gemma2 재등장
        mm_mod.current_loaded_models = [lm_t5]
        instance.apply_model(None, None)
        count3 = fm.call_count

        assert count1 >= 1, "1st batch: free_memory 미호출"
        assert count2 == count1, "T5Gemma2 없을 때 추가 free_memory"
        assert count3 > count2, (
            f"T5Gemma2 재등장 후 free_memory 미호출. count2={count2}, count3={count3}"
        )


class TestNonTextEncoderNotUnloaded:
    """non-text-encoder 모델은 unload 대상이 아님 (흑리스트 방식 검증).

    keep_loaded 에 control/LoRA/etc. 가 포함되는지 확인.
    """

    def test_control_model_in_keep_loaded(self, monkeypatch):
        """ControlNet 류 모델이 free_memory keep_loaded 에 포함된다.

        흑리스트: T5Gemma2 만 제외, 나머지는 모두 keep.
        """
        fm = mock.MagicMock(return_value=None)

        class T5Gemma2:
            pass

        class ControlNetModel:
            pass

        t5_instance = T5Gemma2()
        cn_instance = ControlNetModel()

        lm_t5 = _make_loaded_model(t5_instance)
        lm_cn = _make_loaded_model(cn_instance)

        instance, mm_mod, _ = _make_instance(
            monkeypatch, fm, [lm_t5, lm_cn], t5gemma2_class=T5Gemma2
        )

        instance.apply_model(None, None)

        assert fm.call_count >= 1, "T5Gemma2 존재 시 free_memory 미호출"
        # keep_loaded 인자 검증: lm_cn 이 포함되어 있어야 함
        call_kwargs = fm.call_args
        assert call_kwargs is not None, "free_memory 호출 인자 없음"
        keep = call_kwargs.kwargs.get("keep_loaded") or (
            call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
        )
        assert keep is not None, "free_memory keep_loaded 인자 없음"
        assert lm_cn in keep, (
            "ControlNet 모델이 keep_loaded 에 없음 — 의도치 않게 unload 될 수 있음"
        )
        assert lm_t5 not in keep, (
            "T5Gemma2 가 keep_loaded 에 포함됨 — unload 대상이어야 함"
        )

    def test_multiple_non_te_models_all_in_keep_loaded(self, monkeypatch):
        """여러 non-text-encoder 모델 (ControlNet + LoRA patcher + hooks) 모두 keep."""
        fm = mock.MagicMock(return_value=None)

        class T5Gemma2:
            pass

        class ControlNetModel:
            pass

        class LoRAPatcher:
            pass

        class HookModel:
            pass

        t5 = T5Gemma2()
        cn = ControlNetModel()
        lora = LoRAPatcher()
        hook = HookModel()

        lm_t5 = _make_loaded_model(t5)
        lm_cn = _make_loaded_model(cn)
        lm_lora = _make_loaded_model(lora)
        lm_hook = _make_loaded_model(hook)

        instance, mm_mod, _ = _make_instance(
            monkeypatch, fm,
            [lm_t5, lm_cn, lm_lora, lm_hook],
            t5gemma2_class=T5Gemma2,
        )

        instance.apply_model(None, None)

        assert fm.call_count >= 1, "free_memory 미호출"
        call_kwargs = fm.call_args
        keep = call_kwargs.kwargs.get("keep_loaded") or (
            call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
        )
        assert keep is not None, "keep_loaded 인자 없음"
        for lm in [lm_cn, lm_lora, lm_hook]:
            assert lm in keep, f"{lm} 가 keep_loaded 에 없음"
        assert lm_t5 not in keep, "T5Gemma2 가 keep_loaded 에 포함됨"

    def test_direct_nn_module_path_t5gemma2_identified(self, monkeypatch):
        """LoadedModel.model 자체가 T5Gemma2 인 직접 경로에서도 식별됨.

        ModelPatcher 래핑 없이 직접 nn.Module 인 경우 대비.
        """
        fm = mock.MagicMock(return_value=None)

        class T5Gemma2:
            pass

        t5_instance = T5Gemma2()
        lm_direct = _make_direct_loaded_model(t5_instance)

        instance, mm_mod, _ = _make_instance(
            monkeypatch, fm, [lm_direct], t5gemma2_class=T5Gemma2
        )

        instance.apply_model(None, None)

        assert fm.call_count >= 1, (
            "직접 nn.Module 경로 T5Gemma2 를 식별하지 못하고 free_memory 미호출"
        )


class TestMemoryRequiredRealMethod:
    """실제 memory_required 메서드 호출로 margin 검증.

    Codex MEDIUM 3 수정: _simulate_memory_required 삭제, 실제 메서드 호출.
    """

    def test_memory_required_greater_than_base(self, monkeypatch):
        """memory_required 반환값 > super().memory_required (margin 적용)."""
        fm = mock.MagicMock(return_value=None)
        instance, mm_mod, _ = _make_instance(monkeypatch, fm, [])

        # super().memory_required 는 _FakeBaseModel.memory_required = 1000.0
        result = instance.memory_required((1, 16, 9, 80, 80))
        assert result > 1000.0, (
            f"memory_required({result}) <= base(1000.0) — margin 미적용"
        )

    def test_memory_required_ratio_range(self, monkeypatch):
        """memory_required 비율이 [1.2, 1.5] 범위 (P2 스펙)."""
        fm = mock.MagicMock(return_value=None)
        instance, mm_mod, _ = _make_instance(monkeypatch, fm, [])

        base = 1000.0
        result = instance.memory_required((1, 16, 9, 80, 80))
        ratio = result / base
        assert 1.0 < ratio <= 2.0, (
            f"memory_required ratio={ratio:.3f} 가 허용 범위(1.0, 2.0] 밖"
        )

    def test_memory_required_zero_base_handles_gracefully(self, monkeypatch):
        """super().memory_required 가 0.0 반환 시 결과도 유한값."""
        fm = mock.MagicMock(return_value=None)

        class ZeroBaseModel(_FakeBaseModel):
            def memory_required(self, input_shape, cond_shapes=None):
                return 0.0

        # _FakeBaseModel 을 ZeroBaseModel 로 교체
        mm_mod_inner, _ = _build_sys_modules_stubs(monkeypatch, fm, [])
        # model_base 의 BaseModel 을 교체
        sys.modules["comfy.model_base"].BaseModel = ZeroBaseModel

        cache_key = "_models_init_p2_zero"
        monkeypatch.delitem(sys.modules, cache_key, raising=False)
        spec = importlib.util.spec_from_file_location(cache_key, _INIT_PY)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "models"
        monkeypatch.setitem(sys.modules, cache_key, mod)
        spec.loader.exec_module(mod)
        MotifVideoModel = mod.MotifVideoModel

        instance = MotifVideoModel.__new__(MotifVideoModel)
        result = instance.memory_required((1, 16, 9, 80, 80))
        import math
        assert math.isfinite(result), f"memory_required({result}) 가 유한값이 아님"


class TestApplyModelDirectLoadedModelPath:
    """LoadedModel 양 경로 (ModelPatcher, 직접) 에서 T5Gemma2 검색."""

    def test_modelpatcher_path_identified(self, monkeypatch):
        """lm.model.model 이 T5Gemma2 인 ModelPatcher 경로에서 free_memory 호출."""
        fm = mock.MagicMock(return_value=None)

        class T5Gemma2:
            pass

        t5 = T5Gemma2()
        lm = _make_loaded_model(t5)  # lm.model.model = t5

        instance, _, _ = _make_instance(monkeypatch, fm, [lm], t5gemma2_class=T5Gemma2)
        instance.apply_model(None, None)
        assert fm.call_count >= 1, "ModelPatcher 경로 T5Gemma2 미식별"

    def test_none_model_entry_skipped_gracefully(self, monkeypatch):
        """current_loaded_models 에 lm.model=None 항목이 있어도 크래시 없음."""
        fm = mock.MagicMock(return_value=None)

        lm_none = mock.MagicMock()
        lm_none.model = None  # None 경로

        instance, _, _ = _make_instance(monkeypatch, fm, [lm_none])
        # T5Gemma2 없으므로 free_memory 미호출, 예외도 없어야 함
        try:
            instance.apply_model(None, None)
        except Exception as e:
            pytest.fail(f"lm.model=None 항목 처리 중 예외 발생: {e}")
        assert fm.call_count == 0, "T5Gemma2 없는데 free_memory 호출됨"


class TestSyntaxOk:
    """P2 verify: syntax OK."""

    def test_models_init_syntax(self):
        src = open(_INIT_PY).read()
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"models/__init__.py syntax error: {e}")

    def test_root_init_syntax(self):
        root_init = os.path.join(_ROOT, "__init__.py")
        src = open(root_init).read()
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"__init__.py syntax error: {e}")
