"""
P0.7 블라인드 검증 — clip_name widget 값 고정 테스트.

대상 파일:
  workflows/Motif-2B_T2V_example.json
  workflows/Motif-2B_I2V_example.json

JSON 구조 (litegraph subgraph 형식):
  - MotifTextEncoderLoader 노드는 최상위 nodes 배열이 아니라 두 곳에 위치:
    1. definitions.subgraphs[*].nodes  (subgraph 정의)
    2. extra.groupNodes['motif-default'].nodes  (GroupNode 내부 노드)
  - widgets_values = ['<clip_name>', '<dtype>', '<offload>']

요구사항:
  1. JSON 파싱 성공
  2. MotifTextEncoderLoader 노드 widgets_values[0] == 'motifvideo_t5gemma2.safetensors'
  3. 레거시 값 'motifvideo_t5gemma2/model' 완전 제거
  4. top-level 키 집합 / nodes 배열 훼손 없음
  5. widgets_values[1]='bfloat16', [2]='default' 보존
  6. MotifTextEncoderLoader 노드 개수 변화 없음 (두 위치 합산 >= 2)
"""

import json
import pathlib
import unittest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"

T2V_PATH = WORKFLOWS_DIR / "Motif-2B_T2V_example.json"
I2V_PATH = WORKFLOWS_DIR / "Motif-2B_I2V_example.json"

ENCODER_LOADER_TYPES = {"MotifTextEncoderLoader", "MotifVideoTextEncoderLoader"}
EXPECTED_CLIP_NAME = "motifvideo_t5gemma2.safetensors"
LEGACY_CLIP_NAME_FRAGMENT = "motifvideo_t5gemma2/model"

# widgets_values 기존 포맷: ['<clip_name>', 'bfloat16', 'default']
EXPECTED_DTYPE = "bfloat16"
EXPECTED_OFFLOAD = "default"

# top-level 필수 키 (두 파일 공통)
REQUIRED_TOP_LEVEL_KEYS = {"nodes", "links", "last_node_id", "last_link_id", "definitions"}


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _load(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_encoder_loader_nodes(wf: dict) -> list:
    """MotifTextEncoderLoader / MotifVideoTextEncoderLoader 노드를 모든 위치에서 수집.

    검색 위치:
      1. top-level nodes (litegraph 표준)
      2. definitions.subgraphs[*].nodes  (subgraph 정의 내부)
      3. extra.groupNodes[*].nodes       (GroupNode 내부)
    """
    result = []

    # 1. top-level nodes
    for n in wf.get("nodes", []):
        if n.get("type") in ENCODER_LOADER_TYPES:
            result.append(("nodes", n))

    # 2. definitions.subgraphs
    for sg in wf.get("definitions", {}).get("subgraphs", []):
        for n in sg.get("nodes", []):
            if n.get("type") in ENCODER_LOADER_TYPES:
                result.append(("definitions.subgraphs", n))

    # 3. extra.groupNodes
    for gk, gv in wf.get("extra", {}).get("groupNodes", {}).items():
        for n in gv.get("nodes", []):
            if n.get("type") in ENCODER_LOADER_TYPES:
                result.append((f"extra.groupNodes[{gk!r}]", n))

    return result  # list of (location_str, node_dict)


def _raw(path: pathlib.Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. JSON 파싱 유효성
# ---------------------------------------------------------------------------

class TestJsonParsing(unittest.TestCase):
    """두 workflow JSON 파일이 예외 없이 파싱되어야 한다."""

    def test_t2v_json_parses_ok(self):
        """T2V JSON 파싱 — 예외 없어야 한다."""
        try:
            _load(T2V_PATH)
        except json.JSONDecodeError as exc:
            self.fail(f"T2V JSON 파싱 실패: {exc}")
        except FileNotFoundError:
            self.fail(f"파일 없음: {T2V_PATH}")

    def test_i2v_json_parses_ok(self):
        """I2V JSON 파싱 — 예외 없어야 한다."""
        try:
            _load(I2V_PATH)
        except json.JSONDecodeError as exc:
            self.fail(f"I2V JSON 파싱 실패: {exc}")
        except FileNotFoundError:
            self.fail(f"파일 없음: {I2V_PATH}")

    def test_t2v_top_level_is_dict(self):
        """T2V 최상위 구조는 dict여야 한다."""
        wf = _load(T2V_PATH)
        self.assertIsInstance(wf, dict)

    def test_i2v_top_level_is_dict(self):
        """I2V 최상위 구조는 dict여야 한다."""
        wf = _load(I2V_PATH)
        self.assertIsInstance(wf, dict)


# ---------------------------------------------------------------------------
# 2. 신규 clip_name 값 존재
# ---------------------------------------------------------------------------

class TestNewClipNameValueT2V(unittest.TestCase):
    """T2V: MotifTextEncoderLoader widgets_values[0] == 'motifvideo_t5gemma2.safetensors'."""

    def setUp(self):
        self.wf = _load(T2V_PATH)
        self.nodes_with_loc = _collect_encoder_loader_nodes(self.wf)

    def test_encoder_loader_nodes_exist(self):
        """T2V에 MotifTextEncoderLoader 노드가 definitions 또는 extra.groupNodes에 있어야 한다."""
        self.assertGreater(
            len(self.nodes_with_loc), 0,
            f"T2V: {ENCODER_LOADER_TYPES} 타입 노드를 어디서도 찾을 수 없음",
        )

    def test_all_encoder_loader_nodes_have_expected_clip_name(self):
        """T2V: 모든 encoder loader 노드의 widgets_values[0] 이
        'motifvideo_t5gemma2.safetensors' 여야 한다."""
        failures = []
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            actual = wv[0] if wv else None
            if actual != EXPECTED_CLIP_NAME:
                failures.append(
                    f"[{loc}] node id={node.get('id')} type={node.get('type')!r}: "
                    f"widgets_values[0]={actual!r}, expected={EXPECTED_CLIP_NAME!r}"
                )
        self.assertEqual(failures, [], "\n".join(failures))

    def test_t2v_raw_string_contains_expected_clip_name(self):
        """T2V JSON 파일에 'motifvideo_t5gemma2.safetensors' 문자열이 포함되어야 한다."""
        raw = _raw(T2V_PATH)
        self.assertIn(
            EXPECTED_CLIP_NAME, raw,
            f"T2V JSON에 {EXPECTED_CLIP_NAME!r} 문자열 없음",
        )


class TestNewClipNameValueI2V(unittest.TestCase):
    """I2V: MotifTextEncoderLoader widgets_values[0] == 'motifvideo_t5gemma2.safetensors'."""

    def setUp(self):
        self.wf = _load(I2V_PATH)
        self.nodes_with_loc = _collect_encoder_loader_nodes(self.wf)

    def test_encoder_loader_nodes_exist(self):
        """I2V에 MotifTextEncoderLoader 노드가 definitions 또는 extra.groupNodes에 있어야 한다."""
        self.assertGreater(
            len(self.nodes_with_loc), 0,
            f"I2V: {ENCODER_LOADER_TYPES} 타입 노드를 어디서도 찾을 수 없음",
        )

    def test_all_encoder_loader_nodes_have_expected_clip_name(self):
        """I2V: 모든 encoder loader 노드의 widgets_values[0] 이
        'motifvideo_t5gemma2.safetensors' 여야 한다."""
        failures = []
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            actual = wv[0] if wv else None
            if actual != EXPECTED_CLIP_NAME:
                failures.append(
                    f"[{loc}] node id={node.get('id')} type={node.get('type')!r}: "
                    f"widgets_values[0]={actual!r}, expected={EXPECTED_CLIP_NAME!r}"
                )
        self.assertEqual(failures, [], "\n".join(failures))

    def test_i2v_raw_string_contains_expected_clip_name(self):
        """I2V JSON 파일에 'motifvideo_t5gemma2.safetensors' 문자열이 포함되어야 한다."""
        raw = _raw(I2V_PATH)
        self.assertIn(
            EXPECTED_CLIP_NAME, raw,
            f"I2V JSON에 {EXPECTED_CLIP_NAME!r} 문자열 없음",
        )


# ---------------------------------------------------------------------------
# 3. 레거시 값 완전 제거
# ---------------------------------------------------------------------------

class TestLegacyClipNameRemoved(unittest.TestCase):
    """'motifvideo_t5gemma2/model' 문자열이 두 파일에서 완전히 제거되었는지 확인."""

    def test_t2v_no_legacy_clip_name_in_raw(self):
        """T2V JSON 전체 문자열에 'motifvideo_t5gemma2/model' 없어야 한다."""
        raw = _raw(T2V_PATH)
        self.assertNotIn(
            LEGACY_CLIP_NAME_FRAGMENT, raw,
            f"T2V JSON에 레거시 값 {LEGACY_CLIP_NAME_FRAGMENT!r} 남아 있음",
        )

    def test_i2v_no_legacy_clip_name_in_raw(self):
        """I2V JSON 전체 문자열에 'motifvideo_t5gemma2/model' 없어야 한다."""
        raw = _raw(I2V_PATH)
        self.assertNotIn(
            LEGACY_CLIP_NAME_FRAGMENT, raw,
            f"I2V JSON에 레거시 값 {LEGACY_CLIP_NAME_FRAGMENT!r} 남아 있음",
        )

    def test_t2v_encoder_loader_widgets_values_no_legacy(self):
        """T2V: encoder loader 노드 widgets_values 원소에 레거시 값 없음."""
        wf = _load(T2V_PATH)
        for loc, node in _collect_encoder_loader_nodes(wf):
            for v in (node.get("widgets_values") or []):
                if isinstance(v, str):
                    self.assertNotIn(
                        LEGACY_CLIP_NAME_FRAGMENT, v,
                        f"T2V [{loc}] node id={node.get('id')}: "
                        f"widgets_values에 레거시 값 {v!r}",
                    )

    def test_i2v_encoder_loader_widgets_values_no_legacy(self):
        """I2V: encoder loader 노드 widgets_values 원소에 레거시 값 없음."""
        wf = _load(I2V_PATH)
        for loc, node in _collect_encoder_loader_nodes(wf):
            for v in (node.get("widgets_values") or []):
                if isinstance(v, str):
                    self.assertNotIn(
                        LEGACY_CLIP_NAME_FRAGMENT, v,
                        f"I2V [{loc}] node id={node.get('id')}: "
                        f"widgets_values에 레거시 값 {v!r}",
                    )


# ---------------------------------------------------------------------------
# 4. 다른 필드 훼손 없음 (top-level 키, nodes 배열 길이)
# ---------------------------------------------------------------------------

class TestTopLevelStructureIntegrityT2V(unittest.TestCase):
    """T2V: top-level 키 집합 / 주요 필드 타입 검증."""

    def setUp(self):
        self.wf = _load(T2V_PATH)

    def test_required_top_level_keys_present(self):
        """T2V: 필수 top-level 키가 모두 존재해야 한다."""
        missing = REQUIRED_TOP_LEVEL_KEYS - set(self.wf.keys())
        self.assertEqual(missing, set(), f"T2V top-level 키 누락: {missing}")

    def test_nodes_array_nonempty(self):
        """T2V: nodes 배열이 비어있지 않아야 한다."""
        self.assertGreater(len(self.wf.get("nodes", [])), 0, "T2V nodes 배열이 비어 있음")

    def test_links_array_is_list(self):
        """T2V: links 필드가 리스트 타입이어야 한다."""
        self.assertIsInstance(self.wf.get("links"), list, "T2V links가 리스트가 아님")

    def test_last_node_id_is_integer(self):
        """T2V: last_node_id가 정수여야 한다."""
        val = self.wf.get("last_node_id")
        self.assertIsInstance(val, int, f"T2V last_node_id가 정수가 아님: {val!r}")

    def test_last_link_id_is_integer(self):
        """T2V: last_link_id가 정수여야 한다."""
        val = self.wf.get("last_link_id")
        self.assertIsInstance(val, int, f"T2V last_link_id가 정수가 아님: {val!r}")

    def test_definitions_has_subgraphs(self):
        """T2V: definitions.subgraphs 배열이 존재해야 한다."""
        defs = self.wf.get("definitions", {})
        self.assertIn("subgraphs", defs, "T2V definitions에 subgraphs 키 없음")
        self.assertIsInstance(defs["subgraphs"], list, "T2V definitions.subgraphs가 리스트가 아님")
        self.assertGreater(len(defs["subgraphs"]), 0, "T2V definitions.subgraphs가 비어 있음")


class TestTopLevelStructureIntegrityI2V(unittest.TestCase):
    """I2V: top-level 키 집합 / 주요 필드 타입 검증."""

    def setUp(self):
        self.wf = _load(I2V_PATH)

    def test_required_top_level_keys_present(self):
        """I2V: 필수 top-level 키가 모두 존재해야 한다."""
        missing = REQUIRED_TOP_LEVEL_KEYS - set(self.wf.keys())
        self.assertEqual(missing, set(), f"I2V top-level 키 누락: {missing}")

    def test_nodes_array_nonempty(self):
        """I2V: nodes 배열이 비어있지 않아야 한다."""
        self.assertGreater(len(self.wf.get("nodes", [])), 0, "I2V nodes 배열이 비어 있음")

    def test_links_array_is_list(self):
        """I2V: links 필드가 리스트 타입이어야 한다."""
        self.assertIsInstance(self.wf.get("links"), list, "I2V links가 리스트가 아님")

    def test_last_node_id_is_integer(self):
        """I2V: last_node_id가 정수여야 한다."""
        val = self.wf.get("last_node_id")
        self.assertIsInstance(val, int, f"I2V last_node_id가 정수가 아님: {val!r}")

    def test_last_link_id_is_integer(self):
        """I2V: last_link_id가 정수여야 한다."""
        val = self.wf.get("last_link_id")
        self.assertIsInstance(val, int, f"I2V last_link_id가 정수가 아님: {val!r}")

    def test_definitions_has_subgraphs(self):
        """I2V: definitions.subgraphs 배열이 존재해야 한다."""
        defs = self.wf.get("definitions", {})
        self.assertIn("subgraphs", defs, "I2V definitions에 subgraphs 키 없음")
        self.assertIsInstance(defs["subgraphs"], list, "I2V definitions.subgraphs가 리스트가 아님")
        self.assertGreater(len(defs["subgraphs"]), 0, "I2V definitions.subgraphs가 비어 있음")


# ---------------------------------------------------------------------------
# 5. widgets_values 다른 원소 보존 (dtype, offload)
# ---------------------------------------------------------------------------

class TestWidgetsValuesOtherElementsPreservedT2V(unittest.TestCase):
    """T2V: encoder loader 노드 widgets_values[1]='bfloat16', [2]='default' 보존."""

    def setUp(self):
        self.wf = _load(T2V_PATH)
        self.nodes_with_loc = _collect_encoder_loader_nodes(self.wf)
        if not self.nodes_with_loc:
            self.skipTest(f"T2V: {ENCODER_LOADER_TYPES} 타입 노드 없음")

    def test_widgets_values_length_is_at_least_3(self):
        """T2V: encoder loader widgets_values 원소가 3개 이상이어야 한다."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            self.assertGreaterEqual(
                len(wv), 3,
                f"T2V [{loc}] node id={node.get('id')}: widgets_values 길이 {len(wv)} < 3",
            )

    def test_dtype_element_preserved(self):
        """T2V: widgets_values[1] == 'bfloat16' 이어야 한다."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            if len(wv) >= 2:
                self.assertEqual(
                    wv[1],
                    EXPECTED_DTYPE,
                    f"T2V [{loc}] node id={node.get('id')}: "
                    f"widgets_values[1]={wv[1]!r}, expected={EXPECTED_DTYPE!r}",
                )

    def test_offload_element_preserved(self):
        """T2V: widgets_values[2] == 'default' 이어야 한다."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            if len(wv) >= 3:
                self.assertEqual(
                    wv[2],
                    EXPECTED_OFFLOAD,
                    f"T2V [{loc}] node id={node.get('id')}: "
                    f"widgets_values[2]={wv[2]!r}, expected={EXPECTED_OFFLOAD!r}",
                )

    def test_clip_name_is_string(self):
        """T2V: widgets_values[0] 타입이 str이어야 한다 (int/None 등 타입 혼입 방지)."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            if wv:
                self.assertIsInstance(
                    wv[0], str,
                    f"T2V [{loc}] node id={node.get('id')}: "
                    f"widgets_values[0] 타입={type(wv[0])!r}",
                )


class TestWidgetsValuesOtherElementsPreservedI2V(unittest.TestCase):
    """I2V: encoder loader 노드 widgets_values[1]='bfloat16', [2]='default' 보존."""

    def setUp(self):
        self.wf = _load(I2V_PATH)
        self.nodes_with_loc = _collect_encoder_loader_nodes(self.wf)
        if not self.nodes_with_loc:
            self.skipTest(f"I2V: {ENCODER_LOADER_TYPES} 타입 노드 없음")

    def test_widgets_values_length_is_at_least_3(self):
        """I2V: encoder loader widgets_values 원소가 3개 이상이어야 한다."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            self.assertGreaterEqual(
                len(wv), 3,
                f"I2V [{loc}] node id={node.get('id')}: widgets_values 길이 {len(wv)} < 3",
            )

    def test_dtype_element_preserved(self):
        """I2V: widgets_values[1] == 'bfloat16' 이어야 한다."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            if len(wv) >= 2:
                self.assertEqual(
                    wv[1],
                    EXPECTED_DTYPE,
                    f"I2V [{loc}] node id={node.get('id')}: "
                    f"widgets_values[1]={wv[1]!r}, expected={EXPECTED_DTYPE!r}",
                )

    def test_offload_element_preserved(self):
        """I2V: widgets_values[2] == 'default' 이어야 한다."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            if len(wv) >= 3:
                self.assertEqual(
                    wv[2],
                    EXPECTED_OFFLOAD,
                    f"I2V [{loc}] node id={node.get('id')}: "
                    f"widgets_values[2]={wv[2]!r}, expected={EXPECTED_OFFLOAD!r}",
                )

    def test_clip_name_is_string(self):
        """I2V: widgets_values[0] 타입이 str이어야 한다."""
        for loc, node in self.nodes_with_loc:
            wv = node.get("widgets_values") or []
            if wv:
                self.assertIsInstance(
                    wv[0], str,
                    f"I2V [{loc}] node id={node.get('id')}: "
                    f"widgets_values[0] 타입={type(wv[0])!r}",
                )


# ---------------------------------------------------------------------------
# 6. 노드 수 변화 없음 (MotifTextEncoderLoader 개수)
# ---------------------------------------------------------------------------

class TestEncoderLoaderNodeCount(unittest.TestCase):
    """두 파일 모두 MotifTextEncoderLoader 노드가 최소 2개 (definitions + extra 포함).

    요구사항 원문: "기존 2개씩, subgraph definition placeholder 포함".
    두 위치(definitions.subgraphs, extra.groupNodes)에서 각 1개씩 발견되면 합산 2개.
    """

    EXPECTED_MIN_COUNT = 2

    def test_t2v_encoder_loader_node_count_at_least_2(self):
        """T2V: MotifTextEncoderLoader 노드가 (정의 + groupNode 포함) 2개 이상이어야 한다."""
        wf = _load(T2V_PATH)
        nodes = _collect_encoder_loader_nodes(wf)
        self.assertGreaterEqual(
            len(nodes),
            self.EXPECTED_MIN_COUNT,
            f"T2V encoder loader 노드 수={len(nodes)}, "
            f"expected >= {self.EXPECTED_MIN_COUNT}. "
            f"발견 위치: {[loc for loc, _ in nodes]}",
        )

    def test_i2v_encoder_loader_node_count_at_least_2(self):
        """I2V: MotifTextEncoderLoader 노드가 (정의 + groupNode 포함) 2개 이상이어야 한다."""
        wf = _load(I2V_PATH)
        nodes = _collect_encoder_loader_nodes(wf)
        self.assertGreaterEqual(
            len(nodes),
            self.EXPECTED_MIN_COUNT,
            f"I2V encoder loader 노드 수={len(nodes)}, "
            f"expected >= {self.EXPECTED_MIN_COUNT}. "
            f"발견 위치: {[loc for loc, _ in nodes]}",
        )

    def test_t2v_total_node_count_reasonable(self):
        """T2V: top-level nodes 배열 길이가 1 이상이어야 한다."""
        wf = _load(T2V_PATH)
        self.assertGreater(len(wf.get("nodes", [])), 0, "T2V: nodes 배열이 비어 있음")

    def test_i2v_total_node_count_reasonable(self):
        """I2V: top-level nodes 배열 길이가 1 이상이어야 한다."""
        wf = _load(I2V_PATH)
        self.assertGreater(len(wf.get("nodes", [])), 0, "I2V: nodes 배열이 비어 있음")

    def test_t2v_definitions_subgraph_has_encoder_loader(self):
        """T2V: definitions.subgraphs 내에 MotifTextEncoderLoader 노드가 있어야 한다."""
        wf = _load(T2V_PATH)
        found = [
            n for sg in wf.get("definitions", {}).get("subgraphs", [])
            for n in sg.get("nodes", [])
            if n.get("type") in ENCODER_LOADER_TYPES
        ]
        self.assertGreater(
            len(found), 0,
            "T2V definitions.subgraphs에 MotifTextEncoderLoader 노드 없음",
        )

    def test_i2v_definitions_subgraph_has_encoder_loader(self):
        """I2V: definitions.subgraphs 내에 MotifTextEncoderLoader 노드가 있어야 한다."""
        wf = _load(I2V_PATH)
        found = [
            n for sg in wf.get("definitions", {}).get("subgraphs", [])
            for n in sg.get("nodes", [])
            if n.get("type") in ENCODER_LOADER_TYPES
        ]
        self.assertGreater(
            len(found), 0,
            "I2V definitions.subgraphs에 MotifTextEncoderLoader 노드 없음",
        )

    def test_t2v_group_nodes_has_encoder_loader(self):
        """T2V: extra.groupNodes 내에 MotifTextEncoderLoader 노드가 있어야 한다."""
        wf = _load(T2V_PATH)
        found = [
            n for gv in wf.get("extra", {}).get("groupNodes", {}).values()
            for n in gv.get("nodes", [])
            if n.get("type") in ENCODER_LOADER_TYPES
        ]
        self.assertGreater(
            len(found), 0,
            "T2V extra.groupNodes에 MotifTextEncoderLoader 노드 없음",
        )

    def test_i2v_group_nodes_has_encoder_loader(self):
        """I2V: extra.groupNodes 내에 MotifTextEncoderLoader 노드가 있어야 한다."""
        wf = _load(I2V_PATH)
        found = [
            n for gv in wf.get("extra", {}).get("groupNodes", {}).values()
            for n in gv.get("nodes", [])
            if n.get("type") in ENCODER_LOADER_TYPES
        ]
        self.assertGreater(
            len(found), 0,
            "I2V extra.groupNodes에 MotifTextEncoderLoader 노드 없음",
        )


# ---------------------------------------------------------------------------
# 경계값 / 타입 안전성 추가 케이스
# ---------------------------------------------------------------------------

class TestEdgeCasesClipName(unittest.TestCase):
    """clip_name 값의 경계값 및 타입 이상 케이스."""

    def _get_nodes(self, path: pathlib.Path):
        return _collect_encoder_loader_nodes(_load(path))

    def test_t2v_clip_name_not_empty_string(self):
        """T2V: clip_name이 빈 문자열이 아니어야 한다."""
        for loc, node in self._get_nodes(T2V_PATH):
            wv = node.get("widgets_values") or []
            if wv:
                self.assertNotEqual(
                    wv[0], "",
                    f"T2V [{loc}] node id={node.get('id')}: clip_name이 빈 문자열",
                )

    def test_i2v_clip_name_not_empty_string(self):
        """I2V: clip_name이 빈 문자열이 아니어야 한다."""
        for loc, node in self._get_nodes(I2V_PATH):
            wv = node.get("widgets_values") or []
            if wv:
                self.assertNotEqual(
                    wv[0], "",
                    f"I2V [{loc}] node id={node.get('id')}: clip_name이 빈 문자열",
                )

    def test_t2v_clip_name_not_none(self):
        """T2V: clip_name(widgets_values[0])이 None이 아니어야 한다."""
        for loc, node in self._get_nodes(T2V_PATH):
            wv = node.get("widgets_values") or []
            if wv:
                self.assertIsNotNone(
                    wv[0],
                    f"T2V [{loc}] node id={node.get('id')}: clip_name이 None",
                )

    def test_i2v_clip_name_not_none(self):
        """I2V: clip_name(widgets_values[0])이 None이 아니어야 한다."""
        for loc, node in self._get_nodes(I2V_PATH):
            wv = node.get("widgets_values") or []
            if wv:
                self.assertIsNotNone(
                    wv[0],
                    f"I2V [{loc}] node id={node.get('id')}: clip_name이 None",
                )

    def test_t2v_clip_name_no_directory_separator(self):
        """T2V: clip_name에 '/' 디렉토리 구분자가 없어야 한다 (flat filename 요구)."""
        for loc, node in self._get_nodes(T2V_PATH):
            wv = node.get("widgets_values") or []
            if wv and isinstance(wv[0], str):
                self.assertNotIn(
                    "/", wv[0],
                    f"T2V [{loc}] node id={node.get('id')}: clip_name에 '/' 포함: {wv[0]!r}",
                )

    def test_i2v_clip_name_no_directory_separator(self):
        """I2V: clip_name에 '/' 디렉토리 구분자가 없어야 한다 (flat filename 요구)."""
        for loc, node in self._get_nodes(I2V_PATH):
            wv = node.get("widgets_values") or []
            if wv and isinstance(wv[0], str):
                self.assertNotIn(
                    "/", wv[0],
                    f"I2V [{loc}] node id={node.get('id')}: clip_name에 '/' 포함: {wv[0]!r}",
                )

    def test_t2v_clip_name_ends_with_safetensors(self):
        """T2V: clip_name이 '.safetensors' 확장자로 끝나야 한다."""
        for loc, node in self._get_nodes(T2V_PATH):
            wv = node.get("widgets_values") or []
            if wv and isinstance(wv[0], str) and wv[0]:
                self.assertTrue(
                    wv[0].endswith(".safetensors"),
                    f"T2V [{loc}] node id={node.get('id')}: "
                    f"clip_name이 .safetensors로 끝나지 않음: {wv[0]!r}",
                )

    def test_i2v_clip_name_ends_with_safetensors(self):
        """I2V: clip_name이 '.safetensors' 확장자로 끝나야 한다."""
        for loc, node in self._get_nodes(I2V_PATH):
            wv = node.get("widgets_values") or []
            if wv and isinstance(wv[0], str) and wv[0]:
                self.assertTrue(
                    wv[0].endswith(".safetensors"),
                    f"I2V [{loc}] node id={node.get('id')}: "
                    f"clip_name이 .safetensors로 끝나지 않음: {wv[0]!r}",
                )

    def test_t2v_clip_name_exact_match_not_partial(self):
        """T2V: clip_name이 정확히 EXPECTED_CLIP_NAME 전체와 일치해야 한다 (부분 일치 거부)."""
        for loc, node in self._get_nodes(T2V_PATH):
            wv = node.get("widgets_values") or []
            if wv and isinstance(wv[0], str):
                self.assertEqual(
                    wv[0],
                    EXPECTED_CLIP_NAME,
                    f"T2V [{loc}] node id={node.get('id')}: "
                    f"clip_name 완전 불일치: got {wv[0]!r}, expected {EXPECTED_CLIP_NAME!r}",
                )

    def test_i2v_clip_name_exact_match_not_partial(self):
        """I2V: clip_name이 정확히 EXPECTED_CLIP_NAME 전체와 일치해야 한다."""
        for loc, node in self._get_nodes(I2V_PATH):
            wv = node.get("widgets_values") or []
            if wv and isinstance(wv[0], str):
                self.assertEqual(
                    wv[0],
                    EXPECTED_CLIP_NAME,
                    f"I2V [{loc}] node id={node.get('id')}: "
                    f"clip_name 완전 불일치: got {wv[0]!r}, expected {EXPECTED_CLIP_NAME!r}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
