"""
체크리스트 항목 2.2 "I2V 워크플로우 JSON" 테스트.

범위:
1. JSON 파일 유효성 (파싱 성공)
2. 필수 노드 존재 여부 (type 기반)
3. MotifImageEncode 노드의 입력 연결 구조 검증
4. KSampler의 positive/negative 입력이 MotifImageEncode에서 오는지 검증
5. ModelSamplingSD3 shift 값이 2.5인지 검증
6. 링크 정합성: origin_id / target_id 가 실제 노드 id와 매치

JSON 구조 (litegraph 형식):
  - 최상위: {"last_node_id": ..., "nodes": [...], "links": [...], ...}
  - 노드: {"id": int, "type": str, "inputs": [{"name": str, "link": int|None, ...}],
           "widgets_values": [...], ...}
  - links: [[link_id, origin_node_id, origin_slot, target_node_id, target_slot, type], ...]

주의: 파일 파싱만으로 검증하며 GPU/comfy 의존 없음.
"""

import json
import pathlib
import unittest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
WORKFLOW_PATH = PROJECT_ROOT / "workflows" / "i2v_example.json"

REQUIRED_NODE_TYPES = [
    "LoadImage",
    "MotifImageEncode",
    "MotifTextEncode",
    "EmptyMotifLatent",
    "KSampler",
    "VAEDecode",
    "MotifVAELoader",
    "UNETLoader",
    "MotifTextEncoderLoader",
]


def _load_workflow() -> dict:
    """workflows/i2v_example.json을 파싱하여 반환. 실패 시 예외."""
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _node_list(workflow: dict) -> list:
    """nodes 배열 반환."""
    return workflow.get("nodes", [])


def _node_map(workflow: dict) -> dict:
    """노드 id(int) -> 노드 dict 매핑."""
    return {n["id"]: n for n in _node_list(workflow)}


def _find_nodes_by_type(workflow: dict, node_type: str) -> list:
    return [n for n in _node_list(workflow) if n.get("type") == node_type]


def _input_link_id(node: dict, input_name: str) -> int | None:
    """노드 inputs 배열에서 name이 일치하는 항목의 link id를 반환. 없으면 None."""
    for inp in node.get("inputs") or []:
        if inp.get("name") == input_name:
            return inp.get("link")
    return None


def _link_origin_node_id(workflow: dict, link_id: int) -> int | None:
    """link_id에 해당하는 origin node id 반환. 없으면 None."""
    for lnk in workflow.get("links", []):
        if lnk[0] == link_id:
            return lnk[1]
    return None


# ---------------------------------------------------------------------------
# 1. JSON 유효성
# ---------------------------------------------------------------------------

class TestWorkflowI2VJsonValidity(unittest.TestCase):
    """JSON 파일 자체의 유효성을 검증한다."""

    def test_file_exists(self):
        """workflows/i2v_example.json 파일이 존재해야 한다."""
        self.assertTrue(
            WORKFLOW_PATH.exists(),
            f"파일 없음: {WORKFLOW_PATH}",
        )

    def test_json_parses_without_error(self):
        """파일이 유효한 JSON으로 파싱되어야 한다."""
        try:
            _load_workflow()
        except json.JSONDecodeError as exc:
            self.fail(f"JSON 파싱 실패: {exc}")

    def test_json_top_level_is_dict(self):
        """최상위 구조는 dict여야 한다."""
        workflow = _load_workflow()
        self.assertIsInstance(workflow, dict, "최상위가 dict가 아님")

    def test_json_has_nodes_key(self):
        """최상위에 nodes 키가 존재해야 한다."""
        workflow = _load_workflow()
        self.assertIn("nodes", workflow, "nodes 키 없음")

    def test_json_nodes_is_list(self):
        """nodes 필드는 리스트여야 한다."""
        workflow = _load_workflow()
        self.assertIsInstance(workflow["nodes"], list, "nodes가 리스트가 아님")

    def test_json_not_empty(self):
        """노드가 비어있지 않아야 한다."""
        workflow = _load_workflow()
        self.assertGreater(len(_node_list(workflow)), 0, "nodes가 빈 리스트")


# ---------------------------------------------------------------------------
# 2. 필수 노드 존재 여부
# ---------------------------------------------------------------------------

class TestWorkflowI2VRequiredNodes(unittest.TestCase):
    """I2V에 필요한 모든 노드 타입이 워크플로우에 존재하는지 확인."""

    def setUp(self):
        self.workflow = _load_workflow()

    def _node_types_present(self) -> set:
        return {n.get("type") for n in _node_list(self.workflow)}

    def test_all_required_node_types_exist(self):
        """REQUIRED_NODE_TYPES 의 모든 타입이 워크플로우에 있어야 한다."""
        present = self._node_types_present()
        missing = [t for t in REQUIRED_NODE_TYPES if t not in present]
        self.assertEqual(
            missing,
            [],
            f"누락된 노드 타입: {missing}",
        )

    def test_load_image_node_exists(self):
        """LoadImage 노드가 최소 1개 존재해야 한다."""
        found = _find_nodes_by_type(self.workflow, "LoadImage")
        self.assertGreaterEqual(len(found), 1, "LoadImage 노드 없음")

    def test_motif_image_encode_node_exists(self):
        """MotifImageEncode 노드가 최소 1개 존재해야 한다."""
        found = _find_nodes_by_type(self.workflow, "MotifImageEncode")
        self.assertGreaterEqual(len(found), 1, "MotifImageEncode 노드 없음")

    def test_node_count_at_least_required(self):
        """전체 노드 수가 필수 목록 수 이상이어야 한다."""
        self.assertGreaterEqual(
            len(_node_list(self.workflow)),
            len(REQUIRED_NODE_TYPES),
            "노드 수가 필수 목록보다 적음",
        )


# ---------------------------------------------------------------------------
# 3. MotifImageEncode 노드 입력 구조 검증
# ---------------------------------------------------------------------------

class TestMotifImageEncodeInputs(unittest.TestCase):
    """MotifImageEncode 노드의 입력(positive, negative, vae, image)을 검증."""

    REQUIRED_INPUT_NAMES = {"positive", "negative", "vae", "image"}

    def setUp(self):
        self.workflow = _load_workflow()
        nodes_found = _find_nodes_by_type(self.workflow, "MotifImageEncode")
        self.assertGreaterEqual(
            len(nodes_found), 1,
            "MotifImageEncode 노드가 없어 입력 검증 불가",
        )
        self.image_encode_node = nodes_found[0]

    def _input_names(self, node: dict) -> set:
        return {inp["name"] for inp in (node.get("inputs") or [])}

    def test_image_encode_has_required_input_names(self):
        """MotifImageEncode inputs에 positive, negative, vae, image 이름이 있어야 한다."""
        present = self._input_names(self.image_encode_node)
        missing = self.REQUIRED_INPUT_NAMES - present
        self.assertEqual(
            missing,
            set(),
            f"MotifImageEncode inputs에 누락된 입력 이름: {missing}",
        )

    def test_image_encode_image_input_has_link(self):
        """image 입력의 link id가 None이 아니어야 한다 (실제 연결됨)."""
        link_id = _input_link_id(self.image_encode_node, "image")
        self.assertIsNotNone(link_id, "MotifImageEncode image 입력이 연결되지 않음 (link=None)")

    def test_image_encode_vae_input_has_link(self):
        """vae 입력의 link id가 None이 아니어야 한다."""
        link_id = _input_link_id(self.image_encode_node, "vae")
        self.assertIsNotNone(link_id, "MotifImageEncode vae 입력이 연결되지 않음 (link=None)")

    def test_image_encode_positive_input_has_link(self):
        """positive 입력의 link id가 None이 아니어야 한다."""
        link_id = _input_link_id(self.image_encode_node, "positive")
        self.assertIsNotNone(link_id, "MotifImageEncode positive 입력이 연결되지 않음 (link=None)")

    def test_image_encode_negative_input_has_link(self):
        """negative 입력의 link id가 None이 아니어야 한다."""
        link_id = _input_link_id(self.image_encode_node, "negative")
        self.assertIsNotNone(link_id, "MotifImageEncode negative 입력이 연결되지 않음 (link=None)")

    def test_image_input_origin_is_load_image(self):
        """image 입력의 출처 노드 타입이 LoadImage여야 한다."""
        link_id = _input_link_id(self.image_encode_node, "image")
        self.assertIsNotNone(link_id, "image 링크 없음")
        origin_id = _link_origin_node_id(self.workflow, link_id)
        self.assertIsNotNone(origin_id, f"link {link_id}의 origin 노드를 찾을 수 없음")
        node_map = _node_map(self.workflow)
        origin_node = node_map.get(origin_id)
        self.assertIsNotNone(origin_node, f"origin 노드 id={origin_id} 없음")
        self.assertEqual(
            origin_node.get("type"),
            "LoadImage",
            f"image 입력이 LoadImage에서 오지 않음: {origin_node.get('type')!r}",
        )


# ---------------------------------------------------------------------------
# 4. KSampler positive/negative 입력이 MotifImageEncode에서 오는지 검증
# ---------------------------------------------------------------------------

class TestKSamplerConnectedToImageEncode(unittest.TestCase):
    """KSampler의 positive/negative가 MotifImageEncode에서 와야 한다."""

    def setUp(self):
        self.workflow = _load_workflow()

        ks_nodes = _find_nodes_by_type(self.workflow, "KSampler")
        self.assertGreaterEqual(len(ks_nodes), 1, "KSampler 노드 없음")
        self.ksampler = ks_nodes[0]

        ie_nodes = _find_nodes_by_type(self.workflow, "MotifImageEncode")
        self.assertGreaterEqual(len(ie_nodes), 1, "MotifImageEncode 노드 없음")
        self.ie_id = ie_nodes[0]["id"]

    def test_ksampler_positive_comes_from_image_encode(self):
        """KSampler positive 입력의 출처 노드가 MotifImageEncode여야 한다."""
        link_id = _input_link_id(self.ksampler, "positive")
        self.assertIsNotNone(link_id, "KSampler positive 입력 링크 없음")
        origin_id = _link_origin_node_id(self.workflow, link_id)
        self.assertEqual(
            origin_id,
            self.ie_id,
            f"KSampler positive가 MotifImageEncode({self.ie_id})에서 오지 않음: "
            f"origin_id={origin_id!r}",
        )

    def test_ksampler_negative_comes_from_image_encode(self):
        """KSampler negative 입력의 출처 노드가 MotifImageEncode여야 한다."""
        link_id = _input_link_id(self.ksampler, "negative")
        self.assertIsNotNone(link_id, "KSampler negative 입력 링크 없음")
        origin_id = _link_origin_node_id(self.workflow, link_id)
        self.assertEqual(
            origin_id,
            self.ie_id,
            f"KSampler negative가 MotifImageEncode({self.ie_id})에서 오지 않음: "
            f"origin_id={origin_id!r}",
        )

    def test_ksampler_positive_not_from_text_encode_directly(self):
        """KSampler positive가 MotifTextEncode를 직접 참조하면 안 된다 (ImageEncode 경유 필수)."""
        link_id = _input_link_id(self.ksampler, "positive")
        self.assertIsNotNone(link_id, "KSampler positive 링크 없음")
        origin_id = _link_origin_node_id(self.workflow, link_id)
        node_map = _node_map(self.workflow)
        origin_node = node_map.get(origin_id, {})
        self.assertNotEqual(
            origin_node.get("type"),
            "MotifTextEncode",
            "KSampler positive가 MotifTextEncode를 직접 참조함 (ImageEncode 경유 필요)",
        )

    def test_ksampler_negative_not_from_text_encode_directly(self):
        """KSampler negative가 MotifTextEncode를 직접 참조하면 안 된다."""
        link_id = _input_link_id(self.ksampler, "negative")
        self.assertIsNotNone(link_id, "KSampler negative 링크 없음")
        origin_id = _link_origin_node_id(self.workflow, link_id)
        node_map = _node_map(self.workflow)
        origin_node = node_map.get(origin_id, {})
        self.assertNotEqual(
            origin_node.get("type"),
            "MotifTextEncode",
            "KSampler negative가 MotifTextEncode를 직접 참조함 (ImageEncode 경유 필요)",
        )


# ---------------------------------------------------------------------------
# 5. ModelSamplingSD3 shift 값 검증
# ---------------------------------------------------------------------------

class TestModelSamplingSD3Shift(unittest.TestCase):
    """ModelSamplingSD3 노드의 shift 파라미터가 2.5인지 확인.

    litegraph 형식에서 shift는 inputs 배열이 아닌 widgets_values[0]에 저장된다.
    """

    def setUp(self):
        self.workflow = _load_workflow()

    def test_model_sampling_sd3_node_exists(self):
        """ModelSamplingSD3 노드가 워크플로우에 존재해야 한다."""
        found = _find_nodes_by_type(self.workflow, "ModelSamplingSD3")
        self.assertGreaterEqual(len(found), 1, "ModelSamplingSD3 노드 없음")

    def test_model_sampling_sd3_shift_is_2_5(self):
        """ModelSamplingSD3 노드의 shift 값(widgets_values[0])이 2.5여야 한다."""
        found = _find_nodes_by_type(self.workflow, "ModelSamplingSD3")
        self.assertGreaterEqual(len(found), 1, "ModelSamplingSD3 노드 없음")
        node = found[0]
        widgets = node.get("widgets_values") or []
        self.assertGreater(len(widgets), 0, "ModelSamplingSD3 widgets_values가 비어있음")
        shift_val = widgets[0]
        self.assertAlmostEqual(
            float(shift_val),
            2.5,
            places=5,
            msg=f"shift 값이 2.5가 아님: {shift_val!r}",
        )

    def test_model_sampling_sd3_shift_not_t2v_default(self):
        """I2V shift 값은 T2V 기본값(1.0)과 달라야 한다."""
        found = _find_nodes_by_type(self.workflow, "ModelSamplingSD3")
        if not found:
            self.skipTest("ModelSamplingSD3 노드 없음")
        widgets = found[0].get("widgets_values") or []
        if not widgets:
            self.skipTest("widgets_values 없음")
        shift_val = widgets[0]
        self.assertNotAlmostEqual(
            float(shift_val),
            1.0,
            places=5,
            msg="shift 값이 T2V 기본값(1.0)임 — I2V 전용 값(2.5)으로 설정 필요",
        )


# ---------------------------------------------------------------------------
# 6. 링크 정합성 검증
# ---------------------------------------------------------------------------

class TestWorkflowI2VLinkIntegrity(unittest.TestCase):
    """links 배열의 origin_id / target_id 가 실제 노드 id와 일치하는지 확인.

    link 형식: [link_id, origin_node_id, origin_slot, target_node_id, target_slot, type]
    """

    def setUp(self):
        self.workflow = _load_workflow()
        self.node_ids = {n["id"] for n in _node_list(self.workflow)}

    def test_links_field_exists(self):
        """워크플로우에 links 필드가 존재해야 한다."""
        self.assertIn("links", self.workflow, "links 필드 없음")

    def test_links_is_list(self):
        """links 필드는 리스트여야 한다."""
        self.assertIsInstance(self.workflow.get("links"), list, "links가 리스트가 아님")

    def test_links_not_empty(self):
        """links 리스트가 비어있지 않아야 한다."""
        links = self.workflow.get("links", [])
        self.assertGreater(len(links), 0, "links 배열이 비어 있음 — 노드 연결 없음")

    def test_all_link_origin_ids_exist(self):
        """모든 link의 origin node id가 실제 노드에 존재해야 한다."""
        bad = []
        for lnk in self.workflow.get("links", []):
            if not isinstance(lnk, list) or len(lnk) < 4:
                bad.append(f"잘못된 link 형식: {lnk!r}")
                continue
            origin_id = lnk[1]
            if origin_id not in self.node_ids:
                bad.append(f"link_id={lnk[0]}: origin_id={origin_id} 노드 없음")
        self.assertEqual(bad, [], "origin_id 불일치:\n" + "\n".join(bad))

    def test_all_link_target_ids_exist(self):
        """모든 link의 target node id가 실제 노드에 존재해야 한다."""
        bad = []
        for lnk in self.workflow.get("links", []):
            if not isinstance(lnk, list) or len(lnk) < 4:
                bad.append(f"잘못된 link 형식: {lnk!r}")
                continue
            target_id = lnk[3]
            if target_id not in self.node_ids:
                bad.append(f"link_id={lnk[0]}: target_id={target_id} 노드 없음")
        self.assertEqual(bad, [], "target_id 불일치:\n" + "\n".join(bad))

    def test_no_duplicate_link_ids(self):
        """link id가 중복되지 않아야 한다."""
        links = self.workflow.get("links", [])
        link_ids = [lnk[0] for lnk in links if isinstance(lnk, list) and len(lnk) >= 1]
        duplicates = [i for i in link_ids if link_ids.count(i) > 1]
        self.assertEqual(
            len(link_ids),
            len(set(link_ids)),
            f"중복된 link id: {list(set(duplicates))}",
        )

    def test_node_input_links_reference_valid_link_ids(self):
        """각 노드 inputs의 link 필드가 links 배열에 존재하는 link id를 참조해야 한다."""
        valid_link_ids = {
            lnk[0] for lnk in self.workflow.get("links", [])
            if isinstance(lnk, list) and len(lnk) >= 1
        }
        bad = []
        for node in _node_list(self.workflow):
            for inp in node.get("inputs") or []:
                link_id = inp.get("link")
                if link_id is not None and link_id not in valid_link_ids:
                    bad.append(
                        f"노드 id={node['id']} ({node.get('type')}) "
                        f"inputs[{inp.get('name')!r}] link={link_id} links 배열에 없음"
                    )
        self.assertEqual(bad, [], "입력 링크 참조 불일치:\n" + "\n".join(bad))


# ---------------------------------------------------------------------------
# 경계값 / 이상 입력 방어 테스트
# ---------------------------------------------------------------------------

class TestWorkflowI2VEdgeCases(unittest.TestCase):
    """JSON 구조 이상 케이스 방어 검증."""

    def setUp(self):
        self.workflow = _load_workflow()

    def test_no_null_type_nodes(self):
        """type이 None이거나 빈 문자열인 노드가 없어야 한다."""
        bad = [
            n.get("id") for n in _node_list(self.workflow)
            if not n.get("type")
        ]
        self.assertEqual(bad, [], f"type 없는 노드 id: {bad}")

    def test_node_ids_are_unique(self):
        """노드 id가 중복되지 않아야 한다."""
        ids = [n["id"] for n in _node_list(self.workflow)]
        self.assertEqual(
            len(ids),
            len(set(ids)),
            "중복된 노드 id 존재",
        )

    def test_ksampler_cfg_is_positive_float(self):
        """KSampler cfg 값이 양수여야 한다.

        litegraph 형식: widgets_values = [seed, seed_mode, steps, cfg, sampler, scheduler, denoise]
        cfg는 index 3.
        """
        ks_nodes = _find_nodes_by_type(self.workflow, "KSampler")
        if not ks_nodes:
            self.skipTest("KSampler 없음")
        widgets = ks_nodes[0].get("widgets_values") or []
        self.assertGreater(len(widgets), 3, "KSampler widgets_values 항목 부족 (cfg 없음)")
        cfg = widgets[3]
        self.assertGreater(float(cfg), 0.0, f"cfg가 양수가 아님: {cfg!r}")

    def test_ksampler_cfg_matches_i2v_spec(self):
        """KSampler cfg 값이 I2V 스펙(8.0)과 일치해야 한다."""
        ks_nodes = _find_nodes_by_type(self.workflow, "KSampler")
        if not ks_nodes:
            self.skipTest("KSampler 없음")
        widgets = ks_nodes[0].get("widgets_values") or []
        if len(widgets) <= 3:
            self.skipTest("widgets_values에 cfg 없음")
        cfg = widgets[3]
        self.assertAlmostEqual(
            float(cfg),
            8.0,
            places=5,
            msg=f"KSampler cfg가 I2V 스펙(8.0)과 다름: {cfg!r}",
        )

    def test_empty_latent_node_exists(self):
        """EmptyMotifLatent 노드가 존재해야 한다."""
        found = _find_nodes_by_type(self.workflow, "EmptyMotifLatent")
        self.assertGreaterEqual(len(found), 1, "EmptyMotifLatent 노드 없음")

    def test_vae_decode_node_exists(self):
        """VAEDecode 노드가 존재해야 한다."""
        found = _find_nodes_by_type(self.workflow, "VAEDecode")
        self.assertGreaterEqual(len(found), 1, "VAEDecode 노드 없음")

    def test_all_nodes_have_integer_id(self):
        """모든 노드의 id 필드가 정수여야 한다."""
        bad = [
            n for n in _node_list(self.workflow)
            if not isinstance(n.get("id"), int)
        ]
        self.assertEqual(bad, [], f"id가 정수가 아닌 노드: {[n.get('id') for n in bad]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
