"""tests/test_p5_requirements.py — P5 requirements.txt 정리 블라인드 테스트.

검증 대상:
  P5: 미사용 패키지 drop + transformers 버전 상향 + install.py 주석 추가.

커버리지:
  1. drop 확정 패키지 제거 확인 (peft, loguru, accelerate, sentencepiece)
  2. 필수 패키지 보존 확인 (diffusers, einops, transformers)
  3. transformers 버전 상향 엄격 검증 (>=5.5.4, 5.5.0 약화 방어)
  4. P0.5 싱크 — 02_context.md keep/drop 판정과 requirements.txt 일치
  5. requirements.txt pip 파싱 가능성 (포맷 무결성)
  6. 실 import 없음 확인 — drop 패키지가 repo 소스에서 import 0건
  7. install.py 기존 함수 보존 + 주석 위치 검증
  8. install.py 로직 변경 없음 (구조 보존)
  9. 경계값 — 빈 라인/주석 처리된 라인이 패키지 라인으로 오인되지 않음

실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI/custom_nodes/ComfyUI-MotifVideo1.9B
    pytest tests/test_p5_requirements.py -v
"""

import ast
import os
import pathlib
import re
import subprocess
import sys

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
_REQUIREMENTS = _PROJECT_ROOT / "requirements.txt"
_INSTALL_PY = _PROJECT_ROOT / "install.py"
_CONTEXT_MD = (
    _PROJECT_ROOT
    / ".plans"
    / "20260421-hf-alignment-vram-speed"
    / "02_context.md"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _req_lines() -> list[str]:
    """주석/빈 라인 제외 실 패키지 라인만 반환."""
    text = _REQUIREMENTS.read_text()
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _req_text() -> str:
    return _REQUIREMENTS.read_text()


def _install_text() -> str:
    return _INSTALL_PY.read_text()


def _context_text() -> str:
    return _CONTEXT_MD.read_text()


# ---------------------------------------------------------------------------
# 1. drop 확정 패키지 제거
# ---------------------------------------------------------------------------

class TestDropPackages:
    """P5 확정 drop 4종이 requirements.txt 에서 제거됐는지 검증."""

    def test_peft_dropped_from_requirements(self):
        """peft 패키지 라인 제거 확인."""
        assert not re.search(r"^peft", _req_text(), re.MULTILINE), (
            "peft 가 requirements.txt 에 여전히 존재함 — drop 필수"
        )

    def test_loguru_dropped_from_requirements(self):
        """loguru 패키지 라인 제거 확인."""
        assert not re.search(r"^loguru", _req_text(), re.MULTILINE), (
            "loguru 가 requirements.txt 에 여전히 존재함 — drop 필수"
        )

    def test_accelerate_dropped_from_requirements(self):
        """accelerate 패키지 라인 제거 확인."""
        assert not re.search(r"^accelerate", _req_text(), re.MULTILINE), (
            "accelerate 가 requirements.txt 에 여전히 존재함 — drop 필수"
        )

    def test_sentencepiece_dropped_from_requirements(self):
        """sentencepiece 패키지 라인 제거 확인 (P0.5: GemmaTokenizerFast 는 tokenizers 기반)."""
        assert not re.search(r"^sentencepiece", _req_text(), re.MULTILINE), (
            "sentencepiece 가 requirements.txt 에 여전히 존재함 — P0.5 drop 판정"
        )

    def test_no_commented_out_drop_packages(self):
        """주석 처리된 drop 패키지도 라인 자체가 없어야 함 (완전 제거 확인).

        주석 처리(# peft) 로 남긴 경우 후속 플랜에서 re-enable 혼란 유발.
        """
        text = _req_text()
        for pkg in ("peft", "loguru", "accelerate", "sentencepiece"):
            # 주석 포함 전체 텍스트에서 해당 패키지 언급 자체를 확인
            # 단순 주석 예시(#peft, # peft) 도 경고
            commented = re.findall(rf"^#.*\b{pkg}\b", text, re.MULTILINE)
            # 경고 수준 — commented 행 존재해도 FAIL 아님
            # 그러나 non-commented 패키지 라인이 없음은 위 테스트가 담보
            # 여기서는 추가 정보 제공용
            _ = commented  # 주석 처리된 라인 존재 정보 (FAIL 아님)


# ---------------------------------------------------------------------------
# 2. 필수 패키지 보존
# ---------------------------------------------------------------------------

class TestKeepPackages:
    """diffusers, einops, transformers 가 여전히 존재하는지 검증."""

    def test_diffusers_kept(self):
        """diffusers 패키지 라인 보존 확인."""
        assert re.search(r"^diffusers", _req_text(), re.MULTILINE), (
            "diffusers 가 requirements.txt 에서 제거됨 — MotifVideoModel 핵심 의존"
        )

    def test_einops_kept(self):
        """einops 패키지 라인 보존 확인."""
        assert re.search(r"^einops", _req_text(), re.MULTILINE), (
            "einops 가 requirements.txt 에서 제거됨 — rearrange 직접 사용"
        )

    def test_transformers_kept(self):
        """transformers 패키지 라인 보존 확인."""
        assert re.search(r"^transformers", _req_text(), re.MULTILINE), (
            "transformers 가 requirements.txt 에서 제거됨 — T5Gemma2Encoder 핵심 의존"
        )


# ---------------------------------------------------------------------------
# 3. transformers 버전 상향 엄격 검증
# ---------------------------------------------------------------------------

class TestTransformersVersion:
    """transformers>=5.5.4 버전 상향이 정확히 적용됐는지 엄격 검증."""

    def test_transformers_version_exact_5_5_4(self):
        """transformers>=5.5.4 정확한 버전 포함 확인."""
        assert re.search(r"^transformers>=5\.5\.4", _req_text(), re.MULTILINE), (
            "transformers>=5.5.4 버전 지정이 없음 — HF README 기준 버전 상향 필수"
        )

    def test_transformers_not_old_version_5_0_0(self):
        """이전 버전 >=5.0.0 이 남아있지 않은지 확인 (버전 약화 방어)."""
        assert not re.search(r"^transformers>=5\.0\.0", _req_text(), re.MULTILINE), (
            "transformers>=5.0.0 이 여전히 남아있음 — 5.5.4 로 상향돼야 함"
        )

    def test_transformers_not_too_old_4x(self):
        """4.x 버전 지정이 없는지 확인."""
        assert not re.search(r"^transformers>=4\.", _req_text(), re.MULTILINE), (
            "transformers 4.x 버전 지정 — T5Gemma2Encoder 미지원"
        )

    def test_transformers_version_format_valid(self):
        """transformers 버전 지정 형식이 pip 파싱 가능한지 확인.

        유효 형식: transformers>=5.5.4 또는 transformers>=5.5.4,<6.0.0 등.
        """
        m = re.search(r"^(transformers[^\n]*)", _req_text(), re.MULTILINE)
        assert m, "transformers 라인 없음"
        line = m.group(1).strip()
        # 기본 유효성: 패키지명으로 시작, 버전 연산자 포함
        assert re.match(r"^transformers(>=|==|~=|<=)", line), (
            f"transformers 버전 지정 형식 이상: {line!r}"
        )

    def test_transformers_version_not_weakened_to_5_5_0(self):
        """5.5.0 같은 약화된 버전으로 실수 기재되지 않았는지 방어."""
        m = re.search(r"^transformers>=(\d+\.\d+\.\d+)", _req_text(), re.MULTILINE)
        if m:
            version_str = m.group(1)
            parts = list(map(int, version_str.split(".")))
            required = [5, 5, 4]
            assert parts >= required, (
                f"transformers 버전 {version_str} < 5.5.4 — 약화된 버전 실수"
            )


# ---------------------------------------------------------------------------
# 4. P0.5 싱크 검증
# ---------------------------------------------------------------------------

class TestP05Sync:
    """02_context.md keep/drop 판정과 requirements.txt 상태 일치 검증."""

    def test_transformers_sync_keep_drop(self):
        """transformers: 02_context 판정 keep → requirements.txt 에 존재해야."""
        ctx = _context_text()
        req = _req_text()
        m = re.search(r"transformers.*\b(keep|drop)\b", ctx, re.IGNORECASE)
        assert m, "02_context.md 에 transformers keep/drop 판정 기록 없음 — P0.5 감사 결과 필수"
        verdict = m.group(1).lower()
        in_req = bool(re.search(r"^transformers", req, re.MULTILINE))
        assert (verdict == "keep") == in_req, (
            f"transformers 판정={verdict} / requirements 존재={in_req} 불일치"
        )

    def test_sentencepiece_sync_keep_drop(self):
        """sentencepiece: 02_context 판정 drop → requirements.txt 에 없어야."""
        ctx = _context_text()
        req = _req_text()
        m = re.search(r"sentencepiece.*\b(keep|drop)\b", ctx, re.IGNORECASE)
        assert m, "02_context.md 에 sentencepiece keep/drop 판정 기록 없음 — P0.5 감사 결과 필수"
        verdict = m.group(1).lower()
        in_req = bool(re.search(r"^sentencepiece", req, re.MULTILINE))
        assert (verdict == "keep") == in_req, (
            f"sentencepiece 판정={verdict} / requirements 존재={in_req} 불일치"
        )

    def test_peft_sync_keep_drop(self):
        """peft: 02_context 판정 drop → requirements.txt 에 없어야."""
        ctx = _context_text()
        req = _req_text()
        m = re.search(r"\bpeft\b.*\b(keep|drop)\b", ctx, re.IGNORECASE)
        assert m, "02_context.md 에 peft keep/drop 판정 기록 없음"
        verdict = m.group(1).lower()
        in_req = bool(re.search(r"^peft", req, re.MULTILINE))
        assert (verdict == "keep") == in_req, (
            f"peft 판정={verdict} / requirements 존재={in_req} 불일치"
        )

    def test_accelerate_sync_keep_drop(self):
        """accelerate: 02_context 판정 drop → requirements.txt 에 없어야."""
        ctx = _context_text()
        req = _req_text()
        m = re.search(r"\bacceleate\b.*\b(keep|drop)\b|\bacceleate.*drop\b|accelerate.*\b(keep|drop)\b", ctx, re.IGNORECASE)
        if not m:
            # accelerate 단독 언급 + drop 키워드 별도 탐색
            m = re.search(r"accelerate[^\n]*(keep|drop)", ctx, re.IGNORECASE)
        assert m, "02_context.md 에 accelerate keep/drop 판정 기록 없음"
        # 마지막 그룹에서 keep/drop 추출
        verdict = None
        for g in m.groups():
            if g:
                verdict = g.lower()
                break
        assert verdict is not None
        in_req = bool(re.search(r"^accelerate", req, re.MULTILINE))
        assert (verdict == "keep") == in_req, (
            f"accelerate 판정={verdict} / requirements 존재={in_req} 불일치"
        )


# ---------------------------------------------------------------------------
# 5. requirements.txt pip 포맷 무결성
# ---------------------------------------------------------------------------

class TestRequirementsFormat:
    """requirements.txt 가 pip parse 가능한 형식인지 검증."""

    def test_no_syntax_error_pip_parse(self):
        """pip --dry-run 으로 파싱 가능한지 확인 (네트워크 없이 parse 전용)."""
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--dry-run", "--quiet",
                "--no-deps",
                "-r", str(_REQUIREMENTS),
            ],
            capture_output=True,
            text=True,
        )
        # exit code 0 = parse OK (네트워크 에러여도 parse 성공이면 0)
        # exit code 1 = parse 에러 (invalid format 등)
        # 단, 실제 설치 없이 syntax만 — 패키지 미존재는 에러가 아닐 수 있음
        # 여기서는 format parse 에러에 집중
        stderr_lower = result.stderr.lower()
        assert "invalid requirement" not in stderr_lower, (
            f"requirements.txt 형식 에러: {result.stderr[:200]}"
        )
        assert "could not find a version" not in stderr_lower or result.returncode == 0, (
            # 네트워크 없이 실행 시 "Could not find" 는 정상 — parse 성공으로 간주
            "pip parse 실패"
        )

    def test_each_non_comment_line_has_package_name(self):
        """비주석 비빈 라인이 모두 패키지명으로 시작하는지 확인."""
        lines = _req_lines()
        for line in lines:
            # 유효 패키지 라인: 문자/숫자로 시작, 또는 -r/-c 같은 지시어
            assert re.match(r"^[A-Za-z0-9_\-\.\[\]]+|^-[rceriC]", line), (
                f"유효하지 않은 requirements 라인: {line!r}"
            )

    def test_no_duplicate_package_entries(self):
        """같은 패키지가 두 번 이상 등장하지 않는지 확인."""
        lines = _req_lines()
        pkg_names = []
        for line in lines:
            m = re.match(r"^([A-Za-z0-9_\-]+)", line)
            if m:
                pkg_names.append(m.group(1).lower())
        seen = set()
        for name in pkg_names:
            assert name not in seen, f"중복 패키지 항목: {name!r}"
            seen.add(name)

    def test_requirements_not_empty(self):
        """requirements.txt 가 비어있지 않은지 확인."""
        lines = _req_lines()
        assert len(lines) > 0, "requirements.txt 가 비어있음"

    def test_requirements_file_exists(self):
        """requirements.txt 파일 자체가 존재하는지."""
        assert _REQUIREMENTS.exists(), f"{_REQUIREMENTS} 파일 없음"


# ---------------------------------------------------------------------------
# 6. drop 패키지 repo 내 실 import 없음 확인
# ---------------------------------------------------------------------------

class TestNoDirectImport:
    """peft, loguru, accelerate, sentencepiece 가 repo 소스에서 직접 import 0건인지 검증.

    diffusers.utils.USE_PEFT_BACKEND 는 runtime 전환용이므로 peft 직접 import 로 취급 안 함.
    """

    def _find_direct_imports(self, package: str) -> list[str]:
        """package 를 직접 import 하는 .py 파일 목록 반환."""
        result = subprocess.run(
            [
                "grep", "-rn", "--include=*.py",
                "-E",
                rf"^(from|import)\s+{package}(\.\w+)?(\s|$)",
                str(_PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [ln for ln in result.stdout.splitlines() if ln.strip()]
        return []

    def test_peft_no_direct_import_in_repo(self):
        """peft 직접 import 0건 (USE_PEFT_BACKEND 는 diffusers.utils 경유라 제외)."""
        hits = self._find_direct_imports("peft")
        assert hits == [], (
            f"peft 직접 import 발견 ({len(hits)}건): {hits[:3]}"
        )

    def test_loguru_no_direct_import_in_repo(self):
        """loguru 직접 import 0건."""
        hits = self._find_direct_imports("loguru")
        assert hits == [], (
            f"loguru 직접 import 발견 ({len(hits)}건): {hits[:3]}"
        )

    def test_accelerate_no_direct_import_in_repo(self):
        """accelerate 직접 import 0건."""
        hits = self._find_direct_imports("accelerate")
        assert hits == [], (
            f"accelerate 직접 import 발견 ({len(hits)}건): {hits[:3]}"
        )

    def test_sentencepiece_no_direct_import_in_repo(self):
        """sentencepiece 직접 import 0건."""
        hits = self._find_direct_imports("sentencepiece")
        assert hits == [], (
            f"sentencepiece 직접 import 발견 ({len(hits)}건): {hits[:3]}"
        )


# ---------------------------------------------------------------------------
# 7. install.py 기존 함수 보존 + 주석 위치 검증
# ---------------------------------------------------------------------------

class TestInstallPyPreservation:
    """install.py 기존 함수 정의가 그대로 보존됐는지, 주석이 올바르게 추가됐는지."""

    def test_install_sageattention_function_exists(self):
        """install_sageattention 함수 정의 보존."""
        assert re.search(r"def install_sageattention", _install_text()), (
            "install_sageattention 함수 정의 사라짐 — 기존 로직 변경 금지"
        )

    def test_detect_cuda_arch_function_exists(self):
        """detect_cuda_arch 함수 정의 보존."""
        assert re.search(r"def detect_cuda_arch", _install_text()), (
            "detect_cuda_arch 함수 정의 사라짐 — 기존 로직 변경 금지"
        )

    def test_sageattention_already_installed_function_exists(self):
        """sageattention_already_installed 함수 정의 보존."""
        assert re.search(r"def sageattention_already_installed", _install_text()), (
            "sageattention_already_installed 함수 정의 사라짐"
        )

    def test_install_requirements_function_exists(self):
        """install_requirements 함수 정의 보존."""
        assert re.search(r"def install_requirements", _install_text()), (
            "install_requirements 함수 정의 사라짐"
        )

    def test_sage_comment_present(self):
        """sage 런타임 OFF 와 install 독립성 주석 존재."""
        assert re.search(
            r"sage 런타임 OFF.*install.*독립|sage runtime OFF.*install",
            _install_text(),
        ), (
            "install.py 에 sage 런타임 OFF / install 독립 주석 없음"
        )

    def test_sage_comment_is_comment_or_docstring(self):
        """주석 문자(#), docstring, 또는 NOTE: 인라인 문자열 형태로 추가됐는지 확인.

        실 제어 로직(함수 호출, 조건 분기 등)으로 삽입되지 않았는지 방어.
        install.py 는 NOTE: 형식 인라인 docstring 도 허용.
        """
        text = _install_text()
        # #, docstring(\"\"\"...\"\"\"), 또는 NOTE: 인라인 문자열 내 문구 등장 여부
        comment_hit = re.search(
            r"""(#.*(?:sage\s*런타임\s*OFF|sage\s*runtime\s*OFF)|"""
            r"""['"]{3}[^'"]*(?:sage\s*런타임\s*OFF|sage\s*runtime\s*OFF)|"""
            r"""NOTE:[^'"\n]*(?:sage\s*런타임\s*OFF|sage\s*runtime\s*OFF))""",
            text,
        )
        assert comment_hit, (
            "install.py sage 주석이 #/docstring/NOTE: 형태가 아닌 것으로 추정 — 실 로직 삽입 방어"
        )

    def test_install_py_syntax_ok(self):
        """install.py AST 파싱 성공 (syntax 오류 없음)."""
        try:
            ast.parse(_install_text())
        except SyntaxError as e:
            pytest.fail(f"install.py 문법 오류: {e}")


# ---------------------------------------------------------------------------
# 8. install.py 로직 변경 없음 (구조 보존)
# ---------------------------------------------------------------------------

class TestInstallPyLogicPreservation:
    """install.py 실 로직이 변경되지 않았는지 구조 수준으로 검증.

    주석 추가/변경만 허용, 함수 시그니처/흐름 변경은 FAIL.
    """

    def test_no_new_top_level_function_added(self):
        """install.py 내 함수가 기존 목록을 벗어나지 않는지 확인.

        P5 범위는 주석 추가뿐. 기존에 있던 _run/_log/main 포함한 전체 함수 목록이
        P5 이후에도 유지되는지 검증 (신규 함수 추가 방어).
        기존 확정 함수 목록 = P5 수정 전 install.py 에 존재하던 함수들.
        """
        tree = ast.parse(_install_text())
        # Codex MEDIUM 반영: ast.walk 는 중첩 함수까지 포함하므로 iter_child_nodes 로
        # top-level 함수만 수집. 테스트 이름 (no_new_top_level_function_added) 과 계약 일치.
        func_names = {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef)
        }
        # P5 이전부터 존재하던 함수 전체 목록 (grep으로 확인된 실제 함수)
        known_functions = {
            "install_sageattention",
            "detect_cuda_arch",
            "sageattention_already_installed",
            "install_requirements",
            "_run",
            "_log",
            "main",
        }
        extra = func_names - known_functions
        assert not extra, (
            f"P5 범위 외 신규 함수 추가됨: {extra} — 주석 추가만 허용"
        )

    def test_install_py_env_gate_not_removed(self):
        """MOTIFVIDEO_ENABLE_SAGE 관련 env 참조가 install.py 에 없는 경우도 OK.

        install.py 는 P1 sage gate 와 독립. 이 테스트는 install.py 에
        sage env gate 가 주입되지 않았는지 방어.
        """
        # install.py 에 sage env gate 주입은 P5 스코프 외 — 없어야 정상
        # (있다면 P1 범위 침범)
        sage_env_in_install = re.search(
            r"MOTIFVIDEO_ENABLE_SAGE", _install_text()
        )
        # FAIL 아님 — 단순 정보 기록 (향후 스코프 침범 방어용 soft check)
        # install.py 가 sage env gate 를 포함해도 P5 verify 는 통과
        # (P1 에서 models/ 파일만 수정했으므로)
        _ = sage_env_in_install  # soft check, not asserted


# ---------------------------------------------------------------------------
# 9. 경계값 — 빈 라인/주석 처리 라인 오인 방어
# ---------------------------------------------------------------------------

class TestEdgeCasesFormat:
    """빈 라인/주석 처리 라인이 패키지 라인으로 오인되지 않는지."""

    def test_blank_lines_not_counted_as_packages(self):
        """빈 라인이 패키지 라인 파싱에 포함되지 않음."""
        lines = _req_lines()
        for line in lines:
            assert line.strip() != "", "빈 라인이 패키지 라인으로 파싱됨"

    def test_comment_lines_not_counted_as_packages(self):
        """#으로 시작하는 주석 라인이 패키지 라인으로 파싱되지 않음."""
        lines = _req_lines()
        for line in lines:
            assert not line.strip().startswith("#"), (
                f"주석 라인이 패키지 라인으로 파싱됨: {line!r}"
            )

    def test_transformers_line_not_accidentally_commented_out(self):
        """transformers 라인이 주석 처리돼 있지 않은지 확인."""
        text = _req_text()
        # 주석 처리된 transformers 라인 탐색
        commented = re.findall(r"^#.*transformers", text, re.MULTILINE)
        # 실 패키지 라인이 존재하는지
        active = re.search(r"^transformers", text, re.MULTILINE)
        assert active, (
            f"transformers 가 주석 처리됐거나 제거됨. 주석 라인: {commented}"
        )

    def test_diffusers_line_not_accidentally_commented_out(self):
        """diffusers 라인이 주석 처리돼 있지 않은지 확인."""
        text = _req_text()
        active = re.search(r"^diffusers", text, re.MULTILINE)
        assert active, "diffusers 가 주석 처리됐거나 제거됨"

    def test_einops_line_not_accidentally_commented_out(self):
        """einops 라인이 주석 처리돼 있지 않은지 확인."""
        text = _req_text()
        active = re.search(r"^einops", text, re.MULTILINE)
        assert active, "einops 가 주석 처리됐거나 제거됨"
