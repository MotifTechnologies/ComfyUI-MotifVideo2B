"""
P0.3 블라인드 검증: text_encoders/tokenizer_assets/ 정적 배치 검증
요구사항: tokenizer.json / tokenizer_config.json 이 레포 내 고정 배치,
          GemmaTokenizerFast 로드 및 동작 정상 여부 확인
"""

import hashlib
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
ASSETS_DIR = REPO_ROOT / "text_encoders" / "tokenizer_assets"

# HF 원본 정보 (2026-04-24 기준, P0.3 반입 시점 고정)
EXPECTED_TOKENIZER_SIZE = 33_378_248   # bytes
EXPECTED_CONFIG_SIZE = 780             # bytes
EXPECTED_TOKENIZER_SHA256 = (
    "3220c5bec16e78ddf8e59c08fecdede7e8d31820cb5b3e69f17fed6a29a0b30c"
)
EXPECTED_CONFIG_SHA256 = (
    "e20ed2c2c1398cc0d008bcc972d5df38554b7a44e975479118dcc999183f6e91"
)

# 02_context.md 명시 특수 토큰 id
EXPECTED_BOS = 2
EXPECTED_EOS = 1
EXPECTED_PAD = 0


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. 파일 존재
# ---------------------------------------------------------------------------

class TestFileExistence:
    def test_tokenizer_json_exists(self):
        """tokenizer.json 파일이 지정 경로에 존재해야 한다."""
        assert (ASSETS_DIR / "tokenizer.json").is_file(), (
            f"tokenizer.json 없음: {ASSETS_DIR / 'tokenizer.json'}"
        )

    def test_tokenizer_config_json_exists(self):
        """tokenizer_config.json 파일이 지정 경로에 존재해야 한다."""
        assert (ASSETS_DIR / "tokenizer_config.json").is_file(), (
            f"tokenizer_config.json 없음: {ASSETS_DIR / 'tokenizer_config.json'}"
        )

    def test_assets_dir_exists(self):
        """tokenizer_assets 디렉토리 자체가 존재해야 한다."""
        assert ASSETS_DIR.is_dir(), f"디렉토리 없음: {ASSETS_DIR}"


# ---------------------------------------------------------------------------
# 2. 파일 크기
# ---------------------------------------------------------------------------

class TestFileSize:
    def test_tokenizer_json_size(self):
        """tokenizer.json 크기가 HF 원본(33,378,248 bytes) 과 일치해야 한다."""
        actual = (ASSETS_DIR / "tokenizer.json").stat().st_size
        assert actual == EXPECTED_TOKENIZER_SIZE, (
            f"크기 불일치: 기대 {EXPECTED_TOKENIZER_SIZE}, 실제 {actual}"
        )

    def test_tokenizer_config_json_size(self):
        """tokenizer_config.json 크기가 HF 원본(780 bytes) 과 일치해야 한다."""
        actual = (ASSETS_DIR / "tokenizer_config.json").stat().st_size
        assert actual == EXPECTED_CONFIG_SIZE, (
            f"크기 불일치: 기대 {EXPECTED_CONFIG_SIZE}, 실제 {actual}"
        )


# ---------------------------------------------------------------------------
# 3. 로컬 sha256 (HF 원본과 바이트 동일성 — 오프라인 검증)
# ---------------------------------------------------------------------------

class TestLocalSha256:
    def test_tokenizer_json_sha256(self):
        """tokenizer.json sha256 이 HF 원본과 바이트 단위 동일해야 한다."""
        actual = _sha256(ASSETS_DIR / "tokenizer.json")
        assert actual == EXPECTED_TOKENIZER_SHA256, (
            f"sha256 불일치:\n  기대: {EXPECTED_TOKENIZER_SHA256}\n  실제: {actual}"
        )

    def test_tokenizer_config_json_sha256(self):
        """tokenizer_config.json sha256 이 HF 원본과 바이트 단위 동일해야 한다."""
        actual = _sha256(ASSETS_DIR / "tokenizer_config.json")
        assert actual == EXPECTED_CONFIG_SHA256, (
            f"sha256 불일치:\n  기대: {EXPECTED_CONFIG_SHA256}\n  실제: {actual}"
        )


# HF 원본과의 실시간 비교 테스트는 의도적으로 제외.
# 이유: (a) HF `resolve/main/...` 은 moving target 이어서 업스트림 업데이트 시
# 본 repo 가 이유 없이 깨진다, (b) 네트워크 필수 테스트를 pytest 기본 실행에
# 태우는 것은 hermetic CI 원칙에 어긋난다. 로컬 sha256 (§3) 이 반입 시점의
# 바이트 동일성을 이미 pin 한다.


# ---------------------------------------------------------------------------
# 5. GemmaTokenizerFast 로드
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    from transformers import GemmaTokenizerFast
    return GemmaTokenizerFast.from_pretrained(str(ASSETS_DIR))


class TestTokenizerLoad:
    def test_load_no_exception(self, tokenizer):
        """GemmaTokenizerFast.from_pretrained 이 예외 없이 성공해야 한다."""
        assert tokenizer is not None

    def test_load_correct_type(self, tokenizer):
        """로드된 객체가 GemmaTokenizerFast 인스턴스여야 한다."""
        from transformers import GemmaTokenizerFast
        assert isinstance(tokenizer, GemmaTokenizerFast)


# ---------------------------------------------------------------------------
# 6. 토큰화 동작 (정상 케이스 + 경계값)
# ---------------------------------------------------------------------------

class TestTokenization:
    def test_english_returns_dict_like(self, tokenizer):
        """영어 입력이 dict-like 객체(BatchEncoding)를 반환해야 한다."""
        out = tokenizer("hello")
        assert hasattr(out, "__getitem__"), f"dict-like 아님: {type(out)}"
        assert "input_ids" in out

    def test_english_input_ids_nonempty_ints(self, tokenizer):
        """영어 input_ids 가 1개 이상의 int 로 구성된 리스트여야 한다."""
        ids = tokenizer("hello")["input_ids"]
        assert isinstance(ids, list) and len(ids) >= 1
        assert all(isinstance(i, int) for i in ids), (
            f"int 아닌 원소 포함: {[type(x).__name__ for x in ids]}"
        )

    def test_korean_utf8_tokenized(self, tokenizer):
        """한국어 입력이 길이 1 이상의 input_ids 를 반환해야 한다."""
        ids = tokenizer("안녕하세요")["input_ids"]
        assert isinstance(ids, list) and len(ids) >= 1, (
            f"한국어 토큰화 실패: {ids}"
        )

    def test_empty_string_does_not_raise(self, tokenizer):
        """빈 문자열 입력이 예외 없이 처리되어야 한다 (경계값)."""
        out = tokenizer("")
        assert "input_ids" in out

    def test_long_text_does_not_raise(self, tokenizer):
        """긴 텍스트 입력이 예외 없이 처리되어야 한다 (경계값)."""
        long_text = "a" * 10000
        out = tokenizer(long_text)
        assert "input_ids" in out and len(out["input_ids"]) >= 1

    def test_special_chars_tokenized(self, tokenizer):
        """특수문자가 포함된 입력이 처리되어야 한다."""
        out = tokenizer("!@#$%^&*()")
        assert "input_ids" in out

    def test_none_raises_type_error(self, tokenizer):
        """None 입력 시 TypeError 가 발생해야 한다 (타입 불일치 경계값)."""
        with pytest.raises((TypeError, AttributeError, ValueError)):
            tokenizer(None)


# ---------------------------------------------------------------------------
# 7. 특수 토큰 id (02_context.md 명시값)
# ---------------------------------------------------------------------------

class TestSpecialTokenIds:
    def test_bos_token_id_is_2(self, tokenizer):
        """bos_token_id 가 2 여야 한다 (02_context.md 명시값)."""
        assert tokenizer.bos_token_id == EXPECTED_BOS, (
            f"bos_token_id 기대 {EXPECTED_BOS}, 실제 {tokenizer.bos_token_id}"
        )

    def test_eos_token_id_is_1(self, tokenizer):
        """eos_token_id 가 1 이여야 한다 (02_context.md 명시값)."""
        assert tokenizer.eos_token_id == EXPECTED_EOS, (
            f"eos_token_id 기대 {EXPECTED_EOS}, 실제 {tokenizer.eos_token_id}"
        )

    def test_pad_token_id_is_0(self, tokenizer):
        """pad_token_id 가 0 이여야 한다 (02_context.md 명시값)."""
        assert tokenizer.pad_token_id == EXPECTED_PAD, (
            f"pad_token_id 기대 {EXPECTED_PAD}, 실제 {tokenizer.pad_token_id}"
        )

    def test_special_token_ids_mutually_distinct(self, tokenizer):
        """bos/eos/pad 토큰 id 가 서로 달라야 한다."""
        ids = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
        assert len(set(ids)) == 3, f"특수 토큰 id 중복: {ids}"


# ---------------------------------------------------------------------------
# 8. 디렉토리 크기 (33 MB 범위 회귀 관점)
# ---------------------------------------------------------------------------

class TestDirectorySize:
    def test_assets_dir_size_approx_33mb(self):
        """tokenizer_assets/ 디렉토리 크기가 33 MB 범위(31~36 MB)여야 한다."""
        total = sum(
            f.stat().st_size for f in ASSETS_DIR.rglob("*") if f.is_file()
        )
        mb = total / (1024 * 1024)
        assert 31 <= mb <= 36, (
            f"디렉토리 크기 범위 벗어남: {mb:.1f} MB (기대: 31~36 MB)"
        )
