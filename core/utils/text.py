"""core.utils.text - 텍스트 처리 유틸리티.

약관 텍스트 정규화, 조항 번호 추출 등을 제공합니다.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """텍스트를 NFC 정규화 + 공백 정규화합니다.

    - Unicode NFC 정규화 (한국어 자모 분리 방지)
    - 연속 공백을 단일 공백으로 축소
    - 앞뒤 공백 제거
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_article_reference(ref: str) -> str:
    """약관 조항 참조를 정규화합니다.

    "제 1 조", "제1 조", "제1조" → "제1조"
    "제 3 항", "제3항" → "제3항"
    """
    ref = normalize_text(ref)
    # "제 N 조/항/호/관" 패턴에서 공백 제거
    ref = re.sub(r"제\s*(\d+)\s*(조|항|호|관)", r"제\1\2", ref)
    return ref


def extract_article_numbers(text: str) -> list[str]:
    """텍스트에서 약관 조항 번호를 추출합니다.

    Args:
        text: 입력 텍스트

    Returns:
        ["제1조", "제3항", ...] 형태의 리스트
    """
    text = normalize_text(text)
    pattern = r"제\s*\d+\s*(?:조|항|호|관)"
    matches = re.findall(pattern, text)
    return [clean_article_reference(m) for m in matches]


def is_korean(text: str) -> bool:
    """텍스트에 한국어가 포함되어 있는지 확인합니다."""
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", text))


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """텍스트를 지정 길이로 잘라냅니다.

    Args:
        text: 원본 텍스트
        max_length: 최대 길이 (suffix 포함)
        suffix: 잘린 경우 뒤에 붙는 문자열
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
