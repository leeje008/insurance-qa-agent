"""api.retrieval.query_expansion - Pseudo-Relevance Feedback (PRF) 쿼리 확장.

검색된 상위 문서에서 핵심 용어를 추출하여 키워드 리스트를 확장.
학습 데이터 불필요한 클래식 IR 기법.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# 한국어 불용어 (보험약관에서 흔한 기능어)
_STOP_WORDS = frozenset(
    "의 가 이 은 는 을 를 에 에서 와 과 도 로 으로 만 까지 부터 에게 한테 "
    "그 이 저 것 수 등 및 또는 또한 따라 경우 때 관한 대한 위한 "
    "있다 없다 한다 하는 하여 하고 되는 된다 되어 있는 없는 "
    "제 조 항 호 관 의하여 따르면 규정 규정에".split()
)

# 최소 용어 길이
_MIN_TERM_LENGTH = 2


def extract_expansion_terms(
    documents: list[dict],
    *,
    top_k_terms: int = 5,
    max_docs: int = 3,
) -> list[str]:
    """상위 문서에서 핵심 확장 용어를 추출.

    Args:
        documents: 검색된 문서 리스트 (score 순).
        top_k_terms: 반환할 최대 용어 수.
        max_docs: 사용할 상위 문서 수.

    Returns:
        확장 용어 리스트.
    """
    if not documents:
        return []

    # 상위 문서에서 텍스트 수집
    texts = []
    for doc in documents[:max_docs]:
        content = doc.get("content", "")
        title = doc.get("title", "")
        if title:
            texts.append(title)
        texts.append(content)

    combined = " ".join(texts)

    # 한국어 단어 추출 (2글자 이상 한글 단어)
    words = re.findall(r"[가-힣]{2,}", combined)

    # 불용어 제거 + 빈도 계산
    counter: Counter[str] = Counter()
    for word in words:
        if word not in _STOP_WORDS and len(word) >= _MIN_TERM_LENGTH:
            counter[word] += 1

    # 상위 빈도 용어 반환
    expansion_terms = [term for term, _ in counter.most_common(top_k_terms)]

    logger.debug("PRF 확장 용어: %s (from %d docs)", expansion_terms, min(len(documents), max_docs))

    return expansion_terms


def expand_keywords(
    original_keywords: list[str],
    documents: list[dict],
    *,
    max_expansion: int = 5,
) -> list[str]:
    """기존 키워드에 PRF 확장 용어를 추가.

    중복 제거 후 원본 키워드 + 확장 키워드를 반환.
    """
    expansion = extract_expansion_terms(documents, top_k_terms=max_expansion)

    # 원본 키워드와 중복 제거
    existing = set(kw.lower() for kw in original_keywords)
    new_terms = [t for t in expansion if t.lower() not in existing]

    expanded = original_keywords + new_terms
    if new_terms:
        logger.info("키워드 확장: %s → +%s", original_keywords[:3], new_terms)

    return expanded
