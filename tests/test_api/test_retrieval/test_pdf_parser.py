"""PDF 파서 단위 테스트."""

from __future__ import annotations

from api.retrieval.pdf_parser import (
    _detect_level,
    parse_articles,
    parse_glossary,
)
from core.constants import ArticleLevel

# ---------------------------------------------------------------------------
# _detect_level 테스트
# ---------------------------------------------------------------------------

class TestDetectLevel:
    """조항 수준 감지 테스트."""

    def test_part(self) -> None:
        result = _detect_level("제1관 총칙")
        assert result is not None
        level, number, title = result
        assert level == ArticleLevel.PART
        assert number == "제1관"
        assert title == "총칙"

    def test_article(self) -> None:
        result = _detect_level("제3조 (보험금의 지급)")
        assert result is not None
        level, number, title = result
        assert level == ArticleLevel.ARTICLE
        assert number == "제3조"

    def test_article_no_parens(self) -> None:
        result = _detect_level("제1조 목적")
        assert result is not None
        level, number, title = result
        assert level == ArticleLevel.ARTICLE
        assert number == "제1조"

    def test_paragraph(self) -> None:
        result = _detect_level("① 이 약관에서 사용하는 용어")
        assert result is not None
        level, number, title = result
        assert level == ArticleLevel.PARAGRAPH
        assert number == "제1항"

    def test_paragraph_second(self) -> None:
        result = _detect_level("② 두 번째 항")
        assert result is not None
        assert result[1] == "제2항"

    def test_item(self) -> None:
        result = _detect_level("가. 피보험자가 사망한 경우")
        assert result is not None
        level, number, title = result
        assert level == ArticleLevel.ITEM
        assert number == "가호"

    def test_plain_text(self) -> None:
        result = _detect_level("이것은 일반 텍스트입니다.")
        assert result is None


# ---------------------------------------------------------------------------
# parse_articles 테스트
# ---------------------------------------------------------------------------

class TestParseArticles:
    """조항 파싱 테스트."""

    def test_basic_structure(self) -> None:
        text = """제1관 총칙
제1조 (목적)
이 약관은 보험계약에 관한 사항을 규정합니다.
제2조 (용어의 정의)
이 약관에서 사용하는 용어의 정의는 다음과 같습니다.
① 보험계약자란 보험회사와 계약을 체결하는 자를 말합니다.
② 피보험자란 보험사고의 대상이 되는 자를 말합니다."""

        articles = parse_articles(text)
        assert len(articles) >= 3  # 관, 조, 항

        # 관
        assert articles[0].level == ArticleLevel.PART
        assert articles[0].number == "제1관"

        # 조
        article1 = next(a for a in articles if a.number == "제1조")
        assert article1.level == ArticleLevel.ARTICLE

        article2 = next(a for a in articles if a.number == "제2조")
        assert article2.level == ArticleLevel.ARTICLE

    def test_parent_tracking(self) -> None:
        text = """제1조 (목적)
본 약관의 목적
① 첫 번째 항
② 두 번째 항"""

        articles = parse_articles(text)
        paragraphs = [a for a in articles if a.level == ArticleLevel.PARAGRAPH]
        for p in paragraphs:
            assert p.parent_number == "제1조"

    def test_cross_references(self) -> None:
        text = """제1조 (목적)
이 약관의 목적입니다.
제2조 (적용범위)
이 약관은 제1조의 목적에 따라 적용됩니다."""

        articles = parse_articles(text)
        art2 = next(a for a in articles if a.number == "제2조")
        assert "제1조" in art2.references

    def test_empty_text(self) -> None:
        assert parse_articles("") == []


# ---------------------------------------------------------------------------
# parse_glossary 테스트
# ---------------------------------------------------------------------------

class TestParseGlossary:
    """용어 사전 파싱 테스트."""

    def test_quoted_terms(self) -> None:
        text = """【별표】
「보험계약자」: 보험회사와 보험계약을 체결하는 자
「피보험자」: 보험사고의 대상이 되는 자"""

        glossary = parse_glossary(text)
        assert len(glossary) == 2
        assert glossary[0].term == "보험계약자"
        assert "보험회사" in glossary[0].definition

    def test_ran_pattern(self) -> None:
        text = """용어의 정의
'보험금'이란 보험사고 발생 시 지급되는 금액을 말합니다"""

        glossary = parse_glossary(text)
        assert len(glossary) == 1
        assert glossary[0].term == "보험금"

    def test_no_glossary(self) -> None:
        text = "제1조 (목적)\n이 약관은 목적입니다."
        assert parse_glossary(text) == []


# ---------------------------------------------------------------------------
# split_into_chunks (embedder에서 임포트) 테스트
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    """청크 분할 테스트."""

    def test_short_text_no_split(self) -> None:
        from api.retrieval.embedder import split_into_chunks
        chunks = split_into_chunks("짧은 텍스트", max_tokens=100)
        assert len(chunks) == 1

    def test_long_text_split(self) -> None:
        from api.retrieval.embedder import split_into_chunks
        long_text = "가" * 2000
        chunks = split_into_chunks(long_text, max_tokens=100, overlap_tokens=10)
        assert len(chunks) > 1

    def test_overlap(self) -> None:
        from api.retrieval.embedder import split_into_chunks
        text = "0123456789" * 100  # 1000 chars
        chunks = split_into_chunks(text, max_tokens=200, overlap_tokens=50)
        assert len(chunks) > 1
        # 오버랩으로 인해 청크 시작이 겹쳐야 함
        if len(chunks) >= 2:
            # 두 번째 청크의 시작 부분이 첫 번째 청크에 포함
            assert chunks[1][:10] in chunks[0]
