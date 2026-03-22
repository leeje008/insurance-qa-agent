"""프롬프트 템플릿 테스트."""

from __future__ import annotations

from api.llm.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_VALIDATION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    format_context,
)


class TestPromptTemplates:
    """프롬프트 포맷팅 테스트."""

    def test_query_analysis_format(self) -> None:
        result = QUERY_ANALYSIS_PROMPT.format(question="보험금 청구 방법")
        assert "보험금 청구 방법" in result
        assert "keywords" in result

    def test_answer_generation_format(self) -> None:
        result = ANSWER_GENERATION_PROMPT.format(
            context="제1조 내용", question="질문",
        )
        assert "제1조 내용" in result
        assert "질문" in result

    def test_answer_validation_format(self) -> None:
        result = ANSWER_VALIDATION_PROMPT.format(
            context="조항", question="질문", answer="답변",
        )
        assert "답변" in result
        assert "is_valid" in result


class TestFormatContext:
    """format_context 테스트."""

    def test_single_article(self) -> None:
        articles = [{"number": "제1조", "title": "목적", "content": "본문"}]
        result = format_context(articles)
        assert "[1] 제1조 (목적)" in result
        assert "본문" in result

    def test_multiple_articles(self) -> None:
        articles = [
            {"number": "제1조", "title": "", "content": "A"},
            {"number": "제2조", "title": "적용", "content": "B"},
        ]
        result = format_context(articles)
        assert "[1] 제1조" in result
        assert "[2] 제2조 (적용)" in result
        assert "---" in result

    def test_empty(self) -> None:
        assert format_context([]) == ""
