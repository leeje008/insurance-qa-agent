"""스키마 유효성 검증 테스트."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas.qa_schema import FeedbackRequest, QuestionRequest


class TestQuestionRequest:
    """QuestionRequest 스키마 테스트."""

    def test_valid_question(self) -> None:
        req = QuestionRequest(question="보험금 청구 절차가 어떻게 되나요?")
        assert req.question == "보험금 청구 절차가 어떻게 되나요?"
        assert req.product_id is None

    def test_with_product_id(self) -> None:
        req = QuestionRequest(question="보장 범위는?", product_id=1)
        assert req.product_id == 1

    def test_too_short(self) -> None:
        with pytest.raises(ValidationError):
            QuestionRequest(question="a")

    def test_empty(self) -> None:
        with pytest.raises(ValidationError):
            QuestionRequest(question="")


class TestFeedbackRequest:
    """FeedbackRequest 스키마 테스트."""

    def test_valid_score(self) -> None:
        for score in range(1, 6):
            req = FeedbackRequest(score=score)
            assert req.score == score

    def test_score_too_low(self) -> None:
        with pytest.raises(ValidationError):
            FeedbackRequest(score=0)

    def test_score_too_high(self) -> None:
        with pytest.raises(ValidationError):
            FeedbackRequest(score=6)
