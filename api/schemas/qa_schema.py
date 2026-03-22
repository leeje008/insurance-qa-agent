"""api.schemas.qa_schema - Q&A 요청/응답 스키마."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Q&A 질문 요청."""

    question: str = Field(..., min_length=2, max_length=1000, description="사용자 질문")
    product_id: int | None = Field(None, description="특정 상품 ID로 제한 (없으면 전체)")


class SourceInfo(BaseModel):
    """답변 근거 조항 정보."""

    number: str = Field(..., description="조항 번호 (예: 제3조)")
    title: str = Field("", description="조항 제목")


class AnswerResponse(BaseModel):
    """Q&A 답변 응답."""

    answer: str = Field(..., description="생성된 답변")
    confidence: float = Field(0.0, description="신뢰도 점수 (0~1)")
    sources: list[SourceInfo] = Field(default_factory=list, description="근거 조항 목록")
    log_id: int | None = Field(None, description="Q&A 로그 ID (피드백용)")


class FeedbackRequest(BaseModel):
    """사용자 피드백 요청."""

    score: int = Field(..., ge=1, le=5, description="피드백 점수 (1~5)")


class QALogResponse(BaseModel):
    """Q&A 로그 응답."""

    id: int
    question: str
    answer: str
    confidence: float | None
    feedback_score: int | None
    created_at: datetime

    model_config = {"from_attributes": True}
