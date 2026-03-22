"""api.schemas.admin_schema - 관리자 요청/응답 스키마."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class ProductResponse(BaseModel):
    """보험 상품 응답."""

    id: int
    name: str
    insurance_type: str
    insurer: str
    version: str | None
    effective_date: date | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProductListResponse(BaseModel):
    """상품 목록 응답."""

    products: list[ProductResponse]
    total: int


class IngestRequest(BaseModel):
    """PDF 인제스트 요청."""

    pdf_path: str = Field(..., description="PDF 파일 경로 (서버 내 경로)")
    insurer: str = Field(..., description="보험사명")
    product_name: str = Field("", description="상품명 (비어있으면 파일명 사용)")
    insurance_type: str = Field("other", description="보험 유형")


class IngestResponse(BaseModel):
    """인제스트 결과 응답."""

    product_id: int
    articles: int
    references: int
    glossary: int


class StatsResponse(BaseModel):
    """시스템 통계 응답."""

    total_products: int
    total_articles: int
    total_qa_logs: int
    avg_confidence: float | None
