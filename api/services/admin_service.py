"""api.services.admin_service - 약관 관리 비즈니스 로직."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import InsuranceProduct, PolicyArticle, QALog
from api.db.repositories import policy_repo
from api.ingest import ingest_pdf
from api.schemas.admin_schema import (
    IngestResponse,
    ProductListResponse,
    ProductResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)


async def list_products(session: AsyncSession) -> ProductListResponse:
    """전체 상품 목록 조회."""
    products = await policy_repo.list_products(session)
    return ProductListResponse(
        products=[ProductResponse.model_validate(p) for p in products],
        total=len(products),
    )


async def ingest_policy(
    *,
    pdf_path: str,
    insurer: str,
    product_name: str = "",
    insurance_type: str = "other",
) -> IngestResponse:
    """PDF 약관 인제스트."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 파일 없음: {pdf_path}")

    result = await ingest_pdf(
        path,
        insurer=insurer,
        product_name=product_name,
        insurance_type=insurance_type,
    )

    return IngestResponse(
        product_id=result.get("product_id", 0),
        articles=result.get("articles", 0),
        references=result.get("references", 0),
        glossary=result.get("glossary", 0),
    )


async def get_stats(session: AsyncSession) -> StatsResponse:
    """시스템 통계 조회."""
    product_count = await session.scalar(
        select(func.count()).select_from(InsuranceProduct)
    )
    article_count = await session.scalar(
        select(func.count()).select_from(PolicyArticle)
    )
    qa_count = await session.scalar(
        select(func.count()).select_from(QALog)
    )
    avg_conf = await session.scalar(
        select(func.avg(QALog.confidence)).where(QALog.confidence.is_not(None))
    )

    return StatsResponse(
        total_products=product_count or 0,
        total_articles=article_count or 0,
        total_qa_logs=qa_count or 0,
        avg_confidence=round(float(avg_conf), 3) if avg_conf else None,
    )
