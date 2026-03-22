"""api.routers.admin - 약관 업로드/관리 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_db
from api.schemas.admin_schema import (
    IngestRequest,
    IngestResponse,
    ProductListResponse,
    StatsResponse,
)
from api.services import admin_service

router = APIRouter()


@router.get("/products", response_model=ProductListResponse)
async def list_products(
    db: AsyncSession = Depends(get_db),
) -> ProductListResponse:
    """보험 상품 목록 조회."""
    return await admin_service.list_products(db)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """PDF 약관 인제스트."""
    try:
        return await admin_service.ingest_policy(
            pdf_path=req.pdf_path,
            insurer=req.insurer,
            product_name=req.product_name,
            insurance_type=req.insurance_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/stats", response_model=StatsResponse)
async def stats(
    db: AsyncSession = Depends(get_db),
) -> StatsResponse:
    """시스템 통계 조회."""
    return await admin_service.get_stats(db)
