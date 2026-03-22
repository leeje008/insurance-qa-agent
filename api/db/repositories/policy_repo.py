"""api.db.repositories.policy_repo - 약관/상품 데이터 접근."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.db.models import (
    ArticleReference,
    ArticleSubChunk,
    InsuranceGlossary,
    InsuranceProduct,
    PolicyArticle,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InsuranceProduct CRUD
# ---------------------------------------------------------------------------

async def create_product(
    session: AsyncSession,
    *,
    name: str,
    insurance_type: str,
    insurer: str,
    version: str | None = None,
    effective_date=None,
) -> InsuranceProduct:
    """보험 상품 생성."""
    product = InsuranceProduct(
        name=name,
        insurance_type=insurance_type,
        insurer=insurer,
        version=version,
        effective_date=effective_date,
    )
    session.add(product)
    await session.flush()
    logger.info("상품 생성: id=%d, name=%s", product.id, product.name)
    return product


async def get_product(session: AsyncSession, product_id: int) -> InsuranceProduct | None:
    """상품 ID로 조회."""
    return await session.get(InsuranceProduct, product_id)


async def get_product_by_name(
    session: AsyncSession, name: str, insurer: str
) -> InsuranceProduct | None:
    """상품명 + 보험사로 조회."""
    stmt = select(InsuranceProduct).where(
        InsuranceProduct.name == name,
        InsuranceProduct.insurer == insurer,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_products(session: AsyncSession) -> Sequence[InsuranceProduct]:
    """전체 상품 목록 조회."""
    stmt = select(InsuranceProduct).order_by(InsuranceProduct.id)
    result = await session.execute(stmt)
    return result.scalars().all()


# ---------------------------------------------------------------------------
# PolicyArticle CRUD
# ---------------------------------------------------------------------------

async def create_article(
    session: AsyncSession,
    *,
    product_id: int,
    level: str,
    number: str,
    title: str | None = None,
    content: str,
    parent_id: int | None = None,
    embedding: list[float] | None = None,
) -> PolicyArticle:
    """약관 조항 생성."""
    article = PolicyArticle(
        product_id=product_id,
        level=level,
        number=number,
        title=title,
        content=content,
        parent_id=parent_id,
        embedding=embedding,
    )
    session.add(article)
    await session.flush()
    return article


async def get_articles_by_product(
    session: AsyncSession, product_id: int
) -> Sequence[PolicyArticle]:
    """상품의 모든 조항 조회 (자식 포함)."""
    stmt = (
        select(PolicyArticle)
        .where(PolicyArticle.product_id == product_id)
        .options(selectinload(PolicyArticle.children))
        .order_by(PolicyArticle.id)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_article_by_number(
    session: AsyncSession, product_id: int, number: str
) -> PolicyArticle | None:
    """상품 + 조항 번호로 조회."""
    stmt = select(PolicyArticle).where(
        PolicyArticle.product_id == product_id,
        PolicyArticle.number == number,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# ArticleSubChunk CRUD
# ---------------------------------------------------------------------------

async def create_sub_chunk(
    session: AsyncSession,
    *,
    article_id: int,
    chunk_index: int,
    content: str,
    embedding: list[float] | None = None,
) -> ArticleSubChunk:
    """조항 서브 청크 생성."""
    chunk = ArticleSubChunk(
        article_id=article_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding,
    )
    session.add(chunk)
    await session.flush()
    return chunk


# ---------------------------------------------------------------------------
# ArticleReference CRUD
# ---------------------------------------------------------------------------

async def create_reference(
    session: AsyncSession,
    *,
    source_id: int,
    target_id: int,
) -> ArticleReference:
    """조항 간 참조 관계 생성."""
    ref = ArticleReference(source_id=source_id, target_id=target_id)
    session.add(ref)
    await session.flush()
    return ref


# ---------------------------------------------------------------------------
# InsuranceGlossary CRUD
# ---------------------------------------------------------------------------

async def create_glossary_term(
    session: AsyncSession,
    *,
    term: str,
    definition: str,
    product_id: int | None = None,
) -> InsuranceGlossary:
    """용어 사전 항목 생성."""
    entry = InsuranceGlossary(
        term=term, definition=definition, product_id=product_id,
    )
    session.add(entry)
    await session.flush()
    return entry


async def get_glossary_by_product(
    session: AsyncSession, product_id: int | None = None
) -> Sequence[InsuranceGlossary]:
    """상품별(또는 공통) 용어 사전 조회."""
    if product_id is not None:
        stmt = select(InsuranceGlossary).where(
            (InsuranceGlossary.product_id == product_id)
            | (InsuranceGlossary.product_id.is_(None))
        )
    else:
        stmt = select(InsuranceGlossary).where(
            InsuranceGlossary.product_id.is_(None)
        )
    result = await session.execute(stmt)
    return result.scalars().all()
