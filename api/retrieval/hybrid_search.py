"""api.retrieval.hybrid_search - PGVector 시맨틱 + 키워드 + RRF 하이브리드 검색."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import ArticleSubChunk, PolicyArticle
from core.constants import (
    KEYWORD_WEIGHT,
    RRF_K,
    SEMANTIC_WEIGHT,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 검색 결과 타입
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """하이브리드 검색 결과."""

    article_id: int
    product_id: int
    number: str
    title: str | None
    content: str
    level: str
    score: float
    source: str = ""  # "semantic" | "keyword" | "hybrid"
    sub_chunk_id: int | None = None


# ---------------------------------------------------------------------------
# 시맨틱 검색 (PGVector cosine similarity)
# ---------------------------------------------------------------------------

async def semantic_search(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    product_id: int | None = None,
    top_k: int = TOP_K_RESULTS,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[SearchResult]:
    """PGVector cosine similarity 기반 시맨틱 검색."""
    # cosine distance: 1 - cosine_similarity
    distance = PolicyArticle.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            PolicyArticle.id,
            PolicyArticle.product_id,
            PolicyArticle.number,
            PolicyArticle.title,
            PolicyArticle.content,
            PolicyArticle.level,
            (1 - distance).label("similarity"),
        )
        .where(PolicyArticle.embedding.is_not(None))
    )

    if product_id:
        stmt = stmt.where(PolicyArticle.product_id == product_id)

    stmt = stmt.order_by(distance).limit(top_k * 2)

    result = await session.execute(stmt)
    rows = result.all()

    results: list[SearchResult] = []
    for row in rows:
        sim = float(row.similarity)
        if sim < threshold:
            continue
        results.append(SearchResult(
            article_id=row.id,
            product_id=row.product_id,
            number=row.number,
            title=row.title,
            content=row.content,
            level=row.level,
            score=sim,
            source="semantic",
        ))

    # 서브청크 검색도 수행
    sub_distance = ArticleSubChunk.embedding.cosine_distance(query_embedding)
    sub_stmt = (
        select(
            ArticleSubChunk.id.label("sub_chunk_id"),
            ArticleSubChunk.article_id,
            ArticleSubChunk.content,
            PolicyArticle.product_id,
            PolicyArticle.number,
            PolicyArticle.title,
            PolicyArticle.level,
            (1 - sub_distance).label("similarity"),
        )
        .join(PolicyArticle, ArticleSubChunk.article_id == PolicyArticle.id)
        .where(ArticleSubChunk.embedding.is_not(None))
    )
    if product_id:
        sub_stmt = sub_stmt.where(PolicyArticle.product_id == product_id)

    sub_stmt = sub_stmt.order_by(sub_distance).limit(top_k)

    sub_result = await session.execute(sub_stmt)
    for row in sub_result.all():
        sim = float(row.similarity)
        if sim < threshold:
            continue
        results.append(SearchResult(
            article_id=row.article_id,
            product_id=row.product_id,
            number=row.number,
            title=row.title,
            content=row.content,
            level=row.level,
            score=sim,
            source="semantic",
            sub_chunk_id=row.sub_chunk_id,
        ))

    return results[:top_k]


# ---------------------------------------------------------------------------
# 키워드 검색 (ILIKE 기반)
# ---------------------------------------------------------------------------

async def keyword_search(
    session: AsyncSession,
    keywords: list[str],
    *,
    product_id: int | None = None,
    top_k: int = TOP_K_RESULTS,
) -> list[SearchResult]:
    """키워드 ILIKE 기반 검색."""
    if not keywords:
        return []

    # 각 키워드에 대해 OR 조건으로 검색
    conditions = [
        PolicyArticle.content.ilike(f"%{kw}%") for kw in keywords
    ]
    from sqlalchemy import or_

    stmt = (
        select(
            PolicyArticle.id,
            PolicyArticle.product_id,
            PolicyArticle.number,
            PolicyArticle.title,
            PolicyArticle.content,
            PolicyArticle.level,
        )
        .where(or_(*conditions))
    )

    if product_id:
        stmt = stmt.where(PolicyArticle.product_id == product_id)

    stmt = stmt.limit(top_k * 2)

    result = await session.execute(stmt)
    rows = result.all()

    results: list[SearchResult] = []
    for row in rows:
        # 키워드 매칭 점수: 매칭된 키워드 수 / 전체 키워드 수
        matched = sum(
            1 for kw in keywords if kw.lower() in row.content.lower()
        )
        score = matched / len(keywords) if keywords else 0

        results.append(SearchResult(
            article_id=row.id,
            product_id=row.product_id,
            number=row.number,
            title=row.title,
            content=row.content,
            level=row.level,
            score=score,
            source="keyword",
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def rrf_fusion(
    semantic_results: list[SearchResult],
    keyword_results: list[SearchResult],
    *,
    k: int = RRF_K,
    semantic_weight: float = SEMANTIC_WEIGHT,
    keyword_weight: float = KEYWORD_WEIGHT,
    top_k: int = TOP_K_RESULTS,
) -> list[SearchResult]:
    """RRF 점수 융합.

    RRF(rank) = 1 / (k + rank)
    최종 점수 = semantic_weight * rrf_semantic + keyword_weight * rrf_keyword
    """
    scores: dict[int, float] = {}
    result_map: dict[int, SearchResult] = {}

    for rank, r in enumerate(semantic_results):
        rrf_score = semantic_weight * (1.0 / (k + rank + 1))
        scores[r.article_id] = scores.get(r.article_id, 0) + rrf_score
        result_map[r.article_id] = r

    for rank, r in enumerate(keyword_results):
        rrf_score = keyword_weight * (1.0 / (k + rank + 1))
        scores[r.article_id] = scores.get(r.article_id, 0) + rrf_score
        if r.article_id not in result_map:
            result_map[r.article_id] = r

    # 점수순 정렬
    sorted_ids = sorted(scores, key=lambda aid: scores[aid], reverse=True)

    results: list[SearchResult] = []
    for aid in sorted_ids[:top_k]:
        r = result_map[aid]
        results.append(SearchResult(
            article_id=r.article_id,
            product_id=r.product_id,
            number=r.number,
            title=r.title,
            content=r.content,
            level=r.level,
            score=scores[aid],
            source="hybrid",
            sub_chunk_id=r.sub_chunk_id,
        ))

    return results


# ---------------------------------------------------------------------------
# 하이브리드 검색 (통합)
# ---------------------------------------------------------------------------

async def hybrid_search(
    session: AsyncSession,
    query_embedding: list[float],
    keywords: list[str],
    *,
    product_id: int | None = None,
    top_k: int = TOP_K_RESULTS,
) -> list[SearchResult]:
    """시맨틱 + 키워드 + RRF 하이브리드 검색.

    Args:
        session: DB 세션.
        query_embedding: 질문 임베딩 벡터.
        keywords: 추출된 키워드 목록.
        product_id: 특정 상품으로 제한 (None이면 전체).
        top_k: 반환할 최대 결과 수.

    Returns:
        RRF 점수순 SearchResult 리스트.
    """
    sem_results = await semantic_search(
        session, query_embedding, product_id=product_id, top_k=top_k,
    )
    kw_results = await keyword_search(
        session, keywords, product_id=product_id, top_k=top_k,
    )

    logger.info(
        "검색 결과: semantic=%d, keyword=%d",
        len(sem_results), len(kw_results),
    )

    if not sem_results and not kw_results:
        return []

    if not sem_results:
        return kw_results[:top_k]

    if not kw_results:
        return sem_results[:top_k]

    return rrf_fusion(sem_results, kw_results, top_k=top_k)
