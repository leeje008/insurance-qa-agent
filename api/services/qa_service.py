"""api.services.qa_service - Q&A 비즈니스 로직."""

from __future__ import annotations

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from api.agents.graph import get_pipeline
from api.cache import qa_cache
from api.db.repositories import qa_log_repo
from api.schemas.qa_schema import AnswerResponse, SourceInfo

logger = logging.getLogger(__name__)


async def ask_question(
    question: str,
    *,
    product_id: int | None = None,
    session: AsyncSession,
) -> AnswerResponse:
    """질문을 받아 RAG 파이프라인을 실행하고 답변을 반환.

    1. 캐시 확인 (동일 질문이면 캐시 반환)
    2. LangGraph 파이프라인 실행
    3. Q&A 로그 저장
    4. 캐시 저장 + 응답 반환
    """
    # 1. 캐시 확인
    cache_key = qa_cache.make_key(question, product_id)
    cached = qa_cache.get(cache_key)
    if cached is not None:
        logger.info("캐시 히트: %s", cache_key[:12])
        # 캐시된 답변도 로그에 기록
        log = await qa_log_repo.create_qa_log(
            session,
            question=question,
            answer=cached["answer"],
            sources_json=cached["sources_json"],
            confidence=cached["confidence"],
        )
        await session.commit()
        return AnswerResponse(
            answer=cached["answer"],
            confidence=cached["confidence"],
            sources=cached["sources"],
            log_id=log.id,
        )

    # 2. 파이프라인 실행
    pipeline = get_pipeline()

    result = await pipeline.ainvoke({
        "question": question,
        "product_id": product_id,
        "retry_count": 0,
    })

    answer = result.get("answer", "답변을 생성하지 못했습니다.")
    confidence = result.get("confidence", 0.0)
    sources_json = result.get("sources_json", "[]")

    # 소스 파싱
    try:
        sources_data = json.loads(sources_json)
        sources = [
            SourceInfo(number=s.get("number", ""), title=s.get("title", ""))
            for s in sources_data
        ]
    except (json.JSONDecodeError, TypeError):
        sources = []

    # 3. Q&A 로그 저장
    log = await qa_log_repo.create_qa_log(
        session,
        question=question,
        answer=answer,
        sources_json=sources_json,
        confidence=confidence,
    )
    await session.commit()

    # 4. 캐시 저장
    qa_cache.set(cache_key, {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "sources_json": sources_json,
    })

    return AnswerResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        log_id=log.id,
    )
