"""api.db.repositories.qa_log_repo - Q&A 로그 데이터 접근."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import QALog

logger = logging.getLogger(__name__)


async def create_qa_log(
    session: AsyncSession,
    *,
    question: str,
    answer: str,
    sources_json: str | None = None,
    confidence: float | None = None,
) -> QALog:
    """Q&A 로그 저장."""
    log = QALog(
        question=question,
        answer=answer,
        sources_json=sources_json,
        confidence=confidence,
    )
    session.add(log)
    await session.flush()
    logger.info("QA 로그 생성: id=%d", log.id)
    return log


async def get_qa_log(session: AsyncSession, log_id: int) -> QALog | None:
    """Q&A 로그 조회."""
    return await session.get(QALog, log_id)


async def list_qa_logs(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
) -> Sequence[QALog]:
    """Q&A 로그 목록 조회 (최신순)."""
    stmt = (
        select(QALog)
        .order_by(QALog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def update_feedback(
    session: AsyncSession,
    log_id: int,
    feedback_score: int,
) -> bool:
    """Q&A 로그에 사용자 피드백 점수 업데이트.

    Returns:
        업데이트 성공 여부.
    """
    stmt = (
        update(QALog)
        .where(QALog.id == log_id)
        .values(feedback_score=feedback_score)
    )
    result = await session.execute(stmt)
    if result.rowcount > 0:
        logger.info("피드백 업데이트: log_id=%d, score=%d", log_id, feedback_score)
        return True
    return False
