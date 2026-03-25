"""api.evaluation.failure_analysis - Retrieval 실패 분석 프레임워크.

6가지 에러 택소노미 기반 실패 모드 분류:
1. embedding_mismatch — 키워드 매칭되지만 시맨틱 점수 낮음
2. missing_content — DB에 관련 콘텐츠 없음
3. chunking_error — 정답 조항은 있지만 오답 청크가 상위
4. intent_confusion — 키워드 매칭되지만 의도 불일치
5. cross_reference_gap — 답변에 여러 조항 필요하지만 1개만 검색
6. terminology_gap — 구어체 vs 전문용어 불일치
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import QALog, RetrievalLog

logger = logging.getLogger(__name__)


class FailureMode(StrEnum):
    """검색 실패 모드."""

    EMBEDDING_MISMATCH = "embedding_mismatch"
    MISSING_CONTENT = "missing_content"
    CHUNKING_ERROR = "chunking_error"
    INTENT_CONFUSION = "intent_confusion"
    CROSS_REFERENCE_GAP = "cross_reference_gap"
    TERMINOLOGY_GAP = "terminology_gap"
    UNKNOWN = "unknown"


@dataclass
class FailureCase:
    """개별 실패 케이스."""

    qa_log_id: int
    question: str
    feedback_score: int
    confidence: float
    failure_modes: list[FailureMode] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class FailureReport:
    """실패 분석 리포트."""

    total_failures: int
    mode_counts: dict[str, int]
    mode_percentages: dict[str, float]
    cases: list[FailureCase]
    recommendations: list[str]


async def analyze_failures(
    session: AsyncSession,
    *,
    max_feedback_score: int = 2,
    limit: int = 200,
) -> FailureReport:
    """qa_logs에서 실패 케이스를 수집하고 6가지 모드로 분류.

    Args:
        session: DB 세션.
        max_feedback_score: 실패로 간주할 최대 피드백 점수.
        limit: 분석할 최대 케이스 수.
    """
    # 1. 실패 케이스 수집 (낮은 피드백 또는 낮은 confidence)
    stmt = (
        select(QALog)
        .where(
            (QALog.feedback_score <= max_feedback_score)
            | (QALog.confidence is not None and QALog.confidence < 0.4),  # noqa: E711
        )
        .order_by(QALog.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    failed_logs = result.scalars().all()

    if not failed_logs:
        return FailureReport(
            total_failures=0,
            mode_counts={},
            mode_percentages={},
            cases=[],
            recommendations=["분석할 실패 케이스가 없습니다."],
        )

    cases: list[FailureCase] = []

    for log in failed_logs:
        case = FailureCase(
            qa_log_id=log.id,
            question=log.question,
            feedback_score=log.feedback_score or 0,
            confidence=log.confidence or 0.0,
        )

        # 해당 Q&A의 retrieval_logs 조회
        rl_stmt = (
            select(RetrievalLog)
            .where(RetrievalLog.qa_log_id == log.id)
            .order_by(RetrievalLog.rank_before_rerank)
        )
        rl_result = await session.execute(rl_stmt)
        retrieval_logs = rl_result.scalars().all()

        # 실패 모드 분류
        modes = _classify_failure(log, retrieval_logs)
        case.failure_modes = modes
        case.details = _build_details(log, retrieval_logs)
        cases.append(case)

    # 집계
    mode_counter: Counter[str] = Counter()
    for case in cases:
        for mode in case.failure_modes:
            mode_counter[mode] += 1

    total = len(cases)
    mode_counts = dict(mode_counter.most_common())
    mode_percentages = {
        mode: round(count / total * 100, 1)
        for mode, count in mode_counts.items()
    }

    recommendations = _generate_recommendations(mode_percentages)

    logger.info(
        "실패 분석 완료: %d건 — %s",
        total, mode_counts,
    )

    return FailureReport(
        total_failures=total,
        mode_counts=mode_counts,
        mode_percentages=mode_percentages,
        cases=cases,
        recommendations=recommendations,
    )


def _classify_failure(
    log: QALog,
    retrieval_logs: list[RetrievalLog],
) -> list[FailureMode]:
    """개별 실패 케이스의 실패 모드를 분류."""
    modes: list[FailureMode] = []

    if not retrieval_logs:
        # 검색 결과 자체가 없음
        modes.append(FailureMode.MISSING_CONTENT)
        return modes

    # 점수 분석
    semantic_scores = [
        rl.semantic_score for rl in retrieval_logs
        if rl.semantic_score is not None
    ]
    keyword_scores = [
        rl.keyword_score for rl in retrieval_logs
        if rl.keyword_score is not None
    ]
    used_count = sum(1 for rl in retrieval_logs if rl.was_used_in_answer)

    max_semantic = max(semantic_scores) if semantic_scores else 0
    max_keyword = max(keyword_scores) if keyword_scores else 0
    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0

    # 1. Embedding 불일치: 키워드 점수 높지만 시맨틱 점수 낮음
    if max_keyword > 0.5 and max_semantic < 0.5:
        modes.append(FailureMode.EMBEDDING_MISMATCH)

    # 2. 콘텐츠 부재: 모든 점수가 매우 낮음
    if max_semantic < 0.3 and max_keyword < 0.2:
        modes.append(FailureMode.MISSING_CONTENT)

    # 3. 청킹 오류: 리랭킹 전후 순위 변화가 극심
    rank_changes = [
        abs((rl.rank_before_rerank or 0) - (rl.rank_after_rerank or 0))
        for rl in retrieval_logs
        if rl.rank_before_rerank is not None and rl.rank_after_rerank is not None
    ]
    if rank_changes and max(rank_changes) >= 5:
        modes.append(FailureMode.CHUNKING_ERROR)

    # 4. 의도 혼동: 키워드 매칭은 되지만 confidence가 매우 낮음
    if max_keyword > 0.4 and (log.confidence or 0) < 0.3:
        modes.append(FailureMode.INTENT_CONFUSION)

    # 5. 교차참조 누락: 사용된 문서가 1개뿐인데 confidence 낮음
    if used_count == 1 and (log.confidence or 0) < 0.5:
        modes.append(FailureMode.CROSS_REFERENCE_GAP)

    # 6. 용어 갭: 시맨틱 점수 중간인데 키워드 점수 0
    if avg_semantic > 0.4 and max_keyword < 0.1:
        modes.append(FailureMode.TERMINOLOGY_GAP)

    if not modes:
        modes.append(FailureMode.UNKNOWN)

    return modes


def _build_details(log: QALog, retrieval_logs: list[RetrievalLog]) -> dict:
    """실패 케이스의 상세 정보."""
    semantic_scores = [
        rl.semantic_score for rl in retrieval_logs if rl.semantic_score is not None
    ]
    keyword_scores = [
        rl.keyword_score for rl in retrieval_logs if rl.keyword_score is not None
    ]

    return {
        "num_retrieved": len(retrieval_logs),
        "num_used_in_answer": sum(1 for rl in retrieval_logs if rl.was_used_in_answer),
        "max_semantic_score": max(semantic_scores) if semantic_scores else 0,
        "max_keyword_score": max(keyword_scores) if keyword_scores else 0,
        "confidence": log.confidence or 0,
        "rerank_strategy": retrieval_logs[0].rerank_strategy if retrieval_logs else None,
    }


def _generate_recommendations(mode_percentages: dict[str, float]) -> list[str]:
    """실패 모드 비율에 따른 개선 권고."""
    recs: list[str] = []

    for mode, pct in sorted(mode_percentages.items(), key=lambda x: x[1], reverse=True):
        if pct < 5:
            continue

        if mode == FailureMode.EMBEDDING_MISMATCH:
            recs.append(
                f"[{pct}%] Embedding 불일치 — 도메인 임베딩 파인튜닝 또는 "
                "보험 용어 사전 기반 쿼리 확장 권장",
            )
        elif mode == FailureMode.MISSING_CONTENT:
            recs.append(
                f"[{pct}%] 콘텐츠 부재 — 크롤러로 추가 약관 PDF 수집 필요",
            )
        elif mode == FailureMode.CHUNKING_ERROR:
            recs.append(
                f"[{pct}%] 청킹 오류 — 조항 유형별 청크 크기 최적화 또는 "
                "시맨틱 경계 기반 청킹 적용 권장",
            )
        elif mode == FailureMode.INTENT_CONFUSION:
            recs.append(
                f"[{pct}%] 의도 혼동 — Intent 분류 모델 학습 또는 "
                "intent별 검색 전략 분기 권장",
            )
        elif mode == FailureMode.CROSS_REFERENCE_GAP:
            recs.append(
                f"[{pct}%] 교차참조 누락 — ArticleReference 기반 "
                "관련 조항 자동 확장 권장",
            )
        elif mode == FailureMode.TERMINOLOGY_GAP:
            recs.append(
                f"[{pct}%] 용어 갭 — InsuranceGlossary 기반 "
                "쿼리 용어 치환/확장 권장",
            )

    if not recs:
        recs.append("특정 실패 모드가 지배적이지 않음 — 전반적 품질 개선 필요")

    return recs
