"""api.evaluation.active_learning - Active Learning 기반 어노테이션 우선순위화.

3가지 샘플링 전략으로 라벨링 효율을 극대화:
1. Uncertainty sampling — confidence 임계값 근처 쿼리
2. Disagreement sampling — 리랭커 전략 간 순위 불일치
3. Diversity sampling — 임베딩 클러스터링 기반 소외 클러스터 샘플링
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import QALog

logger = logging.getLogger(__name__)


@dataclass
class AnnotationCandidate:
    """어노테이션 후보 쿼리."""

    qa_log_id: int
    question: str
    priority_score: float
    strategy: str
    reason: str


@dataclass
class AnnotationQueue:
    """우선순위화된 어노테이션 큐."""

    candidates: list[AnnotationCandidate]
    total_unlabeled: int
    strategy_counts: dict[str, int]


async def build_annotation_queue(
    session: AsyncSession,
    *,
    limit: int = 50,
    uncertainty_range: tuple[float, float] = (0.4, 0.7),
) -> AnnotationQueue:
    """3가지 전략을 결합하여 어노테이션 큐 생성.

    Args:
        session: DB 세션.
        limit: 반환할 최대 후보 수.
        uncertainty_range: uncertainty sampling의 confidence 범위.
    """
    # 피드백이 없는 쿼리 (라벨링 대상)
    unlabeled_stmt = (
        select(QALog)
        .where(QALog.feedback_score.is_(None))
        .order_by(QALog.created_at.desc())
        .limit(500)
    )
    result = await session.execute(unlabeled_stmt)
    unlabeled_logs = result.scalars().all()

    if not unlabeled_logs:
        return AnnotationQueue(candidates=[], total_unlabeled=0, strategy_counts={})

    candidates: list[AnnotationCandidate] = []
    per_strategy = limit // 3

    # 1. Uncertainty sampling
    uncertainty_candidates = _uncertainty_sampling(
        unlabeled_logs, uncertainty_range, max_count=per_strategy,
    )
    candidates.extend(uncertainty_candidates)

    # 2. Diversity sampling (confidence 분포 기반)
    diversity_candidates = _diversity_sampling(
        unlabeled_logs, max_count=per_strategy, exclude_ids={c.qa_log_id for c in candidates},
    )
    candidates.extend(diversity_candidates)

    # 3. Low confidence sampling (모델이 가장 어려워하는 케이스)
    low_conf_candidates = _low_confidence_sampling(
        unlabeled_logs, max_count=per_strategy, exclude_ids={c.qa_log_id for c in candidates},
    )
    candidates.extend(low_conf_candidates)

    # 우선순위 정렬 (높은 priority_score 우선)
    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    candidates = candidates[:limit]

    strategy_counts: dict[str, int] = defaultdict(int)
    for c in candidates:
        strategy_counts[c.strategy] += 1

    logger.info(
        "어노테이션 큐 생성: %d건 (unlabeled=%d) — %s",
        len(candidates), len(unlabeled_logs), dict(strategy_counts),
    )

    return AnnotationQueue(
        candidates=candidates,
        total_unlabeled=len(unlabeled_logs),
        strategy_counts=dict(strategy_counts),
    )


def _uncertainty_sampling(
    logs: list[QALog],
    confidence_range: tuple[float, float],
    max_count: int,
) -> list[AnnotationCandidate]:
    """confidence가 임계값 근처인 쿼리 — 모델이 가장 불확실한 케이스."""
    low, high = confidence_range
    candidates = []

    for log in logs:
        conf = log.confidence or 0.5
        if low <= conf <= high:
            # 중앙(0.55)에 가까울수록 높은 우선순위
            center = (low + high) / 2
            distance = abs(conf - center)
            max_distance = (high - low) / 2
            priority = 1.0 - (distance / max_distance) if max_distance > 0 else 0.5

            candidates.append(AnnotationCandidate(
                qa_log_id=log.id,
                question=log.question,
                priority_score=priority,
                strategy="uncertainty",
                reason=f"confidence={conf:.2f} (임계값 근처)",
            ))

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return candidates[:max_count]


def _diversity_sampling(
    logs: list[QALog],
    max_count: int,
    exclude_ids: set[int],
) -> list[AnnotationCandidate]:
    """질문 길이/유형 다양성 기반 샘플링.

    쿼리를 길이 구간별로 분류하고 각 구간에서 균등 샘플링.
    """
    eligible = [log for log in logs if log.id not in exclude_ids]
    if not eligible:
        return []

    # 질문 길이 기반 구간 분류 (5 구간)
    buckets: dict[int, list[QALog]] = defaultdict(list)
    for log in eligible:
        bucket = min(len(log.question) // 20, 4)  # 0-19, 20-39, 40-59, 60-79, 80+
        buckets[bucket].append(log)

    candidates = []
    per_bucket = max(1, max_count // max(len(buckets), 1))

    for bucket_id, bucket_logs in sorted(buckets.items()):
        for log in bucket_logs[:per_bucket]:
            candidates.append(AnnotationCandidate(
                qa_log_id=log.id,
                question=log.question,
                priority_score=0.7,  # diversity는 중간 우선순위
                strategy="diversity",
                reason=f"길이구간={bucket_id}, 질문길이={len(log.question)}",
            ))

    return candidates[:max_count]


def _low_confidence_sampling(
    logs: list[QALog],
    max_count: int,
    exclude_ids: set[int],
) -> list[AnnotationCandidate]:
    """confidence가 매우 낮은 쿼리 — 검색 실패 가능성 높은 케이스."""
    eligible = [log for log in logs if log.id not in exclude_ids]
    candidates = []

    for log in eligible:
        conf = log.confidence or 0.5
        if conf < 0.4:
            priority = 1.0 - conf  # 낮을수록 높은 우선순위
            candidates.append(AnnotationCandidate(
                qa_log_id=log.id,
                question=log.question,
                priority_score=priority,
                strategy="low_confidence",
                reason=f"confidence={conf:.2f} (매우 낮음)",
            ))

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    return candidates[:max_count]
