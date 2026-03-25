"""api.evaluation.dataset - Golden test set 관리."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import QALog

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


@dataclass
class EvalQuery:
    """평가용 쿼리."""

    question: str
    relevant_article_ids: set[int] = field(default_factory=set)
    product_id: int | None = None


async def build_from_feedback(
    session: AsyncSession,
    min_score: int = 4,
    limit: int = 200,
) -> list[EvalQuery]:
    """qa_logs에서 높은 피드백 점수의 데이터로 golden set 생성.

    feedback_score >= min_score인 로그에서 sources_json의 조항을 relevant로 간주.
    """
    stmt = (
        select(QALog)
        .where(QALog.feedback_score >= min_score)
        .where(QALog.sources_json.is_not(None))
        .order_by(QALog.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    logs = result.scalars().all()

    queries: list[EvalQuery] = []
    for log in logs:
        try:
            sources = json.loads(log.sources_json) if log.sources_json else []
        except json.JSONDecodeError:
            continue

        # sources에 article_id가 있으면 사용, 없으면 건너뜀
        article_ids = set()
        for src in sources:
            if "article_id" in src:
                article_ids.add(src["article_id"])

        if article_ids:
            queries.append(EvalQuery(
                question=log.question,
                relevant_article_ids=article_ids,
            ))

    logger.info("피드백 기반 golden set 생성: %d건", len(queries))
    return queries


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[EvalQuery]:
    """JSON 파일에서 golden test set 로드."""
    if not path.exists():
        logger.warning("Golden set 파일 없음: %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        EvalQuery(
            question=item["question"],
            relevant_article_ids=set(item.get("relevant_article_ids", [])),
            product_id=item.get("product_id"),
        )
        for item in data
    ]


def save_golden_set(queries: list[EvalQuery], path: Path = GOLDEN_SET_PATH) -> None:
    """Golden test set을 JSON 파일로 저장."""
    data = [
        {
            "question": q.question,
            "relevant_article_ids": sorted(q.relevant_article_ids),
            "product_id": q.product_id,
        }
        for q in queries
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Golden set 저장: %d건 → %s", len(data), path)
