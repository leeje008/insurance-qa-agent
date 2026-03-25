"""api.evaluation.runner - 평가 실행기 (리랭커 벤치마크 포함)."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from api.evaluation.dataset import EvalQuery
from api.evaluation.metrics import compute_all_metrics
from api.retrieval.reranker import get_reranker, list_rerankers

logger = logging.getLogger(__name__)


async def evaluate_retrieval(
    session: AsyncSession,
    golden_set: list[EvalQuery],
    *,
    rerank_strategy: str | None = None,
    k: int = 5,
) -> dict[str, float]:
    """golden set으로 검색 품질을 평가.

    Args:
        session: DB 세션.
        golden_set: 평가용 쿼리 목록.
        rerank_strategy: 리랭커 전략 (None이면 현재 설정 사용).
        k: 상위 K개 기준 평가.

    Returns:
        메트릭 딕셔너리 (mrr, precision@k, hit_rate@k, ndcg@k).
    """
    from api.retrieval.embedder import EmbeddingClient
    from api.retrieval.hybrid_search import hybrid_search

    reranker = get_reranker(rerank_strategy)

    ranked_lists: list[list[int]] = []
    relevant_sets: list[set[int]] = []

    embedder = EmbeddingClient()
    try:
        for query in golden_set:
            if not query.relevant_article_ids:
                continue

            # 임베딩 생성
            query_embedding = await embedder.embed(query.question)

            # 하이브리드 검색
            results = await hybrid_search(
                session, query_embedding, query.question.split()[:5],
                product_id=query.product_id,
            )

            documents = [
                {
                    "article_id": r.article_id, "product_id": r.product_id,
                    "number": r.number, "title": r.title,
                    "content": r.content, "level": r.level, "score": r.score,
                }
                for r in results
            ]

            # 리랭킹
            reranked = await reranker.rerank(query.question, documents, top_k=k)
            ranked_ids = [r.document["article_id"] for r in reranked]

            ranked_lists.append(ranked_ids)
            relevant_sets.append(query.relevant_article_ids)
    finally:
        await embedder.close()

    if not ranked_lists:
        return {"mrr": 0.0, f"precision@{k}": 0.0, f"hit_rate@{k}": 0.0, f"ndcg@{k}": 0.0}

    metrics = compute_all_metrics(ranked_lists, relevant_sets, k=k)
    metrics["rerank_strategy"] = reranker.name
    logger.info("평가 완료 [%s]: %s", reranker.name, metrics)
    return metrics


async def benchmark_rerankers(
    session: AsyncSession,
    golden_set: list[EvalQuery],
    strategies: list[str] | None = None,
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """모든 리랭커 전략을 golden set으로 벤치마크.

    Returns:
        {"cross_encoder": {"mrr": 0.82, ...}, "noop": {"mrr": 0.65, ...}, ...}
    """
    strategies = strategies or list_rerankers()
    results: dict[str, dict[str, float]] = {}

    for strategy in strategies:
        try:
            metrics = await evaluate_retrieval(
                session, golden_set, rerank_strategy=strategy, k=k,
            )
            results[strategy] = metrics
        except Exception:
            logger.warning("벤치마크 실패: %s", strategy, exc_info=True)
            results[strategy] = {"error": True}

    return results
