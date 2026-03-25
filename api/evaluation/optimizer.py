"""api.evaluation.optimizer - 검색 파라미터 최적화 (RRF 가중치, 유사도 임계값)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from api.evaluation.dataset import EvalQuery
from api.evaluation.metrics import ndcg_at_k
from api.retrieval.hybrid_search import keyword_search, rrf_fusion, semantic_search

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """최적화 결과."""

    best_params: dict[str, float]
    best_score: float
    all_trials: list[dict]


class RetrievalOptimizer:
    """검색 파라미터 최적화."""

    async def optimize_weights(
        self,
        session: AsyncSession,
        golden_set: list[EvalQuery],
        *,
        steps: int = 11,
        k: int = 5,
    ) -> OptimizationResult:
        """SEMANTIC_WEIGHT / KEYWORD_WEIGHT 최적 조합 탐색.

        0.0~1.0 범위에서 steps개 간격으로 그리드 탐색.
        """
        from api.retrieval.embedder import EmbeddingClient

        trials: list[dict] = []
        best_score = -1.0
        best_params: dict[str, float] = {}

        # 쿼리별 검색 결과를 미리 가져옴 (weight만 변경하면서 재평가)
        embedder = EmbeddingClient()
        try:
            query_results = []
            for query in golden_set:
                if not query.relevant_article_ids:
                    continue
                query_embedding = await embedder.embed(query.question)
                sem = await semantic_search(
                    session, query_embedding, product_id=query.product_id,
                )
                kw = await keyword_search(
                    session, query.question.split()[:5], product_id=query.product_id,
                )
                query_results.append((sem, kw, query.relevant_article_ids))
        finally:
            await embedder.close()

        if not query_results:
            return OptimizationResult(best_params={}, best_score=0.0, all_trials=[])

        # 그리드 탐색
        for i in range(steps):
            sem_w = i / (steps - 1)
            kw_w = 1.0 - sem_w

            ranked_lists: list[list[int]] = []
            relevant_sets: list[set[int]] = []

            for sem_results, kw_results, relevant in query_results:
                fused = rrf_fusion(
                    sem_results, kw_results,
                    semantic_weight=sem_w, keyword_weight=kw_w, top_k=k,
                )
                ranked_lists.append([r.article_id for r in fused])
                relevant_sets.append(relevant)

            score = ndcg_at_k(ranked_lists, relevant_sets, k)

            trial = {"semantic_weight": round(sem_w, 2), "keyword_weight": round(kw_w, 2),
                      f"ndcg@{k}": round(score, 4)}
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = {"semantic_weight": sem_w, "keyword_weight": kw_w}

        logger.info(
            "가중치 최적화 완료: sem=%.2f, kw=%.2f, ndcg@%d=%.4f",
            best_params.get("semantic_weight", 0), best_params.get("keyword_weight", 0),
            k, best_score,
        )

        return OptimizationResult(
            best_params=best_params, best_score=best_score, all_trials=trials,
        )

    async def optimize_threshold(
        self,
        session: AsyncSession,
        golden_set: list[EvalQuery],
        *,
        min_threshold: float = 0.3,
        max_threshold: float = 0.9,
        steps: int = 13,
        k: int = 5,
    ) -> OptimizationResult:
        """SIMILARITY_THRESHOLD 최적값 탐색."""
        from api.retrieval.embedder import EmbeddingClient

        trials: list[dict] = []
        best_score = -1.0
        best_params: dict[str, float] = {}

        embedder = EmbeddingClient()
        try:
            query_embeddings = []
            for query in golden_set:
                if not query.relevant_article_ids:
                    continue
                emb = await embedder.embed(query.question)
                query_embeddings.append((emb, query))
        finally:
            await embedder.close()

        if not query_embeddings:
            return OptimizationResult(best_params={}, best_score=0.0, all_trials=[])

        for i in range(steps):
            threshold = min_threshold + (max_threshold - min_threshold) * i / (steps - 1)

            ranked_lists: list[list[int]] = []
            relevant_sets: list[set[int]] = []

            for emb, query in query_embeddings:
                sem = await semantic_search(
                    session, emb, product_id=query.product_id, threshold=threshold,
                )
                kw = await keyword_search(
                    session, query.question.split()[:5], product_id=query.product_id,
                )
                fused = rrf_fusion(sem, kw, top_k=k)
                ranked_lists.append([r.article_id for r in fused])
                relevant_sets.append(query.relevant_article_ids)

            score = ndcg_at_k(ranked_lists, relevant_sets, k)

            trial = {"threshold": round(threshold, 3), f"ndcg@{k}": round(score, 4)}
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = {"similarity_threshold": threshold}

        logger.info(
            "임계값 최적화 완료: threshold=%.3f, ndcg@%d=%.4f",
            best_params.get("similarity_threshold", 0), k, best_score,
        )

        return OptimizationResult(
            best_params=best_params, best_score=best_score, all_trials=trials,
        )
