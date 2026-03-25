"""api.evaluation.metrics - 검색 품질 평가 메트릭 (MRR, NDCG, Hit Rate, Precision)."""

from __future__ import annotations

import math


def mrr(ranked_lists: list[list[int]], relevant_sets: list[set[int]]) -> float:
    """Mean Reciprocal Rank.

    각 쿼리에 대해 첫 번째 관련 문서의 역순위 평균.

    Args:
        ranked_lists: 쿼리별 검색 결과 article_id 리스트.
        relevant_sets: 쿼리별 관련 article_id 집합.
    """
    if not ranked_lists:
        return 0.0

    total = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        for i, doc_id in enumerate(ranked):
            if doc_id in relevant:
                total += 1.0 / (i + 1)
                break

    return total / len(ranked_lists)


def precision_at_k(
    ranked_lists: list[list[int]], relevant_sets: list[set[int]], k: int,
) -> float:
    """Precision @ K.

    상위 K개 결과 중 관련 문서 비율의 평균.
    """
    if not ranked_lists:
        return 0.0

    total = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        top_k = ranked[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        total += hits / k

    return total / len(ranked_lists)


def hit_rate_at_k(
    ranked_lists: list[list[int]], relevant_sets: list[set[int]], k: int,
) -> float:
    """Hit Rate @ K.

    상위 K개에 관련 문서가 하나 이상 포함된 쿼리 비율.
    """
    if not ranked_lists:
        return 0.0

    hits = 0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        top_k = ranked[:k]
        if any(doc_id in relevant for doc_id in top_k):
            hits += 1

    return hits / len(ranked_lists)


def dcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Discounted Cumulative Gain @ K (단일 쿼리)."""
    score = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            score += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0
    return score


def ndcg_at_k(
    ranked_lists: list[list[int]], relevant_sets: list[set[int]], k: int,
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    DCG를 이상적인 DCG(IDCG)로 정규화한 값의 평균.
    """
    if not ranked_lists:
        return 0.0

    total = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        actual_dcg = dcg_at_k(ranked, relevant, k)

        # IDCG: 관련 문서를 모두 상위에 배치한 이상적 순서
        ideal_count = min(len(relevant), k)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

        if ideal_dcg > 0:
            total += actual_dcg / ideal_dcg

    return total / len(ranked_lists)


def compute_all_metrics(
    ranked_lists: list[list[int]], relevant_sets: list[set[int]], k: int = 5,
) -> dict[str, float]:
    """모든 메트릭을 한 번에 계산."""
    return {
        "mrr": mrr(ranked_lists, relevant_sets),
        f"precision@{k}": precision_at_k(ranked_lists, relevant_sets, k),
        f"hit_rate@{k}": hit_rate_at_k(ranked_lists, relevant_sets, k),
        f"ndcg@{k}": ndcg_at_k(ranked_lists, relevant_sets, k),
        "num_queries": len(ranked_lists),
    }
