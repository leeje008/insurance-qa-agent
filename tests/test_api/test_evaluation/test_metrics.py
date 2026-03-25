"""tests.test_api.test_evaluation.test_metrics - 검색 품질 메트릭 테스트."""

from api.evaluation.metrics import (
    compute_all_metrics,
    dcg_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
)


class TestMRR:
    """Mean Reciprocal Rank 테스트."""

    def test_perfect_ranking(self):
        # 첫 번째 결과가 항상 관련 문서
        ranked = [[1, 2, 3], [4, 5, 6]]
        relevant = [{1}, {4}]
        assert mrr(ranked, relevant) == 1.0

    def test_second_position(self):
        ranked = [[2, 1, 3]]
        relevant = [{1}]
        assert mrr(ranked, relevant) == 0.5

    def test_no_relevant(self):
        ranked = [[1, 2, 3]]
        relevant = [{99}]
        assert mrr(ranked, relevant) == 0.0

    def test_empty(self):
        assert mrr([], []) == 0.0

    def test_mixed(self):
        ranked = [[1, 2, 3], [5, 4, 6]]
        relevant = [{1}, {4}]
        # (1.0 + 0.5) / 2 = 0.75
        assert mrr(ranked, relevant) == 0.75


class TestPrecisionAtK:
    """Precision @ K 테스트."""

    def test_all_relevant(self):
        ranked = [[1, 2, 3]]
        relevant = [{1, 2, 3}]
        assert precision_at_k(ranked, relevant, k=3) == 1.0

    def test_none_relevant(self):
        ranked = [[1, 2, 3]]
        relevant = [{99}]
        assert precision_at_k(ranked, relevant, k=3) == 0.0

    def test_partial(self):
        ranked = [[1, 2, 3, 4]]
        relevant = [{1, 3}]
        # top-2: 1 relevant / 2 = 0.5
        assert precision_at_k(ranked, relevant, k=2) == 0.5


class TestHitRateAtK:
    """Hit Rate @ K 테스트."""

    def test_all_hit(self):
        ranked = [[1, 2], [3, 4]]
        relevant = [{1}, {4}]
        assert hit_rate_at_k(ranked, relevant, k=2) == 1.0

    def test_no_hit(self):
        ranked = [[1, 2]]
        relevant = [{99}]
        assert hit_rate_at_k(ranked, relevant, k=2) == 0.0

    def test_partial_hit(self):
        ranked = [[1, 2], [5, 6]]
        relevant = [{1}, {99}]
        assert hit_rate_at_k(ranked, relevant, k=2) == 0.5


class TestDCGAtK:
    """DCG @ K 테스트."""

    def test_first_position(self):
        # 1 / log2(2) = 1.0
        assert abs(dcg_at_k([1], {1}, k=1) - 1.0) < 1e-6

    def test_second_position(self):
        # 0 + 1/log2(3) ≈ 0.6309
        assert abs(dcg_at_k([2, 1], {1}, k=2) - 0.6309) < 0.001

    def test_no_relevant(self):
        assert dcg_at_k([1, 2, 3], {99}, k=3) == 0.0


class TestNDCGAtK:
    """NDCG @ K 테스트."""

    def test_perfect_ranking(self):
        ranked = [[1, 2, 3]]
        relevant = [{1}]
        assert ndcg_at_k(ranked, relevant, k=3) == 1.0

    def test_worst_ranking(self):
        ranked = [[2, 3, 1]]
        relevant = [{1}]
        # DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
        assert ndcg_at_k(ranked, relevant, k=3) == 0.5

    def test_empty(self):
        assert ndcg_at_k([], [], k=5) == 0.0


class TestComputeAllMetrics:
    """compute_all_metrics 통합 테스트."""

    def test_returns_all_keys(self):
        ranked = [[1, 2, 3]]
        relevant = [{1}]
        result = compute_all_metrics(ranked, relevant, k=3)
        assert "mrr" in result
        assert "precision@3" in result
        assert "hit_rate@3" in result
        assert "ndcg@3" in result
        assert result["num_queries"] == 1
