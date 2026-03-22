"""하이브리드 검색 단위 테스트 (RRF 융합 로직)."""

from __future__ import annotations

from api.retrieval.hybrid_search import SearchResult, rrf_fusion


def _make_result(aid: int, score: float, source: str = "semantic") -> SearchResult:
    return SearchResult(
        article_id=aid,
        product_id=1,
        number=f"제{aid}조",
        title=f"조항{aid}",
        content=f"내용{aid}",
        level="article",
        score=score,
        source=source,
    )


class TestRRFFusion:
    """RRF 융합 테스트."""

    def test_empty_inputs(self) -> None:
        result = rrf_fusion([], [])
        assert result == []

    def test_semantic_only(self) -> None:
        sem = [_make_result(1, 0.9), _make_result(2, 0.8)]
        result = rrf_fusion(sem, [], top_k=5)
        assert len(result) == 2
        assert result[0].article_id == 1

    def test_keyword_only(self) -> None:
        kw = [_make_result(3, 0.7, "keyword"), _make_result(4, 0.5, "keyword")]
        result = rrf_fusion([], kw, top_k=5)
        assert len(result) == 2
        assert result[0].article_id == 3

    def test_fusion_boost(self) -> None:
        """같은 article이 양쪽에 있으면 점수가 합산되어 상위에 올라간다."""
        sem = [_make_result(1, 0.9), _make_result(2, 0.8)]
        kw = [_make_result(2, 0.9, "keyword"), _make_result(3, 0.7, "keyword")]

        result = rrf_fusion(sem, kw, top_k=5)
        # article 2가 양쪽에 있으므로 점수가 합산되어 1위
        assert result[0].article_id == 2
        assert result[0].source == "hybrid"

    def test_top_k_limit(self) -> None:
        sem = [_make_result(i, 0.9 - i * 0.1) for i in range(10)]
        result = rrf_fusion(sem, [], top_k=3)
        assert len(result) == 3

    def test_scores_are_positive(self) -> None:
        sem = [_make_result(1, 0.5)]
        kw = [_make_result(2, 0.3, "keyword")]
        result = rrf_fusion(sem, kw, top_k=5)
        for r in result:
            assert r.score > 0
