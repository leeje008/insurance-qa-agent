"""tests.test_api.test_retrieval.test_reranker - 리랭커 시스템 테스트."""

from __future__ import annotations

import pytest

from api.retrieval.reranker import (
    BaseReranker,
    NoOpReranker,
    RerankResult,
    _cosine_similarity,
    compare_rerankers,
    get_reranker,
    list_rerankers,
)

_SAMPLE_DOCS = [
    {"article_id": 1, "number": "제1조", "title": "보장", "content": "화재보험 보장 범위"},
    {"article_id": 2, "number": "제2조", "title": "면책", "content": "면책 사항 목록"},
    {"article_id": 3, "number": "제3조", "title": "청구", "content": "보험금 청구 절차"},
    {"article_id": 4, "number": "제4조", "title": "해지", "content": "계약 해지 조건"},
]


class TestBaseReranker:
    """BaseReranker 인터페이스 테스트."""

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseReranker()


class TestNoOpReranker:
    """NoOp 리랭커 테스트."""

    @pytest.mark.asyncio
    async def test_returns_top_k(self):
        reranker = NoOpReranker()
        results = await reranker.rerank("test", _SAMPLE_DOCS, top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        reranker = NoOpReranker()
        results = await reranker.rerank("test", _SAMPLE_DOCS, top_k=4)
        assert results[0].document["article_id"] == 1
        assert results[3].document["article_id"] == 4

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        reranker = NoOpReranker()
        results = await reranker.rerank("test", [], top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_result_type(self):
        reranker = NoOpReranker()
        results = await reranker.rerank("test", _SAMPLE_DOCS, top_k=1)
        assert isinstance(results[0], RerankResult)
        assert results[0].rank == 0

    @pytest.mark.asyncio
    async def test_scores_are_positive(self):
        reranker = NoOpReranker()
        results = await reranker.rerank("test", _SAMPLE_DOCS, top_k=4)
        assert all(r.score > 0 for r in results)


class TestRegistry:
    """리랭커 레지스트리 테스트."""

    def test_list_rerankers(self):
        rerankers = list_rerankers()
        assert "noop" in rerankers
        assert "cross_encoder" in rerankers
        assert "llm_listwise" in rerankers
        assert "similarity" in rerankers

    def test_get_reranker_noop(self):
        reranker = get_reranker("noop")
        assert isinstance(reranker, NoOpReranker)

    def test_get_reranker_invalid(self):
        with pytest.raises(ValueError, match="알 수 없는 리랭커 전략"):
            get_reranker("nonexistent_strategy")


class TestCompareRerankers:
    """compare_rerankers 테스트."""

    @pytest.mark.asyncio
    async def test_compare_noop_only(self):
        results = await compare_rerankers("테스트", _SAMPLE_DOCS, strategies=["noop"])
        assert "noop" in results
        assert len(results["noop"]) > 0

    @pytest.mark.asyncio
    async def test_compare_returns_all_strategies(self):
        results = await compare_rerankers(
            "테스트", _SAMPLE_DOCS, strategies=["noop"],
        )
        assert len(results) == 1


class TestCosineSimlarity:
    """코사인 유사도 함수 테스트."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 2]) == 0.0
