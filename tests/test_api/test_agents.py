"""에이전트 상태/그래프 구조 테스트."""

from __future__ import annotations

from api.agents.graph import build_graph, should_retry
from api.agents.state import PipelineState


class TestShouldRetry:
    """재시도 판단 로직 테스트."""

    def test_valid_high_confidence(self) -> None:
        state: PipelineState = {
            "question": "test",
            "is_valid": True,
            "confidence": 0.9,
            "retry_count": 0,
        }
        assert should_retry(state) == "end"

    def test_invalid_first_try(self) -> None:
        state: PipelineState = {
            "question": "test",
            "is_valid": False,
            "confidence": 0.3,
            "retry_count": 0,
        }
        assert should_retry(state) == "retry"

    def test_max_retry_reached(self) -> None:
        state: PipelineState = {
            "question": "test",
            "is_valid": False,
            "confidence": 0.2,
            "retry_count": 2,
        }
        assert should_retry(state) == "end"

    def test_low_confidence_but_valid(self) -> None:
        state: PipelineState = {
            "question": "test",
            "is_valid": True,
            "confidence": 0.3,
            "retry_count": 0,
        }
        assert should_retry(state) == "retry"

    def test_exactly_at_threshold(self) -> None:
        state: PipelineState = {
            "question": "test",
            "is_valid": True,
            "confidence": 0.6,
            "retry_count": 0,
        }
        assert should_retry(state) == "end"


class TestBuildGraph:
    """그래프 빌드 테스트."""

    def test_graph_nodes(self) -> None:
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "query_processor", "retriever",
            "answer_generator", "answer_validator",
        }
        assert expected.issubset(node_names)

    def test_graph_compiles(self) -> None:
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None
