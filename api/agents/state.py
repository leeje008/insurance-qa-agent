"""api.agents.state - LangGraph 상태 정의.

query_processor → retriever → answer_generator → answer_validator
"""

from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """RAG 파이프라인 상태."""

    # 입력
    question: str
    product_id: int | None

    # query_processor 출력
    keywords: list[str]
    intent: str
    article_refs: list[str]
    rewritten_query: str
    query_embedding: list[float]

    # retriever 출력
    documents: list[dict[str, Any]]

    # answer_generator 출력
    answer: str

    # answer_validator 출력
    is_valid: bool
    confidence: float
    issues: list[str]
    retry_count: int

    # 최종 결과
    sources_json: str
    error: str | None
