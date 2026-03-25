"""api.retrieval.reranker - 플러거블 리랭커 시스템 (Strategy Pattern + Registry)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from core.constants import RERANK_MODEL, RERANK_TOP_K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 리랭크 결과 타입
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """리랭크 결과."""

    document: dict[str, Any]
    score: float
    rank: int


# ---------------------------------------------------------------------------
# 추상 베이스 클래스
# ---------------------------------------------------------------------------

class BaseReranker(ABC):
    """리랭커 추상 베이스 클래스."""

    name: str = ""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """문서 재순위화."""


# ---------------------------------------------------------------------------
# 레지스트리
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseReranker]] = {}


def register_reranker(cls: type[BaseReranker]) -> type[BaseReranker]:
    """리랭커 클래스를 레지스트리에 등록."""
    _REGISTRY[cls.name] = cls
    return cls


def get_reranker(strategy: str | None = None) -> BaseReranker:
    """설정 또는 지정된 전략으로 리랭커 인스턴스 반환."""
    if strategy is None:
        from api.config import settings
        strategy = settings.rerank_strategy

    if strategy not in _REGISTRY:
        available = ", ".join(_REGISTRY)
        raise ValueError(f"알 수 없는 리랭커 전략: {strategy!r} (사용 가능: {available})")

    return _REGISTRY[strategy]()


def list_rerankers() -> list[str]:
    """등록된 리랭커 전략 목록."""
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# 구현체 1: Cross-Encoder (BAAI/bge-reranker-v2-m3)
# ---------------------------------------------------------------------------

_cross_encoder_model = None


@register_reranker
class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 기반 문서 재순위화.

    BAAI/bge-reranker-v2-m3 — 다국어(한국어 포함), 568M params, CPU 추론 가능.
    """

    name = "cross_encoder"

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents:
            return []

        model = _get_cross_encoder()
        pairs = [(query, doc.get("content", "")) for doc in documents]
        scores = await asyncio.to_thread(model.predict, pairs)

        ranked = sorted(
            zip(documents, scores), key=lambda x: float(x[1]), reverse=True,
        )

        return [
            RerankResult(document=doc, score=float(score), rank=i)
            for i, (doc, score) in enumerate(ranked[:top_k])
        ]


def _get_cross_encoder():
    """Cross-Encoder 모델 싱글톤 로드."""
    global _cross_encoder_model  # noqa: PLW0603
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Cross-Encoder 모델 로드: %s", RERANK_MODEL)
        _cross_encoder_model = CrossEncoder(RERANK_MODEL)
    return _cross_encoder_model


# ---------------------------------------------------------------------------
# 구현체 2: LLM Listwise (Ollama qwen2.5)
# ---------------------------------------------------------------------------

RERANKING_PROMPT = """\
당신은 보험약관 검색 결과의 관련성을 평가하는 전문가입니다.
사용자 질문에 대해 각 문서의 관련성을 0~10점으로 평가하세요.

## 출력 형식 (JSON 배열만 출력)
[{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}, ...]

## 사용자 질문
{question}

## 검색된 문서
{documents}
"""


@register_reranker
class LLMListwiseReranker(BaseReranker):
    """LLM 기반 Listwise 리랭킹 (기존 Ollama 인프라 활용)."""

    name = "llm_listwise"

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents:
            return []

        from api.llm.ollama_client import invoke_with_fallback

        formatted_docs = "\n\n".join(
            f"[{i}] {doc.get('number', '')} {doc.get('title', '')}\n"
            f"{doc.get('content', '')[:500]}"
            for i, doc in enumerate(documents)
        )

        prompt = RERANKING_PROMPT.format(question=query, documents=formatted_docs)

        try:
            raw = (await invoke_with_fallback(prompt)).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            rankings = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            logger.warning("LLM 리랭킹 JSON 파싱 실패, 원본 순서 유지", exc_info=True)
            return [
                RerankResult(document=doc, score=1.0 - i / len(documents), rank=i)
                for i, doc in enumerate(documents[:top_k])
            ]

        # index → score 매핑
        score_map: dict[int, float] = {}
        for entry in rankings:
            idx = entry.get("index", -1)
            score = float(entry.get("score", 0))
            if 0 <= idx < len(documents):
                score_map[idx] = score

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        return [
            RerankResult(document=documents[idx], score=score, rank=i)
            for i, (idx, score) in enumerate(ranked[:top_k])
        ]


# ---------------------------------------------------------------------------
# 구현체 3: Similarity (임베딩 재계산)
# ---------------------------------------------------------------------------

@register_reranker
class SimilarityReranker(BaseReranker):
    """임베딩 코사인 유사도 기반 리랭킹 (기존 embedder 활용)."""

    name = "similarity"

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents:
            return []

        from api.retrieval.embedder import EmbeddingClient

        embedder = EmbeddingClient()
        try:
            query_emb = await embedder.embed(query)
            doc_embs = await embedder.embed_batch(
                [doc.get("content", "")[:1000] for doc in documents],
            )
        finally:
            await embedder.close()

        scored = []
        for doc, doc_emb in zip(documents, doc_embs):
            score = _cosine_similarity(query_emb, doc_emb)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RerankResult(document=doc, score=score, rank=i)
            for i, (doc, score) in enumerate(scored[:top_k])
        ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """코사인 유사도 계산."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# 구현체 4: NoOp (패스스루, 기준선 비교용)
# ---------------------------------------------------------------------------

@register_reranker
class NoOpReranker(BaseReranker):
    """리랭킹 비활성화 (기준선 비교용)."""

    name = "noop"

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        return [
            RerankResult(
                document=doc,
                score=1.0 - i / max(len(documents), 1),
                rank=i,
            )
            for i, doc in enumerate(documents[:top_k])
        ]


# ---------------------------------------------------------------------------
# 구현체 5: LTR (Learning-to-Rank, LightGBM LambdaRank)
# ---------------------------------------------------------------------------

_ltr_model = None


@register_reranker
class LTRReranker(BaseReranker):
    """LightGBM LambdaRank 기반 리랭킹 (학습된 피처 결합)."""

    name = "ltr"

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents:
            return []

        model = _get_ltr_model()
        if model is None or not model.is_trained:
            logger.warning("LTR 모델 미학습 — NoOp 폴백")
            return [
                RerankResult(document=doc, score=1.0 - i / max(len(documents), 1), rank=i)
                for i, doc in enumerate(documents[:top_k])
            ]

        from api.evaluation.ltr import extract_features

        features = extract_features(query, documents)
        import numpy as np

        scores = model.predict(np.array(features, dtype=np.float32))

        ranked = sorted(
            zip(documents, scores), key=lambda x: float(x[1]), reverse=True,
        )

        return [
            RerankResult(document=doc, score=float(score), rank=i)
            for i, (doc, score) in enumerate(ranked[:top_k])
        ]


def _get_ltr_model():
    """LTR 모델 싱글톤 로드."""
    global _ltr_model  # noqa: PLW0603
    if _ltr_model is None:
        from api.evaluation.ltr import LTRModel

        _ltr_model = LTRModel()
        _ltr_model.load()  # 파일 있으면 로드, 없으면 is_trained=False
    return _ltr_model


# ---------------------------------------------------------------------------
# 비교 도구
# ---------------------------------------------------------------------------

async def compare_rerankers(
    query: str,
    documents: list[dict[str, Any]],
    strategies: list[str] | None = None,
    top_k: int = RERANK_TOP_K,
) -> dict[str, list[RerankResult]]:
    """여러 리랭커를 동일 입력에 대해 실행하고 결과를 비교."""
    strategies = strategies or list_rerankers()
    results: dict[str, list[RerankResult]] = {}

    for name in strategies:
        try:
            reranker = get_reranker(name)
            results[name] = await reranker.rerank(query, documents, top_k)
        except Exception:
            logger.warning("리랭커 %s 실행 실패", name, exc_info=True)
            results[name] = []

    return results
