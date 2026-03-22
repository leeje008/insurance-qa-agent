"""api.llm.ollama_client - Ollama API 클라이언트."""

from __future__ import annotations

import logging

from langchain_ollama import ChatOllama, OllamaEmbeddings

from api.config import settings
from core.constants import (
    DEFAULT_LLM_MODEL,
    EMBEDDING_MODEL,
    FALLBACK_LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


def get_llm(
    *,
    model: str | None = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> ChatOllama:
    """ChatOllama LLM 인스턴스 생성."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model or settings.ollama_model or DEFAULT_LLM_MODEL,
        temperature=temperature,
        num_predict=max_tokens,
    )


def get_fallback_llm() -> ChatOllama:
    """폴백 LLM 인스턴스 (경량 모델)."""
    return get_llm(model=FALLBACK_LLM_MODEL)


def get_embeddings(
    *, model: str | None = None,
) -> OllamaEmbeddings:
    """OllamaEmbeddings 인스턴스 생성."""
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=model or settings.embedding_model or EMBEDDING_MODEL,
    )


async def invoke_with_fallback(
    prompt: str,
    *,
    primary: ChatOllama | None = None,
    fallback: ChatOllama | None = None,
) -> str:
    """LLM 호출 — primary 실패 시 fallback 사용.

    Returns:
        LLM 응답 텍스트.
    """
    primary = primary or get_llm()
    try:
        resp = await primary.ainvoke(prompt)
        return resp.content
    except Exception:
        logger.warning("Primary LLM 실패, fallback 시도", exc_info=True)
        fallback = fallback or get_fallback_llm()
        resp = await fallback.ainvoke(prompt)
        return resp.content
