"""api.retrieval.embedder - nomic-embed-text 임베딩 생성."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from api.config import settings
from core.constants import (
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    SUB_CHUNK_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 토큰 추정 (한국어: ~1.5자/토큰, 간이 추정)
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 1.5


def _estimate_tokens(text: str) -> int:
    """텍스트의 토큰 수를 간이 추정."""
    return int(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# 서브 청크 분할
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    *,
    max_tokens: int = SUB_CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """긴 텍스트를 토큰 기준으로 오버랩 청크로 분할.

    Args:
        text: 분할할 텍스트.
        max_tokens: 청크당 최대 토큰 수.
        overlap_tokens: 청크 간 오버랩 토큰 수.

    Returns:
        청크 문자열 리스트.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)
    step = max_chars - overlap_chars

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Ollama 임베딩 클라이언트
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """Ollama nomic-embed-text 임베딩 생성 클라이언트."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.embedding_model or EMBEDDING_MODEL
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        """리소스 정리."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        """단일 텍스트의 임베딩 벡터 생성.

        Returns:
            EMBEDDING_DIM 차원의 float 리스트.
        """
        client = await self._get_client()
        resp = await client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"임베딩 응답에 embeddings 없음: {data}")

        vector = embeddings[0]
        if len(vector) != EMBEDDING_DIM:
            logger.warning(
                "임베딩 차원 불일치: 예상 %d, 실제 %d", EMBEDDING_DIM, len(vector),
            )
        return vector

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """여러 텍스트의 임베딩 벡터 일괄 생성.

        Ollama /api/embed는 input에 리스트를 지원한다.

        Returns:
            각 텍스트에 대한 임베딩 벡터 리스트.
        """
        if not texts:
            return []

        client = await self._get_client()
        resp = await client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": list(texts)},
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            logger.warning(
                "배치 임베딩 수 불일치: 요청 %d, 응답 %d",
                len(texts), len(embeddings),
            )
        return embeddings

    async def embed_with_chunks(
        self,
        text: str,
        *,
        max_tokens: int = SUB_CHUNK_MAX_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> tuple[list[float], list[tuple[int, str, list[float]]]]:
        """텍스트를 청킹하여 전체 + 청크별 임베딩 생성.

        긴 조항의 경우 서브 청크로 분할하여 각각 임베딩을 생성한다.

        Returns:
            (전체 임베딩, [(chunk_index, chunk_text, chunk_embedding), ...])
            청크가 1개면 서브 청크 리스트는 빈 리스트.
        """
        chunks = split_into_chunks(
            text, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
        )

        if len(chunks) <= 1:
            vector = await self.embed(text)
            return vector, []

        # 전체 텍스트 + 청크들 일괄 임베딩
        all_texts = [text, *chunks]
        vectors = await self.embed_batch(all_texts)

        full_vector = vectors[0]
        chunk_results = [
            (i, chunk, vectors[i + 1])
            for i, chunk in enumerate(chunks)
        ]

        logger.info(
            "청크 임베딩 완료: 전체 1 + 서브 %d건", len(chunk_results),
        )
        return full_vector, chunk_results
