"""tests.conftest - pytest 공통 fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture()
def app():
    """테스트용 FastAPI 앱 인스턴스."""
    return create_app()


@pytest.fixture()
def client(app):
    """테스트 HTTP 클라이언트."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_ollama():
    """Ollama LLM 호출 모킹."""
    with patch("api.llm.ollama_client.invoke_with_fallback", new_callable=AsyncMock) as m:
        m.return_value = (
            '{"keywords": ["테스트"], "intent": "general",'
            ' "article_refs": [], "rewritten_query": "테스트 질문"}'
        )
        yield m


@pytest.fixture()
def mock_embedding():
    """EmbeddingClient 모킹."""
    mock_client = AsyncMock()
    mock_client.embed.return_value = [0.1] * 768
    mock_client.embed_batch.return_value = [[0.1] * 768]
    mock_client.close.return_value = None

    with patch("api.retrieval.embedder.EmbeddingClient", return_value=mock_client):
        yield mock_client


@pytest.fixture()
def mock_db_session():
    """AsyncSession 모킹."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session
