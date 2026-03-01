"""tests.conftest - pytest 공통 fixtures."""

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
