"""tests.test_api.test_routers.test_qa - Q&A 라우터 테스트."""


def test_health_check(client):
    """GET /health 정상 응답."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
