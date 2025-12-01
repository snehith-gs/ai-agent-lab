from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_ok():
    """
    Simple sanity test:
    - call GET /health
    - expect 200 and {"status": "ok"}
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
