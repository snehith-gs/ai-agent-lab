from types import SimpleNamespace

from fastapi.testclient import TestClient
from app.main import app
from app.services import llm_service

client = TestClient(app)


def fake_get_client():
    """
    Fake OpenAI client used only in tests.
    It mimics: client.chat.completions.create(...)
    and always returns a dummy reply.
    """

    def create(model, messages, temperature):
        # This shape matches what llm_service expects:
        # resp.choices[0].message["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message={"content": "hi from test"}
                )
            ]
        )

    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def test_chat_basic(monkeypatch):
    """
    Happy-path test for /chat:
    - Patch llm_service.get_client so it uses fake OpenAI
    - Call POST /chat with a simple message
    - Assert 200 and that reply comes from our fake model
    """

    # Replace the real get_client() with our fake one
    monkeypatch.setattr(llm_service, "get_client", fake_get_client)

    payload = {
        "message": "Hello from test",
        "session_id": None,
        "use_rag": False
    }

    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    # We don't depend on exact response shape beyond these:
    assert "reply" in data
    assert data["reply"] == "hi from test"
    assert "session_id" in data
    assert "history" in data

def test_chat_missing_message(monkeypatch):
    """
    If client does not send 'message' field,
    FastAPI/Pydantic should reject with 422.
    """

    monkeypatch.setattr(llm_service, "get_client", fake_get_client)

    # Missing "message"
    resp = client.post("/chat", json={"session_id": None})
    assert resp.status_code == 422
