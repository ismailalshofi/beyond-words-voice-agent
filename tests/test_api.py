import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import llm


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _create(client, **overrides):
    body = {"name": "Nova", "system_prompt": "You help with travel plans.", **overrides}
    res = client.post("/api/assistants", json=body)
    assert res.status_code == 201, res.text
    return res.json()


def test_assistant_crud(client):
    created = _create(client, call_greeting="Hi, Nova here.", chat_greeting="Hey! Where to?")
    assert created["language"] == "en"
    assert created["voice_id"] is None

    assert client.get(f"/api/assistants/{created['id']}").json()["name"] == "Nova"
    assert any(a["id"] == created["id"] for a in client.get("/api/assistants").json())

    updated = client.patch(f"/api/assistants/{created['id']}", json={"name": "Nova 2", "voice_id": "abc"}).json()
    assert updated["name"] == "Nova 2"
    assert updated["voice_id"] == "abc"
    assert updated["system_prompt"] == "You help with travel plans."

    assert client.delete(f"/api/assistants/{created['id']}").status_code == 204
    assert client.get(f"/api/assistants/{created['id']}").status_code == 404


def test_create_requires_name(client):
    assert client.post("/api/assistants", json={"name": ""}).status_code == 422


def test_chat_uses_greeting_then_streams_reply(client, monkeypatch):
    calls = []

    async def fake_stream(messages, *, model=None, max_tokens):
        calls.append(messages)
        for token in ["Paris", " is", " lovely", "."]:
            yield token

    monkeypatch.setattr(llm, "stream_reply", fake_stream)
    assistant = _create(client, chat_greeting="Hey! Where to?")

    with client.websocket_connect(f"/api/assistants/{assistant['id']}/chat") as ws:
        assert ws.receive_json() == {"type": "done", "text": "Hey! Where to?"}

        ws.send_json({"type": "message", "text": "Somewhere in France"})
        deltas = []
        while (event := ws.receive_json())["type"] == "delta":
            deltas.append(event["text"])
        assert event == {"type": "done", "text": "Paris is lovely."}
        assert "".join(deltas) == "Paris is lovely."

    messages = calls[0]
    assert messages[0]["role"] == "system"
    assert "Your name is Nova." in messages[0]["content"]
    assert "You help with travel plans." in messages[0]["content"]
    assert messages[1:] == [
        {"role": "assistant", "content": "Hey! Where to?"},
        {"role": "user", "content": "Somewhere in France"},
    ]


def test_chat_without_greeting_generates_one(client, monkeypatch):
    async def fake_stream(messages, *, model=None, max_tokens):
        assert messages[-1]["content"] == llm.GREETING_INSTRUCTION
        yield "Hello!"

    monkeypatch.setattr(llm, "stream_reply", fake_stream)
    assistant = _create(client)

    with client.websocket_connect(f"/api/assistants/{assistant['id']}/chat") as ws:
        assert ws.receive_json() == {"type": "delta", "text": "Hello!"}
        assert ws.receive_json() == {"type": "done", "text": "Hello!"}


def test_unknown_assistant_closes_socket(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/api/assistants/missing/chat") as ws:
        ws.receive_json()
    assert exc.value.code == 4404
