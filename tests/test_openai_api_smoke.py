import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_startup_agent(monkeypatch):
    monkeypatch.setenv("NANOBOT_API_DISABLE_STARTUP_AGENT", "1")


def test_chat_completions_smoke(monkeypatch):
    from nanobot_api.app import app

    class DummyAgent:
        async def process_direct(self, content: str, session_key: str, channel: str, chat_id: str):
            # Echo to verify prompt extraction + session mapping wiring
            return f"echo:{chat_id}:{content}"

    app.state.agent = DummyAgent()

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "user": "u1",
            "conversation_id": "c1",
            "messages": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "echo:u1:c1:hello"


def test_stream_smoke(monkeypatch):
    from nanobot_api.app import app

    class DummyAgent:
        async def process_direct(self, content: str, session_key: str, channel: str, chat_id: str):
            return "ok-stream"

    app.state.agent = DummyAgent()
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
