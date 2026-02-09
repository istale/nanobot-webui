import os

from fastapi.testclient import TestClient


def test_brain_suggest_direct_reply(monkeypatch):
    """If central brain provides suggested_reply, nanobot should return it and skip agent."""

    from nanobot_api.app import app

    class DummyAgent:
        async def process_direct(self, content: str, session_key: str, channel: str, chat_id: str):
            raise AssertionError("agent.process_direct should not be called")

    app.state.agent = DummyAgent()

    monkeypatch.setenv("NANOBOT_BRAIN_SUGGEST_URL", "http://brain.local/suggest")

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://brain.local/suggest")
        return httpx.Response(200, json={"suggested_reply": "鋼鐵人", "confidence": 0.99})

    transport = httpx.MockTransport(handler)

    real_async_client = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "user": "u1",
            "conversation_id": "c1",
            "messages": [{"role": "user", "content": "opencode 的代稱是什麼？"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "鋼鐵人"

    monkeypatch.setattr(httpx, "AsyncClient", real_async_client)


def test_brain_suggest_hint(monkeypatch):
    """If central brain provides system_hint only, nanobot should pass it into agent prompt."""

    from nanobot_api.app import app

    seen = {}

    class DummyAgent:
        async def process_direct(self, content: str, session_key: str, channel: str, chat_id: str):
            seen["prompt"] = content
            return "ok"

    app.state.agent = DummyAgent()

    monkeypatch.setenv("NANOBOT_BRAIN_SUGGEST_URL", "http://brain.local/suggest")

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"system_hint": "Answer with the alias."})

    transport = httpx.MockTransport(handler)

    real_async_client = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "user": "u1",
            "conversation_id": "c1",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    assert "[SYSTEM]" in seen["prompt"]
    assert "中央大腦（小號）" in seen["prompt"]
    assert "[CENTRAL_HINT]" in seen["prompt"]
    assert "Answer with the alias." in seen["prompt"]

    monkeypatch.setattr(httpx, "AsyncClient", real_async_client)
