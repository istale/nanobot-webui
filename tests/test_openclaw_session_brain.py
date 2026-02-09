import json

from fastapi.testclient import TestClient


def test_openclaw_session_brain_mode(monkeypatch):
    """Nanobot can consult an OpenClaw session via /tools/invoke (mocked).

    We emulate:
    - POST /tools/invoke sessions_send -> ok
    - POST /tools/invoke sessions_history -> returns an assistant message
    """

    from nanobot_api.app import app

    class DummyAgent:
        async def process_direct(self, content: str, session_key: str, channel: str, chat_id: str):
            raise AssertionError("agent.process_direct should not be called")

    app.state.agent = DummyAgent()

    monkeypatch.delenv("NANOBOT_BRAIN_SUGGEST_URL", raising=False)
    monkeypatch.setenv("NANOBOT_OPENCLAW_GATEWAY_URL", "http://gw.local")
    monkeypatch.setenv("NANOBOT_OPENCLAW_TOKEN", "t")
    monkeypatch.setenv("NANOBOT_OPENCLAW_SESSION_KEY", "main")
    monkeypatch.setenv("NANOBOT_OPENCLAW_POLL_TIMEOUT_S", "1.0")
    monkeypatch.setenv("NANOBOT_OPENCLAW_POLL_INTERVAL_S", "0.01")

    import httpx

    calls = {"history": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://gw.local/tools/invoke")
        body = json.loads(request.content.decode("utf-8"))
        tool = body.get("tool")
        if tool == "sessions_send":
            return httpx.Response(200, json={"ok": True, "result": {"sent": True}})
        if tool == "sessions_history":
            calls["history"] += 1
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": {
                        "messages": [
                            {"role": "user", "content": "x"},
                            {"role": "assistant", "content": "鋼鐵人"},
                        ]
                    },
                },
            )
        return httpx.Response(400, json={"ok": False})

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
    assert calls["history"] >= 1

    monkeypatch.setattr(httpx, "AsyncClient", real_async_client)
