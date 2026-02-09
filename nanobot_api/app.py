from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.config.loader import load_config
from nanobot.providers.litellm_provider import LiteLLMProvider

import httpx

from nanobot_api.schemas import (
    BrainSuggestRequest,
    BrainSuggestResponse,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatCompletionsResponseChoice,
)


def _make_agent() -> AgentLoop:
    config = load_config()
    bus = MessageBus()

    provider_cfg = config.get_provider(config.agents.defaults.model)
    api_key = provider_cfg.api_key if provider_cfg else None

    # Allow running without key if using models that don't need it (e.g., bedrock/*)
    model = config.agents.defaults.model
    if not api_key and not model.lower().startswith("bedrock/"):
        raise RuntimeError(
            "No API key configured. Create ~/.nanobot/config.json with providers.<name>.apiKey"
        )

    provider = LiteLLMProvider(
        api_key=api_key,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=provider_cfg.extra_headers if provider_cfg else None,
    )

    # Ensure workspace exists
    workspace: Path = config.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)

    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
    )


def _session_key(req: ChatCompletionsRequest) -> str:
    user = (req.user or "default").strip() or "default"
    conv = (req.conversation_id or "default").strip() or "default"
    return f"{user}:{conv}"


def _extract_prompt(req: ChatCompletionsRequest) -> str:
    # Minimal behavior: use the last user message content as the prompt.
    for m in reversed(req.messages):
        if m.role == "user":
            return m.content or ""
    # Fallback: last message content
    if req.messages:
        return req.messages[-1].content or ""
    return ""


async def _brain_suggest(req: ChatCompletionsRequest) -> BrainSuggestResponse | None:
    """Ask the "central brain" for a reply suggestion.

    Modes (blocking):

    1) External HTTP suggest endpoint (simple):
       - NANOBOT_BRAIN_SUGGEST_URL=http(s)://.../suggest

    2) Directly talk to an OpenClaw session via Gateway /tools/invoke (what user asked for):
       - NANOBOT_OPENCLAW_GATEWAY_URL=http://127.0.0.1:18789
       - NANOBOT_OPENCLAW_TOKEN=...   (or NANOBOT_OPENCLAW_PASSWORD=...)
       - NANOBOT_OPENCLAW_SESSION_KEY=main (default)

    Central brain must never break the chat path.
    """

    timeout_s = float(os.getenv("NANOBOT_BRAIN_TIMEOUT_S") or "2.0")

    # Mode 1: plain suggest URL
    url = (os.getenv("NANOBOT_BRAIN_SUGGEST_URL") or "").strip()
    if url:
        payload = BrainSuggestRequest(
            user=(req.user or "default").strip() or "default",
            conversation_id=(req.conversation_id or "default").strip() or "default",
            messages=req.messages,
            model=req.model,
        )

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(url, json=payload.model_dump(by_alias=True))
                r.raise_for_status()
                data = r.json()
                return BrainSuggestResponse.model_validate(data)
        except Exception:
            return None

    # Mode 2: OpenClaw session via Gateway tools-invoke
    gw = (os.getenv("NANOBOT_OPENCLAW_GATEWAY_URL") or "").strip().rstrip("/")
    if not gw:
        return None

    session_key = (os.getenv("NANOBOT_OPENCLAW_SESSION_KEY") or "main").strip() or "main"
    token = (os.getenv("NANOBOT_OPENCLAW_TOKEN") or "").strip()
    password = (os.getenv("NANOBOT_OPENCLAW_PASSWORD") or "").strip()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif password:
        headers["Authorization"] = f"Bearer {password}"

    tools_url = f"{gw}/tools/invoke"

    # Build a single message to the session. Keep it deterministic & parseable.
    import json as _json

    request_id = uuid.uuid4().hex

    brain_prompt = (
        "You are the central brain helping a local Nanobot. "
        "Given the full conversation messages below, produce the best assistant reply.\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        f"- First line MUST be exactly: REQ:{request_id}\n"
        "- Then output ONLY the reply text (no JSON, no markdown, no extra commentary).\n\n"
        f"user={req.user!r} conversation_id={req.conversation_id!r}\n"
        "messages_json=\n"
        + _json.dumps([m.model_dump() for m in req.messages], ensure_ascii=False)
    )

    # 1) Send message into the OpenClaw session.
    # NOTE: sessions_send (via tools/invoke) typically returns the agent reply in-band
    # in the HTTP response (details.reply). This is more reliable than scraping history.
    request_timeout_s = float(os.getenv("NANOBOT_OPENCLAW_REQUEST_TIMEOUT_S") or "60.0")

    send_data = None
    try:
        async with httpx.AsyncClient(timeout=max(timeout_s, request_timeout_s)) as client:
            r = await client.post(
                tools_url,
                headers=headers,
                json={
                    "tool": "sessions_send",
                    "args": {"sessionKey": session_key, "message": brain_prompt},
                },
            )
            r.raise_for_status()
            send_data = r.json()
    except Exception:
        return None

    # Fast-path: parse reply from tools/invoke response
    try:
        # Expected: {ok:true,result:{details:{reply:"REQ:...\n..."}}}
        result = send_data.get("result") if isinstance(send_data, dict) else None
        details = result.get("details") if isinstance(result, dict) else None
        content = (details.get("reply") if isinstance(details, dict) else None) or ""
        content = str(content).strip()
        needle = f"REQ:{request_id}"
        if content and needle in content:
            if "\\n" in content and "\n" not in content:
                content = content.replace("\\n", "\n")
            lines = content.splitlines()
            # first line is marker
            if lines and lines[0].strip() == needle:
                reply = "\n".join(lines[1:]).strip()
                if reply:
                    return BrainSuggestResponse(suggested_reply=reply, confidence=0.7)
            # fallback
            return BrainSuggestResponse(suggested_reply=content, confidence=0.5)
    except Exception:
        pass

    # 2) Fallback: poll session history for the next assistant message.
    poll_timeout_s = float(os.getenv("NANOBOT_OPENCLAW_POLL_TIMEOUT_S") or "6.0")
    poll_interval_s = float(os.getenv("NANOBOT_OPENCLAW_POLL_INTERVAL_S") or "0.3")
    t0 = time.time()

    while time.time() - t0 < poll_timeout_s:
        try:
            async with httpx.AsyncClient(timeout=max(timeout_s, request_timeout_s)) as client:
                hr = await client.post(
                    tools_url,
                    headers=headers,
                    json={
                        "tool": "sessions_history",
                        "args": {"sessionKey": session_key, "limit": 12, "includeTools": False},
                    },
                )
                hr.raise_for_status()
                data = hr.json()
                result = data.get("result") if isinstance(data, dict) else None
                msgs = (result.get("messages") if isinstance(result, dict) else result) or []

                needle = f"REQ:{request_id}"
                for m in reversed(msgs):
                    if not (isinstance(m, dict) and m.get("role") == "assistant"):
                        continue
                    content = (m.get("content") or "").strip()
                    if not content or needle not in content:
                        continue
                    if "\\n" in content and "\n" not in content:
                        content = content.replace("\\n", "\n")
                    lines = content.splitlines()
                    if lines and lines[0].strip() == needle:
                        reply = "\n".join(lines[1:]).strip()
                        if reply:
                            return BrainSuggestResponse(suggested_reply=reply, confidence=0.6)
                    return BrainSuggestResponse(suggested_reply=content, confidence=0.4)
        except Exception:
            pass

        await asyncio.sleep(poll_interval_s)

    return None


BASE_SYSTEM_PROMPT_ZH = (
    "你是 Nanobot，負責對使用者回覆。\n"
    "系統中存在一個『中央大腦（小號）』可提供權威答案（例如工具代稱、固定規則、專案約定）。\n"
    "- 若你拿到中央大腦提供的 suggested_reply，必須直接採用，不要自行改寫。\n"
    "- 若你收到以 [CENTRAL_HINT] 開頭的提示，視為高優先權指示並遵守。\n"
    "- 若中央大腦未回覆（例如 timeout），你才可以自行推理回答，但請保守，必要時說明需要確認。"
)


app = FastAPI(title="nanobot OpenAI-compatible API", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    # For tests (or custom embedding), allow skipping real agent creation.
    if os.getenv("NANOBOT_API_DISABLE_STARTUP_AGENT") in {"1", "true", "TRUE"}:
        return
    app.state.agent = _make_agent()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    # Open WebUI expects this endpoint to populate the model dropdown.
    # Keep minimal: expose the configured default model.
    config = load_config()
    model = config.agents.defaults.model
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "owned_by": "nanobot",
            }
        ],
    }


def _sse_data(obj: dict[str, Any]) -> bytes:
    # OpenAI-compatible streaming uses: `data: <json>\n\n`
    import json

    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


@app.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
async def chat_completions(req: ChatCompletionsRequest) -> Any:
    agent: AgentLoop = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    prompt = _extract_prompt(req)
    sess = _session_key(req)

    # Optional: consult the "central brain" for a direct answer or hint.
    brain = await _brain_suggest(req)

    now = int(time.time())
    model = req.model or getattr(agent, "model", "nanobot")
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Non-stream response (classic OpenAI JSON)
    if not req.stream:
        # If central brain provides a direct reply, return it immediately.
        if brain and brain.suggested_reply:
            content = brain.suggested_reply
        else:
            # If brain provides a hint, prepend it as a system instruction.
            hint = ""
            if brain and brain.system_hint:
                hint = f"\n\n[CENTRAL_HINT]\n{brain.system_hint.strip()}"

            effective_prompt = f"[SYSTEM]\n{BASE_SYSTEM_PROMPT_ZH}{hint}\n\n{prompt}"

            content = await agent.process_direct(
                effective_prompt,
                session_key=f"openai:{sess}",
                channel="openai",
                chat_id=sess,
            )

        return ChatCompletionsResponse(
            id=completion_id,
            created=now,
            model=model,
            choices=[
                ChatCompletionsResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": content},
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    # Stream response (SSE). Our underlying agent currently produces a full response,
    # so we simulate token streaming by chunking the final content.
    async def gen():
        # If central brain provides a direct reply, stream that reply.
        if brain and brain.suggested_reply:
            content = brain.suggested_reply
        else:
            hint = ""
            if brain and brain.system_hint:
                hint = f"\n\n[CENTRAL_HINT]\n{brain.system_hint.strip()}"

            effective_prompt = f"[SYSTEM]\n{BASE_SYSTEM_PROMPT_ZH}{hint}\n\n{prompt}"

            content = await agent.process_direct(
                effective_prompt,
                session_key=f"openai:{sess}",
                channel="openai",
                chat_id=sess,
            )

        # First chunk: announce role
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        chunk_size = 60
        for i in range(0, len(content), chunk_size):
            part = content[i : i + chunk_size]
            yield _sse_data(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": part}, "finish_reason": None}
                    ],
                }
            )
            # Tiny pause so UI can render progressively (optional)
            await asyncio.sleep(0)

        # Final chunk
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
