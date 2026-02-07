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

from nanobot_api.schemas import ChatCompletionsRequest, ChatCompletionsResponse, ChatCompletionsResponseChoice


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

    now = int(time.time())
    model = req.model or getattr(agent, "model", "nanobot")
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Non-stream response (classic OpenAI JSON)
    if not req.stream:
        content = await agent.process_direct(
            prompt,
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
        content = await agent.process_direct(
            prompt,
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
