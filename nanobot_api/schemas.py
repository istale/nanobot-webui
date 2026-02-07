from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]

    # OpenAI-compatible optional fields
    user: str | None = None
    stream: bool | None = False

    # Non-standard but commonly used by clients (e.g. Open WebUI)
    conversation_id: str | None = Field(default=None, alias="conversationId")

    # Allow extra keys without failing (Open WebUI sends many)
    model_config = {"extra": "allow", "populate_by_name": True}


class ChatCompletionsResponseChoice(BaseModel):
    index: int = 0
    message: dict[str, Any]
    finish_reason: str = "stop"


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionsResponseChoice]
    # usage is optional in OpenAI responses; keep minimal
    usage: dict[str, int] | None = None
