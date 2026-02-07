# Testing: Nanobot OpenAI-compatible API

This repo includes a minimal OpenAI-compatible API server (FastAPI) exposing:

- `POST /v1/chat/completions`

It forwards the final user prompt to `nanobot.agent.loop.AgentLoop.process_direct()` and wraps the result in an OpenAI-style response.

## Prerequisites

1. Create nanobot config on the host (so Docker can mount it):

- Path: `~/.nanobot/config.json`
- Must include a provider apiKey for the model you use.

Example (minimal):

```json
{
  "agents": {"defaults": {"model": "openai/gpt-4o-mini"}},
  "providers": {"openai": {"apiKey": "YOUR_KEY"}}
}
```

2. Make sure `~/.nanobot/workspace` exists (it will be created automatically).

## Run locally (Docker)

```bash
cd agent_dev/nanobot
docker compose -f docker-compose.api.yml up --build
```

Health check:

```bash
curl -s http://localhost:8000/health
```

## cURL end-to-end test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai/gpt-4o-mini",
    "user": "u1",
    "conversation_id": "c1",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in 1 short sentence."}
    ]
  }' | jq .
```

### Session mapping / history acceptance criteria

1. Send two requests with the same `user` + `conversation_id`.
2. The second message should be able to reference context from the first.

Example:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "user": "u1",
    "conversation_id": "c1",
    "messages": [{"role":"user","content":"Remember the code word is BLUE."}]
  }' > /dev/null

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "user": "u1",
    "conversation_id": "c1",
    "messages": [{"role":"user","content":"What is the code word?"}]
  }' | jq -r '.choices[0].message.content'
```

Acceptance:
- Response should contain `BLUE`.

## Open WebUI end-to-end

### One-click

```bash
cd agent_dev/nanobot
./run_webui.sh
```

Stop:

```bash
./stop_webui.sh
```

### Manual

Start both services:

```bash
cd agent_dev/nanobot
docker compose -f docker-compose.openwebui.yml up --build
```

Then:

1. Open <http://localhost:3000>
2. (If `WEBUI_AUTH=false`, it will skip login)
3. Create a new chat and select a model name (it can be any string; the server ignores most fields and uses nanobot config default when `model` is missing).
4. Send a message.

Acceptance:
- Open WebUI shows assistant response.
- Sending a second message in the same chat preserves context.

## Automated tests (pytest)

Smoke test (no external LLM call):

```bash
cd agent_dev/nanobot
python -m pytest -q
```

Acceptance:
- Test suite exits with code 0.
- `tests/test_openai_api_smoke.py` passes by monkeypatching `AgentLoop.process_direct`.
