#!/usr/bin/env bash
set -euo pipefail

# One-click local run for Nanobot OpenAI-compatible API + Open WebUI
#
# Requirements:
# - Docker + docker compose
# - Host config file: ~/.nanobot/config.json (provider apiKey etc.)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing required command: $1" >&2
    exit 1
  }
}

need docker

# Docker compose v2 is `docker compose`
docker compose version >/dev/null 2>&1 || {
  echo "ERROR: docker compose not available (need Docker Compose v2)" >&2
  exit 1
}

CFG="$HOME/.nanobot/config.json"
if [[ ! -f "$CFG" ]]; then
  cat >&2 <<EOF
ERROR: nanobot config not found:
  $CFG

Create it first. Minimal example:
{
  "agents": {"defaults": {"model": "openai/gpt-4o-mini"}},
  "providers": {"openai": {"apiKey": "YOUR_KEY"}}
}
EOF
  exit 1
fi

echo "[run_webui] starting Open WebUI + nanobot OpenAI API"

docker compose -f docker-compose.openwebui.yml up -d --build

cat <<EOF

[run_webui] OK
- Open WebUI: http://localhost:3000
- OpenAI API (nanobot): http://localhost:8000/v1
- Health check: curl -s http://localhost:8000/health

To stop:
  docker compose -f docker-compose.openwebui.yml down

Testing guide:
  docs/TESTING.md
EOF
