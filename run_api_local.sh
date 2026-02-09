#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Load env (contains OpenClaw gateway token + URL)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec "$ROOT_DIR/.venv/bin/uvicorn" nanobot_api.app:app --host 0.0.0.0 --port 8000
