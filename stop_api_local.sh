#!/usr/bin/env bash
set -euo pipefail

# Best-effort: kill uvicorn serving nanobot_api.app on port 8000
pids=$(lsof -t -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true)
if [[ -z "$pids" ]]; then
  echo "No process listening on :8000"
  exit 0
fi

echo "Killing: $pids"
kill $pids || true
