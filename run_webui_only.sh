#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Preflight
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not installed" >&2
  exit 1
fi

if ! docker ps >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: cannot access docker daemon.

Fix (recommended):
  sudo usermod -aG docker $USER
  newgrp docker

Then re-run:
  ./run_webui_only.sh
EOF
  exit 1
fi

# Compose v2
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose v2 not available" >&2
  exit 1
fi

echo "[run_webui_only] starting Open WebUI (points to host nanobot at :8000/v1)"
docker compose -f docker-compose.webui-only.yml up -d

cat <<EOF

[run_webui_only] OK
- Open WebUI: http://localhost:3000
- Nanobot (host): http://localhost:8000/v1

Stop:
  docker compose -f docker-compose.webui-only.yml down
EOF
