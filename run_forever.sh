#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
while true; do
  ./start.sh || true
  echo "[$(date '+%F %T')] bot exited, restart in 5s"
  sleep 5
done
