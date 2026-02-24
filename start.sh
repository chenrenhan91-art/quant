#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "[ERROR] .env not found. Copy .env.example to .env and fill values first."
  exit 1
fi

set -a
source ./.env
set +a

python3 -u trading_bot_simple.py
