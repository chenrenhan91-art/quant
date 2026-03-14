#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

HOST="${MONITOR_HOST:-127.0.0.1}"
PORT="${MONITOR_PORT:-8787}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/monitor_panel.log"
PLIST="./monitor_dashboard.launchd.plist"
LABEL="com.chenrenhan.quant-monitor"
GUI_DOMAIN="gui/$(id -u)"

mkdir -p "$LOG_DIR"

if [[ "$HOST" == "127.0.0.1" && "$PORT" == "8787" && -f "$PLIST" ]]; then
  if ! launchctl print "${GUI_DOMAIN}/${LABEL}" >/dev/null 2>&1; then
    launchctl bootstrap "$GUI_DOMAIN" "$PLIST" >/dev/null 2>&1 || true
  fi
  launchctl enable "${GUI_DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  launchctl kickstart -k "${GUI_DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
else
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] python3 not found."
    exit 1
  fi
  nohup "$PYTHON_BIN" -u ./monitor_panel.py --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
fi

for _ in {1..25}; do
  if curl -fsS "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1; then
    open "http://${HOST}:${PORT}"
    echo "Monitor started: http://${HOST}:${PORT}"
    exit 0
  fi
  sleep 0.3
done

echo "Monitor failed to start. Check logs:"
tail -n 40 "$LOG_FILE" 2>/dev/null || true
tail -n 40 "${LOG_DIR}/monitor_panel.err.log" 2>/dev/null || true
exit 1
