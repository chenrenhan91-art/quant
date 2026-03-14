#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import subprocess
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests


RUNTIME_DIR = Path(__file__).resolve().parent
ENV_FILE = RUNTIME_DIR / ".env"
LOG_FILE = RUNTIME_DIR / "logs" / "launchd.out.log"
LAUNCH_LABEL = "com.chenrenhan.binance-trading-bot"
CACHE_TTL_SEC = 2.0

CACHE_LOCK = threading.Lock()
CACHE_TS = 0.0
CACHE_DATA: dict[str, Any] = {}


def load_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_local_time(ms: Any) -> str:
    try:
        ts = float(ms) / 1000.0
        return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "-"


def env_int(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "1" if default else "0")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def calculate_rqk_simple(closes: list[float], lb: int = 15, w: int = 6) -> float:
    if len(closes) < lb + 1:
        return 0.0
    recent = closes[-lb:]
    high = max(recent)
    low = min(recent)
    current = closes[-1]
    if high == low:
        return 0.0
    rqk = ((current - low) / (high - low)) * 100
    return rqk - 50


def calculate_rsi_simple(closes: list[float], period: int = 21) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(closes: list[float], span: int) -> list[float]:
    if not closes:
        return []
    k = 2 / (span + 1)
    ema = [closes[0]]
    for c in closes[1:]:
        ema.append(c * k + ema[-1] * (1 - k))
    return ema


def calc_rqk_series(closes: list[float], lb: int = 15, r: int = 6) -> list[float]:
    n = len(closes)
    res: list[float] = []
    for i in range(n):
        if i < lb:
            res.append(closes[i])
            continue
        ws = 0.0
        vs = 0.0
        for j in range(max(0, i - lb), i + 1):
            d = (i - j) / lb
            w = (1 + d * d / (2 * r)) ** (-r)
            ws += w
            vs += closes[j] * w
        res.append(vs / ws if ws else closes[i])
    return res


def calc_rsi_series(closes: list[float], period: int = 21) -> list[float]:
    n = len(closes)
    if n == 0:
        return []
    rsi = [50.0] * n
    if n < period + 1:
        return rsi

    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rs = avg_gain / avg_loss if avg_loss else float("inf")
    rsi[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, n):
        d = deltas[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
        rs = avg_gain / avg_loss if avg_loss else float("inf")
        rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def calc_stoch_series(highs: list[float], lows: list[float], closes: list[float], kp: int = 14) -> tuple[list[float], list[float]]:
    k: list[float] = []
    for i in range(len(closes)):
        if i < kp - 1:
            k.append(50.0)
        else:
            wh = max(highs[i - kp + 1:i + 1])
            wl = min(lows[i - kp + 1:i + 1])
            k.append(100 * (closes[i] - wl) / (wh - wl) if wh != wl else 50.0)

    d: list[float] = []
    for i in range(len(k)):
        if i < 2:
            d.append(50.0)
        else:
            d.append((k[i] + k[i - 1] + k[i - 2]) / 3.0)
    return k, d


def calc_macd_hist_series(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> list[float]:
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    if len(macd_line) <= slow:
        return [0.0] * len(closes)

    signal_line = calc_ema(macd_line[slow:], signal)
    signal_line = [0.0] * slow + signal_line
    if len(signal_line) < len(macd_line):
        signal_line.extend([signal_line[-1]] * (len(macd_line) - len(signal_line)))
    elif len(signal_line) > len(macd_line):
        signal_line = signal_line[:len(macd_line)]
    return [m - s for m, s in zip(macd_line, signal_line)]


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def get_process_snapshot() -> dict[str, Any]:
    rc, out, err = run_cmd(["ps", "-axo", "pid,command"])
    lines: list[str] = []
    if rc == 0 and out:
        for raw in out.splitlines():
            line = raw.strip()
            if "trading_bot_simple.py" not in line:
                continue
            lines.append(line)
    runtime_lines: list[str] = []
    extra_lines: list[str] = []
    for line in lines:
        if ".openclaw/workspace/" in line:
            extra_lines.append(line)
            continue
        runtime_lines.append(line)
    return {
        "runtime_processes": runtime_lines,
        "extra_processes": extra_lines,
        "error": err if rc != 0 else "",
    }


def get_launchd_snapshot(label: str) -> dict[str, Any]:
    rc, out, err = run_cmd(["launchctl", "list"])
    if rc != 0:
        return {
            "running": False,
            "pid": None,
            "status_code": None,
            "label": label,
            "error": err or out,
        }
    target_line = ""
    for line in out.splitlines():
        if label in line:
            target_line = line.strip()
            break

    if not target_line:
        return {
            "running": False,
            "pid": None,
            "status_code": None,
            "label": label,
            "error": "label not found in launchctl list",
        }

    parts = target_line.split()
    pid = None
    status_code = None
    if len(parts) >= 3:
        try:
            pid = int(parts[0])
        except ValueError:
            pid = None
        try:
            status_code = int(parts[1])
        except ValueError:
            status_code = None
    return {
        "running": pid is not None and pid > 0,
        "pid": pid,
        "status_code": status_code,
        "label": label,
        "error": "",
    }


def tail_lines(path: Path, count: int = 500) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=count)]


def find_last_line(lines: list[str], keywords: tuple[str, ...]) -> str:
    for line in reversed(lines):
        if any(key in line for key in keywords):
            return line
    return ""


def parse_log_snapshot(log_path: Path) -> dict[str, Any]:
    lines = tail_lines(log_path, count=600)
    session_start = 0
    for idx, line in enumerate(lines):
        if "simplified" in line.lower():
            session_start = idx
        if "简化版交易机器人启动" in line:
            session_start = idx
    active = lines[session_start:] if lines else []
    stat = log_path.stat() if log_path.exists() else None
    return {
        "path": str(log_path),
        "exists": log_path.exists(),
        "updated_at": dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S") if stat else "-",
        "strategy_line": find_last_line(active, ("策略:", "Strategy:")),
        "mode_line": find_last_line(active, ("自动下单:", "AUTO_TRADE")),
        "risk_line": find_last_line(active, ("风控:", "保护单:")),
        "last_price_line": find_last_line(active, ("价格:", "price:")),
        "last_signal_line": find_last_line(active, ("📢 信号", "signal")),
        "last_order_line": find_last_line(active, ("✅ 已开仓", "✅ 平仓成功", "❌ 开仓失败", "❌ 平仓失败")),
        "last_protect_line": find_last_line(active, ("🛡️", "保护单")),
        "last_error_line": find_last_line(active, ("❌", "ERROR", "Error")),
        "recent_lines": active[-40:],
    }


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, proxy_url: str, recv_window: int = 5000):
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self.base = "https://fapi.binance.com"
        self.session = requests.Session()
        self.session.trust_env = False
        if proxy_url:
            self.session.proxies.update({"http": proxy_url, "https": proxy_url})

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        timeout: float = 10.0,
    ) -> tuple[Any, str]:
        payload: dict[str, Any] = dict(params or {})
        if signed:
            payload["timestamp"] = int(time.time() * 1000)
            payload["recvWindow"] = self.recv_window
            query = urlencode(payload, doseq=True)
            signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            payload["signature"] = signature

        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        url = f"{self.base}{endpoint}"
        try:
            resp = self.session.request(method, url, params=payload or None, headers=headers, timeout=timeout)
        except Exception as exc:
            return None, f"request failed: {exc}"

        body = (resp.text or "").strip()
        if not resp.ok:
            short_body = body[:300] + ("..." if len(body) > 300 else "")
            return None, f"HTTP {resp.status_code}: {short_body}"

        if not body:
            return {}, ""
        try:
            return resp.json(), ""
        except ValueError:
            return None, "invalid JSON response"

    def public_get(self, endpoint: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        return self._request("GET", endpoint, params=params, signed=False)

    def signed_get(self, endpoint: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        return self._request("GET", endpoint, params=params, signed=True)


def create_binance_client(env: dict[str, str]) -> tuple[BinanceClient | None, str]:
    api_key = env.get("BINANCE_API_KEY", "")
    api_secret = env.get("BINANCE_API_SECRET", "")
    proxy_url = env.get("BINANCE_PROXY_URL", "")
    recv_window = env_int(env, "BINANCE_RECV_WINDOW", 5000)
    if not api_key or not api_secret:
        return None, "BINANCE_API_KEY/API_SECRET missing in .env"
    return BinanceClient(api_key, api_secret, proxy_url, recv_window=recv_window), ""


def collect_indicator_panel(env: dict[str, str], symbol: str, client: BinanceClient | None) -> dict[str, Any]:
    if client is None:
        return {"ok": False, "error": "missing Binance client"}

    interval = env.get("SIGNAL_INTERVAL", "15m")
    strict_mode = env_bool(env, "STRICT_BACKTEST_MODE", True)
    quality_lookback = env_int(env, "LONG_SCORE2_QUALITY_LOOKBACK_BARS", 1000)
    enable_quality_tiering = env_bool(env, "ENABLE_LONG_SCORE2_QUALITY_TIERING", True)
    kline_limit = max(220, quality_lookback) if enable_quality_tiering else 220
    klines, klines_err = client.public_get(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": kline_limit},
    )
    if klines_err:
        return {"ok": False, "error": f"fetch klines failed: {klines_err}"}
    if not isinstance(klines, list) or len(klines) < 3:
        return {"ok": False, "error": "insufficient klines"}

    confirmed = klines[:-1]
    opens = [float(k[1]) for k in confirmed]
    highs = [float(k[2]) for k in confirmed]
    lows = [float(k[3]) for k in confirmed]
    closes = [float(k[4]) for k in confirmed]
    bar_open_ts = int(confirmed[-1][0])
    bar_close_ts = int(confirmed[-1][6])

    if strict_mode:
        strict_rqk_lb = env_int(env, "STRICT_RQK_LB", 15)
        strict_rqk_w = env_int(env, "STRICT_RQK_W", 6)
        strict_rsi_period = env_int(env, "STRICT_RSI_PERIOD", 21)
        strict_rsi_os = env_float(env, "STRICT_RSI_OVERSOLD", 40.0)
        strict_rsi_ob = env_float(env, "STRICT_RSI_OVERBOUGHT", 60.0)
        strict_macd_fast = env_int(env, "STRICT_MACD_FAST", 12)
        strict_macd_slow = env_int(env, "STRICT_MACD_SLOW", 26)
        strict_macd_sig = env_int(env, "STRICT_MACD_SIG", 9)
        strict_confirm = env_int(env, "STRICT_CONFIRM", 2)
        strict_stoch_k = env_int(env, "STRICT_STOCH_K", 14)
        strict_stoch_os = env_float(env, "STRICT_STOCH_OS", 35.0)
        strict_stoch_ob = env_float(env, "STRICT_STOCH_OB", 65.0)

        min_bars = max(
            strict_macd_slow + strict_macd_sig + 5,
            strict_rsi_period + 2,
            strict_rqk_lb + 2,
            strict_stoch_k + 2,
        )
        if len(closes) < min_bars:
            return {
                "ok": False,
                "error": f"kline bars not enough ({len(closes)}/{min_bars})",
                "mode": "strict",
                "interval": interval,
            }

        rqk = calc_rqk_series(closes, strict_rqk_lb, strict_rqk_w)
        rsi = calc_rsi_series(closes, strict_rsi_period)
        macd = calc_macd_hist_series(closes, strict_macd_fast, strict_macd_slow, strict_macd_sig)
        stoch_k, stoch_d = calc_stoch_series(highs, lows, closes, strict_stoch_k)

        i = len(closes) - 1
        rb = rqk[i] > rqk[i - 1]
        rs = rqk[i] < rqk[i - 1]
        ib = rsi[i] > strict_rsi_os and rsi[i - 1] <= strict_rsi_os
        is_ = rsi[i] < strict_rsi_ob and rsi[i - 1] >= strict_rsi_ob
        mb = macd[i - 1] <= 0 and macd[i] > 0
        ms = macd[i - 1] >= 0 and macd[i] < 0
        sb = stoch_k[i] > stoch_d[i] and stoch_k[i] < strict_stoch_os
        ss = stoch_k[i] < stoch_d[i] and stoch_k[i] > strict_stoch_ob

        long_count = sum([rb, ib, mb, sb])
        short_count = sum([rs, is_, ms, ss])
        long_ready = rb and long_count >= strict_confirm
        short_ready = rs and short_count >= strict_confirm
        signal = "long" if long_ready else "short" if short_ready else None

        return {
            "ok": True,
            "error": "",
            "mode": "strict",
            "interval": interval,
            "bar_open_time": to_local_time(bar_open_ts),
            "bar_close_time": to_local_time(bar_close_ts),
            "confirm": strict_confirm,
            "long_count": long_count,
            "short_count": short_count,
            "long_ready": long_ready,
            "short_ready": short_ready,
            "signal": signal,
            "values": {
                "close": closes[i],
                "rqk": rqk[i],
                "rqk_prev": rqk[i - 1],
                "rsi": rsi[i],
                "rsi_prev": rsi[i - 1],
                "macd_hist": macd[i],
                "macd_hist_prev": macd[i - 1],
                "stoch_k": stoch_k[i],
                "stoch_d": stoch_d[i],
            },
            "conditions": [
                {
                    "name": "RQK Direction",
                    "long": rb,
                    "short": rs,
                    "detail": f"RQK {rqk[i]:.3f} vs prev {rqk[i - 1]:.3f}",
                },
                {
                    "name": "RSI Cross",
                    "long": ib,
                    "short": is_,
                    "detail": f"RSI {rsi[i]:.2f} (OS={strict_rsi_os:.1f}, OB={strict_rsi_ob:.1f})",
                },
                {
                    "name": "MACD Hist Cross 0",
                    "long": mb,
                    "short": ms,
                    "detail": f"MACDh {macd[i]:.4f} vs prev {macd[i - 1]:.4f}",
                },
                {
                    "name": "Stoch K/D",
                    "long": sb,
                    "short": ss,
                    "detail": f"K/D {stoch_k[i]:.2f}/{stoch_d[i]:.2f} (OS={strict_stoch_os:.1f}, OB={strict_stoch_ob:.1f})",
                },
            ],
        }

    rqk_simple = calculate_rqk_simple(closes)
    rsi_simple = calculate_rsi_simple(closes)
    long_rqk = rqk_simple > 10
    short_rqk = rqk_simple < -10
    long_rsi = rsi_simple < 40
    short_rsi = rsi_simple > 60
    long_count = sum([long_rqk, long_rsi])
    short_count = sum([short_rqk, short_rsi])
    long_ready = long_count >= 2
    short_ready = short_count >= 2
    signal = "long" if long_ready else "short" if short_ready else None
    return {
        "ok": True,
        "error": "",
        "mode": "simple",
        "interval": interval,
        "bar_open_time": to_local_time(bar_open_ts),
        "bar_close_time": to_local_time(bar_close_ts),
        "confirm": 2,
        "long_count": long_count,
        "short_count": short_count,
        "long_ready": long_ready,
        "short_ready": short_ready,
        "signal": signal,
        "values": {"close": closes[-1], "rqk": rqk_simple, "rsi": rsi_simple},
        "conditions": [
            {
                "name": "RQK Threshold",
                "long": long_rqk,
                "short": short_rqk,
                "detail": f"RQK {rqk_simple:.3f} (long>10 / short<-10)",
            },
            {
                "name": "RSI Threshold",
                "long": long_rsi,
                "short": short_rsi,
                "detail": f"RSI {rsi_simple:.2f} (long<40 / short>60)",
            },
        ],
    }


def collect_binance_snapshot(symbol: str, client: BinanceClient | None) -> dict[str, Any]:
    if client is None:
        return {"ok": False, "error": "missing Binance client"}

    ping, ping_err = client.public_get("/fapi/v1/ping")
    ticker, ticker_err = client.public_get("/fapi/v1/ticker/price", {"symbol": symbol})
    account, account_err = client.signed_get("/fapi/v2/account")
    position, pos_err = client.signed_get("/fapi/v2/positionRisk", {"symbol": symbol})
    open_orders, orders_err = client.signed_get("/fapi/v1/openOrders", {"symbol": symbol})
    trades, trades_err = client.signed_get("/fapi/v1/userTrades", {"symbol": symbol, "limit": 20})

    errors = [msg for msg in (ping_err, ticker_err, account_err, pos_err, orders_err, trades_err) if msg]

    account_summary: dict[str, Any] = {}
    if isinstance(account, dict):
        account_summary = {
            "can_trade": bool(account.get("canTrade")),
            "available_balance": to_float(account.get("availableBalance")),
            "wallet_balance": to_float(account.get("totalWalletBalance")),
            "margin_balance": to_float(account.get("totalMarginBalance")),
            "unrealized_pnl": to_float(account.get("totalUnrealizedProfit")),
        }

    current_price = to_float((ticker or {}).get("price")) if isinstance(ticker, dict) else 0.0

    positions: list[dict[str, Any]] = []
    source_positions = position if isinstance(position, list) else [position] if isinstance(position, dict) else []
    for pos in source_positions:
        amount = to_float(pos.get("positionAmt"))
        if abs(amount) < 1e-12:
            continue
        positions.append(
            {
                "symbol": pos.get("symbol", symbol),
                "side": "LONG" if amount > 0 else "SHORT",
                "amount": amount,
                "entry_price": to_float(pos.get("entryPrice")),
                "mark_price": to_float(pos.get("markPrice")),
                "unrealized_pnl": to_float(pos.get("unRealizedProfit")),
                "leverage": int(to_float(pos.get("leverage"), default=0)),
                "liquidation_price": to_float(pos.get("liquidationPrice")),
                "margin_type": pos.get("marginType", ""),
                "position_side": pos.get("positionSide", ""),
            }
        )

    parsed_orders: list[dict[str, Any]] = []
    if isinstance(open_orders, list):
        for order in sorted(open_orders, key=lambda item: int(item.get("updateTime") or item.get("time") or 0), reverse=True):
            parsed_orders.append(
                {
                    "order_id": order.get("orderId"),
                    "side": order.get("side"),
                    "type": order.get("type"),
                    "status": order.get("status"),
                    "qty": to_float(order.get("origQty")),
                    "price": to_float(order.get("price")),
                    "stop_price": to_float(order.get("stopPrice")),
                    "reduce_only": bool(order.get("reduceOnly")),
                    "time": to_local_time(order.get("updateTime") or order.get("time")),
                }
            )

    parsed_trades: list[dict[str, Any]] = []
    if isinstance(trades, list):
        for trade in sorted(trades, key=lambda item: int(item.get("time") or 0), reverse=True):
            parsed_trades.append(
                {
                    "time": to_local_time(trade.get("time")),
                    "side": trade.get("side"),
                    "price": to_float(trade.get("price")),
                    "qty": to_float(trade.get("qty")),
                    "realized_pnl": to_float(trade.get("realizedPnl")),
                    "commission": to_float(trade.get("commission")),
                    "order_id": trade.get("orderId"),
                }
            )

    return {
        "ok": len(errors) == 0,
        "error": "; ".join(errors),
        "ping_ok": isinstance(ping, dict),
        "current_price": current_price,
        "account": account_summary,
        "positions": positions,
        "open_orders": parsed_orders,
        "recent_trades": parsed_trades,
    }


def build_snapshot() -> dict[str, Any]:
    env = load_env(ENV_FILE)
    symbol = env.get("SYMBOL", "BTCUSDT") or "BTCUSDT"
    client, client_err = create_binance_client(env)
    bot_process = get_process_snapshot()
    launchd = get_launchd_snapshot(LAUNCH_LABEL)
    logs = parse_log_snapshot(LOG_FILE)
    binance = collect_binance_snapshot(symbol=symbol, client=client)
    indicator_panel = collect_indicator_panel(env=env, symbol=symbol, client=client)
    if client is None:
        if not binance.get("error"):
            binance["error"] = client_err
        if not indicator_panel.get("error"):
            indicator_panel["error"] = client_err

    return {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "bot": {
            "launchd": launchd,
            "processes": bot_process["runtime_processes"],
            "extra_processes": bot_process["extra_processes"],
            "process_error": bot_process["error"],
            "auto_trade": env.get("AUTO_TRADE", ""),
            "strict_backtest_mode": env.get("STRICT_BACKTEST_MODE", ""),
            "signal_interval": env.get("SIGNAL_INTERVAL", ""),
            "proxy_url": env.get("BINANCE_PROXY_URL", ""),
            "proxy_ip_expect": env.get("BINANCE_PROXY_IP", ""),
            "api_key_masked": mask_key(env.get("BINANCE_API_KEY", "")),
            "strategy": {
                "confirm": env.get("STRICT_CONFIRM", ""),
                "sl": env.get("STOP_LOSS_PCT", ""),
                "tp": env.get("TAKE_PROFIT_PCT", ""),
                "score2": env.get("POSITION_PCT_SCORE2", "0.17"),
                "score3": env.get("POSITION_PCT_SCORE3", "0.30"),
                "score4": env.get("POSITION_PCT_SCORE4", "0.30"),
                "long_multiplier": env.get("LONG_SCORE_POSITION_MULTIPLIER", ""),
                "short_multiplier": env.get("SHORT_SCORE_POSITION_MULTIPLIER", ""),
                "target120": env.get("ENABLE_TARGET_STRATEGY_120", ""),
            },
        },
        "binance": binance,
        "indicator_panel": indicator_panel,
        "logs": logs,
    }


def get_snapshot() -> dict[str, Any]:
    global CACHE_TS, CACHE_DATA
    now = time.time()
    with CACHE_LOCK:
        if CACHE_DATA and now - CACHE_TS < CACHE_TTL_SEC:
            return CACHE_DATA
    data = build_snapshot()
    with CACHE_LOCK:
        CACHE_DATA = data
        CACHE_TS = now
    return data


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quant Bot Monitor</title>
  <style>
    :root {
      --bg1: #08131f;
      --bg2: #10273b;
      --panel: rgba(8, 23, 36, 0.88);
      --text: #e9f5ff;
      --muted: #89a4b8;
      --ok: #37d39b;
      --warn: #ffc857;
      --bad: #ff6b6b;
      --line: rgba(139, 176, 201, 0.20);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "SF Pro Display", "Helvetica Neue", sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 500px at -10% -20%, #19486d 0%, transparent 60%),
        radial-gradient(1000px 600px at 110% -30%, #175840 0%, transparent 55%),
        linear-gradient(150deg, var(--bg1), var(--bg2));
      min-height: 100vh;
      letter-spacing: 0.2px;
    }
    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px 16px 24px;
      display: grid;
      gap: 12px;
    }
    .top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(6, 17, 26, 0.70);
      backdrop-filter: blur(8px);
      animation: rise 0.45s ease-out both;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .title {
      font-weight: 700;
      font-size: 20px;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .badge {
      font-size: 12px;
      font-weight: 700;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: #08211a;
      background: var(--ok);
    }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      animation: rise 0.55s ease-out both;
    }
    .card {
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.9px; }
    .value { margin-top: 7px; font-size: 24px; font-weight: 700; }
    .hint { margin-top: 6px; color: var(--muted); font-size: 12px; line-height: 1.45; }
    .layout {
      display: grid;
      gap: 12px;
      grid-template-columns: 1.2fr 1fr;
      animation: rise 0.65s ease-out both;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 14px;
    }
    .panel h3 {
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      color: #d2e9fa;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      text-align: left;
      border-bottom: 1px solid rgba(130, 166, 188, 0.13);
      padding: 8px 6px;
      font-size: 13px;
    }
    th { color: #9ec1d8; font-weight: 600; }
    td { color: #edf7ff; }
    .mono { font-family: "SF Mono", Menlo, Monaco, monospace; }
    pre {
      margin: 0;
      max-height: 320px;
      overflow: auto;
      white-space: pre-wrap;
      background: rgba(4, 11, 18, 0.75);
      border: 1px solid rgba(114, 153, 180, 0.25);
      border-radius: 10px;
      padding: 10px;
      line-height: 1.45;
      font-size: 12px;
    }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .lamp {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.15) inset;
    }
    .lamp.green { background: #2fd792; box-shadow: 0 0 10px rgba(47, 215, 146, 0.75); }
    .lamp.red { background: #ff6b6b; box-shadow: 0 0 10px rgba(255, 107, 107, 0.7); }
    .indicator-summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 8px;
    }
    .mini-box {
      border: 1px solid rgba(132, 170, 193, 0.2);
      border-radius: 10px;
      padding: 10px;
      background: rgba(4, 14, 21, 0.68);
    }
    .mini-title {
      color: #99bdd4;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 4px;
    }
    .mini-value {
      font-size: 18px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .state-cell {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-weight: 600;
    }
    .detail-cell {
      color: #bfd7e8;
      font-size: 12px;
      font-family: "SF Mono", Menlo, Monaco, monospace;
    }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .grid { grid-template-columns: 1fr; }
      .top { flex-direction: column; align-items: flex-start; }
      .meta { justify-content: flex-start; }
      .value { font-size: 20px; }
      .indicator-summary { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="top">
      <div>
        <div class="title">Quant Bot Live Monitor</div>
        <div id="subline" class="hint">Loading...</div>
      </div>
      <div class="meta">
        <div class="badge" id="badge">CONNECTING</div>
        <div id="time">-</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <div class="label">Bot Service</div>
        <div class="value" id="botStatus">-</div>
        <div class="hint" id="botHint">-</div>
      </div>
      <div class="card">
        <div class="label">Auto Trade</div>
        <div class="value" id="autoTrade">-</div>
        <div class="hint" id="strategyHint">-</div>
      </div>
      <div class="card">
        <div class="label">BTCUSDT Price</div>
        <div class="value" id="price">-</div>
        <div class="hint" id="pnlHint">-</div>
      </div>
      <div class="card">
        <div class="label">Open Position</div>
        <div class="value" id="positionCount">-</div>
        <div class="hint" id="ordersHint">-</div>
      </div>
    </section>

    <section class="panel">
      <h3>Current Candle Indicator Board</h3>
      <div class="indicator-summary">
        <div class="mini-box">
          <div class="mini-title">Long Conditions</div>
          <div class="mini-value" id="longSummary">-</div>
        </div>
        <div class="mini-box">
          <div class="mini-title">Short Conditions</div>
          <div class="mini-value" id="shortSummary">-</div>
        </div>
        <div class="mini-box">
          <div class="mini-title">Current Signal Gate</div>
          <div class="mini-value" id="signalSummary">-</div>
        </div>
      </div>
      <div class="hint" id="indicatorMeta">-</div>
      <table>
        <thead>
          <tr><th>Condition</th><th>Long</th><th>Short</th><th>Detail</th></tr>
        </thead>
        <tbody id="indicatorBody"></tbody>
      </table>
    </section>

    <section class="layout">
      <div class="panel">
        <h3>Account Summary</h3>
        <table>
          <tbody id="accountBody"></tbody>
        </table>
      </div>
      <div class="panel">
        <h3>Latest Bot Signals</h3>
        <table>
          <tbody id="signalBody"></tbody>
        </table>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h3>Positions</h3>
        <table>
          <thead>
            <tr><th>Side</th><th>Qty</th><th>Entry</th><th>Mark</th><th>UPnL</th><th>Lev</th></tr>
          </thead>
          <tbody id="positionsBody"></tbody>
        </table>
      </div>
      <div class="panel">
        <h3>Open Orders</h3>
        <table>
          <thead>
            <tr><th>Time</th><th>Side</th><th>Type</th><th>Qty</th><th>Price/Stop</th><th>Status</th></tr>
          </thead>
          <tbody id="ordersBody"></tbody>
        </table>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h3>Recent Trades</h3>
        <table>
          <thead>
            <tr><th>Time</th><th>Side</th><th>Qty</th><th>Price</th><th>Realized PnL</th><th>Fee</th></tr>
          </thead>
          <tbody id="tradesBody"></tbody>
        </table>
      </div>
      <div class="panel">
        <h3>Recent Logs (current run)</h3>
        <pre id="logs">Waiting for log stream...</pre>
      </div>
    </section>
  </div>

  <script>
    const fmt2 = (n) => {
      const v = Number(n);
      if (!Number.isFinite(v)) return "-";
      return v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    };
    const fmt4 = (n) => {
      const v = Number(n);
      if (!Number.isFinite(v)) return "-";
      return v.toLocaleString("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
    };
    const safe = (v) => (v === null || v === undefined || v === "" ? "-" : String(v));
    const yesNo = (v) => (v ? "YES" : "NO");
    const lamp = (on) => `<span class="lamp ${on ? "green" : "red"}"></span>`;

    function setBadge(ok, text) {
      const badge = document.getElementById("badge");
      badge.textContent = text;
      badge.style.background = ok ? "var(--ok)" : "var(--bad)";
    }

    function row(key, val, cls = "") {
      return `<tr><td class="label">${key}</td><td class="${cls} mono">${val}</td></tr>`;
    }

    function render(data) {
      const binance = data.binance || {};
      const bot = data.bot || {};
      const launchd = bot.launchd || {};
      const account = binance.account || {};
      const logs = data.logs || {};
      const indicator = data.indicator_panel || {};
      const positions = Array.isArray(binance.positions) ? binance.positions : [];
      const orders = Array.isArray(binance.open_orders) ? binance.open_orders : [];
      const trades = Array.isArray(binance.recent_trades) ? binance.recent_trades : [];

      const launchOk = Boolean(launchd.running) && Array.isArray(bot.processes) && bot.processes.length > 0;
      setBadge(launchOk && binance.ok, launchOk && binance.ok ? "HEALTHY" : "ATTENTION");
      document.getElementById("time").textContent = `Updated: ${safe(data.timestamp)}`;
      document.getElementById("subline").textContent = `Symbol ${safe(data.symbol)} | API ${safe(bot.api_key_masked)} | Proxy ${safe(bot.proxy_url)}`;

      document.getElementById("botStatus").textContent = launchOk ? "RUNNING" : "STOPPED";
      document.getElementById("botStatus").className = `value ${launchOk ? "ok" : "bad"}`;
      const extraCount = Array.isArray(bot.extra_processes) ? bot.extra_processes.length : 0;
      document.getElementById("botHint").textContent =
        `launchd pid=${safe(launchd.pid)} | active process=${(bot.processes || []).length} | extra process=${extraCount}`;

      const autoTradeOn = safe(bot.auto_trade) === "1";
      document.getElementById("autoTrade").textContent = autoTradeOn ? "ENABLED" : "DISABLED";
      document.getElementById("autoTrade").className = `value ${autoTradeOn ? "warn" : ""}`;
      const strategy = bot.strategy || {};
      document.getElementById("strategyHint").textContent =
        `confirm=${safe(strategy.confirm)} | SL=${safe(strategy.sl)} TP=${safe(strategy.tp)} | xL=${safe(strategy.long_multiplier)} xS=${safe(strategy.short_multiplier)}`;

      document.getElementById("price").textContent = binance.current_price ? `$${fmt2(binance.current_price)}` : "-";
      document.getElementById("pnlHint").textContent =
        `U-PnL: ${fmt2(account.unrealized_pnl)} | Wallet: ${fmt2(account.wallet_balance)} | Available: ${fmt2(account.available_balance)}`;

      document.getElementById("positionCount").textContent = String(positions.length);
      document.getElementById("ordersHint").textContent = `Open orders: ${orders.length} | Recent trades: ${trades.length}`;

      const longCount = Number(indicator.long_count || 0);
      const shortCount = Number(indicator.short_count || 0);
      const confirmN = Number(indicator.confirm || 0);
      const longReady = Boolean(indicator.long_ready);
      const shortReady = Boolean(indicator.short_ready);
      const mode = safe(indicator.mode);
      const signal = safe(indicator.signal || "none").toUpperCase();
      const conds = Array.isArray(indicator.conditions) ? indicator.conditions : [];
      const totalConds = conds.length || Math.max(confirmN, 1);

      document.getElementById("longSummary").innerHTML = `${lamp(longReady)} ${longCount}/${totalConds}${confirmN ? ` (need ${confirmN})` : ""}`;
      document.getElementById("shortSummary").innerHTML = `${lamp(shortReady)} ${shortCount}/${totalConds}${confirmN ? ` (need ${confirmN})` : ""}`;
      const signalLamp = signal === "LONG" || signal === "SHORT";
      document.getElementById("signalSummary").innerHTML = `${lamp(signalLamp)} ${signal}`;
      document.getElementById("indicatorMeta").textContent =
        indicator.ok
          ? `mode=${mode} | interval=${safe(indicator.interval)} | candle ${safe(indicator.bar_open_time)} ~ ${safe(indicator.bar_close_time)}`
          : `indicator unavailable: ${safe(indicator.error)}`;

      const indicatorRows = conds.length
        ? conds.map((c) => `<tr>
            <td>${safe(c.name)}</td>
            <td><span class="state-cell">${lamp(Boolean(c.long))}<span>${Boolean(c.long) ? "PASS" : "MISS"}</span></span></td>
            <td><span class="state-cell">${lamp(Boolean(c.short))}<span>${Boolean(c.short) ? "PASS" : "MISS"}</span></span></td>
            <td class="detail-cell">${safe(c.detail)}</td>
          </tr>`).join("")
        : `<tr><td colspan="4" class="hint">${safe(indicator.error || "No indicator data.")}</td></tr>`;
      document.getElementById("indicatorBody").innerHTML = indicatorRows;

      const accountRows = [
        row("Can Trade", yesNo(account.can_trade), account.can_trade ? "ok" : "bad"),
        row("Available Balance", fmt2(account.available_balance)),
        row("Wallet Balance", fmt2(account.wallet_balance)),
        row("Margin Balance", fmt2(account.margin_balance)),
        row("Unrealized PnL", fmt2(account.unrealized_pnl), Number(account.unrealized_pnl) >= 0 ? "ok" : "bad"),
        row("Ping", yesNo(binance.ping_ok), binance.ping_ok ? "ok" : "bad"),
        row("API Error", safe(binance.error || "-"), binance.error ? "bad" : "ok")
      ];
      document.getElementById("accountBody").innerHTML = accountRows.join("");

      const signalRows = [
        row("Strategy", safe(logs.strategy_line)),
        row("Mode", safe(logs.mode_line)),
        row("Risk", safe(logs.risk_line)),
        row("Price line", safe(logs.last_price_line)),
        row("Signal line", safe(logs.last_signal_line)),
        row("Order line", safe(logs.last_order_line)),
        row("Protection", safe(logs.last_protect_line)),
        row("Last error", safe(logs.last_error_line), logs.last_error_line ? "bad" : "ok"),
        row("Log updated", safe(logs.updated_at))
      ];
      document.getElementById("signalBody").innerHTML = signalRows.join("");

      const posRows = positions.length
        ? positions.map((p) => `<tr>
            <td class="${p.side === "LONG" ? "ok" : "bad"}">${safe(p.side)}</td>
            <td>${fmt4(p.amount)}</td>
            <td>${fmt2(p.entry_price)}</td>
            <td>${fmt2(p.mark_price)}</td>
            <td class="${Number(p.unrealized_pnl) >= 0 ? "ok" : "bad"}">${fmt2(p.unrealized_pnl)}</td>
            <td>${safe(p.leverage)}x</td>
          </tr>`).join("")
        : `<tr><td colspan="6" class="hint">No open positions.</td></tr>`;
      document.getElementById("positionsBody").innerHTML = posRows;

      const orderRows = orders.length
        ? orders.map((o) => `<tr>
            <td>${safe(o.time)}</td>
            <td>${safe(o.side)}</td>
            <td>${safe(o.type)}</td>
            <td>${fmt4(o.qty)}</td>
            <td>${fmt2(o.price)} / ${fmt2(o.stop_price)}</td>
            <td>${safe(o.status)}</td>
          </tr>`).join("")
        : `<tr><td colspan="6" class="hint">No open orders.</td></tr>`;
      document.getElementById("ordersBody").innerHTML = orderRows;

      const tradeRows = trades.length
        ? trades.map((t) => `<tr>
            <td>${safe(t.time)}</td>
            <td>${safe(t.side)}</td>
            <td>${fmt4(t.qty)}</td>
            <td>${fmt2(t.price)}</td>
            <td class="${Number(t.realized_pnl) >= 0 ? "ok" : "bad"}">${fmt2(t.realized_pnl)}</td>
            <td>${fmt4(t.commission)}</td>
          </tr>`).join("")
        : `<tr><td colspan="6" class="hint">No recent trades.</td></tr>`;
      document.getElementById("tradesBody").innerHTML = tradeRows;

      const recent = Array.isArray(logs.recent_lines) ? logs.recent_lines : [];
      document.getElementById("logs").textContent = recent.length ? recent.join("\\n") : "No logs found.";
    }

    async function refresh() {
      try {
        const res = await fetch(`/api/status?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        render(data);
      } catch (err) {
        setBadge(false, "OFFLINE");
        document.getElementById("subline").textContent = `Dashboard error: ${err.message}`;
      }
    }

    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "QuantMonitor/1.0"

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML_PAGE)
            return
        if path == "/api/status":
            self._send_json(get_snapshot())
            return
        if path == "/healthz":
            self._send_json({"ok": True, "time": dt.datetime.now().isoformat()})
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local monitor panel for quant trading bot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"[monitor] listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
