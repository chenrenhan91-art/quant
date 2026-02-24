#!/usr/bin/env python3
"""
简化版量化交易机器人 - 使用 requests 连接 Binance API
"""

import os
import time
import hmac
import hashlib
import atexit
import fcntl
import json
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from urllib.parse import urlencode

import requests

session = requests.Session()
session.trust_env = False  # 只使用显式配置的代理，避免系统环境干扰

# ============== 配置 ==============
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')
PROXY_URL = os.getenv('BINANCE_PROXY_URL', 'http://127.0.0.1:7890')
EXPECTED_PROXY_IP = os.getenv('BINANCE_PROXY_IP', '103.36.25.66')
RECV_WINDOW = int(os.getenv('BINANCE_RECV_WINDOW', '5000'))
REQUEST_RETRIES = int(os.getenv('BINANCE_REQUEST_RETRIES', '2'))
RETRY_SLEEP_SEC = float(os.getenv('BINANCE_RETRY_SLEEP_SEC', '0.6'))

SYMBOL = 'BTCUSDT'
POSITION_PCT = 0.25
LEVERAGE = 10
STRICT_BACKTEST_MODE = os.getenv('STRICT_BACKTEST_MODE', '1').lower() in ('1', 'true', 'yes', 'on')
SIGNAL_INTERVAL = os.getenv('SIGNAL_INTERVAL', '15m')
AUTO_TRADE = os.getenv('AUTO_TRADE', '1').lower() in ('1', 'true', 'yes', 'on')
ALLOW_REVERSE = os.getenv('ALLOW_REVERSE', '1').lower() in ('1', 'true', 'yes', 'on')
ALLOW_CONTINUOUS_ADD = os.getenv('ALLOW_CONTINUOUS_ADD', '1').lower() in ('1', 'true', 'yes', 'on')
ONE_ORDER_PER_CANDLE = os.getenv('ONE_ORDER_PER_CANDLE', '1').lower() in ('1', 'true', 'yes', 'on')
MIN_MARGIN_USDT = float(os.getenv('MIN_MARGIN_USDT', '20'))
MIN_NOTIONAL_USDT = float(os.getenv('MIN_NOTIONAL_USDT', '100'))
MAX_SLIPPAGE_PCT = float(os.getenv('MAX_SLIPPAGE_PCT', '0.10'))
ENABLE_ORDER_NOTIFY = os.getenv('ENABLE_ORDER_NOTIFY', '1').lower() in ('1', 'true', 'yes', 'on')
WECHAT_SCKEY = os.getenv('WECHAT_SCKEY', '').strip()
ENABLE_PROTECTIVE_ORDERS = os.getenv('ENABLE_PROTECTIVE_ORDERS', '1').lower() in ('1', 'true', 'yes', 'on')
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.008'))
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.012'))
PROTECTIVE_WORKING_TYPE = os.getenv('PROTECTIVE_WORKING_TYPE', 'MARK_PRICE').upper()
MAX_POSITION_OVERSHOOT_PCT = float(os.getenv('MAX_POSITION_OVERSHOOT_PCT', '5'))

# 动态出场与动态仓位（默认对齐“更均衡”参数）
ENABLE_DYNAMIC_STRATEGY = os.getenv('ENABLE_DYNAMIC_STRATEGY', '1').lower() in ('1', 'true', 'yes', 'on')
DYN_ATR_PERIOD = int(os.getenv('DYN_ATR_PERIOD', '14'))
DYN_ADX_PERIOD = int(os.getenv('DYN_ADX_PERIOD', '14'))
DYN_EMA_SPAN = int(os.getenv('DYN_EMA_SPAN', '200'))

DYN_ATR_TARGET_PCT = float(os.getenv('DYN_ATR_TARGET_PCT', '0.50'))  # ATR%目标值，用于仓位归一
DYN_ATR_MIN_SCALE = float(os.getenv('DYN_ATR_MIN_SCALE', '0.05'))
DYN_ADX_TREND = float(os.getenv('DYN_ADX_TREND', '22'))
DYN_ADX_CHOP = float(os.getenv('DYN_ADX_CHOP', '17'))
DYN_ATR_HIGH_VOL_PCT = float(os.getenv('DYN_ATR_HIGH_VOL_PCT', '1.2'))
DYN_MULT_TREND = float(os.getenv('DYN_MULT_TREND', '0.65'))
DYN_MULT_NEUTRAL = float(os.getenv('DYN_MULT_NEUTRAL', '0.35'))
DYN_MULT_CHOP = float(os.getenv('DYN_MULT_CHOP', '0.40'))
DYN_MULT_HIGHVOL = float(os.getenv('DYN_MULT_HIGHVOL', '0.20'))
DYN_BLOCK_HIGHVOL = os.getenv('DYN_BLOCK_HIGHVOL', '0').lower() in ('1', 'true', 'yes', 'on')

DYN_K_SL = float(os.getenv('DYN_K_SL', '6.0'))
DYN_SL_MIN = float(os.getenv('DYN_SL_MIN', '0.020'))
DYN_SL_MAX = float(os.getenv('DYN_SL_MAX', '0.120'))
DYN_RR_TREND = float(os.getenv('DYN_RR_TREND', '3.0'))
DYN_RR_NEUTRAL = float(os.getenv('DYN_RR_NEUTRAL', '2.0'))
DYN_RR_CHOP = float(os.getenv('DYN_RR_CHOP', '1.6'))
DYN_RR_HIGHVOL = float(os.getenv('DYN_RR_HIGHVOL', '1.2'))
DYN_TP_MIN = float(os.getenv('DYN_TP_MIN', '0.020'))
DYN_TP_MAX = float(os.getenv('DYN_TP_MAX', '0.250'))

DYN_TRAIL_ENABLE = os.getenv('DYN_TRAIL_ENABLE', '1').lower() in ('1', 'true', 'yes', 'on')
DYN_TRAIL_TRIGGER_R = float(os.getenv('DYN_TRAIL_TRIGGER_R', '1.5'))
DYN_TRAIL_K_ATR = float(os.getenv('DYN_TRAIL_K_ATR', '2.0'))

STRICT_RQK_LB = int(os.getenv('STRICT_RQK_LB', '15'))
STRICT_RQK_W = int(os.getenv('STRICT_RQK_W', '6'))
STRICT_RSI_PERIOD = int(os.getenv('STRICT_RSI_PERIOD', '21'))
STRICT_RSI_OVERSOLD = float(os.getenv('STRICT_RSI_OVERSOLD', '40'))
STRICT_RSI_OVERBOUGHT = float(os.getenv('STRICT_RSI_OVERBOUGHT', '60'))
STRICT_MACD_FAST = int(os.getenv('STRICT_MACD_FAST', '12'))
STRICT_MACD_SLOW = int(os.getenv('STRICT_MACD_SLOW', '26'))
STRICT_MACD_SIG = int(os.getenv('STRICT_MACD_SIG', '9'))
STRICT_CONFIRM = int(os.getenv('STRICT_CONFIRM', '2'))
STRICT_STOCH_K = int(os.getenv('STRICT_STOCH_K', '14'))

if STRICT_BACKTEST_MODE:
    # 与 2026-02-22 标准回测参数保持一致
    ALLOW_REVERSE = False
    ALLOW_CONTINUOUS_ADD = False
    ONE_ORDER_PER_CANDLE = True
    ENABLE_PROTECTIVE_ORDERS = True
    STOP_LOSS_PCT = float(os.getenv('STRICT_STOP_LOSS_PCT', '0.04'))
    TAKE_PROFIT_PCT = float(os.getenv('STRICT_TAKE_PROFIT_PCT', '0.06'))
    # 回测里滑点是执行模型，不是下单过滤。严格模式关闭过滤以避免漏单。
    MAX_SLIPPAGE_PCT = float(os.getenv('STRICT_MAX_SLIPPAGE_FILTER_PCT', '0'))

TIME_OFFSET_MS = 0
LOCK_FILE = os.getenv('BOT_LOCK_FILE', './trading_bot_simple.lock')
LOCK_FD = None
SYMBOL_RULE_CACHE = {}
POSITION_MODE_CACHE = None
POSITION_MODE_CACHE_TS = 0.0
POSITION_MODE_CACHE_TTL_SEC = int(os.getenv('POSITION_MODE_CACHE_TTL_SEC', '300'))
LAST_REQUEST_ERROR = ''
LAST_EXECUTED_SIGNAL_KEY = None
DYNAMIC_STATE_FILE = os.getenv('DYNAMIC_STATE_FILE', './dynamic_strategy_state.json')
DYNAMIC_STATE = {'LONG': None, 'SHORT': None}


if PROXY_URL:
    session.proxies.update({'http': PROXY_URL, 'https': PROXY_URL})


def release_lock():
    global LOCK_FD
    if LOCK_FD:
        try:
            fcntl.flock(LOCK_FD, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            LOCK_FD.close()
        except OSError:
            pass
        LOCK_FD = None


def acquire_single_instance_lock():
    global LOCK_FD
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    LOCK_FD = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'⚠️ 检测到已有运行实例，退出防止重复下单: {LOCK_FILE}')
        raise SystemExit(0)
    LOCK_FD.write(str(os.getpid()))
    LOCK_FD.flush()
    atexit.register(release_lock)


def get_timestamp():
    return int(time.time() * 1000) + TIME_OFFSET_MS


def to_pairs(params):
    if not params:
        return []
    if isinstance(params, dict):
        return list(params.items())  # 保持插入顺序，签名和发送顺序一致
    return list(params)


def sign(param_pairs, secret):
    query = urlencode(param_pairs, doseq=True)
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def get_base_urls(endpoint):
    if endpoint.startswith('/fapi/'):
        return ['https://fapi.binance.com']
    return [
        'https://api.binance.com',
        'https://api1.binance.com',
        'https://api2.binance.com',
        'https://api3.binance.com',
        'https://api4.binance.com',
        'https://api-gcp.binance.com',
    ]


def request(method, endpoint, params=None, signed=False, timeout=15):
    global LAST_REQUEST_ERROR
    headers = {'X-MBX-APIKEY': API_KEY}
    base_pairs = to_pairs(params)
    errors = []

    for base_url in get_base_urls(endpoint):
        url = f'{base_url}{endpoint}'
        for _ in range(max(1, REQUEST_RETRIES)):
            pairs = list(base_pairs)
            if signed:
                pairs.append(('timestamp', get_timestamp()))
                pairs.append(('recvWindow', RECV_WINDOW))
                pairs.append(('signature', sign(pairs, API_SECRET)))
            try:
                r = session.request(method.upper(), url, params=pairs or None, headers=headers, timeout=timeout)
                if 200 <= r.status_code < 300:
                    LAST_REQUEST_ERROR = ''
                    return r.json() if r.text else {}
                body = (r.text or '').replace('\n', ' ').strip()
                if len(body) > 260:
                    body = body[:260] + '...'
                errors.append(f'{base_url} -> HTTP {r.status_code} {body}')
                # 参数/权限错误不必重试同一节点，避免无意义重复下单请求
                if 400 <= r.status_code < 500 and r.status_code not in (418, 429):
                    break
            except Exception as e:
                errors.append(f'{base_url} -> {e}')
            time.sleep(RETRY_SLEEP_SEC)

    if errors:
        LAST_REQUEST_ERROR = errors[-1]
        print(f'❌ 请求失败 [{endpoint}]: {" | ".join(errors[-3:])}')
    return None


def sync_server_time():
    global TIME_OFFSET_MS
    data = request('GET', '/fapi/v1/time')
    if not data or 'serverTime' not in data:
        return False
    local_now = int(time.time() * 1000)
    TIME_OFFSET_MS = int(data['serverTime']) - local_now
    return True


def get_position_mode(force_refresh=False):
    global POSITION_MODE_CACHE, POSITION_MODE_CACHE_TS
    now = time.time()
    if (
        not force_refresh
        and POSITION_MODE_CACHE is not None
        and now - POSITION_MODE_CACHE_TS < POSITION_MODE_CACHE_TTL_SEC
    ):
        return POSITION_MODE_CACHE

    data = request('GET', '/fapi/v1/positionSide/dual', signed=True, timeout=15)
    if not data or 'dualSidePosition' not in data:
        if POSITION_MODE_CACHE is not None:
            return POSITION_MODE_CACHE
        print('⚠️ 无法读取持仓模式，默认按单向模式处理')
        return False

    POSITION_MODE_CACHE = bool(data['dualSidePosition'])
    POSITION_MODE_CACHE_TS = now
    return POSITION_MODE_CACHE


def show_proxy_egress_ip():
    try:
        r = session.get('https://api.ipify.org', timeout=10)
        egress_ip = r.text.strip()
        print(f'🌐 代理出口IP: {egress_ip}')
        if EXPECTED_PROXY_IP and egress_ip != EXPECTED_PROXY_IP:
            print(f'⚠️ 警告: 当前出口IP与白名单IP不一致 (期望 {EXPECTED_PROXY_IP})')
        return egress_ip
    except Exception as e:
        print(f'⚠️ 无法获取代理出口IP: {e}')
        return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_dynamic_state():
    global DYNAMIC_STATE
    try:
        if os.path.exists(DYNAMIC_STATE_FILE):
            with open(DYNAMIC_STATE_FILE, 'r') as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                DYNAMIC_STATE['LONG'] = obj.get('LONG')
                DYNAMIC_STATE['SHORT'] = obj.get('SHORT')
    except Exception as e:
        print(f'⚠️ 读取动态策略状态失败，已忽略: {e}')


def save_dynamic_state():
    tmp_file = f'{DYNAMIC_STATE_FILE}.tmp'
    try:
        with open(tmp_file, 'w') as f:
            json.dump(DYNAMIC_STATE, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, DYNAMIC_STATE_FILE)
    except Exception as e:
        print(f'⚠️ 保存动态策略状态失败，已忽略: {e}')
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except OSError:
            pass


def side_key(signal=None, position_side=None):
    if position_side in ('LONG', 'SHORT'):
        return position_side
    if signal == 'long':
        return 'LONG'
    return 'SHORT'


def clear_dynamic_state_if_flat():
    position = get_position(SYMBOL)
    rules = get_symbol_rules(SYMBOL)
    if not position or not rules:
        return
    min_qty = rules['min_qty']
    long_open = position['long_amount'] >= min_qty
    short_open = position['short_amount'] >= min_qty
    net_open = abs(position['net_amount']) >= min_qty

    changed = False
    if not long_open and (not net_open or position['net_amount'] <= 0):
        if DYNAMIC_STATE.get('LONG') is not None:
            DYNAMIC_STATE['LONG'] = None
            changed = True
    if not short_open and (not net_open or position['net_amount'] >= 0):
        if DYNAMIC_STATE.get('SHORT') is not None:
            DYNAMIC_STATE['SHORT'] = None
            changed = True
    if changed:
        save_dynamic_state()


def calc_atr_series(highs, lows, closes, period=14):
    n = len(closes)
    if n == 0:
        return []
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    atr = [tr[0]]
    alpha = 1.0 / max(1, period)
    for i in range(1, n):
        atr.append(atr[-1] * (1 - alpha) + tr[i] * alpha)
    return atr


def calc_adx_series(highs, lows, closes, period=14):
    n = len(closes)
    if n == 0:
        return []
    plus_dm = [0.0]
    minus_dm = [0.0]
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))

    alpha = 1.0 / max(1, period)
    sm_tr = [tr[0]]
    sm_plus = [plus_dm[0]]
    sm_minus = [minus_dm[0]]
    for i in range(1, n):
        sm_tr.append(sm_tr[-1] * (1 - alpha) + tr[i] * alpha)
        sm_plus.append(sm_plus[-1] * (1 - alpha) + plus_dm[i] * alpha)
        sm_minus.append(sm_minus[-1] * (1 - alpha) + minus_dm[i] * alpha)

    dx = []
    for i in range(n):
        if sm_tr[i] <= 1e-12:
            dx.append(0.0)
            continue
        pdi = 100.0 * sm_plus[i] / sm_tr[i]
        mdi = 100.0 * sm_minus[i] / sm_tr[i]
        den = pdi + mdi
        dx.append(0.0 if den <= 1e-12 else 100.0 * abs(pdi - mdi) / den)

    adx = [dx[0]]
    for i in range(1, n):
        adx.append(adx[-1] * (1 - alpha) + dx[i] * alpha)
    return adx


def classify_regime(price, ema_now, ema_prev, adx_now, atr_pct_now):
    if atr_pct_now >= DYN_ATR_HIGH_VOL_PCT:
        return 'highvol'
    if adx_now >= DYN_ADX_TREND and ((price > ema_now and ema_now >= ema_prev) or (price < ema_now and ema_now <= ema_prev)):
        return 'trend'
    if adx_now <= DYN_ADX_CHOP:
        return 'chop'
    return 'neutral'


def rr_by_regime(regime):
    if regime == 'trend':
        return DYN_RR_TREND
    if regime == 'neutral':
        return DYN_RR_NEUTRAL
    if regime == 'chop':
        return DYN_RR_CHOP
    return DYN_RR_HIGHVOL


def build_dynamic_context(closes, highs, lows):
    if not closes:
        return None
    i = len(closes) - 1
    atr = calc_atr_series(highs, lows, closes, DYN_ATR_PERIOD)
    adx = calc_adx_series(highs, lows, closes, DYN_ADX_PERIOD)
    ema = calc_ema(closes, DYN_EMA_SPAN)
    price = closes[i]
    atr_now = atr[i] if i < len(atr) else 0.0
    atr_ratio = (atr_now / price) if price > 0 else 0.0
    atr_pct_now = atr_ratio * 100.0
    adx_now = adx[i] if i < len(adx) else 0.0
    ema_now = ema[i] if i < len(ema) else price
    ema_prev = ema[i - 1] if i > 0 and i - 1 < len(ema) else ema_now

    regime = classify_regime(price, ema_now, ema_prev, adx_now, atr_pct_now)

    scale = 1.0
    if DYN_ATR_TARGET_PCT > 0:
        v = max(1e-9, atr_pct_now)
        scale = min(1.0, DYN_ATR_TARGET_PCT / v)
        scale = max(DYN_ATR_MIN_SCALE, scale)

    if regime == 'trend':
        scale *= DYN_MULT_TREND
    elif regime == 'neutral':
        scale *= DYN_MULT_NEUTRAL
    elif regime == 'chop':
        scale *= DYN_MULT_CHOP
    elif regime == 'highvol':
        scale *= DYN_MULT_HIGHVOL
    scale = clamp(scale, 0.05, 1.0)

    sl_pct = clamp(DYN_K_SL * atr_ratio, DYN_SL_MIN, DYN_SL_MAX)
    rr = rr_by_regime(regime)
    tp_pct = clamp(sl_pct * rr, DYN_TP_MIN, DYN_TP_MAX)

    return {
        'price': price,
        'atr': atr_now,
        'atr_ratio': atr_ratio,
        'atr_pct': atr_pct_now,
        'adx': adx_now,
        'ema': ema_now,
        'regime': regime,
        'size_scale': scale,
        'effective_position_pct': POSITION_PCT * scale,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'rr': rr,
        'allow_entry': not (DYN_BLOCK_HIGHVOL and regime == 'highvol'),
    }


def send_wechat(title, content):
    if not ENABLE_ORDER_NOTIFY:
        return False
    if not WECHAT_SCKEY:
        print('⚠️ 未配置 WECHAT_SCKEY，跳过推送')
        return False
    try:
        resp = requests.post(
            f'https://sctapi.ftqq.com/{WECHAT_SCKEY}.send',
            data={'title': title, 'desp': content},
            timeout=20,
        )
        data = resp.json() if resp.text else {}
        ok = resp.status_code == 200 and data.get('code') == 0
        if not ok:
            print(f'⚠️ 推送失败: HTTP {resp.status_code} {data}')
        return ok
    except Exception as e:
        print(f'⚠️ 推送异常: {e}')
        return False


def get_klines(symbol, interval='15m', limit=100):
    r = request('GET', '/fapi/v1/klines', {'symbol': symbol, 'interval': interval, 'limit': limit})
    return r


def get_balance():
    return request('GET', '/fapi/v2/account', signed=True)


def get_latest_price(symbol):
    data = request('GET', '/fapi/v1/ticker/price', {'symbol': symbol}, timeout=10)
    if not data or 'price' not in data:
        return None
    try:
        return float(data['price'])
    except (TypeError, ValueError):
        return None


def set_leverage(symbol, leverage):
    return request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': leverage}, signed=True, timeout=20)


def place_order(
    symbol,
    side,
    order_type,
    quantity=None,
    price=None,
    reduce_only=False,
    position_side=None,
    stop_price=None,
    close_position=False,
    working_type=None,
    price_protect=None,
):
    params = {
        'symbol': symbol,
        'side': side.upper(),
        'type': order_type.upper(),
    }
    if quantity is not None:
        params['quantity'] = quantity
    if price is not None:
        params['price'] = price
        params['timeInForce'] = 'GTC'
    if stop_price is not None:
        params['stopPrice'] = stop_price
    if close_position:
        params['closePosition'] = 'true'
    if position_side:
        params['positionSide'] = position_side
    if reduce_only and not position_side:
        params['reduceOnly'] = 'true'
    if working_type:
        params['workingType'] = working_type
    if price_protect is not None:
        params['priceProtect'] = 'TRUE' if price_protect else 'FALSE'
    return request('POST', '/fapi/v1/order', params, signed=True, timeout=20)


def get_symbol_rules(symbol):
    if symbol in SYMBOL_RULE_CACHE:
        return SYMBOL_RULE_CACHE[symbol]

    data = request('GET', '/fapi/v1/exchangeInfo', timeout=20)
    if not data or 'symbols' not in data:
        return None

    for s in data['symbols']:
        if s.get('symbol') != symbol:
            continue
        step_size = 0.001
        min_qty = 0.001
        tick_size = 0.1
        for f in s.get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                step_size = float(f.get('stepSize', step_size))
                min_qty = float(f.get('minQty', min_qty))
            elif f.get('filterType') == 'PRICE_FILTER':
                tick_size = float(f.get('tickSize', tick_size))
        SYMBOL_RULE_CACHE[symbol] = {'step_size': step_size, 'min_qty': min_qty, 'tick_size': tick_size}
        return SYMBOL_RULE_CACHE[symbol]
    return None


def format_quantity(raw_qty, step_size):
    step = Decimal(str(step_size))
    qty = (Decimal(str(raw_qty)) / step).to_integral_value(rounding=ROUND_DOWN) * step
    s = format(qty, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s if s else '0'


def format_price(raw_price, tick_size, rounding=ROUND_DOWN):
    tick = Decimal(str(tick_size))
    price = (Decimal(str(raw_price)) / tick).to_integral_value(rounding=rounding) * tick
    s = format(price, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s if s else '0'


def get_open_orders(symbol):
    data = request('GET', '/fapi/v1/openOrders', {'symbol': symbol}, signed=True, timeout=20)
    return data if isinstance(data, list) else []


def cancel_order(symbol, order_id):
    return request('DELETE', '/fapi/v1/order', {'symbol': symbol, 'orderId': order_id}, signed=True, timeout=20)


def place_algo_order(symbol, side, order_type, quantity, trigger_price, position_side=None):
    params = {
        'symbol': symbol,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': quantity,
        'triggerPrice': trigger_price,
        'algoType': 'CONDITIONAL',
    }
    if position_side:
        params['positionSide'] = position_side
    if PROTECTIVE_WORKING_TYPE in ('MARK_PRICE', 'CONTRACT_PRICE'):
        params['workingType'] = PROTECTIVE_WORKING_TYPE
    return request('POST', '/fapi/v1/algoOrder', params, signed=True, timeout=20)


def get_open_algo_orders(symbol):
    data = request('GET', '/fapi/v1/openAlgoOrders', {'symbol': symbol}, signed=True, timeout=20)
    return data if isinstance(data, list) else []


def cancel_algo_order(symbol, algo_id):
    return request('DELETE', '/fapi/v1/algoOrder', {'symbol': symbol, 'algoId': algo_id}, signed=True, timeout=20)


def _match_position_side(order, position_side):
    order_side = order.get('positionSide')
    if position_side:
        return order_side == position_side
    return order_side in (None, '', 'BOTH')


def _order_trigger_price(order):
    for k in ('triggerPrice', 'stopPrice', 'activatePrice'):
        v = order.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def get_protective_orders(symbol, position_side=None):
    out = {'SL': None, 'TP': None}
    orders = get_open_algo_orders(symbol)
    for o in orders:
        if not _match_position_side(o, position_side):
            continue
        otype = (o.get('orderType') or o.get('type') or '').upper()
        if otype in ('STOP_MARKET', 'STOP'):
            out['SL'] = o
        elif otype in ('TAKE_PROFIT_MARKET', 'TAKE_PROFIT'):
            out['TP'] = o
    return out


def get_protective_order_state(symbol, position_side=None):
    orders = get_protective_orders(symbol, position_side=position_side)
    return bool(orders.get('SL')), bool(orders.get('TP'))


def cancel_protective_orders(symbol, position_side=None):
    orders = get_open_algo_orders(symbol)
    cancelled = 0
    for o in orders:
        if not _match_position_side(o, position_side):
            continue
        otype = (o.get('orderType') or o.get('type') or '').upper()
        if otype not in ('STOP_MARKET', 'STOP', 'TAKE_PROFIT_MARKET', 'TAKE_PROFIT'):
            continue
        oid = o.get('algoId')
        if not oid:
            continue
        if cancel_algo_order(symbol, oid):
            cancelled += 1
    if cancelled:
        print(f'🧹 已撤销保护单: {cancelled} 笔 (positionSide={position_side or "BOTH"})')
    return cancelled


def ensure_protective_orders(signal, entry_price, position_qty, position_side=None, force_replace=False, dynamic_ctx=None):
    if not ENABLE_PROTECTIVE_ORDERS:
        return True
    if STOP_LOSS_PCT <= 0 or TAKE_PROFIT_PCT <= 0:
        return True
    if not entry_price or entry_price <= 0:
        print('⚠️ 无法确定持仓均价，跳过保护单设置')
        return False

    rules = get_symbol_rules(SYMBOL)
    if not rules:
        print('❌ 无法读取交易规则，保护单设置失败')
        return False
    qty = format_quantity(abs(position_qty), rules['step_size'])
    if float(qty) < rules['min_qty']:
        print(f'⚠️ 持仓数量过小，跳过保护单设置: {qty}')
        return False

    tick_size = rules.get('tick_size', 0.1)
    mode_dynamic = ENABLE_DYNAMIC_STRATEGY
    side_name = side_key(signal=signal, position_side=position_side)
    existing_orders = get_protective_orders(SYMBOL, position_side=position_side)

    # 读取或初始化该方向动态状态
    state = DYNAMIC_STATE.get(side_name)
    entry_changed = False
    if state and state.get('entry_price'):
        old_entry = float(state.get('entry_price', 0.0))
        if old_entry > 0:
            entry_changed = abs(old_entry - entry_price) / old_entry > 0.002
    if force_replace or state is None or entry_changed:
        inferred_sl = None
        inferred_tp = None
        sl_o = existing_orders.get('SL')
        tp_o = existing_orders.get('TP')
        sl_trg = _order_trigger_price(sl_o) if sl_o else None
        tp_trg = _order_trigger_price(tp_o) if tp_o else None
        if sl_trg and tp_trg and entry_price > 0:
            inferred_sl = abs(entry_price - sl_trg) / entry_price
            inferred_tp = abs(tp_trg - entry_price) / entry_price

        if mode_dynamic and dynamic_ctx:
            sl_pct = float(dynamic_ctx.get('sl_pct', STOP_LOSS_PCT))
            tp_pct = float(dynamic_ctx.get('tp_pct', TAKE_PROFIT_PCT))
        elif inferred_sl and inferred_tp:
            sl_pct = inferred_sl
            tp_pct = inferred_tp
        else:
            sl_pct = STOP_LOSS_PCT
            tp_pct = TAKE_PROFIT_PCT

        state = {
            'entry_price': float(entry_price),
            'sl_pct': float(sl_pct),
            'tp_pct': float(tp_pct),
            'best_price': float(entry_price),
            'trail_enable': bool(mode_dynamic and DYN_TRAIL_ENABLE),
            'trail_trigger_r': DYN_TRAIL_TRIGGER_R,
            'trail_k_atr': DYN_TRAIL_K_ATR,
            'last_regime': (dynamic_ctx or {}).get('regime'),
            'last_atr_ratio': float((dynamic_ctx or {}).get('atr_ratio', 0.0)),
            'updated_ts': time.time(),
        }
        if signal == 'long':
            state['sl_price'] = entry_price * (1.0 - state['sl_pct'])
            state['tp_price'] = entry_price * (1.0 + state['tp_pct'])
        else:
            state['sl_price'] = entry_price * (1.0 + state['sl_pct'])
            state['tp_price'] = entry_price * (1.0 - state['tp_pct'])
        DYNAMIC_STATE[side_name] = state
        save_dynamic_state()

    # 使用最新价格推进 trailing 状态
    state_changed = False
    market_price = None
    if dynamic_ctx and dynamic_ctx.get('price'):
        market_price = float(dynamic_ctx['price'])
    else:
        market_price = get_latest_price(SYMBOL)
    if market_price and market_price > 0:
        old_best = state['best_price']
        if signal == 'long':
            state['best_price'] = max(state['best_price'], market_price)
        else:
            state['best_price'] = min(state['best_price'], market_price)
        if state['best_price'] != old_best:
            state_changed = True

    atr_ratio_now = float((dynamic_ctx or {}).get('atr_ratio', state.get('last_atr_ratio', 0.0)))
    if atr_ratio_now > 0:
        state['last_atr_ratio'] = atr_ratio_now

    base_sl = entry_price * (1.0 - state['sl_pct']) if signal == 'long' else entry_price * (1.0 + state['sl_pct'])
    tp_price_raw = entry_price * (1.0 + state['tp_pct']) if signal == 'long' else entry_price * (1.0 - state['tp_pct'])
    sl_price_raw = float(state.get('sl_price', base_sl))
    if signal == 'long':
        sl_price_raw = max(sl_price_raw, base_sl)
    else:
        sl_price_raw = min(sl_price_raw, base_sl)

    if state.get('trail_enable') and atr_ratio_now > 0 and state.get('best_price'):
        trigger_dist = entry_price * state['sl_pct'] * float(state.get('trail_trigger_r', DYN_TRAIL_TRIGGER_R))
        trail_k = float(state.get('trail_k_atr', DYN_TRAIL_K_ATR))
        if signal == 'long' and (state['best_price'] - entry_price) >= trigger_dist:
            trail_sl = state['best_price'] * (1.0 - trail_k * atr_ratio_now)
            if trail_sl > sl_price_raw:
                sl_price_raw = trail_sl
                state_changed = True
        if signal == 'short' and (entry_price - state['best_price']) >= trigger_dist:
            trail_sl = state['best_price'] * (1.0 + trail_k * atr_ratio_now)
            if trail_sl < sl_price_raw:
                sl_price_raw = trail_sl
                state_changed = True

    state['sl_price'] = sl_price_raw
    state['tp_price'] = tp_price_raw
    state['updated_ts'] = time.time()
    if state_changed:
        DYNAMIC_STATE[side_name] = state
        save_dynamic_state()

    if signal == 'long':
        exit_side = 'SELL'
        sl_price = format_price(sl_price_raw, tick_size, ROUND_DOWN)
        tp_price = format_price(tp_price_raw, tick_size, ROUND_UP)
    else:
        exit_side = 'BUY'
        sl_price = format_price(sl_price_raw, tick_size, ROUND_UP)
        tp_price = format_price(tp_price_raw, tick_size, ROUND_DOWN)

    target_sl = float(sl_price)
    target_tp = float(tp_price)
    cur_sl = _order_trigger_price(existing_orders.get('SL')) if existing_orders.get('SL') else None
    cur_tp = _order_trigger_price(existing_orders.get('TP')) if existing_orders.get('TP') else None
    tol = max(float(tick_size) * 0.5, 1e-8)
    same_sl = cur_sl is not None and abs(cur_sl - target_sl) <= tol
    same_tp = cur_tp is not None and abs(cur_tp - target_tp) <= tol

    if same_sl and same_tp and not force_replace:
        return True

    has_sl, has_tp = bool(existing_orders.get('SL')), bool(existing_orders.get('TP'))
    if has_sl or has_tp or force_replace:
        cancel_protective_orders(SYMBOL, position_side=position_side)

    sl_res = place_algo_order(
        SYMBOL,
        exit_side,
        'STOP_MARKET',
        quantity=qty,
        trigger_price=sl_price,
        position_side=position_side,
    )
    if sl_res is None:
        print(f'❌ 止损单设置失败: {LAST_REQUEST_ERROR or "API无返回"}')
        return False

    tp_res = place_algo_order(
        SYMBOL,
        exit_side,
        'TAKE_PROFIT_MARKET',
        quantity=qty,
        trigger_price=tp_price,
        position_side=position_side,
    )
    if tp_res is None:
        print(f'❌ 止盈单设置失败: {LAST_REQUEST_ERROR or "API无返回"}')
        return False

    if mode_dynamic:
        print(
            f'🛡️ 动态保护单已设置: side={signal.upper()} | regime={state.get("last_regime")} '
            f'| SL={sl_price} ({state["sl_pct"]*100:.2f}%) | TP={tp_price} ({state["tp_pct"]*100:.2f}%) '
            f'| trail={state.get("trail_trigger_r", DYN_TRAIL_TRIGGER_R):.2f}R/{state.get("trail_k_atr", DYN_TRAIL_K_ATR):.2f}ATR'
        )
    else:
        print(
            f'🛡️ 保护单已设置: side={signal.upper()} '
            f'| SL={sl_price} (-{STOP_LOSS_PCT*100:.2f}%) '
            f'| TP={tp_price} (+{TAKE_PROFIT_PCT*100:.2f}%)'
        )
    return True


def get_position(symbol):
    positions = request('GET', '/fapi/v2/positionRisk', signed=True, timeout=20)
    if positions is None:
        return None
    result = {
        'net_amount': 0.0,
        'long_amount': 0.0,
        'short_amount': 0.0,
        'entry_price': 0.0,
        'long_entry_price': 0.0,
        'short_entry_price': 0.0,
        'unrealized_pnl': 0.0,
    }
    found = False
    for p in positions:
        if p.get('symbol') == symbol:
            found = True
            side = p.get('positionSide', 'BOTH')
            amt = float(p.get('positionAmt', 0) or 0)
            pnl = float(p.get('unRealizedProfit', 0) or 0)
            result['unrealized_pnl'] += pnl
            if side == 'LONG':
                result['long_amount'] = abs(amt)
                result['long_entry_price'] = float(p.get('entryPrice', 0) or 0)
            elif side == 'SHORT':
                result['short_amount'] = abs(amt)
                result['short_entry_price'] = float(p.get('entryPrice', 0) or 0)
            else:
                result['net_amount'] = amt
                result['entry_price'] = float(p.get('entryPrice', 0) or 0)
    if not found:
        return {
            'net_amount': 0.0,
            'long_amount': 0.0,
            'short_amount': 0.0,
            'entry_price': 0.0,
            'long_entry_price': 0.0,
            'short_entry_price': 0.0,
            'unrealized_pnl': 0.0,
        }
    if result['net_amount'] == 0.0 and (result['long_amount'] > 0 or result['short_amount'] > 0):
        result['net_amount'] = result['long_amount'] - result['short_amount']
    return result


def calculate_order_quantity(last_price, position_pct_override=None):
    rules = get_symbol_rules(SYMBOL)
    if not rules:
        print('❌ 无法获取交易规则（step/minQty）')
        return None

    account = get_balance()
    if account is None:
        print('❌ 无法获取账户余额，跳过下单')
        return None

    available = float(account.get('availableBalance', 0) or 0)
    use_pct = POSITION_PCT if position_pct_override is None else max(0.0, float(position_pct_override))
    margin = available * use_pct
    if margin < MIN_MARGIN_USDT:
        print(f'⚠️ 可用保证金过低: {margin:.2f} USDT，跳过下单')
        return None

    notional = margin * LEVERAGE
    if notional < MIN_NOTIONAL_USDT:
        print(f'⚠️ 名义价值过低: {notional:.2f} USDT，跳过下单')
        return None

    raw_qty = Decimal(str(notional / last_price))
    step = Decimal(str(rules['step_size']))

    qty_down = (raw_qty / step).to_integral_value(rounding=ROUND_DOWN) * step
    qty_up = (raw_qty / step).to_integral_value(rounding=ROUND_UP) * step

    candidates = []
    for qty_dec in {qty_down, qty_up}:
        qty_str = format_quantity(float(qty_dec), rules['step_size'])
        qty = float(qty_str)
        if qty < rules['min_qty']:
            continue
        cand_notional = qty * last_price
        if cand_notional < MIN_NOTIONAL_USDT:
            continue
        actual_margin = cand_notional / LEVERAGE
        overshoot_pct = max(0.0, (actual_margin - margin) / margin * 100) if margin > 0 else 0.0
        deviation_pct = abs(actual_margin - margin) / margin * 100 if margin > 0 else 0.0
        candidates.append(
            {
                'qty_str': qty_str,
                'qty': qty,
                'notional': cand_notional,
                'actual_margin': actual_margin,
                'overshoot_pct': overshoot_pct,
                'deviation_pct': deviation_pct,
            }
        )

    if not candidates:
        print('⚠️ 就近档位后无可用下单数量，跳过')
        return None

    allowed = [c for c in candidates if c['overshoot_pct'] <= MAX_POSITION_OVERSHOOT_PCT]
    if not allowed:
        print(
            f'⚠️ 就近档位会超目标仓位 +{MAX_POSITION_OVERSHOOT_PCT:.2f}% 限制，跳过下单 '
            f'(目标保证金 {margin:.2f} USDT)'
        )
        return None

    chosen = min(allowed, key=lambda c: (c['deviation_pct'], c['overshoot_pct']))
    print(
        f'📏 仓位档位: 目标保证金 {margin:.2f} USDT ({use_pct*100:.2f}%) -> 实际 {chosen["actual_margin"]:.2f} USDT '
        f'| 偏离 {chosen["deviation_pct"]:.2f}% | 上浮 {chosen["overshoot_pct"]:.2f}% '
        f'| 数量 {chosen["qty_str"]}'
    )
    return chosen['qty_str']


def check_slippage(signal, signal_price):
    if MAX_SLIPPAGE_PCT <= 0:
        return signal_price

    market_price = get_latest_price(SYMBOL)
    if not market_price:
        print('⚠️ 无法获取实时价格，跳过本次开仓')
        send_wechat(
            f'⚠️ 开仓已跳过 {SYMBOL}',
            f'原因: 无法获取实时价格\n信号: {signal.upper()}\n策略价: {signal_price:.2f}',
        )
        return None

    slippage_pct = abs(market_price - signal_price) / signal_price * 100
    if slippage_pct > MAX_SLIPPAGE_PCT:
        print(
            f'⚠️ 滑点超阈值，跳过开仓: 策略价={signal_price:.2f}, '
            f'实时价={market_price:.2f}, 滑点={slippage_pct:.3f}% > {MAX_SLIPPAGE_PCT:.3f}%'
        )
        send_wechat(
            f'⚠️ 开仓已跳过 {SYMBOL}',
            (
                f'原因: 滑点过大\n信号: {signal.upper()}\n'
                f'策略价: {signal_price:.2f}\n实时价: {market_price:.2f}\n'
                f'滑点: {slippage_pct:.3f}% (阈值 {MAX_SLIPPAGE_PCT:.3f}%)'
            ),
        )
        return None
    return market_price


def maintain_existing_position_protection(dynamic_ctx=None):
    if not AUTO_TRADE or not ENABLE_PROTECTIVE_ORDERS:
        return
    clear_dynamic_state_if_flat()

    rules = get_symbol_rules(SYMBOL)
    if not rules:
        return
    position = get_position(SYMBOL)
    if not position:
        return

    dual_mode = get_position_mode()
    if dual_mode:
        has_long = position['long_amount'] >= rules['min_qty']
        has_short = position['short_amount'] >= rules['min_qty']
        if has_long and not has_short:
            ensure_protective_orders(
                'long',
                position['long_entry_price'],
                position['long_amount'],
                position_side='LONG',
                force_replace=False,
                dynamic_ctx=dynamic_ctx,
            )
        elif has_short and not has_long:
            ensure_protective_orders(
                'short',
                position['short_entry_price'],
                position['short_amount'],
                position_side='SHORT',
                force_replace=False,
                dynamic_ctx=dynamic_ctx,
            )
    else:
        net_amt = position['net_amount']
        if abs(net_amt) >= rules['min_qty']:
            side = 'long' if net_amt > 0 else 'short'
            ensure_protective_orders(
                side,
                position['entry_price'],
                abs(net_amt),
                position_side=None,
                force_replace=False,
                dynamic_ctx=dynamic_ctx,
            )


def execute_trade_signal(signal, last_price, strategy_ctx=None):
    if ENABLE_DYNAMIC_STRATEGY and strategy_ctx and not strategy_ctx.get('allow_entry', True):
        print(f'⚠️ 当前regime={strategy_ctx.get("regime")} 且已配置禁止入场，跳过开仓')
        send_wechat(
            f'⚠️ 开仓已跳过 {SYMBOL}',
            f'原因: 高波动regime过滤\n信号: {signal.upper()}\nregime: {strategy_ctx.get("regime")}',
        )
        return False

    desired_side = 'BUY' if signal == 'long' else 'SELL'
    dual_mode = get_position_mode()
    position = get_position(SYMBOL)
    if position is None:
        print('❌ 无法读取持仓信息，跳过下单')
        return False

    rules = get_symbol_rules(SYMBOL)
    if not rules:
        print('❌ 无法读取交易规则，跳过下单')
        return False

    current_side = None
    current_qty = 0.0
    current_position_side = None

    if dual_mode:
        has_long = position['long_amount'] >= rules['min_qty']
        has_short = position['short_amount'] >= rules['min_qty']
        if has_long and has_short:
            print('⚠️ 检测到双向持仓(同时有LONG/SHORT)，该策略仅支持单向持仓，跳过本次信号')
            send_wechat(
                f'⚠️ 信号已跳过 {SYMBOL}',
                '检测到账户同时存在 LONG/SHORT 持仓，策略避免在双向混合仓位下自动交易。',
            )
            return False
        if has_long:
            current_side = 'long'
            current_qty = position['long_amount']
            current_position_side = 'LONG'
        elif has_short:
            current_side = 'short'
            current_qty = position['short_amount']
            current_position_side = 'SHORT'
    else:
        net_amt = position['net_amount']
        if abs(net_amt) >= rules['min_qty']:
            current_side = 'long' if net_amt > 0 else 'short'
            current_qty = abs(net_amt)

    if current_side:
        if signal == current_side:
            if not ALLOW_CONTINUOUS_ADD:
                print(f'🧭 已有同向持仓({current_side})，跳过重复开仓')
                if dual_mode:
                    current_entry = position['long_entry_price'] if current_side == 'long' else position['short_entry_price']
                else:
                    current_entry = position['entry_price']
                ensure_protective_orders(
                    current_side,
                    current_entry,
                    current_qty,
                    position_side=current_position_side if dual_mode else None,
                    force_replace=False,
                    dynamic_ctx=strategy_ctx,
                )
                return False
            print(f'➕ 同向信号({signal})，执行连续加仓')
        if not ALLOW_REVERSE:
            print(f'⚠️ 检测到反向信号({signal})，但未开启反手，保持当前持仓')
            return False

        if signal != current_side:
            cancel_protective_orders(SYMBOL, position_side=current_position_side if dual_mode else None)
            close_side = 'SELL' if current_side == 'long' else 'BUY'
            close_qty = format_quantity(abs(current_qty), rules['step_size'])
            print(f'🔁 反向信号，先平仓: {close_side} {close_qty}')
            close_res = place_order(
                SYMBOL,
                close_side,
                'MARKET',
                close_qty,
                reduce_only=(not dual_mode),
                position_side=current_position_side if dual_mode else None,
            )
            if close_res is None:
                reason = LAST_REQUEST_ERROR or 'API无返回'
                print(f'❌ 平仓失败，取消本次开仓: {reason}')
                send_wechat(
                    f'❌ 平仓失败 {SYMBOL}',
                    f'信号: {signal.upper()}\n方向: {close_side}\n数量: {close_qty}\n原因: {reason}',
                )
                return False
            print(f'✅ 平仓成功: orderId={close_res.get("orderId")}')
            send_wechat(
                f'✅ 平仓成功 {SYMBOL}',
                (
                    f'信号: {signal.upper()}\n方向: {close_side}\n数量: {close_qty}\n'
                    f'orderId: {close_res.get("orderId")}'
                ),
            )
            time.sleep(1)

    safe_open_price = check_slippage(signal, last_price)
    if not safe_open_price:
        return False

    use_pos_pct = None
    if ENABLE_DYNAMIC_STRATEGY and strategy_ctx:
        use_pos_pct = strategy_ctx.get('effective_position_pct', POSITION_PCT)
        print(
            f'🎯 动态仓位: scale={strategy_ctx.get("size_scale", 1.0):.4f} '
            f'| 有效仓位={use_pos_pct*100:.2f}% | regime={strategy_ctx.get("regime")}'
        )
    order_qty = calculate_order_quantity(safe_open_price, position_pct_override=use_pos_pct)
    if not order_qty:
        return False

    open_res = place_order(
        SYMBOL,
        desired_side,
        'MARKET',
        order_qty,
        position_side=('LONG' if signal == 'long' else 'SHORT') if dual_mode else None,
    )
    if open_res is None:
        reason = LAST_REQUEST_ERROR or 'API无返回'
        print(f'❌ 开仓失败: {desired_side} {order_qty} | {reason}')
        send_wechat(
            f'❌ 开仓失败 {SYMBOL}',
            (
                f'信号: {signal.upper()}\n方向: {desired_side}\n数量: {order_qty}\n'
                f'参考价: {safe_open_price:.2f}\n原因: {reason}'
            ),
        )
        return False
    print(f'✅ 已开仓: {desired_side} {order_qty} | orderId={open_res.get("orderId")}')
    time.sleep(1)
    latest_position = get_position(SYMBOL)
    entry_price = safe_open_price
    protect_qty = float(order_qty)
    if latest_position:
        if dual_mode:
            side_entry = latest_position['long_entry_price'] if signal == 'long' else latest_position['short_entry_price']
            side_qty = latest_position['long_amount'] if signal == 'long' else latest_position['short_amount']
        else:
            side_entry = latest_position['entry_price']
            side_qty = abs(latest_position['net_amount'])
        if side_entry and side_entry > 0:
            entry_price = side_entry
        if side_qty and side_qty > 0:
            protect_qty = side_qty

    ensure_protective_orders(
        signal,
        entry_price,
        protect_qty,
        position_side=('LONG' if signal == 'long' else 'SHORT') if dual_mode else None,
        force_replace=True,
        dynamic_ctx=strategy_ctx,
    )
    send_wechat(
        f'✅ 开仓成功 {SYMBOL}',
        (
            f'信号: {signal.upper()}\n方向: {desired_side}\n数量: {order_qty}\n'
            f'参考价: {safe_open_price:.2f}\n'
            f'regime: {(strategy_ctx or {}).get("regime")}\n'
            f'动态SL/TP: {((strategy_ctx or {}).get("sl_pct", STOP_LOSS_PCT)*100):.2f}% / '
            f'{((strategy_ctx or {}).get("tp_pct", TAKE_PROFIT_PCT)*100):.2f}%\n'
            f'orderId: {open_res.get("orderId")}'
        ),
    )
    return True

# ============== 策略 ==============
def calculate_rqk_simple(closes, lb=15, w=6):
    if len(closes) < lb + 1:
        return 0
    recent = closes[-lb:]
    high = max(recent)
    low = min(recent)
    current = closes[-1]
    if high == low:
        return 0
    rqk = ((current - low) / (high - low)) * 100
    return rqk - 50


def calculate_rsi_simple(closes, period=21):
    if len(closes) < period + 1:
        return 50
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(closes, span):
    if not closes:
        return []
    k = 2 / (span + 1)
    ema = [closes[0]]
    for c in closes[1:]:
        ema.append(c * k + ema[-1] * (1 - k))
    return ema


def calc_rqk_series(closes, lb=15, r=6):
    n = len(closes)
    res = []
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


def calc_rsi_series(closes, period=21):
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

    rs = avg_gain / avg_loss if avg_loss else float('inf')
    rsi[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, n):
        d = deltas[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
        rs = avg_gain / avg_loss if avg_loss else float('inf')
        rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def calc_stoch_series(highs, lows, closes, kp=14):
    k = []
    for i in range(len(closes)):
        if i < kp - 1:
            k.append(50.0)
        else:
            wh = max(highs[i - kp + 1:i + 1])
            wl = min(lows[i - kp + 1:i + 1])
            k.append(100 * (closes[i] - wl) / (wh - wl) if wh != wl else 50.0)

    d = []
    for i in range(len(k)):
        if i < 2:
            d.append(50.0)
        else:
            d.append((k[i] + k[i - 1] + k[i - 2]) / 3.0)
    return k, d


def calc_macd_hist_series(closes, fast=12, slow=26, signal=9):
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


def generate_signal():
    klines = get_klines(SYMBOL, interval=SIGNAL_INTERVAL, limit=220)
    if not klines:
        return None, None, None, None

    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    bar_open_ts = int(klines[-1][0])
    last_price = closes[-1]
    dynamic_ctx = build_dynamic_context(closes, highs, lows) if ENABLE_DYNAMIC_STRATEGY else None

    if STRICT_BACKTEST_MODE:
        min_bars = max(STRICT_MACD_SLOW + STRICT_MACD_SIG + 5, STRICT_RSI_PERIOD + 2, STRICT_RQK_LB + 2, STRICT_STOCH_K + 2)
        if len(closes) < min_bars:
            print(f'⚠️ K线不足({len(closes)}/{min_bars})，暂不生成信号')
            return None, last_price, bar_open_ts, dynamic_ctx

        rqk = calc_rqk_series(closes, STRICT_RQK_LB, STRICT_RQK_W)
        rsi = calc_rsi_series(closes, STRICT_RSI_PERIOD)
        macd_hist = calc_macd_hist_series(closes, STRICT_MACD_FAST, STRICT_MACD_SLOW, STRICT_MACD_SIG)
        stoch_k, stoch_d = calc_stoch_series(highs, lows, closes, STRICT_STOCH_K)

        i = len(closes) - 1
        rb = rqk[i] > rqk[i - 1]
        rs = rqk[i] < rqk[i - 1]

        ib = rsi[i] > STRICT_RSI_OVERSOLD and rsi[i - 1] <= STRICT_RSI_OVERSOLD
        is_ = rsi[i] < STRICT_RSI_OVERBOUGHT and rsi[i - 1] >= STRICT_RSI_OVERBOUGHT

        mb = macd_hist[i - 1] <= 0 and macd_hist[i] > 0
        ms = macd_hist[i - 1] >= 0 and macd_hist[i] < 0

        sb = stoch_k[i] > stoch_d[i] and stoch_k[i] < 30
        ss = stoch_k[i] < stoch_d[i] and stoch_k[i] > 70

        bc = sum([rb, ib, mb, sb])
        sc = sum([rs, is_, ms, ss])
        rqk_dir = '↑' if rb else ('↓' if rs else '=')
        dyn_txt = ''
        if dynamic_ctx:
            dyn_txt = (
                f" | Regime:{dynamic_ctx['regime']} | ATR:{dynamic_ctx['atr_pct']:.3f}% "
                f"| 动态SL/TP:{dynamic_ctx['sl_pct']*100:.2f}%/{dynamic_ctx['tp_pct']*100:.2f}% "
                f"| 仓位系数:{dynamic_ctx['size_scale']:.3f}"
            )
        print(
            f'价格: ${last_price:.2f} | RQK:{rqk_dir} | RSI:{rsi[i]:.1f} '
            f'| MACDh:{macd_hist[i]:.4f} | K/D:{stoch_k[i]:.1f}/{stoch_d[i]:.1f} '
            f'| 确认: 多{bc}/空{sc}{dyn_txt}'
        )

        if rb and bc >= STRICT_CONFIRM:
            return 'long', last_price, bar_open_ts, dynamic_ctx
        if rs and sc >= STRICT_CONFIRM:
            return 'short', last_price, bar_open_ts, dynamic_ctx
        return None, last_price, bar_open_ts, dynamic_ctx

    rqk = calculate_rqk_simple(closes)
    rsi = calculate_rsi_simple(closes)

    dyn_txt = ''
    if dynamic_ctx:
        dyn_txt = (
            f" | Regime:{dynamic_ctx['regime']} | ATR:{dynamic_ctx['atr_pct']:.3f}% "
            f"| 动态SL/TP:{dynamic_ctx['sl_pct']*100:.2f}%/{dynamic_ctx['tp_pct']*100:.2f}% "
            f"| 仓位系数:{dynamic_ctx['size_scale']:.3f}"
        )
    print(f'价格: ${last_price:.2f} | RQK: {rqk:.1f} | RSI: {rsi:.1f}{dyn_txt}')

    if rqk > 10 and rsi < 40:
        return 'long', last_price, bar_open_ts, dynamic_ctx
    if rqk < -10 and rsi > 60:
        return 'short', last_price, bar_open_ts, dynamic_ctx
    return None, last_price, bar_open_ts, dynamic_ctx

# ============== 主循环 ==============
acquire_single_instance_lock()

print('🚀 简化版交易机器人启动')
if STRICT_BACKTEST_MODE:
    print(
        '策略: 严格回测一致模式 '
        '(RQK+RSI+MACD+Stochastic, confirm=2, 15m)'
    )
else:
    print(f'策略: RQK + RSI | 仓位: {POSITION_PCT*100}% | 杠杆: {LEVERAGE}x')
print(f'代理: {PROXY_URL or "未配置"}')
print(f'自动下单: {"开启" if AUTO_TRADE else "关闭（仅信号）"} | 反手: {"开启" if ALLOW_REVERSE else "关闭"}')
print(f'连续加仓: {"开启" if ALLOW_CONTINUOUS_ADD else "关闭"} | 单K线仅1单: {"开启" if ONE_ORDER_PER_CANDLE else "关闭"}')
print(f'风控: 最大滑点 {MAX_SLIPPAGE_PCT:.3f}% | 下单推送: {"开启" if ENABLE_ORDER_NOTIFY else "关闭"}')
print(f'仓位档位: 就近取整 | 最大上浮 {MAX_POSITION_OVERSHOOT_PCT:.2f}%')
print(
    f'保护单: {"开启" if ENABLE_PROTECTIVE_ORDERS else "关闭"} '
    f'| 止损 {STOP_LOSS_PCT*100:.2f}% | 止盈 {TAKE_PROFIT_PCT*100:.2f}% | 触发价源 {PROTECTIVE_WORKING_TYPE}'
)
if ENABLE_DYNAMIC_STRATEGY:
    print(
        '动态策略: 开启 '
        f'| k_sl={DYN_K_SL:.2f} | RR={DYN_RR_TREND:.1f}/{DYN_RR_NEUTRAL:.1f}/{DYN_RR_CHOP:.1f}/{DYN_RR_HIGHVOL:.1f} '
        f'| trail={DYN_TRAIL_TRIGGER_R:.2f}R/{DYN_TRAIL_K_ATR:.2f}ATR'
    )
    print(
        f'动态仓位: ATR目标{DYN_ATR_TARGET_PCT:.2f}% min_scale={DYN_ATR_MIN_SCALE:.2f} '
        f'| Regime乘数 T/N/C/H={DYN_MULT_TREND:.2f}/{DYN_MULT_NEUTRAL:.2f}/{DYN_MULT_CHOP:.2f}/{DYN_MULT_HIGHVOL:.2f}'
    )

if not API_KEY or not API_SECRET:
    print('❌ 缺少 API KEY/SECRET，请设置 BINANCE_API_KEY / BINANCE_API_SECRET')
    raise SystemExit(1)

show_proxy_egress_ip()

if sync_server_time():
    print(f'⏱️ 时间偏移: {TIME_OFFSET_MS} ms')
else:
    print('⚠️ 时间同步失败，继续使用本地时间')

ping_futures = request('GET', '/fapi/v1/ping')
if ping_futures is None:
    print('❌ Binance 期货连通性检测失败，请检查代理和白名单')
    raise SystemExit(1)
ping_spot = request('GET', '/api/v3/ping')
if ping_spot is None:
    print('⚠️ 现货节点不可用，已继续使用期货节点运行')

balance = get_balance()
if balance is None:
    print('❌ 账户鉴权失败，请检查 API 权限/白名单/IP')
    raise SystemExit(1)
print('✅ 账户鉴权成功')

# 设置杠杆
lv_result = set_leverage(SYMBOL, LEVERAGE)
if lv_result:
    print(f'✅ 杠杆设置成功: {lv_result}')
else:
    print('⚠️ 杠杆设置失败，将继续仅运行信号监控')

load_dynamic_state()
clear_dynamic_state_if_flat()

while True:
    try:
        signal, last_price, signal_bar_ts, strategy_ctx = generate_signal()
        maintain_existing_position_protection(dynamic_ctx=strategy_ctx)
        if signal:
            print(f'📢 信号: {signal.upper()}')
            if AUTO_TRADE and last_price:
                signal_key = f'{signal}:{signal_bar_ts}'
                if ONE_ORDER_PER_CANDLE and signal_bar_ts and signal_key == LAST_EXECUTED_SIGNAL_KEY:
                    print('🕒 同一根K线信号已下单，跳过本轮避免重复开仓')
                else:
                    executed = execute_trade_signal(signal, last_price, strategy_ctx=strategy_ctx)
                    if executed and signal_bar_ts:
                        LAST_EXECUTED_SIGNAL_KEY = signal_key
            else:
                print('🧪 自动下单未开启，仅记录信号')
        else:
            print('📡 等待信号...')
        
        time.sleep(60)
    except KeyboardInterrupt:
        print('\n🛑 停止')
        break
    except Exception as e:
        print(f'❌ 错误: {e}')
        time.sleep(30)
