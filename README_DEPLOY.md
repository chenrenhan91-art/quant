# Trading Bot Server Bundle (Current Live Strategy)

## Files
- `trading_bot_simple.py`: bot main program
- `.env.example`: runtime config template (copy to `.env`)
- `requirements.txt`: python dependency
- `start.sh`: start once
- `run_forever.sh`: auto-restart loop
- `systemd/trading-bot.service`: Linux service template

## Quick start
1. Install Python 3.10+ and pip
2. `pip3 install -r requirements.txt`
3. `cp .env.example .env`
4. Fill required values in `.env`:
   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET`
   - `BINANCE_PROXY_URL` (if needed)
5. Start:
   - `./start.sh` (foreground)
   - or `./run_forever.sh` (auto restart)

## Optional systemd setup
1. Upload bundle to `/opt/trading-bot`
2. Copy service file:
   - `cp systemd/trading-bot.service /etc/systemd/system/`
3. Reload and start:
   - `systemctl daemon-reload`
   - `systemctl enable trading-bot`
   - `systemctl start trading-bot`
   - `systemctl status trading-bot`

## Current strategy profile (synchronized)
- Dynamic enabled
- `k_sl=6.0`
- `RR=3.0/2.0/1.6/1.2`
- `trail=1.5R/2.0ATR`
- Regime multipliers `0.65/0.35/0.40/0.20`
- Base position `25%`, leverage `10x`

## Notes
- This bundle contains no API secret values by default.
- Set firewall/proxy and exchange API whitelist before live run.
- First run on paper or tiny size is strongly recommended.
