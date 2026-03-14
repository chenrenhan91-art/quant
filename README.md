# lianghua monitor panel

This repository contains the local monitoring panel for the Binance quant bot.

## Included files

- `monitor_panel.py`: local real-time monitor web service
- `launch_monitor_dashboard.sh`: start script
- `monitor_dashboard.launchd.plist`: launchd service sample for macOS

## Security notes

- Do not commit `.env`, API keys, or account secrets.
- Keep the panel bound to `127.0.0.1` unless you fully understand exposure risks.

## Run locally

```bash
./launch_monitor_dashboard.sh
```

Open:

```text
http://127.0.0.1:8787
```

## Mobile real-time view

GitHub Pages is static and cannot directly display local real-time bot status.
If mobile real-time access is required, expose this local service through a secure tunnel/VPN and keep authentication enabled.
