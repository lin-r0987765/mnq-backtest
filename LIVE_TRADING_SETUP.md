# Live Trading Bridge

## Recommended path

For this repo, the lowest-friction path is:

1. TradingView strategy alert
2. Python webhook bridge
3. MT5 executor

If you want a cleaner broker-native route for real QQQ execution, Interactive Brokers or Alpaca is usually a better fit than MT5 because symbol mapping is simpler. This iteration implements TradingView -> Python -> MT5 because it matches the tools already available in this project.

## Files

- `run_live_trading.py`
- `run_live_control.py`
- `live_trading.example.json`
- `requirements-live.txt`
- `src/live/`
- `pine/ORB_AutoTrade_v33.pine`

## Supported modes

- `paper`: logs signals and tracks paper positions locally
- `mt5`: sends market orders to MetaTrader 5 through the official Python package

## Built-in live risk controls

- kill switch file blocks new `entry` orders but still allows `exit` / `close_all`
- per-symbol entry cooldown
- per-symbol daily entry limit
- per-symbol open position limit
- paper-mode daily realized loss cap
- persistent duplicate signal protection across process restarts

## Position sizing

- `position_sizing_mode = "fixed"`: use `default_volume`, or webhook `qty` if `allow_signal_qty_override=true`
- `position_sizing_mode = "risk_pct"`: bridge computes qty from `entry price`, `stop_loss`, `account_equity`, `risk_per_trade_pct`, and `max_notional_pct`
- if `allow_signal_qty_override=true`, webhook `qty` can still override Python sizing
- if `allow_default_fallback=true`, missing `price/stop_loss` falls back to `default_volume`
- recommended for ORB bridge: let TradingView send the stop, and let Python decide qty

## Setup

1. Copy `live_trading.example.json` to your own config file and fill in:
   - `shared_secret`
   - `symbol_map`
   - `position_sizing_mode`
   - `account_equity / risk_per_trade_pct / max_notional_pct`
   - MT5 terminal/login/server/password if you use `mt5`
2. Install MT5 Python package if needed:

```powershell
pip install -r requirements-live.txt
```

3. Start the bridge:

```powershell
python run_live_trading.py --config live_trading.example.json
```

4. Check health:

```powershell
curl http://127.0.0.1:8000/health
```

5. Manual control:

```powershell
python run_live_control.py --config live_trading.example.json --status
python run_live_control.py --config live_trading.example.json --disable-entries --reason "manual halt"
python run_live_control.py --config live_trading.example.json --enable-entries
```

## TradingView alert

Use `pine/ORB_AutoTrade_v33.pine` and create an alert with:

- Condition: `Any alert() function call`
- Webhook URL: your public HTTPS webhook endpoint

The bridge expects JSON like this:

```json
{
  "secret": "change-this-shared-secret",
  "strategy": "ORB",
  "symbol": "QQQ",
  "timeframe": "5",
  "action": "entry",
  "side": "buy",
  "qty": 10,
  "price": 585.12,
  "stop_loss": 582.40,
  "take_profit": 590.20,
  "order_id": "ORB_LONG",
  "comment": "ORB Long",
  "event_time": "2026-04-04T14:30:00Z"
}
```

In `risk_pct` mode, `qty` becomes a fallback or manual override, not the final required size.

## Security notes

- Keep broker credentials only on the Python side. Do not put your MT5 password into TradingView alert payloads.
- Use a unique shared secret.
- Start with `paper` mode, then demo MT5, then live funds.
- TradingView webhooks require a reachable HTTPS endpoint. For local testing, start with `paper` mode and expose the bridge through a reverse proxy or tunnel.

## MT5 notes

- `symbol_map` is important because TradingView and MT5 symbol names may differ.
- This bridge sends market orders and closes matching open positions on `exit` or `close_all`.
- Test in demo first.
