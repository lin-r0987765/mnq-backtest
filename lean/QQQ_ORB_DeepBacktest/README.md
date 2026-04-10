# QQQ ORB Deep Backtest

This project contains the active QuantConnect / LEAN entry points for the
official `QQQ` ORB baseline.

## Active files

- `QQQ_ORB_WebIDE.py`
- `main.py`

Use `QQQ_ORB_WebIDE.py` when pasting into QuantConnect Web IDE.
Use `main.py` when running the LEAN CLI project locally.

## Current baseline markers

```python
SYMBOL = "QQQ"
SCRIPT_VERSION = "v18-prev-day-mom3"
REGIME_MODE = "prev_day_up_and_mom3_positive"
MAX_ENTRIES_PER_SESSION = 1
ENTRY_END_HOUR_UTC = 17
```

## Typical LEAN wrapper command

```powershell
python .\run_deep_backtest.py --check-only --print-command
```

Current wrapper defaults:

- `symbol=QQQ`
- `backtest-name="QQQ ORB Deep 8Y"`
- `trade_quantity=10`
- `orb_bars=4`
- `profit_ratio=3.5`
- `breakout_confirm_pct=0.0003`
- `trailing_pct=0.013`
- `close_before_min=10`

## Notes

- The active workflow is `ORB-only`
- The repo was rolled back from a temporary `MNQ` branch to the official `QQQ` baseline
- Any future candidate should be compared against the accepted `QQQ` baseline, not the archived `MNQ` side branch
