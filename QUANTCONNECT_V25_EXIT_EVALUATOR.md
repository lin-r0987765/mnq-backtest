# QuantConnect V25 Exit Evaluator

Use this when you want to evaluate the next `post-v25` exit mechanism directly in QuantConnect without overwriting the official baseline file.

## File To Paste

- `lean/QQQ_ORB_DeepBacktest/QQQ_V25_ProfitLock_ProxyAnalyzer_WebIDE.py`

Do not use this file as the official baseline strategy. The official baseline remains:

- `lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py`

## What It Does

It runs the normal `QQQ v25-timegated-be` strategy logic and adds one conservative research-only exit overlay:

- activate a persistent profit lock only if unrealized move reaches `1.50 x ORB range`
- once active, hold a stop floor at `entry + 0.25 x ORB range` for longs
- once active, hold a stop ceiling at `entry - 0.25 x ORB range` for shorts
- keep that profit lock active even after the `180 minute` breakeven gate expires

This file is meant to answer one narrow question:

- does the strongest remaining `post-v25` local near-miss survive a real QC rerun?

## Log Markers To Look For

After the backtest completes, confirm these lines exist:

- `QQQ v25 profit-lock evaluator init | ...`
- `version=v25-profit-lock-qc-evaluator`
- `baseline_reference=v25-timegated-be`
- `profit_lock_trigger_mult=1.5`
- `profit_lock_level_mult=0.25`
- `profit_lock_activations=...`
- `profit_lock_stop_exits=...`
- `same_bar_eod_reentry_count = 0`

## Recommended Flow

1. Paste `QQQ_V25_ProfitLock_ProxyAnalyzer_WebIDE.py` into QuantConnect `main.py`.
2. Run the normal `2017-04-03 -> 2026-04-02` backtest.
3. Save the usual four files back into `QuantConnect results/2017-2026`:
   - `<BacktestName>.json`
   - `<BacktestName>_logs.txt`
   - `<BacktestName>_orders.csv`
   - `<BacktestName>_trades.csv`
4. Analyze the rerun before launching any new official candidate.

## Important Limitation

This evaluator is not a promotion by itself. It is a research-only QC rerun for the strongest remaining `post-v25` exit hypothesis.
