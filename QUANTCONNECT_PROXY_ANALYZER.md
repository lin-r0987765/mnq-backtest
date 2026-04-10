# QuantConnect Proxy Analyzer

Use this when you want QuantConnect to run the official `v18-prev-day-mom3` strategy and print slow-trend family proxy summaries directly into `Logs`, without downloading raw history from the Object Store.

## File To Paste

- `lean/QQQ_ORB_DeepBacktest/QQQ_V18_SlowTrend_ProxyAnalyzer_WebIDE.py`

Do not use this file as the official baseline strategy. It is an analysis tool.

## What It Does

It runs the normal `QQQ v18` strategy logic, records the completed trades inside the algorithm, merges those trades with daily slow-trend features, and prints candidate summaries for:

- `mom5_positive`
- `mom7_positive`
- `mom10_positive`
- `close_above_sma5`
- `close_above_sma8`
- `close_above_sma10`
- `close_above_sma15`
- `close_above_sma20`

The current repo version avoids the earlier QuantConnect history trap and the later session-alignment trap:

- daily slow-trend features are built at analysis time
- not during `initialize()`
- so QuantConnect's future-history clipping should not zero out the baseline/candidate summaries
- each completed daily bar is aligned to the next trading session
- so proxy filters now match the regime timing used by the live `v18` strategy much more closely

## Log Markers To Look For

After the backtest completes, the important lines are:

- `QQQ slow-trend proxy baseline | ...`
- `QQQ slow-trend proxy candidate | label=...`
- `QQQ slow-trend proxy compact-json | ...`

## Recommended Flow

1. Paste `QQQ_V18_SlowTrend_ProxyAnalyzer_WebIDE.py` into QuantConnect `main.py`.
2. Run the backtest over the normal `2017-04-03 -> 2026-04-02` range.
3. Copy the baseline line, the candidate lines, and the compact JSON line from `Logs`.
4. Put those logs back into the repo conversation so the next iteration can continue without downloading raw bars.

## Important Limitation

This is still a `QC proxy` analysis, not a separate full rerun for every candidate. It is meant to replace the expensive raw-history export step, not to replace the final promotion gate.
