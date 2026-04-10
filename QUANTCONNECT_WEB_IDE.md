# QUANTCONNECT_WEB_IDE.md

## Current QC State

Official baseline to paste into QuantConnect Web IDE:
- `lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py`

Current repo version:
- `v26-profit-lock`

Accepted baseline in state files:
- `v26-profit-lock`

Active QC candidate right now:
- none

Research-only QC evaluator pending right now:
- none

Latest accepted baseline rerun:
- `Square Blue Termite`

Latest analyzed research bundle:
- `Retrospective Red Orange Whale`
- version:
  - `v26-orb-reentry-qc-evaluator`
- verdict:
  - mixed analyzer-only result
  - do not launch `v27` from this bundle

Latest ORB re-entry branch postmortem:
- `results/qc_regime_prototypes/qc_orb_reentry_postmortem.json`

Separate multi-file research lane:
- `BLUESHIFT_RESEARCH_WORKFLOW.md`
- use Blueshift for larger prototype scaffolds
- keep QuantConnect as the only promotion authority

## If You Need A Clean Official Rerun

Use backtest range:
- `2017-04-03 -> 2026-04-02`

Before clicking run, confirm these markers exist in code:
```python
SCRIPT_VERSION = "v26-profit-lock"
REGIME_MODE = "prev_day_up_and_mom3_positive"
TRAILING_PCT = 0.013
BREAKEVEN_TRIGGER_MULT = 1.25
BREAKEVEN_ACTIVE_MINUTES = 180
PROFIT_LOCK_TRIGGER_MULT = 1.50
PROFIT_LOCK_LEVEL_MULT = 0.25
MAX_ENTRIES_PER_SESSION = 1
ENTRY_END_HOUR_UTC = 17
```

After the run, confirm `Logs` contain:
- `version=v26-profit-lock`
- `symbol=QQQ`
- `security_type=equity`
- `regime_mode=prev_day_up_and_mom3_positive`
- `breakeven_trigger_mult=1.25`
- `breakeven_active_minutes=180`
- `profit_lock_trigger_mult=1.5`
- `profit_lock_level_mult=0.25`
- `same_bar_eod_reentry_count = 0`

Save these files back into `QuantConnect results/2017-2026`:
- `<BacktestName>.json`
- `<BacktestName>_logs.txt`
- `<BacktestName>_orders.csv`
- `<BacktestName>_trades.csv`

## What Not To Do

- do not rerun `v22-mom5-positive`; it is rejected
- do not rerun `v23-wide-trail`; it is rejected
- do not rerun `v24-no-trail`; it is rejected
- do not revert repo QC files back to `v18`
- do not compare future ideas only against `v18`; future candidates must beat `v26`
- do not launch a new official candidate directly from Alpaca local evidence alone
- do not launch `v27` directly from `Retrospective Red Orange Whale`
- do not rerun the same ORB re-entry evaluator expecting auto-promotion without new branch-specific evidence

## If You Need The Last Research-Only QC Proxy For Branch Debugging

Paste this file into QuantConnect Web IDE:
- `lean/QQQ_ORB_DeepBacktest/QQQ_V26_ORBReentry_ProxyAnalyzer_WebIDE.py`

Run range:
- `2017-04-03 -> 2026-04-02`

Before clicking run, confirm these markers exist in code:
```python
SCRIPT_VERSION = "v26-orb-reentry-qc-evaluator"
BREAKEVEN_TRIGGER_MULT = 1.25
BREAKEVEN_ACTIVE_MINUTES = 180
PROFIT_LOCK_TRIGGER_MULT = 1.50
PROFIT_LOCK_LEVEL_MULT = 0.25
ORB_REENTRY_ARM_PROGRESS_MULT = 1.00
ORB_REENTRY_DEPTH_MULT = 0.25
ORB_REENTRY_CONFIRM_BARS = 1
```

After the run, confirm `Logs` contain:
- `version=v26-orb-reentry-qc-evaluator`
- `baseline_reference=v26-profit-lock`
- `orb_reentry_arm_progress_mult=1.0`
- `orb_reentry_depth_mult=0.25`
- `orb_reentry_confirm_bars=1`
- `same_bar_eod_reentry_count = 0`

This evaluator is no longer a pending mainline step. Use it only if you explicitly want to debug the ORB re-entry branch again.
