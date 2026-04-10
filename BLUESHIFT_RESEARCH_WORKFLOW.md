# BLUESHIFT_RESEARCH_WORKFLOW.md

## Purpose

Use Blueshift as a **research / prototyping lane** for this ORB project, while keeping QuantConnect as the only official 10-year promotion authority.

The workflow split is:
- Blueshift: iterate faster on multi-file research strategies and evaluators
- QuantConnect: decide whether a candidate is promoted

## Workspace Layout

The repo now includes a Blueshift-ready workspace skeleton under:
- [blueshift](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift)

Inside that folder:
- [orb_v26_runtime.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/blueshift_library/orb_v26_runtime.py)
  - shared runtime for the Blueshift ORB lane
- [v26_profit_lock_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_profit_lock_blueshift.py)
  - Blueshift baseline entry
- [v26_orb_reentry_evaluator_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_orb_reentry_evaluator_blueshift.py)
  - research-only Blueshift evaluator for structural ORB re-entry

When copying to Blueshift, use the contents of the local `blueshift/` folder as the workspace root so that:
- `blueshift_library/` stays a top-level package
- the strategy entry files can import `blueshift_library.orb_v26_runtime`

## Current Blueshift Files

### Baseline

Use:
- [v26_profit_lock_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_profit_lock_blueshift.py)

Markers:
```python
script_version = "v26-profit-lock-blueshift"
baseline_reference = "v26-profit-lock"
research_only = False
orb_reentry_enabled = False
```

### Research Evaluator

Use:
- [v26_orb_reentry_evaluator_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_orb_reentry_evaluator_blueshift.py)

Markers:
```python
script_version = "v26-orb-reentry-evaluator-blueshift"
baseline_reference = "v26-profit-lock"
research_only = True
orb_reentry_enabled = True
orb_reentry_arm_progress_mult = 1.0
orb_reentry_depth_mult = 0.25
orb_reentry_confirm_bars = 1
```

## Shared Runtime Behavior

The shared runtime keeps the same high-level branch structure as the current QC baseline:
- symbol: `QQQ`
- long-only regime: `prev_day_up_and_mom3_positive`
- ORB bars: `4`
- breakout confirmation: `0.03%`
- trailing stop: `1.3%`
- time-gated breakeven: `1.25x range`, active for `180 minutes`
- profit lock: trigger `1.50x range`, lock `0.25x range`

The Blueshift evaluator then layers structural ORB re-entry on top of that baseline.

## Practical Notes

- This lane is for research only.
- Do not promote directly from Blueshift results.
- Always bring any promising idea back to QuantConnect for the 10-year promotion decision.
- The Blueshift runtime is intentionally split into a shared library so future evaluators do not hit single-file editing pain again.

## Recommended Next Use

If you want a clean Blueshift baseline run:
- start with [v26_profit_lock_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_profit_lock_blueshift.py)

If you want to debug the structural ORB re-entry branch on Blueshift:
- use [v26_orb_reentry_evaluator_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_orb_reentry_evaluator_blueshift.py)

If Blueshift shows something clearly stronger:
1. write the result into the repo state files
2. port only the promising branch back into the QC workflow
3. require a proper QC rerun before any launch decision
