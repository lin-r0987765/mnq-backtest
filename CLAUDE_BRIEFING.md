# CLAUDE_BRIEFING.md

## Snapshot

- Iteration: `320`
- Time: `2026-04-09 UTC+8`
- Version: `3.50.233`
- Engine: `ORB production + ICT development`
- Mode: `RESEARCH`
- Theme: `qualified continuation density depth refinement: keep 93 trades while improving return and PF again.`

## What Changed This Round

- Refined the qualified continuation density helper by tightening the revisit depth from `0.5` to `0.35`
- Replayed the high-density lane and improved it from `93 trades / +11.8319% / PF 1.8131` to `93 trades / +11.9309% / PF 1.8204`
- Kept the `40 trades / +13.4177% / PF 3.1879` pure reversal branch as the middle-density reference
- Synced the repo state forward to iteration 320

## Key Insight

The strategic framing is unchanged: ORB is still the production baseline, while ICT is the approved path for a more radical architecture. The new insight this round is that the high-density lane can still be improved without loosening its trade count: revisit-depth refinement now lifts the `93-trade` branch to `+11.9309% / PF 1.8204`, while the `40-trade` pure reversal branch remains the cleaner middle-density alternative.

## Current Status

Accepted baseline now is:
- `v26-profit-lock`

Latest accepted rerun:
- `Square Blue Termite`

There is currently no active ORB candidate.

There is currently no pending QC evaluator rerun.

Current ICT development baseline is:
- `src/strategies/ict_entry_model.py`

Preferred local research lane remains:
- [qqq_5m_alpaca.csv](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/alpaca/normalized/qqq_5m_alpaca.csv)
- [qqq_1d_alpaca.csv](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/alpaca/normalized/qqq_1d_alpaca.csv)
- [spy_5m_alpaca.csv](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/alpaca/normalized/spy_5m_alpaca.csv)
- [spy_1d_alpaca.csv](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/alpaca/normalized/spy_1d_alpaca.csv)

Preferred multi-file research lane still includes:
- [v26_profit_lock_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_profit_lock_blueshift.py)
- [v26_orb_reentry_evaluator_blueshift.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/v26_orb_reentry_evaluator_blueshift.py)
- [orb_v26_runtime.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/blueshift/blueshift_library/orb_v26_runtime.py)

QC promotion remains on the existing 10-year QuantConnect workflow only.
ICT does not replace that rule.

## User Next Step

If iteration continues without a newer QC bundle, the repo should stay on:
- [QQQ_ORB_WebIDE.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py)

Do not launch directly from:
- [Retrospective Red Orange Whale_logs.txt](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/QuantConnect%20results/2017-2026/Retrospective%20Red%20Orange%20Whale_logs.txt)
- [Geeky Fluorescent Yellow Alligator.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/QuantConnect%20results/2017-2026/Geeky%20Fluorescent%20Yellow%20Alligator.json)

The next correct step is:
- keep repo baseline on [QQQ_ORB_WebIDE.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py)
- stop spending the main iteration budget on nearby ORB mechanism layering
- continue the ICT lane one deterministic feature at a time
- keep the active lite frontier fixed at `18 trades / +0.4353% / PF 5.0247` as the clean quality reference
- treat the new pure reversal-FVG balanced branch as the middle-density reference:
  - `40 trades / +13.4177% / PF 3.1879`
- keep the qualified continuation density branch as the high-density reference:
  - `93 trades / +11.9309% / PF 1.8204`
- treat the rolling quick-density branch as completed and rejected for promotion:
  - `109 trades / -0.9467% / PF 0.4595`
- treat the first swing-structure repair replay as partially restorative but still not promotion-ready:
  - `swing_threshold = 2 -> 47 trades / -0.4325% / PF 0.4141`
  - `swing_threshold = 3 -> 20 trades / -0.0865% / PF 0.6166`
  - `swing_threshold = 4 -> 7 trades / -0.0375% / PF 0.4731`
- treat longer recovery on top of the swing branch as rejected for now:
  - `swing_3_recovery_15 -> 20 trades / -0.0865% / PF 0.6166`
  - `swing_3_recovery_20 -> 25 trades / -0.1052% / PF 0.6016`
  - `swing_3_recovery_30 -> 29 trades / -0.1321% / PF 0.5678`
  - `swing_3_dual_12_30 -> 29 trades / -0.1321% / PF 0.5678`
- keep prioritizing swing-based architecture work before returning to nearby rolling-structure parameter sweeps
- treat context-filter counts as independent diagnostics now:
  - the same rejected sweep can increment daily-bias, premium/discount, and other filter counters together
- treat the lite baseline replay as standardized infrastructure now:
  - `strict = 8 trades / +7.7249%`
  - `lite = 14 trades / +10.6136%`
  - `T07` has been removed from the implementation table
- treat the strict baseline summary as standardized infrastructure now:
  - `8 trades / +7.7249% / PF 0.0`
  - `fvg_entries = 8`
  - `accepted_sweeps = 273`
  - `T01` has been removed from the implementation table
- treat stale roadmap rows for `Phase 7`, `T00`, `T02`, and `T03` as cleaned backlog rather than active work
- treat position-sizing compare as completed and remove it from the implementation table backlog
- prioritize the lite-funnel choke points before opening any new continuation lane
- treat the combined reversal + continuation lane as completed and rejected, and remove it from the implementation table backlog
- treat `structure_8` as a density-only candidate and not a promoted helper
- treat mild SMT relaxations (`0.0018 / 0.0020`) as plateau survivors and not as promoted helpers
- treat revisit-depth recalibration as completed and remove it from the implementation table backlog
- treat displacement-body recalibration as completed and remove it from the implementation table backlog
- keep `kill-zone specialization`, `higher-timeframe daily bias`, `premium / discount context`, `breaker fallback`, `IFVG fallback`, `external-liquidity gating`, `SMT divergence gating`, `AMD / market-maker path gating`, `macro timing window gating`, `previous-session anchor gating`, and `session-specific dealing-array refinement` as implemented features, not yet ICT promotion events
- prioritize the next ICT calibration steps in this order:
  - preserve the robust survivor base:
    - `previous-session anchor`
    - `external liquidity`
  - preserve the first robust extension:
    - `session-array refinement`
  - preserve the first robust geometry extension:
    - `liq_sweep_threshold = 0.0008`
  - preserve the slow-recovery extension on the stronger `NY-only` frontier:
    - `liq_sweep_recovery_bars = 4`
  - preserve the first robust structure extension on that stronger frontier:
    - `structure_lookback = 12`
  - preserve the latest robust FVG-geometry extension on that stronger frontier:
    - `fvg_min_gap_pct = 0.0006`
  - preserve the latest robust consequent-encroachment extension on that stronger frontier:
    - `fvg_revisit_depth_ratio = 0.5`
  - treat nearby CHOCH-score retries on the structure-aware frontier as plateaued
  - treat nearby displacement-body retries on the structure-aware frontier as survivor-only
  - treat nearby order-block lookback retests on the structure-aware frontier as survivor-only
  - treat nearby OTE geometry and score retries on the structure-aware frontier as plateaued
  - treat nearby breaker-block lookback retries on the structure-aware frontier as plateaued
  - treat nearby IFVG lookback retries on the structure-aware frontier as plateaued
  - treat nearby liquidity-sweep threshold retries on the structure-aware frontier as survivor-only
  - treat nearby swing-threshold retries on the structure-aware frontier as plateaued
  - treat nearby FVG max-age retries on the structure-aware frontier as survivor-only
  - treat nearby FVG-gap retries on the structure-aware frontier as non-priority:
    - `0.0008` preserves the old base
    - `0.0012+` weakens the lane
  - treat nearby order-block body-quality retests on the structure-aware frontier as survivor-only
  - treat nearby liquidity-pool lookback retests on the structure-aware frontier as survivor-only
  - treat nearby reclaim-strength retests on the slow-recovery `NY-only` frontier as survivor-only
  - treat nearby score-threshold retests on the slow-recovery `NY-only` frontier as survivor-only
  - preserve the first robust SMT extension:
    - `smt_lookback = 10`
  - preserve the first robust heavier-context survivor:
    - `premium / discount`
  - treat `macro timing` only as a secondary optional branch:
    - `macro_early_shifted`
  - treat `kill zones` only as a secondary optional branch:
    - `kill_zones_broader`
  - treat `AMD` only as a thin optional branch:
    - `amd_short_and_soft`
  - treat nearby external-liquidity geometry as plateaued
  - treat nearby previous-session-anchor retries as plateaued
  - treat nearby premium/discount retries as plateaued
  - treat nearby session-array window retries on the slow-recovery `NY-only` frontier as survivor-only
  - keep session-array refinement as part of the validated frontier
  - treat `daily bias` as explicitly rejected on this premium-enabled base
  - on
