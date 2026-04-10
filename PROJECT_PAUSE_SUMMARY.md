# Project Pause Summary

Paused on `2026-04-10` before GitHub archival push.

## Status

- ORB remains the more production-ready research lane.
- ICT remains an active research lane, with the most recent work split into:
  - `1m` MTF top-down experiments on regularized data
  - classic ICT simplification studies
  - a new `FVG + Fibonacci retracement` research lane

## Latest ICT Research Snapshot

### 1m MTF direction

- Current regularized long-only candidate:
  - `41 trades / +4.2782% / PF 2.4459`
- Walk-forward is still not strong enough for live promotion:
  - holdout remains negative

### FVG + Fib retracement direction

- Research rule:
  - wait for a swing pullback into Fibonacci `0.5 - 0.79`
  - require a valid `FVG` inside that retracement window
  - only trade off that `FVG`
- Current baseline result:
  - `21 trades / +1.2109% / PF 1.3766`
- Control without fib gate:
  - `63 trades / -0.0174% / PF 1.1465`
- Interpretation:
  - the `0.5 - 0.79` fib gate improves quality
  - the narrower `0.618 - 0.79` gate is too restrictive

## Key Files

- Main ICT strategy:
  - `src/strategies/ict_entry_model.py`
- FVG + Fib baseline study:
  - `research/ict/analyze_ict_fvg_fib_retracement_baseline.py`
- ICT regression tests:
  - `tests/test_ict_strategy.py`
- Research outputs:
  - `results/qc_regime_prototypes/`

## Data Notes

- Large local `1m` regularized CSVs are intentionally excluded from git because they exceed practical GitHub limits.
- Smaller regularized timeframe files and summary metadata can be re-added later if needed.

## Recommended Resume Order

1. Continue the simplified `FVG + Fib 0.5 - 0.79` lane.
2. Run a small frontier on:
   - `fvg_min_gap_pct`
   - `swing_threshold`
   - retracement window width
3. Only after that, decide whether to reintroduce `1m` execution refinement.
