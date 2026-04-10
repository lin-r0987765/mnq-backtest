# ICT Promotion Memo

## Status

Decision: `DO_NOT_PROMOTE_YET`

Date:
- `2026-04-09`

Primary source alignment:
- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\ict\ICT 交易策略詳解與教學.pdf`

Official production baseline remains:
- `v26-profit-lock`
- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\lean\QQQ_ORB_DeepBacktest\QQQ_ORB_WebIDE.py`

## Compared Lanes

### 1. Strict ICT benchmark

Reference artifact:
- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\results\qc_regime_prototypes\ict_strict_baseline_summary.json`

Standardized replay:
- `8 trades`
- `+7.7249%`
- `max_drawdown_pct = -1.5262%`
- `fvg_entries = 8`
- `accepted_sweeps = 273`

Assessment:
- The strict lane is profitable and mechanically coherent.
- It is far too sparse to satisfy the 500-trades roadmap.
- It should remain the benchmark lane, not the promotion candidate.

### 2. Active lite reversal frontier

Reference artifacts:
- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\results\qc_regime_prototypes\ict_lite_retest_round2.json`
- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\results\qc_regime_prototypes\ict_position_sizing_compare.json`

Active frontier geometry:
- `smt_threshold = 0.0015`
- `liq_sweep_threshold = 0.0006`
- `fvg_revisit_depth_ratio = 0.5`
- `fvg_revisit_min_delay_bars = 2`
- `displacement_body_min_pct = 0.1`

Research-scale signal replay:
- `18 trades`
- `+0.3529%`
- `PF 4.6792`

Economic replay under engine-backed roadmap sizing:
- `capital_100pct_min40`
- `18 trades`
- `+12.6548%`
- `PF 5.8559`

Assessment:
- This is the strongest current ICT candidate.
- It materially improves economic replay versus the strict lane.
- It still misses the density gates by a wide margin.

### 3. Combined reversal + continuation lane

Reference artifact:
- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\results\qc_regime_prototypes\ict_combined_lanes.json`

Replay:
- `1464 trades`
- `-6.0444%`
- `PF 0.6733`

Assessment:
- This lane is explicitly rejected.
- It solves trade count by overtrading, not by robust edge expansion.

## OOS Evidence

Reference artifact:
- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\results\qc_regime_prototypes\ict_walk_forward_results.json`

Walk-forward summary on the active lite frontier:
- `67 folds`
- `avg_holdout_return_pct = +0.0053%`
- `positive_holdout_fold_pct = 16.4179%`
- `holdout_trade_total = 16`

Assessment:
- OOS is positive on average, but weak and sparse.
- This is supportive evidence for continued research.
- It is not strong enough for strategy promotion.

## Promotion Gate Review

Gate A:
- Target `>= 100 trades`
- Current best lite frontier: `18 trades`
- Result: `FAIL`

Gate B:
- Target `>= 250 trades`
- Result: `FAIL`

Gate C:
- Target `>= 350 trades`
- Result: `FAIL`

Gate D:
- Target `>= 500 trades` with positive OOS
- Result: `FAIL`

Gate E:
- OOS stability strong enough for promotion
- Current walk-forward is mixed and sparse
- Result: `FAIL`

## Decision

Promotion verdict:
- `REJECT_PROMOTION_FOR_NOW`

Reason:
- The lite ICT lane has promising structure and positive local/OOS evidence.
- The lane is still far below the roadmap density gates.
- Combined-lane fusion has already been rejected.
- ORB remains the only production-qualified lane.

## Next Approved Direction

Keep:
- ORB as production baseline
- strict ICT as benchmark lane
- lite reversal as the active research lane

Do not do:
- do not promote ICT into production
- do not reopen naive continuation fusion
- do not spend the next rounds on already plateaued strict-frontier filters

Do next:
- continue density/conversion work on the active lite frontier
- prioritize sweep-to-shift and armed-setup conversion bottlenecks
- revisit promotion only after trade-density gates move materially upward
