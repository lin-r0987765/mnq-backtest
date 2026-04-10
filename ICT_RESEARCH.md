# ICT Research Notes

This document captures the research basis for the repo's first systematic
ICT-style entry model. The goal is not to claim a canonical implementation of
ICT, but to translate recurring ICT concepts into a deterministic, testable
rule set.

## Current Benchmarks

- strict benchmark: `8 trades / +0.2119% / PF Infinity`
- quality frontier: `18 trades / +0.4353% / PF 5.0247`
- qualified reversal balance branch: `40 trades / +13.4177% / PF 3.1879`
- qualified continuation density branch: `93 trades / +11.9309% / PF 1.8204`

## Research Summary

The online material consistently points to a sequence like this:

1. Identify liquidity resting above highs or below lows.
2. Wait for price to run that liquidity and return back inside the range.
3. Confirm a market structure shift after the stop hunt.
4. Use a displacement feature such as a fair value gap or order block as the
   delivery zone for entry.

An additional repo-local source now exists in:

- `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\ict\ICT 交易策略詳解與教學.pdf`

That document reinforces several higher-order ICT ideas that were only lightly
represented before:

- daily bias and higher-timeframe narrative first
- kill zones and macro timing windows
- displacement quality as a mandatory filter
- AMD / Power of 3 sequencing
- breaker blocks, inversion FVGs, and SMT as later-stage refinements

That sequence is especially explicit in the ICT 2024 mentorship lecture notes,
which describe:

- relative equal highs/lows
- a liquidity grab
- a close back through the opposing swing level
- entry on the consequent encroachment of the marked gap / block

## Sources

These were the main references used for the first pass:

- ICT abbreviations and concepts:
  [innercircletrader.net tutorial](https://innercircletrader.net/id/tutorial/ict-abbreviations/)
- ICT fair value gap tutorial:
  [innercircletrader.net tutorial](https://innercircletrader.net/tutorials/fair-value-gap-trading-strategy/)
- ICT market structure shift tutorial:
  [innercircletrader.net tutorial](https://innercircletrader.net/tutorials/ict-market-structure-shift/)
- ICT bullish order block tutorial:
  [innercircletrader.net tutorial](https://innercircletrader.net/id/tutorial/ict-bullish-order-block/)
- ICT 2024 mentorship lecture 2 notes:
  [innercircletrader.net tutorial](https://innercircletrader.net/tutorials/ict-2024/mentorship-lecture-2/)

## Important Interpretation Notes

ICT concepts are often discretionary. Different teachers and communities use
the same labels with slightly different trade execution rules.

Because of that, the implementation in this repo should be treated as an
inference from the source material, not as "the one true ICT definition."

The repo intentionally uses a conservative mechanical translation:

- Liquidity pool:
  recent rolling high / low rather than visual equal highs / equal lows
- Liquidity sweep:
  wick beyond that pool plus a close back inside
- Market structure shift / CHOCH:
  close through the opposite recent structure level after the sweep
- Fair value gap:
  three-candle imbalance between candle 1 and candle 3
- Order block:
  last opposite candle before the displacement that confirms the shift
- Entry:
  first revisit of the FVG, or OB if no FVG is available

## Repo Mapping

The first experimental implementation lives in:

- `src/strategies/ict_entry_model.py`

It uses the existing ICT-style config knobs already present in `config.py`,
including:

- `STRUCTURE_LOOKBACK`
- `OB_LOOKBACK`
- `OB_BODY_MIN_PCT`
- `FVG_MIN_GAP_PCT`
- `FVG_MAX_AGE`
- `LIQ_SWEEP_LOOKBACK`
- `LIQ_SWEEP_THRESHOLD`
- `LIQ_SWEEP_RECOVERY_BARS`
- `SCORE_CHOCH`
- `OTE_FIB_LOW`
- `OTE_FIB_HIGH`

## Current Scope

The current strategy is a safe first pass, not a finished "full ICT model."

Included now:

- liquidity sweep detection
- market structure shift confirmation
- newest-valid FVG preference inside the shift window so FVG recency now matches OB / breaker / IFVG recency
- single-event structure-confirmation scoring so a post-sweep reversal no longer receives both BOS and CHOCH credit on the same bar
- terminal `StrategyResult` return restored after full-loop execution so refactors cannot silently turn `generate_signals` into a `None` producer
- explicit displacement body-quality gating on the post-sweep structure-shift candle
- explicit FVG consequent-encroachment / revisit-depth gating on the entry trigger
- explicit FVG origin-lag gating between structure-shift confirmation and the first valid FVG zone
- explicit FVG origin-body gating on the candle that actually creates the FVG
- explicit FVG origin close-position gating on the candle that actually creates the FVG
- explicit FVG origin opposite-wick gating on the candle that actually creates the FVG
- explicit FVG close-back recovery gating on the entry trigger
- independent context-filter reject accounting so diagnostics can measure the true overlap of daily-bias, premium/discount, AMD, prev-session, external-liquidity, SMT, macro, and kill-zone filters on the same rejected sweep
- quick-density-repair helper wiring for the first four-parameter 500-trades fast-repair branch
- quick-density-repair branch replay against the strict and active lite frontiers, confirming that the same four-parameter bundle expands density to `109 trades` but collapses expectancy to `-0.9467% / PF 0.4595`
- confirmed swing-high / swing-low structure references as an alternative to rolling max/min structure levels, with first replay evidence that swing structure helps the quick branch materially but still does not recover positive expectancy
- longer recovery windows on top of that first swing branch, including `20`, `30`, and a dual-speed `12 -> 30` path, do not restore positive expectancy and therefore should not be treated as the next rescue lever
- multiple simultaneous same-direction pending setups with bounded per-direction capacity, so same-day setup density can now be tested directly instead of being implicitly limited to one pending setup per side
- FVG-first entry logic
- OB fallback logic
- breaker block fallback logic
- inversion FVG fallback logic
- external-liquidity gating on top of the existing sweep detector
- SMT divergence gating via optional peer high/low columns
- AMD / market-maker path gating via opening accumulation, manipulation, and midpoint reclaim
- macro timing window gating nested inside the broader kill-zone schedule
- previous-session anchor gating via prior session midpoint and untouched opposite-side liquidity
- session-specific dealing-array refinement by timing window
- peer-symbol data integration for real SMT backtests
- optional OTE confluence scoring
- session gating
- optional kill-zone specialization
- conservative higher-timeframe daily bias filter
- conservative premium / discount context filter
- premium / discount now supports hard-reject vs soft-score modes, with the
  first roadmap-aligned strict no-SMT replay confirming that softening the
  filter only raises density from `11` to `12` trades while weakening return
  from `+0.1727%` to `+0.1241%`, so this axis is density-only rather than a
  promoted frontier extension
- session-array refinement now also supports hard-reject vs soft-score modes,
  with the first roadmap-aligned strict no-SMT replay confirming that
  softening delivery-timing rejects raises density from `11` to `16` trades
  but weakens return from `+0.1727%` to `+0.1184%`, so this axis is also
  density-only rather than a promoted frontier extension
- prev-session-anchor gating now also supports hard-reject vs soft-score modes,
  with the first roadmap-aligned strict no-SMT replay confirming that
  hard, soft, and filter-off variants all converge to the same realized result
  of `11 trades / +0.1727% / PF 3.9924`, so this axis is plateau / survivor-only
  rather than a promoted frontier extension
- same-zone re-entry is now a real strategy capability on armed delivery arrays,
  and the first roadmap-aligned active-lite replay confirms that re-entry does
  rearm stop-outs (`reentry_stop_rearms = 2`) but still plateaus at the exact
  same realized result of `18 trades / +0.4353% / PF 5.0247`, so this axis is
  infrastructure-complete but not a promoted frontier extension
- continuation-style newer-FVG refreshes are now also a real strategy capability
  on armed delivery arrays, and the first roadmap-aligned active-lite replay
  confirms that continuation raises density from `18` to `22` trades but weakens
  return from `+0.4353%` to `+0.3363%` (`PF 3.0037`), so this axis is density-only
  rather than a promoted frontier extension

Explicitly not modeled yet:

- premium / discount arrays beyond the simple rolling-range context filter
- advanced multi-timeframe narrative alignment beyond the simple daily bias gate
- broader paired-data empirical SMT evaluation / profile calibration
- broader paired-data sweep-geometry calibration

## Current Frontier Note

The active paired ICT frontier is still intentionally aligned to the local PDF
sequence:

1. sweep liquidity
2. confirm MSS / CHOCH
3. require usable displacement
4. enter on an FVG-led revisit

Important correctness note:

- the earlier frontier snapshots around `+0.2057%` were measured before the
  newest-FVG and no-double-count structure-score repairs
- after those P0 repairs, the corrected structure-frontier base currently
  replays at `7 trades / +0.1956% / PF Infinity` on the broader paired
  `QQQ + SPY` lane
- the first post-repair recalibration now confirms `fvg_revisit_depth_ratio =
  0.5` still leads at `7 trades / +0.1968% / PF Infinity`
- treat that `+0.1968%` replay as the current post-repair frontier reference
- a follow-up post-repair `FVG origin body` recalibration now fully plateaus:
  `0.0/0.1/0.2/0.3/0.4/0.5` all replay the same `7 trades / +0.1968% / PF Infinity`
- a further post-repair `FVG origin close-position` recalibration now confirms
  `0.60/0.70` preserve the same `7 trades / +0.1968% / PF Infinity`, while
  `0.80+` weakens the lane, so this axis is survivor-only rather than frontier-forming
- a further post-repair `FVG origin opposite-wick` recalibration now confirms
  `0.40` preserves the same `7 trades / +0.1968% / PF Infinity`, while tighter
  wick caps weaken the lane, so this axis is also survivor-only rather than frontier-forming
- a further post-repair `FVG origin range-vs-ATR` recalibration now confirms
  `0.50/0.75` preserve the same `7 trades / +0.1968% / PF Infinity`, while
  `1.00+` weakens the lane, so this axis is also survivor-only rather than frontier-forming
- a further post-repair `FVG origin body-vs-ATR` recalibration now confirms
  `0.50` preserves the same `7 trades / +0.1968% / PF Infinity`, while
  `0.75+` weakens the lane, so this axis is also survivor-only rather than frontier-forming
- a further post-repair `FVG origin lag` recalibration now fully plateaus:
  `0/1/2/3/4/5` all replay the same `7 trades / +0.1968% / PF Infinity`
- a further post-repair `displacement range-vs-ATR` recalibration now confirms
  `0.50/0.75` preserve the same `7 trades / +0.1968% / PF Infinity`, while
  `1.00+` weakens the lane, so this axis is also survivor-only rather than frontier-forming
- a further post-repair `displacement body quality` recalibration now confirms
  `0.10/0.20` improves the lane from `7 trades / +0.1968% / PF Infinity` to
  `8 trades / +0.2119% / PF Infinity`, while `0.30+` weakens the lane
- that light displacement-body extension is now promoted into the active paired
  ICT helper, so `+0.2119% / PF Infinity` is the current corrected frontier
  reference for future local calibration
- a follow-up post-repair `structure shift close buffer` recalibration now
  confirms `0.05/0.10` preserve the exact same `8 trades / +0.2119% / PF Infinity`,
  while `0.15+` weakens the lane, so this axis is survivor-only rather than
  frontier-forming on top of the stronger corrected helper
- a follow-up post-repair `FVG retest touch cap` recalibration now confirms
  only `fvg_touch_cap_5` preserves the exact same `8 trades / +0.2119% / PF Infinity`,
  while tighter caps progressively weaken the lane, so this axis is also
  survivor-only rather than frontier-forming on top of the stronger corrected helper
- a follow-up post-repair `FVG revisit delay` recalibration now reconfirms
  `fvg_revisit_min_delay_bars = 3` as the best robust delay at the exact same
  `8 trades / +0.2119% / PF Infinity`, while `1/2` and `4+` are weaker

## 500-Trades Walk-Forward Note

- ICT-specific walk-forward now lives in:
  - [run_ict_walk_forward.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/run_ict_walk_forward.py)
- active lite frontier replay:
  - `train=60 / validation=20 / holdout=20 / step=20`
  - `67` folds
  - `avg_validation_return_pct = +0.0053%`
  - `avg_holdout_return_pct = +0.0053%`
  - `avg_holdout_sharpe_ratio = 4.8830`
  - `positive_holdout_fold_pct = 16.4179%`
  - `holdout_trade_total = 16`
- verdict:
  - `ICT_WALK_FORWARD_MIXED_BUT_POSITIVE`
- interpretation:
  - the active lite frontier stays slightly positive on average OOS, but most folds are still zero-trade windows
  - walk-forward is therefore implemented evidence for the roadmap, not a pass condition for promotion

## 500-Trades Promotion Memo

- the formal decision memo now lives in:
  - [ICT_PROMOTION_MEMO.md](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/ICT_PROMOTION_MEMO.md)
- comparison captured in that memo:
  - strict benchmark: `8 trades / +7.7249%`
  - active lite frontier: `18 trades / +0.3529% / PF 4.6792`
  - engine-backed economic replay: `18 trades / +12.6548%`
  - combined lane: `1464 trades / -6.0444% / PF 0.6733`
  - walk-forward: `67` folds with `avg_holdout_return_pct = +0.0053%`
- verdict:
  - `DO_NOT_PROMOTE_YET`
- interpretation:
  - ICT now has a proper promotion decision artifact
  - the lite branch remains promising research, but not a production challenger yet

## 500-Trades Phase Backlog Hygiene

- the implementation table previously still listed these already-completed high-level phases as pending:
  - `Phase 0`
  - `Phase 1`
  - `Phase 2`
  - `Phase 3`
  - `Phase 4`
  - `Phase 5`
- completion evidence already existed in:
  - strict frontier baseline replay artifacts
  - [analyze_ict_frontier_funnel.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_frontier_funnel.py)
  - [analyze_ict_lite_reversal_baseline.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_reversal_baseline.py)
  - [analyze_ict_lite_geometry_round1.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_geometry_round1.py)
  - [analyze_ict_lite_retest_round2.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_retest_round2.py)
  - [analyze_ict_lite_session_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_session_density.py)
- those stale phase rows were removed from the old 500-trades roadmap before that file was retired

## 500-Trades Retest Backlog Hygiene

- the implementation table also had one stale retest-priority row:
  - `P2 / fvg_revisit_min_delay_bars`
- that row was no longer live backlog because:
  - [analyze_ict_lite_retest_round2.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_retest_round2.py) is already complete
  - the active lite frontier is already promoted on `fvg_revisit_min_delay_bars = 2`
- the table also still listed shipped analyzer rows as if they were future additions:
  - [analyze_ict_lite_reversal_baseline.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_reversal_baseline.py)
  - [analyze_ict_lite_geometry_round1.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_geometry_round1.py)
  - [analyze_ict_lite_retest_round2.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_retest_round2.py)
  - [analyze_ict_lite_session_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_session_density.py)
- those stale rows were removed from the old 500-trades roadmap before that file was retired

## 500-Trades Reward:Risk Gate Note

- active lite RR-gate replay now lives in:
  - [analyze_ict_lite_rr_gate.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_rr_gate.py)
- result on the active lite frontier:
  - `lite_frontier_base = 18 trades / +0.3529% / PF 4.6792`
  - `rr_gate_off_control = 18 trades / +0.3529% / PF 4.6792`
  - `tp_rr_1p0_no_gate_control = 18 trades / +0.1819% / PF 3.2134`
  - `tp_rr_1p0_gate_1p5_control = 0 trades / rr_filtered_entries = 108`
  - `rr_gate_4p5 = 0 trades / rr_filtered_entries = 108`
- verdict:
  - `RR_GATE_IMPLEMENTED_AND_PLATEAUED_ON_LITE_FRONTIER`
- interpretation:
  - the promoted lite frontier already clears the repo-approved `Reward:Risk >= 1.5:1` gate
  - the new gate is still real because low-target controls are filtered out completely
  - RR gating should now be treated as implemented infrastructure, not as an open backlog item

## 500-Trades Engine Sizing Note

- engine-backed sizing replay now lives in:
  - [analyze_ict_position_sizing_compare.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_position_sizing_compare.py)
- sizing support now exists directly in:
  - [engine.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/src/backtest/engine.py)
- result on the active lite frontier:
  - `research_fixed_10 = 18 trades / +0.3816% / PF 5.4911`
  - `fixed_40_shares = 18 trades / +1.5359% / PF 5.5555`
  - `capital_50pct_min40 = 18 trades / +6.1746% / PF 5.9878`
  - `capital_100pct_min40 = 18 trades / +12.6548% / PF 5.8559`
- verdict:
  - `POSITION_SIZING_IMPACT_CONFIRMED`
- interpretation:
  - T04 is now completed at the engine layer
  - sizing changes economic return materially, but does not solve the density bottleneck
  - future rounds should not reread sizing as an open roadmap item

## 500-Trades Lite Helper Note

- the repo-approved lite helper already lives in:
  - [ict_entry_model.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/src/strategies/ict_entry_model.py)
  - `build_ict_lite_reversal_profile_params(...)`
- contract confirmed by:
  - [test_ict_profile_builders.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/test_ict_profile_builders.py)
- helper behavior:
  - turns off `prev_session_anchor`
  - turns off `external_liquidity`
  - turns off `session_array_refinement`
  - turns off `macro`
  - turns off `AMD`
  - turns off `kill_zone`
  - leaves `SMT` opt-in rather than mandatory
- interpretation:
  - `T06` was already implemented before this round
  - this round closes the backlog bookkeeping gap by removing that stale table row

## 500-Trades Lite Baseline Standardized Replay

- the standardized lite baseline replay now lives in:
  - [analyze_ict_lite_reversal_baseline.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_reversal_baseline.py)
  - [ict_lite_reversal_baseline.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_reversal_baseline.json)
- fixed trading spec:
  - `position_size_mode = capital_pct`
  - `capital_usage_pct = 1.0`
  - `min_shares = 40`
  - `Reward:Risk >= 1.5:1`
- standardized replay result:
  - strict frontier:
    - `8 trades / +7.7249% / PF 0.0`
  - lite reversal baseline:
    - `14 trades / +10.6136% / PF 9.9873`
- verdict:
  - `LITE_ICT_REVERSAL_BASELINE_IMPROVES_DENSITY_BUT_STAYS_BELOW_100_TRADES`
- interpretation:
  - `T07` is now completed on the implementation table's real trading spec rather than the old `10 shares` research convention
  - signal density is unchanged, but the economic replay is materially larger under the standardized engine rules
  - this backlog item should stay closed unless the whole 500-trades branch is re-standardized end-to-end

## 500-Trades Strict Baseline Summary

- the standardized strict baseline summary now lives in:
  - [analyze_ict_strict_baseline_summary.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_strict_baseline_summary.py)
  - [ict_strict_baseline_summary.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_strict_baseline_summary.json)
- fixed trading spec:
  - `position_size_mode = capital_pct`
  - `capital_usage_pct = 1.0`
  - `min_shares = 40`
  - `Reward:Risk >= 1.5:1`
- standardized strict benchmark:
  - `8 trades / +7.7249% / PF 0.0`
  - `fvg_entries = 8`
  - `bullish_sweeps = 151`
  - `bearish_sweeps = 122`
  - `accepted_sweeps = 273`
- interpretation:
  - `T01` is now completed and should stay off the table
  - later roadmap rounds should reference this JSON instead of re-deriving the strict benchmark ad hoc

## 500-Trades Backlog Hygiene

- the implementation table had stale pending rows for completed work:
  - `Phase 7`
  - `T00`
  - `T02`
  - `T03`
- completion evidence already existed in repo through:
  - [run_ict_walk_forward.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/run_ict_walk_forward.py)
  - [analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_displacement_calibration.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_displacement_calibration.py)
  - [analyze_ict_frontier_funnel.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_frontier_funnel.py)
  - metadata counters in [ict_entry_model.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/src/strategies/ict_entry_model.py)
- those stale rows were removed from the old 500-trades roadmap before that file was retired, so future rounds read only live backlog

Current repo-approved paired frontier:

- `short-SMT`
- `premium / discount`
- `session-array refinement`
- `NY-only core session gate`
- `liq_sweep_recovery_bars = 4`
- `structure_lookback = 12`
- `liq_sweep_threshold = 0.0008`
- `fvg_min_gap_pct = 0.0006`
- `fvg_revisit_depth_ratio = 0.5`
- `fvg_revisit_min_delay_bars = 3`
- `displacement_body_min_pct = 0.10`
- `fvg_max_age = 20`

Latest local conclusion:

- `fvg_origin_max_lag_bars` is now implemented as a real guard between the
  structure-shift bar and the first acceptable FVG origin
- `2/3/4/5` preserve the frontier
- `1` weakens the lane
- verdict: `ROBUST_FVG_ORIGIN_LAG_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `fvg_origin_body_min_pct` is now implemented as a real displacement-quality
  guard on the actual displacement candle (`zone_index - 1`) rather than the
  third confirmation candle
- corrected calibration shape:
  - `0.10/0.20/0.30` preserve the frontier
  - `0.40+` weakens the lane
- verdict: `ROBUST_FVG_ORIGIN_BODY_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `fvg_origin_close_position_min_pct` is now implemented as a real close-strength
  guard on the actual displacement candle that creates the FVG
- calibration shape:
  - `0.60/0.70` survive only as thinner variants
  - `0.80+` weakens the lane materially
- verdict: `ROBUST_FVG_ORIGIN_CLOSE_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `fvg_origin_opposite_wick_max_pct` is now implemented as a real opposite-wick
  purity guard on the actual displacement candle that creates the FVG
- calibration shape:
  - `0.20/0.30/0.40` preserve the frontier exactly
  - `0.10/0.08` survive only as thinner variants
  - `0.05` weakens the lane further
- verdict: `ROBUST_FVG_ORIGIN_WICK_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `fvg_origin_range_atr_mult` is now implemented as a real origin-candle
  range-vs-ATR quality guard on the actual displacement candle that creates the
  FVG
- calibration shape:
  - `0.50/0.75` preserve the frontier exactly
  - `1.00` survives only as a thinner variant
  - `1.25+` weaken the lane further
- verdict: `ROBUST_FVG_ORIGIN_RANGE_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `fvg_origin_body_atr_mult` is now implemented as a real origin-candle
  directional-body-vs-ATR quality guard on the actual displacement candle that
  creates the FVG
- calibration shape:
  - `0.50/0.75` survive only as nearby robust continuations
  - `1.00` survives only as a thinner variant
  - `1.10+` weaken the lane further
- verdict: `ROBUST_FVG_ORIGIN_BODY_ATR_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `structure_shift_close_buffer_ratio` is now recalibrated on top of the
  stronger corrected helper that already includes `displacement_body_min_pct = 0.10`
- calibration shape:
  - `0.05/0.10` preserve the exact same `8 trades / +0.2119% / PF Infinity`
  - `0.15+` weakens the lane
- verdict: `ROBUST_SHIFT_BUFFER_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `fvg_max_retest_touches` is now recalibrated on top of the stronger
  corrected helper that already includes `displacement_body_min_pct = 0.10`
- calibration shape:
  - `fvg_touch_cap_5` preserves the exact same `8 trades / +0.2119% / PF Infinity`
  - tighter caps progressively weaken the lane
- verdict: `ROBUST_FVG_TOUCH_CAP_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
- `fvg_revisit_min_delay_bars` is now recalibrated on top of the stronger
  corrected helper that already includes `displacement_body_min_pct = 0.10`
- calibration shape:
  - `fvg_revisit_delay_3` remains the best robust delay at `8 trades / +0.2119% / PF Infinity`
  - `1/2` are weaker and `4+` weaken the lane progressively
- verdict: `ROBUST_FVG_REVISIT_DELAY_EXTENSION_IDENTIFIED_ON_STRUCTURE_BASE`
- strict frontier funnel instrumentation is now available through
  [analyze_ict_frontier_funnel.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_frontier_funnel.py)
- current strict frontier funnel:
  - `829` raw sweeps
  - `273` accepted sweeps
  - `25` shift candidates
  - `10` armed setups
  - `8` retest candidates
  - `8` entries
- top blockers:
  - `premium_discount_filtered_setups = 287`
  - `smt_filtered_sweeps = 137`
  - `prev_session_anchor_filtered_setups = 84`
  - `external_liquidity_filtered_sweeps = 47`
  - `delivery_missing_shifts = 14`
  - `session_array_filtered_shifts = 7`
- interpretation:
  - the active corrected frontier is not mainly dying at the last-mile entry trigger
  - the first 500-trades optimization should branch into a lighter reversal lane
    that strips the heaviest pre-arm context blockers

## Current Calibration Insight

The first real paired-data calibration now has two layers of evidence:

1. `results/qc_regime_prototypes/ict_paired_profile_calibration.json`
   - full-stack `QQQ + SPY` ICT profile produced `0 trades`
   - one-by-one relaxed context variants also produced `0 trades`
2. `results/qc_regime_prototypes/ict_paired_activation_frontier.json`
   - `context_relaxed_bundle` still produced `0 trades`
   - `minimal_structure_default` still produced `0 trades`
   - trades only appeared after materially loosening sweep geometry
   - those activated variants were still negative

Interpretation:

- the broader paired lane is no longer stuck at zero activity
- the first robust context survivors are now known:
  - `previous-session anchor`
  - `external liquidity`
- `premium / discount` and `session-array refinement` remain viable but thinner
- `AMD` and heavier bundles still collapse activity back to zero
- the first survivor-bundle pairwise calibration is now complete
- the first robust extension on top of the survivor base is `session-array refinement`
- the first sweep / SMT geometry calibration is now complete
- the first robust geometry extension on top of the survivor-plus-session-array base is `liq_sweep_threshold = 0.0008`
- the first deeper SMT recalibration is now complete
- the first robust SMT extension on top of the survivor-plus-session-array-loose-sweep base is `smt_lookback = 10`
- the first controlled context reintroduction on top of the short-SMT base is now complete
- the first robust heavier-context survivor on top of the short-SMT base is `premium / discount`
- `macro timing` survives only as a secondary optional branch on top of the short-SMT premium base
- `kill zones` also survive only as a secondary optional branch on top of the short-SMT premium base, and only with broader windows
- `AMD` survives only as a thin optional branch on top of the short-SMT premium base, and only after relaxing midpoint reclaim plus accumulation / manipulation settings
- nearby `external liquidity` geometry tweaks are now plateaued on top of the short-SMT premium base: lookback / tolerance changes preserve the same result but do not improve it
- nearby `previous-session anchor` on/off and tolerance retests are now also plateaued on top of the short-SMT premium base: every tested variant preserves the same `5 trades / +0.1375% / PF Infinity`, including `anchor_off_control`
- nearby `premium / discount` lookback and neutral-band retests are now also plateaued on top of the short-SMT premium base: every robust tested variant preserves the same `5 trades / +0.1375% / PF Infinity`, including `premium_off_control`
- calibrated `displacement body quality` on top of the stronger structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_displacement_calibration.json`
  - `displacement_body_min_pct = 0.10` and `0.20` preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - `0.30+` progressively weakens the lane
  - verdict:
    - `ROBUST_DISPLACEMENT_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - displacement quality is now an explicit mechanical rule in the repo, not just an implied narrative through FVG presence
    - nearby displacement-body retries should not be prioritized until the frontier changes materially
- calibrated `FVG revisit depth / consequent encroachment` on top of the stronger structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_revisit_depth_calibration.json`
  - `fvg_revisit_depth_ratio = 0.50` improves the lane from `7 trades / +0.1831% / PF Infinity` to `7 trades / +0.2042% / PF Infinity`
  - `0.25` preserves activity but weakens return
  - `0.75` and `1.0` thin and weaken the lane
  - verdict:
    - `ROBUST_FVG_REVISIT_DEPTH_EXTENSION_IDENTIFIED_ON_STRUCTURE_BASE`
  - interpretation:
    - consequent encroachment is now a real mechanical rule in the repo, not just a narrative reference from the PDF
    - this is the latest robust extension on top of the structure-aware frontier
- calibrated `FVG close-back recovery` on top of the stronger CE-extended structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_close_recovery_calibration.json`
  - `fvg_rejection_close_ratio = 0.25` stays positive but weaker than the current frontier
  - `0.50+` progressively thins and weakens the lane
  - verdict:
    - `ROBUST_FVG_CLOSE_RECOVERY_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - close-back recovery is now a real mechanical rule in the repo, but on top of the stronger CE frontier it is survivor-only rather than a new extension
- calibrated `FVG rejection wick` on top of the stronger CE-extended structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_rejection_wick_calibration.json`
  - `fvg_rejection_wick_ratio = 0.10` stays robust but weaker than the current frontier at `7 trades / +0.1930% / PF 152.7443`
  - `0.20+` progressively thins and weakens the lane
  - verdict:
    - `ROBUST_FVG_REJECTION_WICK_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - rejection wick quality is now a real mechanical rule in the repo, which keeps the implementation closer to the PDF's revisit -> reaction logic
    - on top of the stronger CE frontier it is survivor-only rather than a new extension
- calibrated `FVG rejection body` on top of the stronger CE-extended structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_rejection_body_calibration.json`
  - `fvg_rejection_body_min_pct = 0.10` stays robust but weaker than the current frontier at `7 trades / +0.1905% / PF Infinity`
  - `0.20+` progressively thins and weakens the lane
  - verdict:
    - `ROBUST_FVG_REJECTION_BODY_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - reaction-body quality is now a real mechanical rule in the repo, which keeps the implementation closer to the PDF's revisit -> reaction / displacement logic
    - on top of the stronger CE frontier it is survivor-only rather than a new extension
- calibrated `structure-shift close-through buffer` on top of the stronger CE-extended structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_shift_buffer_calibration.json`
  - `structure_shift_close_buffer_ratio = 0.05`, `0.10`, and `0.15` preserve the exact same `7 trades / +0.2042% / PF Infinity`
  - `0.20+` progressively thins and weakens the lane
  - verdict:
    - `ROBUST_SHIFT_BUFFER_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - close-through MSS / CHOCH quality is now a real mechanical rule in the repo, which keeps the implementation closer to the PDF's sweep -> close through opposing swing -> displacement sequence
    - on top of the stronger CE frontier it is survivor-only rather than a new extension
- calibrated `FVG revisit delay` on top of the stronger CE-extended structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_revisit_delay_calibration.json`
  - `fvg_revisit_min_delay_bars = 3` improves the lane from `7 trades / +0.2042% / PF Infinity` to `7 trades / +0.2057% / PF Infinity`
  - `2` preserves the old base
  - `4+` progressively weakens the lane
  - verdict:
    - `ROBUST_FVG_REVISIT_DELAY_EXTENSION_IDENTIFIED_ON_STRUCTURE_BASE`
  - interpretation:
    - revisit timing is now a real mechanical rule in the repo, which keeps the implementation closer to the PDF's displacement -> revisit -> reaction sequence
    - this is the latest robust extension on top of the stronger CE frontier
- calibrated `FVG retest-touch cap` on top of the stronger delayed-revisit CE-extended structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_touch_cap_calibration.json`
  - `fvg_max_retest_touches = 5` preserves the exact same `7 trades / +0.2057% / PF Infinity`
  - tighter caps progressively thin and weaken the lane
  - verdict:
    - `ROBUST_FVG_TOUCH_CAP_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - first-touch purity is now a real mechanical rule in the repo, which keeps the implementation closer to the PDF's idea that old, over-tapped imbalances degrade
    - on top of the stronger delayed-revisit CE frontier it is survivor-only rather than a new extension
- calibrated `displacement range / ATR` on top of the stronger delayed-revisit CE-extended structure-aware slow-recovery NY-only frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_displacement_range_calibration.json`
  - robust survivors:
    - `displacement_range_atr_0p50`
    - `displacement_range_atr_0p75`
    - both preserve the exact same `7 trades / +0.2057% / PF Infinity`
  - weaker:
    - `displacement_range_atr_1p00 -> 6 trades / +0.1660% / PF Infinity`
    - `displacement_range_atr_1p25 -> 5 trades / +0.1019% / PF Infinity`
    - `displacement_range_atr_1p50 -> 4 trades / +0.0712% / PF Infinity`
  - verdict:
    - `ROBUST_DISPLACEMENT_RANGE_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - displacement expansion relative to ATR is now a real mechanical rule in the repo, which keeps the implementation closer to the PDF's emphasis on true displacement after the sweep and shift
    - on top of the stronger delayed-revisit CE frontier it is survivor-only rather than a new extension
- `session-array refinement` remains a meaningful robust extension on top of the short-SMT premium base:
  - turning it off weakens the lane to `6 trades / +0.1226% / PF 19.5545`
  - default and broader session-array window variants restore the stronger `5 trades / +0.1375% / PF Infinity`
  - shifted-later windows fall back to the weaker control result
- recalibrated `macro timing` on top of the stronger short-SMT premium-plus-session-array frontier still survives only as a secondary optional branch:
  - best survivor is `macro_early_shifted`
  - result is `3 trades / +0.0493% / PF Infinity`
  - this remains well below the frontier
- calibrated `delivery-array composition / scoring` on top of the stronger short-SMT premium-plus-session-array frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_delivery_calibration.json`
  - every tested variant preserved the exact same `5 trades / +0.1375% / PF Infinity`
  - every realized entry remained `FVG-only`
  - verdict:
    - `DELIVERY_COMPOSITION_PLATEAU_ON_SESSION_ARRAY_BASE`
  - interpretation:
    - nearby delivery-array score or family retries are plateaued on the current frontier
    - the current paired ICT frontier is already behaving like a narrow FVG-led lane
- calibrated `OTE geometry / score` on top of the stronger short-SMT premium-plus-session-array frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ote_calibration.json`
  - every tested variant preserved the exact same `5 trades / +0.1375% / PF Infinity`
  - even `ote_off_control` matched the frontier result
  - verdict:
    - `OTE_CALIBRATION_PLATEAU_ON_SESSION_ARRAY_BASE`
  - interpretation:
    - nearby OTE geometry or score retries are plateaued on the current frontier
    - OTE is not the next edge source on top of the current paired ICT stack
- calibrated `FVG geometry / freshness` on top of the stronger short-SMT premium-plus-session-array frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_fvg_calibration.json`
  - best non-base robust survivor:
    - `fvg_tighter_gap`
    - `5 trades / +0.1375% / PF Infinity`
  - weaker but still positive:
    - `fvg_shorter_age`
    - `4 trades / +0.1364% / PF Infinity`
  - verdict:
    - `ROBUST_FVG_SURVIVOR_BUT_NOT_FRONTIER_ON_SESSION_ARRAY_BASE`
  - interpretation:
    - nearby FVG geometry / freshness retests do not produce a new frontier
    - this face can survive, but it does not beat the current paired ICT stack
- calibrated `session-array window geometry` on top of the stronger slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_session_array_calibration.json`
  - robust survivors:
    - `broader_imbalance`
    - `broader_structural`
    - `dual_broader`
    - all preserve the exact same `6 trades / +0.1627% / PF Infinity` result as the base
  - weaker controls:
    - `session_array_off_control`
    - `shifted_later`
    - both weaken to `8 trades / +0.1233% / PF 7.2221`
  - verdict:
    - `ROBUST_SESSION_ARRAY_SURVIVOR_BUT_NOT_FRONTIER_ON_SLOW_RECOVERY_BASE`
  - interpretation:
    - session-array refinement remains part of the frontier
    - but nearby session-array window retests do not create a new edge on top of the stronger slow-recovery base
- calibrated `score / confluence-threshold geometry` on top of the stronger slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_score_calibration.json`
  - robust survivors:
    - `min_score_5`
    - `min_score_7`
    - `min_score_5_lower_ote`
    - `min_score_7_higher_fvg`
    - all preserve the exact same `6 trades / +0.1627% / PF Infinity` result as the base
  - weaker variants:
    - `min_score_8`
    - `min_score_9`
    - both thin the lane to `1 trade / +0.0011% / PF Infinity`
  - verdict:
    - `ROBUST_SCORE_SURVIVOR_BUT_NOT_FRONTIER_ON_SLOW_RECOVERY_BASE`
  - interpretation:
    - the current slow-recovery frontier is already passing the relevant score gate cleanly
    - nearby score-threshold retests do not create a new edge on top of the stronger base
- calibrated `liquidity sweep reclaim-strength geometry` on top of the stronger slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_reclaim_calibration.json`
  - robust survivor:
    - `reclaim_0p05`
    - preserves the exact same `6 trades / +0.1627% / PF Infinity` result as the base
  - thinner but still positive:
    - `reclaim_0p10`
    - `reclaim_0p15`
    - both reduce the lane to `5 trades / +0.1616% / PF Infinity`
  - weaker:
    - `reclaim_0p20`
    - `reclaim_0p25`
    - `reclaim_0p30`
    - all reduce the lane to `4 trades / +0.1409% / PF Infinity`
  - verdict:
    - `ROBUST_RECLAIM_SURVIVOR_BUT_NOT_FRONTIER_ON_SLOW_RECOVERY_BASE`
  - interpretation:
    - the current slow-recovery frontier already captures enough close-back strength after the sweep
    - nearby reclaim-strength retests do not create a new edge on top of the stronger base
- calibrated `structure_lookback` on top of the stronger slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_calibration.json`
  - best robust extension:
    - `structure_12`
    - `7 trades / +0.1826% / PF Infinity`
  - co-winner:
    - `structure_16`
    - `7 trades / +0.1826% / PF Infinity`
  - weaker but still positive:
    - `structure_24`
    - `5 trades / +0.1517% / PF Infinity`
  - thinner:
    - `structure_28`
    - `4 trades / +0.1417% / PF Infinity`
    - `structure_32`
    - `3 trades / +0.1035% / PF Infinity`
  - verdict:
    - `ROBUST_STRUCTURE_EXTENSION_IDENTIFIED_ON_SLOW_RECOVERY_BASE`
  - interpretation:
    - the paired ICT frontier now moves forward again on the PDF-aligned sweep -> structure-shift path
    - use `structure_lookback = 12` as the repo-approved tie-break helper over co-winner `16`
- calibrated `liquidity-pool lookback` on top of the stronger structure-aware slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_liquidity_pool_calibration.json`
  - preserving but non-improving variants:
    - `liq_pool_lookback_30`
    - `liq_pool_lookback_40`
    - `liq_pool_lookback_60`
    - `liq_pool_lookback_80`
    - `liq_pool_lookback_100`
    - all preserve `7 trades / +0.1826% / PF Infinity`
  - verdict:
    - `ROBUST_LIQUIDITY_POOL_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - the structure-aware frontier already captures enough liquidity-pool geometry
    - nearby pool-lookback retests do not create a new edge on top of the stronger base
- calibrated `order-block body quality` on top of the stronger structure-aware slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_ob_body_calibration.json`
  - preserving but non-improving variants:
    - `ob_body_0p20`
    - `ob_body_0p25`
    - `ob_body_0p35`
    - `ob_body_0p40`
    - `ob_body_0p50`
    - all preserve `7 trades / +0.1826% / PF Infinity`
  - important note:
    - `ob_entries = 0` across every tested variant
  - verdict:
    - `ROBUST_OB_BODY_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - the structure-aware frontier is still entirely FVG-led
    - nearby order-block body-quality retests do not create a new edge on top of the stronger base
- calibrated `order-block lookback` on top of the stronger structure-aware slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_ob_lookback_calibration.json`
  - preserving but non-improving variants:
    - `ob_lookback_8`
    - `ob_lookback_12`
    - `ob_lookback_20`
    - `ob_lookback_24`
    - `ob_lookback_30`
    - all preserve `7 trades / +0.1826% / PF Infinity`
  - important note:
    - `ob_entries = 0` across every tested variant
  - verdict:
    - `ROBUST_OB_LOOKBACK_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - the structure-aware frontier is still entirely FVG-led
    - nearby order-block lookback retests do not create a new edge on top of the stronger base
- calibrated `SMT threshold geometry` on top of the stronger short-SMT premium-plus-session-array frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_smt_threshold_calibration.json`
  - best non-base robust survivor:
    - `smt_looser_0p0012`
    - `5 trades / +0.1375% / PF Infinity`
  - weaker:
    - `smt_tighter_0p0007`
    - `4 trades / +0.0993% / PF Infinity`
  - control:
    - `smt_off_control`
    - `7 trades / +0.0818% / PF 2.4215`
  - verdict:
    - `ROBUST_SMT_THRESHOLD_SURVIVOR_BUT_NOT_FRONTIER_ON_SESSION_ARRAY_BASE`
  - interpretation:
    - nearby SMT-threshold retests do not produce a new frontier
    - loosening can preserve the frontier, tightening degrades it, and removing SMT entirely lowers quality
- recalibrated `kill zones` on top of the stronger short-SMT premium-plus-session-array frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_kill_zone_calibration.json`
  - best non-base robust survivor:
    - `kill_zones_broader`
    - `5 trades / +0.1375% / PF Infinity`
  - thinner survivor:
    - `kill_zones_shifted`
    - `2 trades / +0.0882% / PF Infinity`
  - rejected variants:
    - `kill_zones_default`
    - `kill_zones_ny_am_only`
    - `kill_zones_ny_only`
    - all collapsed to `0 trades`
  - verdict:
    - `ROBUST_KILL_ZONE_SURVIVOR_BUT_NOT_FRONTIER_ON_SESSION_ARRAY_BASE`
  - interpretation:
    - nearby kill-zone retests do not produce a new frontier
    - only broader windows preserve the current frontier; default and narrow windows still over-restrict the lane
- calibrated `broad session gating` on top of the stronger short-SMT premium-plus-session-array frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_session_gate_calibration.json`
  - best robust extension:
    - `ny_only_core`
    - `5 trades / +0.1420% / PF Infinity`
  - preserving but non-improving variants:
    - `trade_sessions_off_control`
    - `broader_sessions`
    - `narrower_london_ny_overlap`
    - all `5 trades / +0.1375% / PF Infinity`
  - thin only:
    - `london_only_core`
    - `1 trade / +0.0596% / PF Infinity`
  - verdict:
    - `ROBUST_SESSION_GATE_EXTENSION_IDENTIFIED_ON_SESSION_ARRAY_BASE`
  - interpretation:
    - the paired ICT frontier can move forward to the `NY-only core` session-gated stack
    - nearby broad-session retests should now be judged against that stronger session-gated base
- calibrated `daily bias` still collapses the short-SMT premium base back to `0 trades`, even after shorter-lookback and softer-threshold retests
- calibrated `CHOCH score` on top of the stronger structure-aware slow-recovery `NY-only` frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_choch_score_calibration.json`
  - all tested variants:
    - `0`
    - `1`
    - `2`
    - `4`
    - `5`
    - `6`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `CHOCH_SCORE_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - this round closed the implementation gap where `SCORE_CHOCH` existed in config but was not wired into strategy scoring
    - the repo is now more faithful to the PDF's sweep -> CHOCH / MSS -> displacement narrative
    - nearby CHOCH-score retries do not create a new edge on top of the stronger frontier
- the next step should be to preserve that premium-enabled short-SMT profile and only then revisit heavier context from that stronger base one filter at a time, excluding daily-bias-first, kill-zone-first, AMD-first, nearby external-liquidity-geometry retries, nearby previous-session-anchor retries, and nearby premium-discount retries

## Next Suggested Iteration Path

The safest next steps are:

1. Run the new ICT strategy as an experimental module, not as a replacement
   for the ORB baseline.
2. Validate whether the sweep -> shift -> FVG / OB flow produces stable trades
   on local data first.
3. Before layering yet another concept, use the real peer-symbol plumbing to
   run broader paired-data SMT evaluation and calibrate the current full-stack profile.
4. With local Alpaca `QQQ + SPY` now available, start from the robust survivor base:
   `previous-session anchor + external liquidity`.
5. Preserve the survivor base:
   - `previous-session anchor`
   - `external liquidity`
6. Preserve the first robust extension:
   - `session-array refinement`
7. Preserve the first robust geometry extension:
   - `liq_sweep_threshold = 0.0008`
8. Preserve the first robust SMT extension:
   - `smt_lookback = 10`
9. Preserve the first robust heavier-context survivor:
   - `premium / discount`
10. Treat `macro timing` only as a secondary optional branch:
    - `macro_early_shifted`
11. Treat `kill zones` only as a secondary optional branch:
    - `kill_zones_broader`
12. Treat `AMD` only as a thin optional branch:
    - `amd_short_and_soft`
13. Treat nearby `external liquidity` geometry as plateaued on the premium-enabled short-SMT base.
14. Treat nearby `previous-session anchor` retries as plateaued on the premium-enabled short-SMT base.
15. Treat nearby `premium / discount` retries as plateaued on the premium-enabled short-SMT base.
16. Keep `session-array refinement` as part of the validated frontier on the premium-enabled short-SMT base.
17. Treat `daily bias` as explicitly over-restrictive on the premium-enabled short-SMT base.
18. Treat nearby session-array window retests on the slow-recovery `NY-only`
    frontier as survivor-only, not as the next frontier.
19. Treat nearby score-threshold retests on the slow-recovery `NY-only`
    frontier as survivor-only, not as the next frontier.
20. Treat nearby reclaim-strength retests on the slow-recovery `NY-only`
    frontier as survivor-only, not as the next frontier.
21. Preserve the first robust structure extension on top of the slow-recovery `NY-only` frontier:
    - `structure_lookback = 12`
22. Treat nearby structure-lookback retries on the stronger frontier as exhausted
    unless the frontier changes materially.
23. Treat nearby liquidity-pool lookback retries on the stronger frontier as exhausted
    unless the frontier changes materially.
24. Treat nearby order-block body-quality retries on the stronger frontier as exhausted
    unless the frontier changes materially.
25. Treat nearby order-block lookback retries on the stronger frontier as exhausted
    unless the frontier changes materially.
26. Treat nearby OTE geometry and score retries on the stronger frontier as plateaued
    unless the frontier changes materially.
27. Treat nearby breaker-block lookback retries on the stronger frontier as plateaued
    unless the frontier changes materially.
28. Treat nearby IFVG lookback retries on the stronger frontier as plateaued
    unless the frontier changes materially.
29. Treat nearby liquidity-sweep threshold retries on the stronger frontier as survivor-only
    unless the frontier changes materially.
30. Treat nearby swing-threshold retries on the stronger frontier as plateaued
    unless the frontier changes materially.
31. Treat nearby FVG max-age retries on the stronger frontier as survivor-only
    unless the frontier changes materially.
32. Treat nearby FVG-gap retries on the stronger frontier as non-priority after the new 0.0006 extension
    unless the frontier changes materially.
33. Only after that, revisit controlled context reintroduction or add one
    additional ICT component at a time.

- calibrated `OTE geometry / score` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_ote_calibration.json`
  - every tested variant preserved the exact same `7 trades / +0.1826% / PF Infinity`
  - even `ote_off_control` matched the frontier result exactly
  - verdict:
    - `OTE_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - nearby OTE geometry or score retries are plateaued on the stronger structure-aware frontier
    - OTE is not the next edge source on top of the current paired ICT stack
- calibrated `breaker-block lookback` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_breaker_lookback_calibration.json`
  - every tested variant preserved the exact same `7 trades / +0.1826% / PF Infinity`
  - `breaker_entries = 0` across every tested variant
  - verdict:
    - `BREAKER_LOOKBACK_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - nearby breaker-block lookback retries are plateaued on the stronger structure-aware frontier
    - breaker blocks are not the next edge source on top of the current paired ICT stack
- calibrated `IFVG lookback` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_ifvg_lookback_calibration.json`
  - every tested variant preserved the exact same `7 trades / +0.1826% / PF Infinity`
  - `ifvg_entries = 0` across every tested variant
  - verdict:
    - `IFVG_LOOKBACK_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - nearby IFVG-lookback retries are plateaued on the stronger structure-aware frontier
    - IFVG is not the next edge source on top of the current paired ICT stack
- calibrated `liquidity-sweep threshold` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_sweep_threshold_calibration.json`
  - best non-base survivors:
    - `sweep_threshold_0p0006`
    - `sweep_threshold_0p0007`
    - both preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - tighter thresholds weaken the lane:
    - `0.0009 -> 6 trades / +0.1726%`
    - `0.0010 -> 5 trades / +0.1440%`
    - `0.0012 -> 4 trades / +0.0799%`
  - verdict:
    - `ROBUST_SWEEP_THRESHOLD_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - nearby sweep-threshold retries remain geometry-sensitive but do not beat the stronger structure-aware frontier
    - keep `liq_sweep_threshold = 0.0008` as the repo-approved base on this frontier
- calibrated `swing-threshold geometry` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_swing_threshold_calibration.json`
  - every tested variant preserved the exact same `7 trades / +0.1826% / PF Infinity`
  - verdict:
    - `SWING_THRESHOLD_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - nearby swing-threshold retries are plateaued on the stronger structure-aware frontier
    - swing-threshold is not the next edge source on top of the current paired ICT stack
- calibrated `FVG max-age` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_age_calibration.json`
  - best non-base robust survivors:
    - `fvg_age_25`
    - `fvg_age_30`
    - `fvg_age_40`
    - all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - shorter ages weaken the lane:
    - `fvg_age_15 -> 6 trades / +0.1815% / PF Infinity`
    - `fvg_age_10 -> 5 trades / +0.1609% / PF Infinity`
  - verdict:
    - `ROBUST_FVG_AGE_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE`
  - interpretation:
    - nearby FVG-age retries remain freshness-sensitive but do not beat the stronger structure-aware frontier
    - keep `fvg_max_age = 20` as the repo-approved base on this frontier
- calibrated `FVG min-gap geometry` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_gap_calibration.json`
  - robust extension:
    - `fvg_gap_0p0006`
    - improves the lane from `7 trades / +0.1826% / PF Infinity` to `7 trades / +0.1831% / PF Infinity`
  - equal survivor:
    - `fvg_gap_0p0008`
    - preserves the old base exactly
  - weaker tighter cluster:
    - `fvg_gap_0p0012`
    - `fvg_gap_0p0015`
    - `fvg_gap_0p0020`
    - all weaken the lane to `6 trades / +0.1627% / PF Infinity`
  - verdict:
    - `ROBUST_FVG_GAP_EXTENSION_IDENTIFIED_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier remains explicitly FVG-led
    - `fvg_min_gap_pct = 0.0006` is now the repo-approved FVG geometry on that frontier
- calibrated `take-profit RR` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_rr_calibration.json`
  - plateau result:
    - `rr_3p0`
    - `rr_3p5`
    - `rr_4p5`
    - `rr_5p0`
    - `rr_6p0`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `RR_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier is not currently sensitive to nearby take-profit RR changes
    - RR is not the next edge source on top of this paired ICT stack
- calibrated `stop-loss ATR multiple` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_stop_calibration.json`
  - plateau result:
    - `stop_atr_1p5`
    - `stop_atr_1p75`
    - `stop_atr_2p25`
    - `stop_atr_2p5`
    - `stop_atr_3p0`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `STOP_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier is not currently sensitive to nearby stop-loss ATR changes
    - stop-loss ATR is not the next edge source on top of this paired ICT stack
- calibrated `ATR period` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_atr_period_calibration.json`
  - plateau result:
    - `atr_period_10`
    - `atr_period_12`
    - `atr_period_16`
    - `atr_period_18`
    - `atr_period_20`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `ATR_PERIOD_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier is not currently sensitive to nearby ATR-period changes
    - ATR period is not the next edge source on top of this paired ICT stack
- calibrated `liquidity-sweep lookback` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_sweep_lookback_calibration.json`
  - plateau result:
    - `sweep_lookback_30`
    - `sweep_lookback_40`
    - `sweep_lookback_60`
    - `sweep_lookback_70`
    - `sweep_lookback_90`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `SWEEP_LOOKBACK_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier is not currently sensitive to nearby liquidity-sweep lookback changes
    - liquidity-sweep lookback is not the next edge source on top of this paired ICT stack
- calibrated `BOS score` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_bos_score_calibration.json`
  - plateau result:
    - `bos_score_1`
    - `bos_score_3`
    - `bos_score_4`
    - `bos_score_5`
    - `bos_score_6`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `BOS_SCORE_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier is not currently sensitive to nearby BOS-score changes
    - BOS score is not the next edge source on top of this paired ICT stack
- calibrated `liquidity-sweep score` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_sweep_score_calibration.json`
  - plateau result:
    - `sweep_score_2`
    - `sweep_score_4`
    - `sweep_score_5`
    - `sweep_score_6`
    - `sweep_score_7`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `SWEEP_SCORE_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier is not currently sensitive to nearby liquidity-sweep score changes
    - liquidity-sweep score is not the next edge source on top of this paired ICT stack
- calibrated `FVG score` on top of the stronger structure-aware frontier:
  - file:
    - `results/qc_regime_prototypes/ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_score_calibration.json`
  - plateau result:
    - `fvg_score_1`
    - `fvg_score_3`
    - `fvg_score_4`
    - `fvg_score_5`
    - `fvg_score_6`
    - all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - verdict:
    - `FVG_SCORE_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE`
  - interpretation:
    - the stronger structure-aware frontier is not currently sensitive to nearby FVG-score changes
    - FVG score is not the next edge source on top of this paired ICT stack

## Current Development Lane

The active deterministic ICT roadmap is now:

1. Preserve `v26-profit-lock` as the production ORB baseline.
2. Treat ICT as a separate revolutionary strategy lane.
3. Keep the current mechanical core:
   - liquidity sweep
   - MSS
   - displacement proxy via FVG / OB / breaker / IFVG
   - external-liquidity gating
   - SMT divergence gating
   - peer-symbol data integration for real SMT backtests
   - AMD / market-maker path gating
   - macro timing window gating
   - previous-session anchor gating
   - session-specific dealing-array refinement
   - OTE scoring
   - kill-zone specialization
   - conservative daily bias
   - conservative premium / discount context
4. Add only one ICT feature per round.
5. Require synthetic unit tests for every new ICT feature.
6. Use Alpaca and Blueshift as research lanes before any QC promotion attempt.
7. Do not let ICT overwrite the ORB production baseline until it earns that
   right through separate validation.

## 500-Trades Roadmap Branch

- strict frontier funnel benchmark now lives in:
  - [ict_frontier_funnel.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_frontier_funnel.json)
- first lite reversal baseline now lives in:
  - [ict_lite_reversal_baseline.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_reversal_baseline.json)
- strict frontier benchmark:
  - `8 trades / +0.2119% / PF Infinity`
- lite reversal baseline:
  - `14 trades / +10.6136% / PF 9.9873`
- verdict:
  - `LITE_ICT_REVERSAL_BASELINE_IMPROVES_DENSITY_BUT_STAYS_BELOW_100_TRADES`
- density interpretation:
  - stripping the heaviest pre-arm context blockers improves both activity and return
  - the first 500-trades gate is still far away:
    - `86` more trades are needed to clear `100`
  - the main blocker on the lite lane is now:
    - `smt_filtered_sweeps = 301`
- this means the next roadmap step should stop optimizing premium / previous-session / external-liquidity density and instead decide whether SMT should be relaxed, shortened, or split into a later reintroduction pass on the lite lane

## Lite SMT Density Pass

- SMT-density analysis now lives in:
  - [ict_lite_smt_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_smt_density.json)
- lite baseline:
  - `14 trades / +10.6136% / PF 9.9873`
- first robust lite SMT extension:
  - `smt_threshold = 0.0015`
  - `16 trades / +0.2902% / PF 4.6325`
- density-only control:
  - `SMT off -> 23 trades / +0.1108% / PF 1.4937`
- verdict:
  - `ROBUST_LITE_SMT_EXTENSION_IDENTIFIED`
- interpretation:
  - the lite lane should keep SMT, but in a looser form
  - the first clean density lift is not `SMT off`; it is `SMT on` with a relaxed threshold

## Lite Geometry Round 1

- lite geometry round 1 now lives in:
  - [ict_lite_geometry_round1.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_geometry_round1.json)
- relaxed-SMT lite base:
  - `16 trades / +0.2902% / PF 4.6325`
- first robust lite geometry extension:
  - `liq_sweep_threshold = 0.0006`
  - `18 trades / +0.3229% / PF 4.3357`
- density-only looser sweep variants:
  - `0.0005 -> 24 trades / +0.2075% / PF 2.1191`
  - `0.0003 -> 38 trades / +0.2093% / PF 1.6001`
- verdict:
  - `ROBUST_LITE_GEOMETRY_EXTENSION_IDENTIFIED`
- interpretation:
  - the first clean lite geometry edge is sweep geometry, not structure lookback
  - the roadmap should now take the stronger `18-trade` lite frontier into the retest-focused round

## Lite Retest Round 2

- lite retest round 2 now lives in:
  - [ict_lite_retest_round2.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_retest_round2.json)
- lite geometry base:
  - `18 trades / +0.3229% / PF 4.3357`
- first robust lite retest extension:
  - `fvg_revisit_min_delay_bars = 2`
  - `18 trades / +0.3529% / PF 4.6792`
- positive but weaker delay survivors:
  - `delay_0 -> 18 trades / +0.3433% / PF 4.2505`
  - `delay_1 -> 18 trades / +0.3433% / PF 4.2505`
- density-only control:
  - `depth_0p00 -> 20 trades / +0.1989% / PF 2.6385`
- verdict:
  - `ROBUST_LITE_RETEST_EXTENSION_IDENTIFIED`
- interpretation:
  - the first clean retest-aware density edge is slightly faster revisit timing, not removing revisit depth
  - the 500-trades roadmap should now carry the stronger lite frontier into the session-density phase

## Lite Session Density

- lite session-density round now lives in:
  - [ict_lite_session_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_session_density.json)
- active lite retest frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- best density-only session variant:
  - `ny_14_21`
  - `19 trades / +0.3395% / PF 4.2127`
- weaker but clustered session-density controls:
  - `session_off_control -> 19 trades / +0.3327% / PF 4.1125`
  - `ny_13_21 -> 19 trades / +0.3327% / PF 4.1125`
  - `broader_sessions -> 19 trades / +0.3327% / PF 4.1125`
  - `london_ny_overlap -> 19 trades / +0.3327% / PF 4.1125`
- verdict:
  - `LITE_SESSION_DENSITY_EXTENSION_ONLY`
- interpretation:
  - session broadening can add one trade on the lite lane, but it weakens return enough that the active frontier should stay on the faster-retest NY-core profile
  - the next 500-trades roadmap branch should move to continuation / setup-density rather than spending more rounds widening sessions

## Lite Setup Density

- lite setup-density round now lives in:
  - [ict_lite_setup_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_setup_density.json)
- active lite retest frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- best density-only setup variant:
  - `fvg_max_age = 60`
  - `20 trades / +0.3324% / PF 4.1996`
- moderate surviving setup-density controls:
  - `fvg_max_age = 30 -> 19 trades / +0.3335% / PF 4.2256`
  - `fvg_max_age = 40 -> 19 trades / +0.3335% / PF 4.2256`
- verdict:
  - `LITE_SETUP_DENSITY_EXTENSION_ONLY`
- interpretation:
  - longer armed-setup life can add trades on the lite lane, but it gives back too much return to replace the faster-retest frontier
  - the next 500-trades roadmap branch should move into the explicit continuation-lane compare

## Continuation Lane Compare

- continuation lane compare now lives in:
  - [ict_continuation_lane_compare.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_continuation_lane_compare.json)
- active lite reversal frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- continuation proxy lane:
  - `1465 trades / -6.3814% / PF 0.6524`
- trade delta vs lite reversal frontier:
  - `+1447`
- verdict:
  - `CONTINUATION_LANE_REJECTED`
- interpretation:
  - a naive `BOS -> FVG retest` continuation proxy creates huge trade density, but the quality collapse is far too severe to justify promotion
  - continuation should only be revisited with a materially stricter architecture, not by relaxing the current proxy

## Lite Frontier Funnel

- lite frontier funnel now lives in:
  - [ict_lite_frontier_funnel.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_funnel.json)
- active lite frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- stage counts:
  - `1229 raw sweeps -> 1036 accepted sweeps -> 57 shift candidates -> 36 armed setups -> 20 retest candidates -> 18 entries`
- first roadmap gap:
  - still `82` trades short of the `100`-trade gate
- largest stage collapse:
  - `accepted sweeps -> shift candidates`
  - `1036 -> 57`
  - drop count `979`
- top recorded filters:
  - `smt_filtered_sweeps = 191`
  - `fvg_depth_filtered_retests = 23`
  - `delivery_missing_shifts = 19`
- verdict:
  - `LITE_FRONTIER_FUNNEL_IDENTIFIED_NEXT_CHOKE_POINT`
- interpretation:
  - the active lite branch is not primarily failing at the last-step entry trigger
  - the next 500-trades optimization should target sweep-to-shift conversion and residual SMT choke points before opening another new lane

## Lite Shift Density

- lite shift-density round now lives in:
  - [ict_lite_shift_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_shift_density.json)
- active lite frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- best density-only shift variant:
  - `structure_lookback = 8`
  - `23 trades / +0.3189% / PF 3.3014`
- other notable variants:
  - `structure_10 -> 20 trades / +0.3241% / PF 3.6012`
  - `structure_16 -> 17 trades / +0.3674% / PF 5.6854`
- conversion effect:
  - base shift candidates:
    - `57`
  - `structure_8` shift candidates:
    - `77`
  - base armed setups:
    - `36`
  - `structure_8` armed setups:
    - `48`
- verdict:
  - `LITE_SHIFT_DENSITY_EXTENSION_ONLY`
- interpretation:
  - shorter structure lookback really does increase sweep-to-shift conversion on the lite branch
  - but the density gain is not clean enough to replace the stronger faster-retest frontier
  - `structure_8` should be treated as a density candidate, not a promoted repo frontier

## Lite Frontier SMT Recalibration

- lite frontier SMT recalibration now lives in:
  - [ict_lite_frontier_smt_recalibration.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_smt_recalibration.json)
- active lite frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- plateau SMT survivors:
  - `smt_threshold = 0.0018 -> 18 trades / +0.3529% / PF 4.6792`
  - `smt_threshold = 0.0020 -> 18 trades / +0.3529% / PF 4.6792`
- density-only SMT variants:
  - `smt_threshold = 0.0025 -> 22 trades / +0.2915% / PF 2.7478`
  - `SMT off -> 25 trades / +0.1730% / PF 1.6870`
- verdict:
  - `LITE_FRONTIER_SMT_DENSITY_EXTENSION_ONLY`
- interpretation:
  - SMT is no longer the clean promotion lever on the active lite frontier
  - mild threshold relaxation reduces raw SMT filtering but does not beat the current lane
  - aggressive relaxation can add density, but not cleanly enough to justify helper promotion

## Lite Frontier Revisit Depth

- lite frontier revisit-depth recalibration now lives in:
  - [ict_lite_frontier_revisit_depth.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_revisit_depth.json)
- active lite frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- best density-only revisit-depth variant:
  - `depth_0p00 -> 20 trades / +0.2547% / PF 3.0260`
- weaker survivor:
  - `depth_0p25 -> 18 trades / +0.3136% / PF 4.2283`
- stricter depth gates:
  - `depth_0p75 -> 15 trades / +0.2340% / PF 3.4133`
  - `depth_1p00 -> 13 trades / +0.1434% / PF 2.5468`
- verdict:
  - `LITE_FRONTIER_REVISIT_DEPTH_DENSITY_EXTENSION_ONLY`
- interpretation:
  - revisit-depth is no longer a clean frontier-promotion lever on the active lite branch
  - removing the depth gate can add trades, but only by weakening quality too much
  - the current `0.5` depth remains the best promoted tradeoff

## Lite Frontier Displacement

- lite frontier displacement recalibration now lives in:
  - [ict_lite_frontier_displacement.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_displacement.json)
- active lite frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- plateau survivors:
  - `displacement_0p15 -> 18 trades / +0.3529% / PF 4.6792`
  - `displacement_0p20 -> 18 trades / +0.3529% / PF 4.6792`
- weaker survivors:
  - `displacement_0p00 -> 17 trades / +0.3378% / PF 4.4993`
  - `displacement_0p05 -> 17 trades / +0.3378% / PF 4.4993`
- verdict:
  - `LITE_FRONTIER_DISPLACEMENT_SURVIVOR_BUT_NOT_EXTENSION`
- interpretation:
  - displacement-body gating is no longer a clean promotion lever on the active lite branch
  - the current `0.10` setting is already on the stable plateau and does not need more retries

## Combined Lanes

- combined-lane compare now lives in:
  - [ict_combined_lanes.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_combined_lanes.json)
- active lite reversal frontier:
  - `18 trades / +0.3529% / PF 4.6792`
- combined reversal + continuation lane:
  - `1464 trades / -6.0444% / PF 0.6733`
- lane mix:
  - `reversal_entries = 14`
  - `continuation_entries = 1450`
  - `same_bar_lane_conflicts = 6`
- verdict:
  - `COMBINED_LANE_REJECTED`
- interpretation:
  - naive lane fusion does increase density, but it inherits almost all of the continuation proxy's overtrading weakness
  - the 500-trades roadmap should treat combined-lane fusion as completed-and-rejected and return to improving lite reversal conversion instead

## Position Sizing Compare

- position-sizing compare now lives in:
  - [ict_position_sizing_compare.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_position_sizing_compare.json)
- active lite frontier signal stack:
  - `18 trades / 72.2222% win rate`
- sizing variants:
  - `research_fixed_10 -> +0.3529% / PF 4.7959`
  - `fixed_40_shares -> +1.4118% / PF 4.7959`
  - `capital_50pct_min40 -> +5.7033% / PF 5.2290`
  - `capital_100pct_min40 -> +11.6677% / PF 5.1253`
- verdict:
  - `POSITION_SIZING_IMPACT_CONFIRMED`
- interpretation:
  - the active lite frontier remains sparse, but the current 10-share research convention clearly understates its economic return
  - sizing should now be treated as an explicit comparison dimension in ICT research reporting rather than as a hidden presentation choice

## Strict Funnel Lifecycle Counters

- strict frontier lifecycle leakage is now instrumented directly in [ict_entry_model.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/src/strategies/ict_entry_model.py)
- refreshed artifact:
  - [ict_frontier_funnel.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_frontier_funnel.json)
- new lifecycle counts:
  - `sweep_expired_before_shift = 263`
  - `sweep_blocked_by_existing_pending = 18`
  - `armed_setup_expired_before_retest = 2`
- interpretation:
  - the largest newly-measured strict-frontier leak is sweep setups expiring before they can confirm shift
  - the 500-trades roadmap should keep prioritizing sweep-to-shift conversion before spending more rounds on downstream retest purity

## Lite Frontier Revisit Delay

- active-lite revisit-delay recalibration now lives in:
  - [analyze_ict_lite_frontier_revisit_delay.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_frontier_revisit_delay.py)
  - [ict_lite_frontier_revisit_delay.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_revisit_delay.json)
- promoted active lite frontier base:
  - `fvg_revisit_min_delay_bars = 2`
  - `18 trades / +0.4353% / PF 5.0247`
- survivor-only variants:
  - `1 -> 18 trades / +0.4304% / PF 4.5887`
  - `3 -> 18 trades / +0.4092% / PF 4.6924`
- thinner higher-delay branches:
  - `4 -> 17 trades / +0.4211% / PF 5.3819`
  - `5 -> 17 trades / +0.3735% / PF 5.0699`
- interpretation:
  - the promoted active lite frontier still wants the faster but not immediate `2-bar` revisit delay
  - both shallower and slower delay variants stay positive, but none beat the promoted base cleanly enough to replace it
  - this closes the remaining `P3-3 / fvg_revisit_min_delay_bars` backlog row in the 500-trades table

## Lite Frontier Revisit Depth

- active-lite revisit-depth recalibration now lives in:
  - [analyze_ict_lite_frontier_revisit_depth.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_frontier_revisit_depth.py)
  - [ict_lite_frontier_revisit_depth.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_revisit_depth.json)
- promoted active lite frontier base:
  - `fvg_revisit_depth_ratio = 0.5`
  - `18 trades / +0.4353% / PF 5.0247`
- density-only lower-depth branch:
  - `0.00 -> 20 trades / +0.3496% / PF 3.3842`
- survivor-only variants:
  - `0.25 -> 18 trades / +0.4085% / PF 4.7221`
  - `0.75 -> 15 trades / +0.3164% / PF 3.7587`
  - `1.00 -> 13 trades / +0.2259% / PF 2.8922`
- interpretation:
  - the promoted lite frontier still wants the stricter `0.5` revisit-depth gate
  - shallower revisit requirements do add density, but they do not preserve enough return to replace the promoted base
  - this closes the remaining `P3-2 / fvg_revisit_depth_ratio` backlog row in the 500-trades table

## Lite Frontier FVG Gap

- active-lite FVG-gap recalibration now lives in:
  - [analyze_ict_lite_frontier_fvg_gap.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_frontier_fvg_gap.py)
  - [ict_lite_frontier_fvg_gap.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_fvg_gap.json)
- promoted active lite frontier base:
  - `18 trades / +0.4353% / PF 5.0247`
- density-only lower-gap variants:
  - `0.0003 -> 20 trades / +0.3379% / PF 4.0458`
  - `0.0005 -> 19 trades / +0.3524% / PF 4.6611`
- thinner survivor:
  - `0.0012 -> 17 trades / +0.4154% / PF 4.8477`
- interpretation:
  - the active lite frontier is now the tighter `fvg_min_gap_pct = 0.0010` stack
  - looser FVG-gap settings can add one or two trades, but they give back too much return to replace the promoted base
  - this closes the remaining `P3-1 / fvg_min_gap_pct` backlog row in the 500-trades table as completed density-only evidence around the promoted base

## Lite Frontier Sweep Threshold

- active-lite sweep-threshold recalibration now lives in:
  - [analyze_ict_lite_frontier_sweep_threshold.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_frontier_sweep_threshold.py)
  - [ict_lite_frontier_sweep_threshold.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_frontier_sweep_threshold.json)
- promoted active lite frontier base:
  - `liq_sweep_threshold = 0.0006`
  - `18 trades / +0.4353% / PF 5.0247`
- best density-only variants:
  - `0.0003 -> 38 trades / +0.3231% / PF 1.7261`
  - `0.0004 -> 28 trades / +0.3492% / PF 2.1417`
- survivor-only variant:
  - `0.0007 -> 17 trades / +0.4498% / PF 6.1051`
- interpretation:
  - looser sweep thresholds can materially increase trade count, but not cleanly enough to replace the promoted base
  - tighter `0.0007` improves return slightly, but gives up one trade and therefore does not qualify as the next lite-frontier promotion
  - this closes the remaining `P1 / liq_sweep_threshold` backlog row in the 500-trades table

## Lite Frontier Structure Lookback

- active-lite structure-lookback recalibration now lives in:
  - [analyze_ict_lite_shift_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_shift_density.py)
  - [ict_lite_shift_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_shift_density.json)
- promoted active lite frontier base:
  - `structure_lookback = 12`
  - `18 trades / +0.4353% / PF 5.0247`
- best density-only variants:
  - `structure_8 -> 24 trades / +0.3928% / PF 3.4265`
  - `structure_10 -> 20 trades / +0.4065% / PF 3.8671`
- survivor-only variant:
  - `structure_16 -> 17 trades / +0.4498% / PF 6.1051`
- interpretation:
  - shorter structure windows can increase shift candidates and trade count, but not cleanly enough to replace the promoted base
  - longer structure windows can improve return slightly, but only by thinning the lane
  - this closes the remaining `P1 / structure_lookback` backlog row in the 500-trades table

## Lite Frontier Session Range

- active-lite session-range recalibration now lives in:
  - [analyze_ict_lite_session_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_session_density.py)
  - [ict_lite_session_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_session_density.json)
- promoted active lite frontier base:
  - `NY-only`
  - `18 trades / +0.4353% / PF 5.0247`
- best density-only variants:
  - `ny_14_21 -> 19 trades / +0.4219% / PF 4.5237`
  - `session_off_control -> 19 trades / +0.4151% / PF 4.4203`
- interpretation:
  - broader session windows can add one trade, but not cleanly enough to replace the promoted base
  - the current `NY-only` session gate therefore remains the clean promoted session setting on the active lite frontier
  - this closes the remaining `P3 / session 範圍` backlog row in the 500-trades table

## Structural Daily Bias

- structural daily-bias replay now lives in:
  - [analyze_ict_structural_daily_bias.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_structural_daily_bias.py)
  - [ict_structural_daily_bias.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_structural_daily_bias.json)
- active lite frontier base:
  - `18 trades / +0.4353% / PF 5.0247`
- statistical daily-bias control:
  - `5 trades / +0.2074% / PF 12.1773`
  - `daily_bias_filtered_setups = 884`
- structural daily-bias threshold `1`:
  - `9 trades / +0.1826% / PF 5.5052`
  - `daily_bias_filtered_setups = 640`
- structural daily-bias threshold `2`:
  - `0 trades / +0.0000% / PF Infinity`
  - `daily_bias_filtered_setups = 1352`
- verdict:
  - `STRUCTURAL_DAILY_BIAS_SURVIVOR_BUT_NOT_EXTENSION`
- interpretation:
  - both higher-timeframe bias modes are now real infrastructure, but neither beats the promoted lite frontier
  - structural bias is less destructive than the old statistical mode, yet it still cuts the lane from `18` trades to `9`
  - this closes the roadmap's `P2-3` multi-timeframe daily-bias backlog without promoting the feature

## Lite Pending Capacity

- active-lite pending-capacity replay lives in:
  - [analyze_ict_lite_pending_capacity.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_pending_capacity.py)
  - [ict_lite_pending_capacity.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_pending_capacity.json)
- active lite frontier base:
  - `18 trades / +0.4353% / PF 5.0247`
- plateau variants:
  - `pending_cap_2 -> 18 trades / +0.4353% / PF 5.0247`
  - `pending_cap_3 -> 18 trades / +0.4353% / PF 5.0247`
  - `pending_cap_4 -> 18 trades / +0.4353% / PF 5.0247`
- lifecycle impact:
  - `sweep_blocked_by_existing_pending` drops from `122` to `7 / 0 / 0`
  - `sweep_expired_before_shift` rises from `1000` to `1110 / 1116 / 1116`
- verdict:
  - `LITE_PENDING_CAPACITY_PLATEAU_ON_ACTIVE_FRONTIER`
- interpretation:
  - multi-pending capacity is real infrastructure because it changes setup lifecycle counts materially
  - it still does not change the promoted active-lite realized result, so it should not remain a live roadmap optimization row

## Lite Score Quality Repair

- quality-aware score repair replay now lives in:
  - [analyze_ict_lite_score_quality_repair.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_score_quality_repair.py)
  - [ict_lite_score_quality_repair.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_score_quality_repair.json)
- active lite frontier base:
  - `18 trades / +0.4353% / PF 5.0247`
- quality-aware scoring variants:
  - `quality_score_6 / 7 / 8 / 9 -> 18 trades / +0.4353% / PF 5.0247`
  - `quality_score_10 -> 18 trades / +0.4353% / PF 5.0247`
- quality-system impact:
  - `score_quality_boosted_shifts` rises from `0` to `36 / 36 / 36 / 36 / 38`
  - `score_quality_bonus_total` rises from `0.0` to `88.1168 / 88.1168 / 88.1168 / 88.1168 / 92.5654`
  - `score_filtered_shifts` finally moves from `0` to `7` at `min_score_to_trade = 10`
- verdict:
  - `QUALITY_SCORE_SYSTEM_REPAIRED_BUT_NOT_EXTENSION`
- interpretation:
  - shift scoring now meaningfully reflects sweep depth, displacement strength, and FVG gap size instead of behaving like a pure static stamp
  - the repaired score stack can filter weak shifts, but it still does not improve the promoted active-lite realized result enough to replace the current frontier

## Active Lite Swing Structure

- active-lite swing-structure replay now lives in:
  - [analyze_ict_lite_active_swing_structure.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_active_swing_structure.py)
  - [ict_lite_active_swing_structure.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_active_swing_structure.json)
- rolling promoted base:
  - `18 trades / +0.4353% / PF 5.0247`
- swing density branch:
  - `swing_threshold = 2 -> 22 trades / +0.3397% / PF 2.7580`
- swing survivor branch:
  - `swing_threshold = 3 -> 14 trades / +0.3817% / PF 5.4217`
- farther swing variants:
  - `swing_threshold = 1 -> 45 trades / +0.1622% / PF 1.2742`
  - `swing_threshold = 4 -> 7 trades / +0.1843% / PF 4.1886`
- verdict:
  - `ACTIVE_LITE_SWING_STRUCTURE_DENSITY_EXTENSION_ONLY`
- interpretation:
  - confirmed swing high/low structure references are now a real tested branch on the promoted lite stack
  - `swing_threshold = 2` opens the lane, but quality drops too much to replace the rolling base
  - `swing_threshold = 3` preserves quality better, but it thins the lane instead of improving it

## Active Lite Dual-Speed Recovery

- active-lite dual-speed recovery replay now lives in:
  - [analyze_ict_lite_dual_speed_recovery.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_dual_speed_recovery.py)
  - [ict_lite_dual_speed_recovery.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_dual_speed_recovery.json)
- standardized fast-only base:
  - `18 trades / +13.8395% / PF 6.3079`
- dual-speed density branches:
  - `dual_speed_6 -> 28 trades / +10.2286% / PF 2.9682`
  - `dual_speed_8 -> 34 trades / +9.2985% / PF 2.4896`
  - `dual_speed_12 -> 45 trades / +10.0200% / PF 2.2037`
- lifecycle effect:
  - `sweep_expired_before_shift` improves from `1000` to `959 / 924 / 861`
  - `slow_recovery_entries` rises from `0` to `10 / 16 / 27`
- verdict:
  - `DUAL_SPEED_RECOVERY_DENSITY_EXTENSION_ONLY`
- interpretation:
  - longer recovery windows do open the lane materially on the promoted lite branch
  - none of the dual-speed variants beat the fast-only base on total return or profit factor
  - this closes the roadmap's `P0-2` recovery-window backlog and also confirms `sweep_expired_before_shift` is a live counter, not a broken always-zero bug

## Active Lite Structure Lookback

- active-lite structure-lookback replay now lives in:
  - [analyze_ict_lite_shift_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_shift_density.py)
  - [ict_lite_shift_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_shift_density.json)
- promoted rolling base:
  - `structure_lookback = 12`
  - `18 trades / +0.4353% / PF 5.0247`
- density branches:
  - `structure_8 -> 24 trades / +0.3928% / PF 3.4265`
  - `structure_10 -> 20 trades / +0.4065% / PF 3.8671`
- thinner survivor branches:
  - `structure_16 -> 17 trades / +0.4498% / PF 6.1051`
  - `structure_20 -> 15 trades / +0.3957% / PF 5.6252`
- verdict:
  - `ACTIVE_LITE_STRUCTURE_LOOKBACK_DENSITY_EXTENSION_ONLY`
- interpretation:
  - shorter structure windows do open the sweep-to-shift funnel on the promoted lite stack
  - none of the shorter variants beat the return profile of the promoted base cleanly enough to replace it
  - the older roadmap rows for `P0-3` and the temporary rolling-structure bug framing should now be treated as completed density-only evidence rather than live backlog

## Qualified Continuation Density Branch

- qualified continuation density replay now lives in:
  - [analyze_ict_lite_qualified_continuation_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_qualified_continuation_density.py)
  - [ict_lite_qualified_continuation_density.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_lite_qualified_continuation_density.json)
- capital-based quality reference:
  - `active_lite_frontier_base -> 18 trades / +13.8395% / PF 6.3079`
- qualified continuation density branches:
  - `continuation_on -> 22 trades / +10.8236% / PF 3.9857`
  - `dual_speed_12_continuation -> 53 trades / +6.2218% / PF 1.6786`
  - `balanced_density_candidate -> 60 trades / +8.0209% / PF 1.8128`
  - first promoted density branch:
    - `qualified_continuation_density -> 99 trades / +4.9880% / PF 1.4148`
  - refined density frontier:
    - SMT-refined branch:
      - `qualified_continuation_density -> 94 trades / +10.3653% / PF 1.6903`
    - exit-refined branch:
      - `qualified_continuation_density -> 94 trades / +11.3181% / PF 1.7294`
    - timing-refined branch:
      - `qualified_continuation_density -> 93 trades / +11.8319% / PF 1.8131`
- lifecycle effect on the promoted density winner:
  - `continuation_entries = 62`
  - `slow_recovery_entries = 58`
  - `sweep_expired_before_shift = 790`
- verdict:
  - `QUALIFIED_CONTINUATION_DENSITY_BRANCH_IDENTIFIED`
- interpretation:
  - the 18-trade active lite frontier remains the clean quality branch
  - ICT now also has a distinct positive higher-density lane instead of being trapped at `18 trades`
  - tightening SMT on that lane keeps the branch in the `90+ trades` regime while materially improving return and profit factor
  - tightening the branch's take-profit target from `4.0R` to `3.0R` improves total return again without giving up any trades
  - slowing the qualified branch's retest timing from `3` to `4` bars improves total return and PF again, even though it gives up one trade
  - nearby `100+ trade` variants exist, but they still trail this stronger `93-trade / +11.8319%` branch on return quality
  - the next 500-trades push should now find a `100+ trade` variant that can actually beat this stronger timing-refined branch rather than reopening the old roadmap shell

