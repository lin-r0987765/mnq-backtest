## ICT Development Workflow

This file defines how ICT work should be developed in this repository.

It exists because the repo now has two distinct lanes:

- ORB production lane:
  - official baseline remains `v26-profit-lock`
  - QuantConnect 10-year workflow is still the only promotion authority
- ICT development lane:
  - intended for genuinely different trade structure research
  - must not overwrite the ORB production lane until it earns that right

### Source Hierarchy

Use these sources in order:

1. `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\ict\ICT 交易策略詳解與教學.pdf`
2. `ICT_RESEARCH.md`
3. `src/strategies/ict_entry_model.py`
4. `test_ict_strategy.py`

### ICT Development Principles

1. Start from narrative, then convert to deterministic rules.
2. Implement one new ICT concept per round.
3. Add tests before calling the round complete.
4. Prefer explicit rule mapping over discretionary interpretation.
5. Do not change ORB production files when working only on ICT research.

### Deterministic Mapping Ladder

The intended build order is:

1. Liquidity map
   - identify external highs/lows or rolling pools
   - distinguish which side is likely to be targeted
   - current repo implementation now includes a conservative external-liquidity gate
2. Sweep
   - require actual run on liquidity and return / rejection logic
3. Structure shift
   - confirm MSS / CHOCH after the sweep
4. Displacement quality
   - require strong move quality, not just a tiny break
5. Delivery array
   - FVG first
   - OB fallback
   - breaker fallback
   - IFVG fallback
6. Time and session logic
   - broad sessions
   - kill-zone specialization
   - macro timing windows
   - later: deeper macro and event refinement
7. Higher-timeframe narrative
   - daily bias
   - premium / discount context
   - AMD / market-maker path logic
   - previous-session liquidity map / dealing-range anchors
   - later: deeper macro timing refinement
8. Session-specific dealing arrays
   - imbalance windows can prefer FVG / IFVG
   - structural windows can prefer OB / breaker
9. Cross-market confirmation
   - SMT divergence via peer high/low series
   - peer-symbol data integration is now implemented
   - next: broader paired-data empirical SMT testing, profile calibration, controlled context-filter reintroduction, survivor-bundle pairwise calibration, robust sweep-geometry calibration, and then robust SMT recalibration before reopening heavier context

### Validation Ladder

Every ICT feature should pass this ladder:

1. Synthetic unit tests
2. Local replay / backtest on available data
3. Alpaca lane if the script supports it
4. Blueshift multi-file prototype if the logic grows beyond simple local runs
5. QuantConnect research evaluator only after the local evidence is coherent
6. Separate QC promotion decision only if the ICT lane becomes mature enough

### Active Priority Queue

Current implemented ICT components:

- liquidity sweep detection
- market structure shift confirmation
- FVG-first entry
- OB fallback
- breaker fallback
- IFVG fallback
- external-liquidity gating
- SMT divergence gating
- peer-symbol data integration for real SMT backtests
- AMD / market-maker path gating
- macro timing window gating
- previous-session anchor gating
- session-specific dealing-array refinement
- peer-symbol data integration for real SMT backtests
- OTE scoring
- explicit displacement body-quality gating on structure-shift confirmation
- explicit FVG consequent-encroachment / revisit-depth gating on the FVG-led entry trigger
- explicit FVG origin-lag gating between structure-shift confirmation and the first valid FVG zone
- explicit FVG origin-body gating on the candle that creates the FVG
- explicit FVG origin body-vs-ATR gating on the candle that creates the FVG
- explicit FVG origin close-position gating on the candle that creates the FVG
- explicit FVG origin opposite-wick gating on the candle that creates the FVG
- explicit FVG origin range-vs-ATR gating on the candle that creates the FVG
- newest-valid FVG selection inside the structure-shift window so FVG recency matches OB / breaker / IFVG
- single-event structure-confirmation scoring so BOS and CHOCH are not double-counted on the same reversal
- explicit FVG rejection-wick gating on the FVG-led entry trigger
- explicit FVG rejection-body gating on the FVG-led entry trigger
- explicit structure-shift close-through buffer gating on the post-sweep MSS / CHOCH confirmation layer
- explicit FVG revisit-delay gating on the FVG-led retest trigger
- explicit FVG retest-touch cap gating on the FVG-led retest trigger
- explicit displacement range-vs-ATR gating on the post-sweep structure-shift confirmation layer
- multi-pending setup state management so same-direction sweep density can be calibrated directly without hard-coding a single pending setup per side
- terminal `StrategyResult` return restored and now covered by regression protection so `generate_signals` cannot silently fall off the end of the function after refactors
- broad session gating
- optional kill-zone specialization
- conservative higher-timeframe daily bias
- conservative premium / discount context
- repo-approved ICT research profile wiring

### 500-Trades Combined-Lane Note

- analyzer-only combined-lane fusion is now complete through:
  - [analyze_ict_combined_lanes.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_combined_lanes.py)
- result:
  - active lite reversal frontier stays at `18 trades / +0.3529% / PF 4.6792`
  - merged reversal + continuation lane degrades to `1464 trades / -6.0444% / PF 0.6733`
- implication:
  - do not spend more mainline iteration budget on naive lane fusion
  - the 500-trades roadmap should return to improving lite reversal conversion instead

### 500-Trades Position-Sizing Note

- analyzer-only sizing compare is now available through:
  - [analyze_ict_position_sizing_compare.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_position_sizing_compare.py)
- result on the active lite frontier:
  - `research_fixed_10 -> +0.3529%`
  - `fixed_40_shares -> +1.4118%`
  - `capital_50pct_min40 -> +5.7033%`
  - `capital_100pct_min40 -> +11.6677%`
- implication:
  - current tiny fixed-size research reporting materially understates the branch's economic return
  - sizing still does not solve the density bottleneck, so future rounds should separate economic replay questions from conversion work
- lite-frontier funnel stage instrumentation for the active `18-trade` 500-trades branch

### 500-Trades Walk-Forward Note

- ICT-specific walk-forward is now available through:
  - [run_ict_walk_forward.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/run_ict_walk_forward.py)
- result on the active lite frontier:
  - `67` folds with `train=60 / validation=20 / holdout=20 / step=20`
  - `avg_validation_return_pct = +0.0053%`
  - `avg_holdout_return_pct = +0.0053%`
  - `positive_holdout_fold_pct = 16.4179%`
  - `holdout_trade_total = 16`
- implication:
  - the active lite frontier is not OOS-dead
  - but it remains too sparse and too inconsistent to count as promotion-ready
  - future rounds should go back to density / conversion work instead of reopening the walk-forward backlog item

### 500-Trades Promotion Memo

- the phase-8 decision artifact now exists:
  - [ICT_PROMOTION_MEMO.md](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/ICT_PROMOTION_MEMO.md)
- verdict:
  - `DO_NOT_PROMOTE_YET`
- why:
  - strict ICT remains too sparse
  - active lite frontier is still only `18 trades`
  - combined-lane fusion is rejected
  - walk-forward evidence is positive but too thin for production promotion
- workflow implication:
  - keep ORB as the only production-qualified lane
  - keep lite reversal as the active ICT research lane
  - do not reopen the promotion backlog item until density gates move materially

### 500-Trades Phase Backlog Hygiene

- stale roadmap rows for completed high-level phases have now been cleaned out of the table:
  - `Phase 0`
  - `Phase 1`
  - `Phase 2`
  - `Phase 3`
  - `Phase 4`
  - `Phase 5`
- those phases were already backed by shipped analyzers and result artifacts
- future rounds should treat the table as live backlog only, not as a second copy of already-finished milestone history

### 500-Trades Retest Backlog Hygiene

- the table had a stale retest-priority row for `fvg_revisit_min_delay_bars`
- that row is now removed because the lite retest round is already complete and the active lite frontier is already promoted on delay `2`
- the table also had stale analyzer inventory rows for already-shipped lite analyzers
- those rows are now removed so later rounds do not confuse shipped tooling with pending work

### 500-Trades Reward:Risk Gate Note

- analyzer-backed RR gate replay is now available through:
  - [analyze_ict_lite_rr_gate.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_rr_gate.py)
- result on the active lite frontier:
  - `lite_frontier_base = 18 trades / +0.3529% / PF 4.6792`
  - `rr_gate_off_control = 18 trades / +0.3529% / PF 4.6792`
  - `tp_rr_1p0_no_gate_control = 18 trades / +0.1819% / PF 3.2134`
  - `tp_rr_1p0_gate_1p5_control = 0 trades / rr_filtered_entries = 108`
- implication:
  - the repo-approved `Reward:Risk >= 1.5:1` rule is now real strategy logic, not a table note
  - the active lite frontier already clears that gate, so RR should not be treated as the next density lever
  - remove RR gating from the backlog and move the roadmap back to sweep-to-shift / retest conversion work

### 500-Trades Engine Sizing Note

- engine-backed sizing replay is now available through:
  - [analyze_ict_position_sizing_compare.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_position_sizing_compare.py)
- sizing support now exists directly in:
  - [engine.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/src/backtest/engine.py)
- result on the active lite frontier:
  - `research_fixed_10 = +0.3816%`
  - `fixed_40_shares = +1.5359%`
  - `capital_50pct_min40 = +6.1746%`
  - `capital_100pct_min40 = +12.6548%`
- implication:
  - T04 is now complete at the engine layer
  - sizing is real infrastructure now, not analyzer-only math
  - future rounds should return to conversion / density work rather than rereading sizing as pending

### 500-Trades Lite Helper Note

- the lite helper backlog item is now explicitly confirmed complete through:
  - [ict_entry_model.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/src/strategies/ict_entry_model.py)
  - [test_ict_profile_builders.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/test_ict_profile_builders.py)
- confirmed helper behavior:
  - disables `prev_session_anchor`
  - disables `external_liquidity`
  - disables `session_array_refinement`
  - disables `macro`
  - disables `AMD`
  - disables `kill_zone`
  - keeps `SMT` opt-in rather than a mandatory pass condition
- implication:
  - `T06` should no longer be treated as an open backlog item
  - future rounds should spend iteration budget on true density / conversion work instead

### 500-Trades Lite Baseline Standardized Replay

- the lite baseline replay is now standardized through:
  - [analyze_ict_lite_reversal_baseline.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_reversal_baseline.py)
- fixed trading spec now used in that replay:
  - `capital_pct`
  - `100% capital usage`
  - `min 40 shares`
  - existing strategy-level `Reward:Risk >= 1.5:1`
- standardized replay result:
  - strict frontier:
    - `8 trades / +7.7249%`
  - lite reversal baseline:
    - `14 trades / +10.6136%`
- implication:
  - `T07` should no longer be treated as an open backlog item
  - this closes the initial lite replay bookkeeping gap without changing the active lite frontier signal stack

### 500-Trades Strict Baseline Summary

- the strict benchmark summary is now standardized through:
  - [analyze_ict_strict_baseline_summary.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_strict_baseline_summary.py)
- standardized strict benchmark block:
  - `8 trades / +7.7249% / PF 0.0`
  - `fvg_entries = 8`
  - `accepted_sweeps = 273`
  - fixed trading spec:
    - `capital_pct`
    - `100% capital usage`
    - `min 40 shares`
    - existing strategy-level `Reward:Risk >= 1.5:1`
- implication:
  - `T01` should no longer be treated as an open backlog item
  - later 500-trades rounds now have a single reusable strict benchmark artifact

### 500-Trades Backlog Hygiene

- stale roadmap rows for completed work have now been cleaned out of the table:
  - `Phase 7`
  - `T00`
  - `T02`
  - `T03`
- evidence for those items already exists in:
  - [run_ict_walk_forward.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/run_ict_walk_forward.py)
  - [analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_displacement_calibration.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_displacement_calibration.py)
  - [analyze_ict_frontier_funnel.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_frontier_funnel.py)
  - funnel metadata in [ict_entry_model.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/src/strategies/ict_entry_model.py)
- future rounds should spend budget on still-open density and promotion work rather than re-reading already-shipped backlog rows as pending

500-trades roadmap note:

- after the continuation proxy was rejected, the active lite frontier funnel now shows:
  - `1229 raw sweeps -> 1036 accepted sweeps -> 57 shift candidates -> 36 armed setups -> 20 retest candidates -> 18 entries`
- the main structural choke point is now `accepted sweeps -> shift candidates`
- the largest explicit recorded filter on the active lite frontier is still `smt_filtered_sweeps = 191`
- before opening any new lane, prioritize sweep-to-shift conversion and residual SMT density on the current lite branch
- the first direct sweep-to-shift calibration confirms:
  - `structure_lookback = 8` can raise shift candidates from `57` to `77` and trades from `18` to `23`
  - but this is density-only, because return falls from `+0.3529%` to `+0.3189%`
  - keep the active lite frontier unchanged and treat shorter structure lookback as a density candidate only
- the next direct SMT recalibration on the active lite frontier confirms:
  - `smt_threshold = 0.0018 / 0.0020` preserves the exact same `18 trades / +0.3529% / PF 4.6792`
  - `smt_threshold = 0.0025` and `SMT off` add density, but only with clear quality degradation
  - keep the active lite frontier unchanged and move the roadmap past SMT as a promotion lever
- the next direct revisit-depth recalibration on the active lite frontier confirms:
  - `depth_0p00` can raise the lane to `20 trades`, but only by weakening return to `+0.2547%`
  - `depth_0p25` stays weaker than the active frontier without adding density
  - keep the active lite frontier unchanged and treat revisit-depth as completed on the backlog
- the next direct displacement-body recalibration on the active lite frontier confirms:
  - `0.15 / 0.20` preserve the exact same `18 trades / +0.3529% / PF 4.6792`
  - `0.00 / 0.05` weaken the lane slightly
  - keep the active lite frontier unchanged and treat displacement-body as completed on the backlog
- repo-approved paired-survivor profile wiring
- repo-approved paired-survivor-plus-session-array profile wiring
- repo-approved paired-survivor-plus-session-array-loose-sweep profile wiring
- repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt profile wiring
- repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt-premium profile wiring
- repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt-premium-ny-only profile wiring
- repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt-premium-ny-only-slow-recovery-short-structure profile wiring
- repo-approved lite-ict-reversal baseline wiring for the 500-trades roadmap
- repo-approved lite-ict-reversal-relaxed-smt frontier wiring for the 500-trades roadmap
- repo-approved lite-ict-reversal-relaxed-smt-looser-sweep frontier wiring for the 500-trades roadmap
- repo-approved lite-ict-reversal-relaxed-smt-looser-sweep-faster-retest frontier wiring for the 500-trades roadmap

Current missing higher-value components:

- current 500-trades roadmap step after the strict-frontier funnel:
  - the first `lite_ict_reversal` baseline now exists and strips the heaviest pre-arm blockers:
    - `premium / discount`
    - `previous-session anchor`
    - `external liquidity`
    - `session-array refinement`
    - `AMD`
    - `macro`
    - `kill zones`
  - it preserves the core path:
    - sweep
    - MSS / CHOCH
    - displacement quality
    - FVG-led revisit
    - `SMT` as the remaining paired guardrail
  - it improves density from `8` to `14` trades, but still misses the first `100`-trade gate
  - the first lite SMT-density pass is now complete:
    - keeping SMT enabled but relaxing `smt_threshold` to `0.0015` improves the lite lane from `14` to `16` trades and from `+0.2832%` to `+0.2902%`
    - turning SMT fully off only acts as a density control:
      - `23 trades / +0.1108% / PF 1.4937`
  - the first lite geometry round is now complete:
    - loosening `liq_sweep_threshold` to `0.0006` improves the relaxed-SMT lite lane from `16` to `18` trades and from `+0.2902%` to `+0.3229%`
    - looser thresholds such as `0.0003 / 0.0005` create more trades but weaken quality too much to become the new frontier
  - the first lite retest round is now complete:
    - reducing `fvg_revisit_min_delay_bars` from `3` to `2` preserves the `18`-trade count and improves the lite lane from `+0.3229% / PF 4.3357` to `+0.3529% / PF 4.6792`
    - `delay_0 / delay_1` are also positive survivors, but weaker than `delay_2`
    - removing revisit depth entirely only acts as a density control:
      - `20 trades / +0.1989% / PF 2.6385`
  - the first lite session-density round is now complete:
    - widening the session window to `ny_14_21` improves activity from `18` to `19` trades, but weakens total return from `+0.3529%` to `+0.3395%`
    - broader session variants and full session-off control cluster into the same weaker density-only family:
      - `19 trades / +0.3327% / PF 4.1125`
  - the first lite setup-density round is now complete:
    - extending `fvg_max_age` to `60` improves activity from `18` to `20` trades, but weakens total return from `+0.3529%` to `+0.3324%`
    - moderate setup windows `30 / 40` also survive only as density controls:
      - `19 trades / +0.3335% / PF 4.2256`
  - the first continuation-lane compare is now complete:
    - a naive `BOS -> FVG retest` continuation proxy massively increases density:
      - `1465 trades`
    - but the quality collapses:
      - `-6.3814% / PF 0.6524`
  - the current continuation proxy is therefore rejected
  - the next roadmap branch should keep the lite reversal frontier active and only revisit continuation if the trigger architecture becomes materially stricter
- controlled context reintroduction beyond premium/discount on top of the short-SMT paired base
- single-filter daily-bias calibration on top of the short-SMT premium base is now complete and rejected
- single-filter kill-zone calibration on top of the short-SMT premium base is now complete and only survives as a secondary optional branch
- single-filter AMD calibration on top of the short-SMT premium base is now complete and only survives as a thin optional branch
- external-liquidity geometry calibration on top of the short-SMT premium base is now complete and shows a plateau, not a new edge
- previous-session-anchor calibration on top of the short-SMT premium base is now complete and also shows a plateau, not a new edge
- premium/discount geometry calibration on top of the short-SMT premium base is now complete and also shows a plateau, not a new edge
- session-array calibration on top of the short-SMT premium base is now complete and confirms that session-array remains a meaningful robust extension; broader window retests do not beat the default
- macro timing recalibration on top of the short-SMT premium-plus-session-array frontier is now complete and confirms that macro remains only a secondary optional branch
- delivery-array composition / scoring calibration on top of the short-SMT premium-plus-session-array frontier is now complete and confirms a plateau:
  - all tested variants preserve the same best result
  - actual realized entries remain `FVG-only`
  - do not prioritize nearby delivery-array score or family retries
- FVG origin-lag calibration on top of the stronger delayed-revisit CE frontier is now complete and confirms a survivor, not a new frontier:
  - `2/3/4/5` preserve the same best result
  - `1` weakens the lane
  - do not prioritize nearby FVG origin-lag retries before a material frontier change
- FVG origin-body calibration on top of the stronger delayed-revisit CE frontier is now complete and confirms a survivor, not a new frontier:
  - correctness repair now makes it evaluate the true displacement candle
  - `0.10/0.20/0.30` preserve the frontier
  - `0.40+` weakens the lane
  - do not prioritize nearby FVG origin-body retries before a material frontier change
- FVG origin range-vs-ATR calibration on top of the stronger delayed-revisit CE frontier is now complete and confirms a survivor, not a new frontier:
  - `0.50/0.75` preserve the exact same best result
  - `1.00` stays positive but thinner
  - `1.25+` weaken the lane further
  - do not prioritize nearby FVG origin range-vs-ATR retries before a material frontier change
- FVG origin body-vs-ATR calibration on top of the stronger delayed-revisit CE frontier is now complete and confirms a survivor, not a new frontier:
  - `0.50/0.75` stay positive but thinner than the active frontier
  - `1.00` stays positive but weaker again
  - `1.10+` thin and weaken the lane further
  - do not prioritize nearby FVG origin body-vs-ATR retries before a material frontier change
- OTE geometry / score calibration on top of the short-SMT premium-plus-session-array frontier is now complete and confirms a plateau:
  - all tested variants preserve the same best result
  - even turning OTE scoring off does not change the frontier
  - do not prioritize nearby OTE geometry or score retries
- FVG geometry / freshness calibration on top of the short-SMT premium-plus-session-array frontier is now complete and confirms a robust survivor but not a new frontier:
  - tighter / looser gap and longer-age variants preserve the same best result
  - shorter-age variants stay positive but weaken the lane
  - do not prioritize nearby FVG geometry or freshness retries as the next frontier
- SMT-threshold geometry calibration on top of the short-SMT premium-plus-session-array frontier is now complete and confirms a robust survivor but not a new frontier:
  - looser thresholds preserve the same best result
  - tighter thresholds stay positive but weaken the lane
  - turning SMT off increases activity but lowers quality
  - do not prioritize nearby SMT-threshold retries as the next frontier
- kill-zone calibration on top of the short-SMT premium-plus-session-array frontier is now complete and confirms a robust survivor but not a new frontier:
  - broader windows preserve the same best result
  - shifted windows stay positive but much thinner
  - default and narrower windows collapse the lane
  - do not prioritize nearby kill-zone retries as the next frontier
- broad session-gating calibration on top of the short-SMT premium-plus-session-array frontier is now complete and confirms the first robust session extension:
  - `ny_only_core` improves the frontier while preserving trade count
  - broader or fully disabled session gating only preserves the old result
  - `london_only_core` is positive but too thin
  - future nearby session-window retries should be judged against the new `NY-only core` base
- macro timing calibration on top of the new `NY-only core` frontier is now complete and confirms it remains secondary:
  - best non-base survivor is `macro_early_shifted`
  - result is `3 trades / +0.0493% / PF Infinity`
  - the `NY-only` base remains better at `5 trades / +0.1420% / PF Infinity`
  - do not prioritize nearby macro-window retries as the next frontier
- sweep-geometry calibration on top of the new `NY-only core` frontier is now complete and confirms the next robust extension:
  - best robust survivor is `sweep_slower_recovery_4`
  - result is `6 trades / +0.1627% / PF Infinity`
  - co-winner `sweep_longer_and_slower` matches the same best result
  - faster-recovery variants weaken the lane
  - future nearby sweep retries should be judged against the new slow-recovery NY-only base
- SMT-lookback calibration on top of the slow-recovery NY-only frontier is now complete and confirms survivor-only behavior:
  - nearby SMT lookback retests preserve the exact same best result
  - turning SMT off increases activity but lowers quality
  - do not prioritize nearby SMT lookback retries as the next frontier
- NY session-boundary calibration on top of the slow-recovery NY-only frontier is now complete and confirms survivor-only behavior:
  - `ny_close_19` and `ny_close_21` preserve the exact same best result
  - shifting the NY open earlier or disabling sessions weakens the lane slightly
  - shifting the NY open later materially thins the lane
  - do not prioritize nearby session-boundary retries as the next frontier
- session-array window calibration on top of the slow-recovery NY-only frontier is now complete and confirms survivor-only behavior:
  - `broader_imbalance`, `broader_structural`, and `dual_broader` preserve the exact same best result
  - `session_array_off_control` and `shifted_later` weaken the lane to `8 trades / +0.1233% / PF 7.2221`
  - do not prioritize nearby session-array window retries as the next frontier
- score / confluence-threshold calibration on top of the slow-recovery NY-only frontier is now complete and confirms survivor-only behavior:
  - `min_score_5`, `min_score_7`, `min_score_5_lower_ote`, and `min_score_7_higher_fvg` preserve the exact same best result
  - `min_score_8` and `min_score_9` thin the lane to `1 trade / +0.0011% / PF Infinity`
  - do not prioritize nearby score-threshold retries as the next frontier
- liquidity-sweep reclaim-strength calibration on top of the slow-recovery NY-only frontier is now complete and confirms survivor-only behavior:
  - `reclaim_0p05` preserves the exact same best result
  - `reclaim_0p10` and `reclaim_0p15` stay positive but thinner
  - `reclaim_0p20` through `reclaim_0p30` weaken the lane further
  - do not prioritize nearby reclaim-strength retries as the next frontier
- displacement-body calibration on top of the stronger structure-aware frontier is now complete and confirms survivor-only behavior:
  - `displacement_body_min_pct = 0.10` and `0.20` preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - `0.30+` progressively weakens the lane
  - this keeps the repo aligned with the PDF's displacement-quality emphasis while confirming that nearby displacement retries are not a new frontier
  - do not prioritize nearby displacement-body retries as the next frontier unless the frontier changes materially
- FVG revisit-depth / consequent-encroachment calibration on top of the stronger structure-aware frontier is now complete and confirms the next robust extension:
  - `fvg_revisit_depth_ratio = 0.5` improves the lane from `7 trades / +0.1831% / PF Infinity` to `7 trades / +0.2042% / PF Infinity`
  - `0.25` stays positive but weaker
  - `0.75` and `1.0` thin and weaken the lane
  - this keeps the repo aligned with the PDF's emphasis on consequent encroachment of the marked gap
  - use `fvg_revisit_depth_ratio = 0.5` as the repo-approved CE geometry on this frontier
  - do not prioritize nearby FVG revisit-depth retries as the next frontier unless the frontier changes materially
- FVG close-back recovery calibration on top of the stronger CE-extended structure-aware frontier is now complete and confirms survivor-only behavior:
  - `fvg_rejection_close_ratio = 0.25` stays positive but weaker than the current frontier
  - `0.50+` progressively thins and weakens the lane
  - this keeps the repo aligned with the PDF's rejection / reaction logic while confirming that nearby close-recovery retries are not a new frontier
  - use `fvg_rejection_close_ratio = 0.0` as the repo-approved base on this frontier
  - do not prioritize nearby FVG close-recovery retries as the next frontier unless the frontier changes materially
- FVG rejection-wick calibration on top of the stronger CE-extended structure-aware frontier is now complete and confirms survivor-only behavior:
  - `fvg_rejection_wick_ratio = 0.10` stays robust but weaker than the current frontier at `7 trades / +0.1930% / PF 152.7443`
  - `0.20+` progressively thins and weakens the lane
  - this keeps the repo aligned with the PDF's revisit -> reaction / rejection emphasis while confirming that nearby wick-quality retries are not a new frontier
  - use `fvg_rejection_wick_ratio = 0.0` as the repo-approved base on this frontier
  - do not prioritize nearby FVG rejection-wick retries as the next frontier unless the frontier changes materially
- FVG rejection-body calibration on top of the stronger CE-extended structure-aware frontier is now complete and confirms survivor-only behavior:
  - `fvg_rejection_body_min_pct = 0.10` stays robust but weaker than the current frontier at `7 trades / +0.1905% / PF Infinity`
  - `0.20+` progressively thins and weakens the lane
  - this keeps the repo aligned with the PDF's revisit -> reaction / displacement emphasis while confirming that nearby reaction-body retries are not a new frontier
  - use `fvg_rejection_body_min_pct = 0.0` as the repo-approved base on this frontier
  - do not prioritize nearby FVG rejection-body retries as the next frontier unless the frontier changes materially
- structure-shift close-buffer calibration on top of the stronger corrected helper with `displacement_body_min_pct = 0.10` is now complete and confirms survivor-only behavior:
  - `structure_shift_close_buffer_ratio = 0.05 / 0.10` preserve the exact same `8 trades / +0.2119% / PF Infinity`
  - `0.15+` weakens the lane
  - this keeps the repo aligned with the PDF's sweep -> close through opposing swing -> displacement logic while confirming that nearby close-buffer retries are not a new frontier
  - use `structure_shift_close_buffer_ratio = 0.0` as the repo-approved base on this stronger corrected helper
  - do not prioritize nearby structure-shift close-buffer retries as the next frontier unless the frontier changes materially
- FVG retest touch-cap calibration on top of the stronger corrected helper with `displacement_body_min_pct = 0.10` is now complete and confirms survivor-only behavior:
  - `fvg_max_retest_touches = 5` preserves the exact same `8 trades / +0.2119% / PF Infinity`
  - tighter caps progressively thin and weaken the lane
  - this keeps the repo aligned with the PDF's revisit / first-touch purity intuition while confirming that nearby touch-cap retries are not a new frontier
  - use `fvg_max_retest_touches = 0` as the repo-approved base on this stronger corrected helper
  - do not prioritize nearby FVG retest touch-cap retries as the next frontier unless the frontier changes materially
- FVG revisit-delay calibration on top of the stronger corrected helper with `displacement_body_min_pct = 0.10` is now complete and reconfirms the existing extension:
  - `fvg_revisit_min_delay_bars = 3` remains the best robust delay at `8 trades / +0.2119% / PF Infinity`
  - `1 / 2` are weaker base-style delays and `4+` progressively weaken the lane
  - this keeps the repo aligned with the PDF's displacement -> revisit -> reaction sequence while confirming that delayed revisit remains part of the active corrected frontier
  - keep `fvg_revisit_min_delay_bars = 3` as the repo-approved revisit timing on this stronger corrected helper
- FVG revisit-delay calibration on top of the stronger CE-extended structure-aware frontier is now complete and confirms the next robust extension:
  - `fvg_revisit_min_delay_bars = 3` improves the lane from `7 trades / +0.2042% / PF Infinity` to `7 trades / +0.2057% / PF Infinity`
  - `2` preserves the old base
  - `4+` progressively weaken the lane
  - this keeps the repo aligned with the PDF's displacement -> revisit -> reaction sequence while confirming that not every revisit should be immediate
  - use `fvg_revisit_min_delay_bars = 3` as the repo-approved revisit timing on this frontier
  - do not prioritize nearby revisit-delay retries as the next frontier unless the frontier changes materially
- FVG retest-touch cap calibration on top of the stronger delayed-revisit CE-extended structure-aware frontier is now complete and confirms survivor-only behavior:
  - `fvg_max_retest_touches = 5` preserves the exact same `7 trades / +0.2057% / PF Infinity`
  - tighter caps progressively thin and weaken the lane
  - this keeps the repo aligned with the PDF's first-touch / stale-imbalance intuition while confirming that nearby touch-cap retries are not a new frontier
  - use `fvg_max_retest_touches = 0` as the repo-approved base on this frontier
  - do not prioritize nearby touch-cap retries as the next frontier unless the frontier changes materially
- displacement range-vs-ATR calibration on top of the stronger delayed-revisit CE-extended structure-aware frontier is now complete and confirms survivor-only behavior:
  - `displacement_range_atr_mult = 0.50` and `0.75` preserve the exact same `7 trades / +0.2057% / PF Infinity`
  - `1.00+` progressively thin and weaken the lane
  - this keeps the repo aligned with the PDF's emphasis on true displacement after the sweep and structure shift while confirming that nearby displacement-range retries are not a new frontier
  - use `displacement_range_atr_mult = 0.0` as the repo-approved base on this frontier
  - do not prioritize nearby displacement-range retries as the next frontier unless the frontier changes materially
- structure-lookback calibration on top of the slow-recovery NY-only frontier is now complete and confirms the next robust extension:
  - `structure_12` and `structure_16` both improve the lane to `7 trades / +0.1826% / PF Infinity`
  - `structure_24` stays positive but weaker
  - `structure_28` and `structure_32` are thinner and weaker
  - use `structure_lookback = 12` as the repo-approved tie-break helper
  - do not prioritize nearby structure-lookback retries as the next frontier unless the frontier changes materially
- liquidity-pool lookback calibration on top of the stronger structure-aware frontier is now complete and confirms survivor-only behavior:
  - `30`, `40`, `60`, `80`, and `100` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - do not prioritize nearby liquidity-pool lookback retries as the next frontier unless the frontier changes materially
- order-block body-quality calibration on top of the stronger structure-aware frontier is now complete and confirms survivor-only behavior:
  - `0.20`, `0.25`, `0.35`, `0.40`, and `0.50` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - `ob_entries` remain `0` across all tested variants
  - do not prioritize nearby order-block body-quality retries as the next frontier unless the frontier changes materially
- order-block lookback calibration on top of the stronger structure-aware frontier is now complete and confirms survivor-only behavior:
  - `8`, `12`, `20`, `24`, and `30` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - `ob_entries` remain `0` across all tested variants
  - do not prioritize nearby order-block lookback retries as the next frontier unless the frontier changes materially
- OTE geometry / score calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - turning OTE off, tightening or loosening the OTE fib band, and moving `score_ote_zone` up or down all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - do not prioritize nearby OTE geometry or score retries as the next frontier unless the frontier changes materially
- breaker-block lookback calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `8`, `12`, `20`, `24`, and `30` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - `breaker_entries` remain `0` across all tested variants
  - do not prioritize nearby breaker-block lookback retries as the next frontier unless the frontier changes materially
- IFVG lookback calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `8`, `12`, `24`, `30`, and `40` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - `ifvg_entries` remain `0` across all tested variants
  - do not prioritize nearby IFVG lookback retries as the next frontier unless the frontier changes materially
- liquidity-sweep threshold calibration on top of the stronger structure-aware frontier is now complete and confirms survivor-only behavior:
  - `0.0006` and `0.0007` preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - tighter thresholds from `0.0009` upward progressively weaken the lane
  - do not prioritize nearby sweep-threshold retries as the next frontier unless the frontier changes materially
- swing-threshold calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `2`, `4`, `5`, and `6` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - do not prioritize nearby swing-threshold retries as the next frontier unless the frontier changes materially
- FVG max-age calibration on top of the stronger structure-aware frontier is now complete and confirms survivor-only behavior:
  - `25`, `30`, and `40` preserve the exact same `7 trades / +0.1826% / PF Infinity`
  - shorter ages `15` and `10` thin and weaken the lane
  - do not prioritize nearby FVG-age retries as the next frontier unless the frontier changes materially
- FVG min-gap calibration on top of the stronger structure-aware frontier is now complete and confirms the next robust extension:
  - `0.0006` improves the lane from `7 trades / +0.1826% / PF Infinity` to `7 trades / +0.1831% / PF Infinity`
  - `0.0008` preserves the old base
  - `0.0012+` weakens the lane
  - use `fvg_min_gap_pct = 0.0006` as the repo-approved FVG geometry on this frontier
  - do not prioritize nearby FVG-gap retries as the next frontier unless the frontier changes materially
- take-profit RR calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `3.0`, `3.5`, `4.5`, `5.0`, and `6.0` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - do not prioritize nearby RR retries as the next frontier unless the frontier changes materially
- stop-loss ATR calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `1.5`, `1.75`, `2.25`, `2.5`, and `3.0` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - do not prioritize nearby stop-loss ATR retries as the next frontier unless the frontier changes materially
- ATR-period calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `10`, `12`, `16`, `18`, and `20` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - do not prioritize nearby ATR-period retries as the next frontier unless the frontier changes materially
- liquidity-sweep lookback calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `30`, `40`, `60`, `70`, and `90` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - do not prioritize nearby liquidity-sweep lookback retries as the next frontier unless the frontier changes materially
- BOS-score calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `1`, `3`, `4`, `5`, and `6` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - do not prioritize nearby BOS-score retries as the next frontier unless the frontier changes materially
- liquidity-sweep score calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `2`, `4`, `5`, `6`, and `7` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - do not prioritize nearby liquidity-sweep score retries as the next frontier unless the frontier changes materially
- FVG-score calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `1`, `3`, `4`, `5`, and `6` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - do not prioritize nearby FVG-score retries as the next frontier unless the frontier changes materially
- CHOCH-score calibration on top of the stronger structure-aware frontier is now complete and confirms plateau behavior:
  - `0`, `1`, `2`, `4`, `5`, and `6` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
  - this round also closes the implementation gap where `SCORE_CHOCH` existed in config but was not wired into strategy scoring
  - do not prioritize nearby CHOCH-score retries as the next frontier unless the frontier changes materially
- keep macro timing as optional unless it materially beats the premium base
- do not prioritize daily-bias-first, kill-zone-first, AMD-first, nearby external-liquidity-geometry retries, nearby previous-session-anchor retries, or nearby premium/discount retries on the premium-enabled short-SMT base unless the paired-data geometry stack changes materially
- do not prioritize nearby delivery-array score or family retries on the premium-enabled short-SMT-plus-session-array base unless the paired-data geometry stack changes materially
- do not prioritize nearby OTE geometry or score retries on the premium-enabled short-SMT-plus-session-array base unless the paired-data geometry stack changes materially
- deeper multi-timeframe narrative refinement beyond the current daily-bias / premium-discount / AMD stack

### Round-Completion Rule

An ICT round is not complete until all of these are updated:

- `SCHEDULED_PROMPT_MULTI_AGENT.md`
- `AGENT_HANDOFF.md`
- `CLAUDE_BRIEFING.md`
- `metrics.json`
- `config.py`
- `iteration_logs/iteration_<N>_*.txt`

If the round only changes ICT research and not ORB production:

- keep the ORB baseline unchanged
- say that explicitly in the state files

### 500-Trades Strict Funnel Lifecycle Instrumentation

- strict frontier metadata now explicitly tracks:
  - `sweep_blocked_by_existing_pending`
  - `sweep_expired_before_shift`
  - `armed_setup_expired_before_retest`
- refreshed strict funnel result:
  - [ict_frontier_funnel.json](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/results/qc_regime_prototypes/ict_frontier_funnel.json)
- current counts:
  - `sweep_expired_before_shift = 263`
  - `sweep_blocked_by_existing_pending = 18`
  - `armed_setup_expired_before_retest = 2`
- workflow implication:
  - the next 500-trades rounds should continue targeting sweep-to-shift conversion and setup lifecycle throughput
  - do not re-open the completed lifecycle-counter backlog items from the implementation table

### 500-Trades Lite FVG-Gap Promotion

- active-lite FVG-gap recalibration now exists through:
  - [analyze_ict_lite_frontier_fvg_gap.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_frontier_fvg_gap.py)
- promoted result:
  - base active-lite frontier: `18 trades / +0.3529% / PF 4.6792`
  - promoted gap setting: `fvg_min_gap_pct = 0.0010`
  - new active-lite frontier: `18 trades / +0.4353% / PF 5.0247`
- workflow implication:
  - the active lite helper should now inherit the tighter `0.0010` gap
  - do not re-open the completed `P1 / fvg_min_gap_pct` row in the implementation table

### 500-Trades Lite Sweep-Threshold Recalibration

- active-lite sweep-threshold recalibration now exists through:
  - [analyze_ict_lite_frontier_sweep_threshold.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_frontier_sweep_threshold.py)
- result on the promoted active-lite frontier:
  - `liq_sweep_threshold = 0.0006 -> 18 trades / +0.4353% / PF 5.0247`
  - `0.0003 -> 38 trades / +0.3231% / PF 1.7261`
  - `0.0004 -> 28 trades / +0.3492% / PF 2.1417`
  - `0.0005 -> 25 trades / +0.2951% / PF 2.0796`
  - `0.0007 -> 17 trades / +0.4498% / PF 6.1051`
- workflow implication:
  - `0.0006` remains the clean promoted active-lite sweep setting because it preserves both density and quality
  - looser sweep thresholds are now explicitly classified as density-only variants
  - do not re-open the completed `P1 / liq_sweep_threshold` row in the implementation table

### 500-Trades Lite Structure-Lookback Recalibration

- active-lite structure-lookback recalibration now exists through:
  - [analyze_ict_lite_shift_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_shift_density.py)
- result on the promoted active-lite frontier:
  - `structure_lookback = 12 -> 18 trades / +0.4353% / PF 5.0247`
  - `structure_8 -> 24 trades / +0.3928% / PF 3.4265`
  - `structure_10 -> 20 trades / +0.4065% / PF 3.8671`
  - `structure_16 -> 17 trades / +0.4498% / PF 6.1051`
- workflow implication:
  - `12` remains the clean promoted active-lite structure setting because it preserves the strongest density/quality balance
  - shorter structure windows are now explicitly classified as density-only variants
  - do not re-open the completed `P1 / structure_lookback` row in the implementation table

### 500-Trades Lite Session-Range Recalibration

- active-lite session-range recalibration now exists through:
  - [analyze_ict_lite_session_density.py](C:/Users/LIN/Desktop/progamming/python/claude/mnq-backtest/analyze_ict_lite_session_density.py)
- result on the promoted active-lite frontier:
  - `NY-only -> 18 trades / +0.4353% / PF 5.0247`
  - `ny_14_21 -> 19 trades / +0.4219% / PF 4.5237`
  - `session_off_control -> 19 trades / +0.4151% / PF 4.4203`
  - `broader_sessions -> 19 trades / +0.4151% / PF 4.4203`
- workflow implication:
  - the current `NY-only` session setting remains the clean promoted active-lite session base
  - broader session variants are now explicitly classified as density-only
  - do not re-open the completed `P3 / session 範圍` row in the implementation table

