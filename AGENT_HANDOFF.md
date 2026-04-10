# AGENT_HANDOFF.md

Current scheduled-agent handoff for the dual-track workflow:

- ORB production baseline remains the official QuantConnect lane.
- ICT is now the revolutionary development lane and must be tracked as a separate strategy architecture.

---

## Current State

| Field | Value |
|------|------|
| Status | `OFFICIAL_BASELINE_V26_PROFIT_LOCK_AND_ICT_DENSITY_BRANCH_DEPTH_REFINED` |
| Agent ID | `scheduled_20260410_320` |
| Updated At | `2026-04-09 UTC+8` |
| Mode | `RESEARCH` |
| Outcome | `v26-profit-lock` remains the official accepted ORB baseline. The strict ICT frontier still benchmarks at `8 trades / +0.2119% / PF Infinity`, the active lite quality frontier remains `18 trades / +0.4353% / PF 5.0247`, the qualified reversal balance branch remains at `40 trades / +13.4177% / PF 3.1879`, and the qualified continuation density branch is now refined to `93 trades / +11.9309% / PF 1.8204` under standardized capital-based replay. This round improves the high-density lane again without promoting ICT over ORB. |

### Round 320 Delta

- Tightened the qualified continuation density branch's revisit depth from `0.5` to `0.35`
- Replayed the branch and improved it from `93 trades / +11.8319% / PF 1.8131` to `93 trades / +11.9309% / PF 1.8204`
- Kept `18 trades / +0.4353% / PF 5.0247` as the clean quality lane and `40 trades / +13.4177% / PF 3.1879` as the middle-density reversal lane

---

## Recent Confirmed Strategy State

### ICT development lane
- Primary source document:
  - `C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\ict\ICT 交易策略詳解與教學.pdf`
- Workflow:
  - `ICT_DEVELOPMENT_WORKFLOW.md`
- Current implementation file:
  - `src/strategies/ict_entry_model.py`
  - `src/data/fetcher.py`
  - `run_backtest.py`
- Included now:
  - liquidity sweep detection
  - MSS confirmation
  - FVG-first entry
  - OB fallback
  - breaker fallback
  - IFVG fallback
  - external-liquidity gating
  - SMT divergence gating
  - AMD / market-maker path gating
  - macro timing window gating
  - previous-session anchor gating
  - session-specific dealing-array refinement
  - explicit CHOCH scoring on top of the post-sweep structure shift
  - explicit displacement body-quality gating on top of the post-sweep structure shift
  - explicit FVG consequent-encroachment / revisit-depth gating on top of the FVG-led entry trigger
  - explicit FVG close-back recovery gating on top of the FVG-led entry trigger
  - explicit FVG rejection-wick gating on top of the FVG-led entry trigger
  - explicit FVG rejection-body gating on top of the FVG-led entry trigger
  - explicit structure-shift close-through buffer gating on top of the post-sweep MSS / CHOCH confirmation
  - explicit FVG revisit-delay gating on top of the FVG-led retest trigger
  - explicit FVG retest-touch cap gating on top of the FVG-led retest trigger
  - explicit FVG origin-lag gating on top of the structure-shift -> FVG handoff
  - explicit FVG origin-body gating on top of the displacement -> FVG creation layer
  - explicit FVG origin body-vs-ATR gating on top of the displacement -> FVG creation layer
  - explicit FVG origin close-position gating on top of the displacement -> FVG creation layer
  - explicit FVG origin opposite-wick gating on top of the displacement -> FVG creation layer
  - explicit FVG origin range-vs-ATR gating on top of the displacement -> FVG creation layer
  - FVG detector now prefers the most recent valid gap in the shift window
  - structure-confirmation scoring now treats BOS / CHOCH as a single event instead of double-counting both
  - explicit displacement range-vs-ATR gating on top of the post-sweep structure-shift confirmation
  - New York trading-day boundary handling for EOD flatten logic
  - strict-frontier funnel stage instrumentation for sweep -> accepted sweep -> shift candidate -> armed setup -> retest candidate -> entry
  - peer-symbol data integration for real SMT backtests
  - OTE scoring
  - broad session gating
  - optional kill-zone specialization
  - conservative higher-timeframe daily bias
  - conservative premium / discount context
  - repo-approved paired-survivor-plus-session-array profile wiring
  - repo-approved paired-survivor-plus-session-array-loose-sweep profile wiring
  - repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt profile wiring
  - repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt-premium profile wiring
  - repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt-premium-ny-only profile wiring
  - repo-approved paired-survivor-plus-session-array-loose-sweep-short-smt-premium-ny-only-slow-recovery-short-structure profile wiring
- Newly implemented this round:
  - prev-session-anchor gating now supports `hard` vs `soft` filter modes plus an optional shift-score penalty
  - added the roadmap-aligned strict soft-prev-session helper, analyzer, and regression test
  - replayed the strict no-SMT context lane:
    - `strict_prev_anchor_hard_base = 11 trades / +0.1727% / PF 3.9924`
    - `prev_anchor_soft_penalty_0 = 11 trades / +0.1727% / PF 3.9924`
    - `prev_anchor_soft_penalty_2 = 11 trades / +0.1727% / PF 3.9924`
    - `prev_anchor_soft_penalty_3 = 11 trades / +0.1727% / PF 3.9924`
    - `prev_anchor_filter_off_control = 11 trades / +0.1727% / PF 3.9924`
  - interpretation:
    - softening prev-session-anchor mismatches changes diagnostics and accepted sweeps
    - but it does not change realized trades or expectancy on the strict lane
    - treat this axis as completed and plateau / survivor-only, not promoted
  - synced the repo state forward to iteration 300
- Previously implemented:
  - `USE_PREV_SESSION_ANCHOR_FILTER`
  - `PREV_SESSION_ANCHOR_TOLERANCE`
  - `USE_MACRO_TIMING_WINDOWS`
  - `MACRO_TIMEZONE`
  - `MACRO_WINDOWS`
  - `USE_AMD_FILTER`
  - `AMD_ACCUMULATION_BARS`
  - `AMD_MANIPULATION_THRESHOLD`
  - `AMD_REQUIRE_MIDPOINT_RECLAIM`
  - `USE_SMT_FILTER`
  - `SMT_LOOKBACK`
  - `SMT_THRESHOLD`
  - `USE_EXTERNAL_LIQUIDITY_FILTER`
  - `EXTERNAL_LIQUIDITY_LOOKBACK`
  - `EXTERNAL_LIQUIDITY_TOLERANCE`
  - `IFVG_LOOKBACK`
  - `SCORE_IFVG`
  - `BREAKER_LOOKBACK`
  - `SCORE_BREAKER_BLOCK`
  - `USE_PREMIUM_DISCOUNT_FILTER`
  - `PREMIUM_DISCOUNT_LOOKBACK`
  - `PREMIUM_DISCOUNT_NEUTRAL_BAND`
  - `USE_DAILY_BIAS_FILTER`
  - `DAILY_BIAS_LOOKBACK`
  - `DAILY_BIAS_BULL_THRESHOLD`
  - `DAILY_BIAS_BEAR_THRESHOLD`
  - `USE_KILL_ZONES`
  - `KILL_ZONE_TIMEZONE`
  - `LONDON_KILL_START / END`
  - `NY_AM_KILL_START / END`
  - `NY_PM_KILL_START / END`
- Current missing higher-order ICT features:
  - controlled context reintroduction beyond premium/discount on top of the short-SMT paired base
  - macro timing remains a secondary optional branch, not the main frontier
  - daily bias should no longer be treated as the next heavier-context candidate on this base
  - deeper multi-timeframe narrative refinement beyond the current daily-bias / premium-discount / AMD / previous-session-anchor / session-array stack
- Newly implemented this round:
  - `analyze_ict_survivor_pairwise_calibration.py`
  - `build_ict_paired_survivor_plus_session_array_params(...)`
- Tests:
  - `test_ict_strategy.py`
  - now includes kill-zone, daily-bias, premium/discount, breaker, IFVG, external-liquidity, SMT, AMD, macro timing, previous-session anchor, session-array refinement, and strict peer-data SMT coverage
  - `test_data_pipeline.py`
  - now includes strict peer-column merge coverage
- First real smoke test:
  - local input: `qqq_5m.csv`
  - peer input: `SPY` via yfinance
  - profile: `build_ict_research_profile_params(enable_smt=True)`
  - result: `0 trades`
  - interpretation: the full ICT stack is now wired correctly, but currently too restrictive on the short local sample
- Paired activation frontier:
  - file: `results/qc_regime_prototypes/ict_paired_activation_frontier.json`
  - broader Alpaca paired lane:
    - primary: `alpaca/normalized/qqq_5m_alpaca.csv`
    - peer: `alpaca/normalized/spy_5m_alpaca.csv`
  - `full_stack_smt`: `0 trades`
  - `context_relaxed_bundle`: `7 trades`, `+0.1088%`, `profit_factor 1.9155`
  - `minimal_structure_default`: `10 trades`, `+0.0209%`, `profit_factor 1.1029`
  - `minimal_structure_very_loose`: `96 trades`, `+0.0476%`, `profit_factor 1.0408`
  - `minimal_structure_loose_sweep`: `32 trades`, `-0.1432%`
  - interpretation:
    - the ICT lane is not structurally dead on broader paired data
    - the full stack is still too restrictive
    - a context-relaxed paired profile is now active and positive
    - the next calibration step should reintroduce context filters one cluster at a time
- Controlled context-filter reintroduction:
  - file: `results/qc_regime_prototypes/ict_context_reintroduction.json`
  - robust survivors:
    - `reintro_prev_session_anchor`: `7 trades`, `+0.1088%`, `profit_factor 1.9155`
    - `reintro_external_liquidity`: `7 trades`, `+0.1088%`, `profit_factor 1.9155`
  - thinner but still positive:
    - `reintro_premium_discount`: `5 trades`, `+0.0444%`, `profit_factor 1.9873`
    - `reintro_session_array`: `4 trades`, `+0.0503%`, `profit_factor 1.9895`
  - sparse survivors only:
    - `reintro_kill_zones`: `1 trade`, `+0.1130%`
    - `reintro_macro_timing`: `1 trade`, `+0.0382%`
  - collapse cases:
    - `reintro_amd`: `0 trades`
    - `reintro_context_core`: `0 trades`
    - `reintro_timing_bundle`: `0 trades`
  - interpretation:
    - the next paired-data calibration should start from the survivor base, not from the fully relaxed bundle
    - the first survivor base is:
      - `previous-session anchor`
      - `external liquidity`
    - next step:
      - preserve the survivor base:
        - `previous-session anchor`
        - `external liquidity`
      - preserve the first robust pairwise extension:
        - `session-array refinement`
      - preserve the first robust geometry extension:
        - `liq_sweep_threshold = 0.0008`
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
      - treat nearby external-liquidity geometry tweaks as plateaued:
        - no tested lookback / tolerance variant beat the current base
      - explicitly treat `daily bias` as over-restrictive on the premium-enabled short-SMT base:
        - all calibrated variants collapsed to `0 trades`
      - treat nearby previous-session-anchor on/off and tolerance tweaks as plateaued:
        - every tested variant preserved the same `5 trades / +0.1375% / PF Infinity`
        - `anchor_off_control` matched the premium base exactly
      - treat nearby premium/discount lookback and neutral-band tweaks as plateaued:
        - every robust tested variant preserved the same `5 trades / +0.1375% / PF Infinity`
        - `premium_off_control` matched the premium base exactly
      - keep session-array refinement in the frontier:
        - turning it off weakens the lane
        - broader window retests can survive but do not beat the default
      - keep macro timing in the secondary bucket:
        - `macro_early_shifted` survives
        - but it remains materially weaker than the current frontier
      - only after that, revisit heavier context reintroduction one filter at a time

### Official accepted baseline: `v26-profit-lock`
- Latest accepted rerun: `Square Blue Termite`
- Trades: `405`
- Orders: `810`
- Net Profit: `5.924%`
- Sharpe: `-3.192`
- Win Rate: `56.0%`
- Drawdown: `1.100%`
- Rolling 6m positive Sharpe: `86.0%`
- Rolling 12m positive Sharpe: `96.8%`
- Positive years: `9`

### Latest analyzed research bundle: `Geeky Fluorescent Yellow Alligator`
- Version: `v26-orb-trail-qc-evaluator`
- Classification: `healthy analyzer-only real backtest, rejected / no launch`
- Trades: `405`
- Orders: `810`
- Net Profit: `5.36%`
- Win Rate: `56.5%`
- Drawdown: `1.40%`
- Comparison vs official baseline:
  - Alpaca local ORB-range trail signal did not survive on QC
  - net profit worsened materially
  - drawdown worsened
  - verdict: do not launch `v27`

### ORB re-entry QC postmortem
- File:
  - `results/qc_regime_prototypes/qc_orb_reentry_postmortem.json`
- Analyzer:
  - `analyze_qc_orb_reentry_postmortem.py`
- Key findings:
  - `orb_reentry_exit_count = 12`
  - ORB re-entry exit bucket realized pnl: `-$757.67`
  - ORB re-entry exit bucket was negative in every active year it appeared
  - yearly realized pnl:
    - `2018: -121.66`
    - `2019: -122.36`
    - `2020: -39.56`
    - `2021: -55.26`
    - `2022: -113.28`
    - `2023: -121.39`
    - `2024: -89.35`
    - `2025: -51.18`
    - `2026: -43.63`
- Interpretation:
  - the branch improved aggregate headline pnl, but not through obviously positive realized ORB re-entry exits
  - without accepted baseline raw trades in workspace, this remains a mixed analyzer-only signal rather than a launch-ready candidate

### Alpaca-backed local reference lane
- Normalized files:
  - `alpaca/normalized/qqq_5m_alpaca.csv`
  - `alpaca/normalized/qqq_1d_alpaca.csv`
- Actual coverage:
  - `2020-07-27 -> 2026-04-07`
  - `110817` five-minute bars
- Baseline reference:
  - `results/qc_regime_prototypes/alpaca_v26_reference_analysis.json`
  - `1044 trades`
  - total pnl `+111.0817`
  - profit factor `1.1039`
  - positive years `3/7`
  - time folds `2 positive / 2 negative`

### Alpaca structural ORB re-entry local branch
- File:
  - `results/qc_regime_prototypes/local_orb_v26_orb_reentry_exit_alpaca.json`
- Best variant:
  - `orb_reentry_after_1.000x_depth_0.25x_confirm_1bar`
- Local result:
  - pnl delta `+16.6986`
  - profit factor `1.1214`
  - improved time folds `4/4`
  - improved years `7/7`
  - clipped winners `0`
  - saved losses `23`
- Local verdict:
  - still the strongest post-`v26` local branch
  - but QC proxy has now been analyzed and remains mixed

### Profit-lock-gated ORB re-entry refinement
- File:
  - `results/qc_regime_prototypes/local_orb_v26_orb_reentry_profitlock_gated_alpaca.json`
- Idea:
  - only allow the structural ORB re-entry exit after persistent profit lock is already active
  - optionally wait `0/1/2` extra bars after profit-lock activation
- Result:
  - every tested variant produced `0` ORB re-entry exits on Alpaca
  - every tested variant had `pnl delta = +0.0000`
  - verdict: `ALPACA_LOCAL_REJECTED`
- Interpretation:
  - the structural ORB re-entry branch appears to derive all of its effect from pre-profit-lock behavior
  - profit-lock gating does not rescue the mixed QC signal
  - do not build a Blueshift evaluator or another QC proxy for this refinement

### ORB-range trailing stop (READY_FOR_QC_PROXY)
- File:
  - `results/qc_regime_prototypes/local_orb_v26_orb_trail_alpaca.json`
- Script:
  - `analyze_local_orb_v26_orb_trail.py`
- Idea:
  - replace the fixed 1.3% trailing stop with ORB-range-denominated trail
  - normalises the one inconsistent stop component in the v26 architecture
  - `trail_stop = best_price - K * orb_range` instead of `best_price * (1 - 0.013)`
- Result:
  - best variant: `orb_trail_2.0x`
  - pnl delta: `+8.7417`
  - improved time folds: `3/4`
  - improved years: `5/7`
  - saved losses: `48`
  - clipped winners: `15`
  - profit factor: `1.1119` (baseline `1.1039`)
  - win rate: `50.1%` (baseline `49.2%`)
  - verdict: `QC_REJECTED`
- QC evaluator result (`Geeky Fluorescent Yellow Alligator`):
  - net profit: 5.36% vs baseline 6.27% ??worse by 0.91%
  - drawdown: 1.40% vs baseline 1.10% ??worse
  - win rate: 56.5% vs 55% ??marginal
  - PF: 1.3195
  - 405 trades, version-matched, no reentry issues
  - Alpaca local signal did not survive on 10-year QC bar data
  - this confirms the pattern seen with mid-trade ratchet: local improvements vanish on QC

### ORB hold ratchet equivalence diagnosis
- File:
  - `results/qc_regime_prototypes/local_orb_v26_orb_hold_ratchet_alpaca.json`
- Best variant:
  - `orb_hold_ratchet_after_1.000x_floorOffset_0.25x_after_0bar`
- Local result:
  - pnl delta `+16.6986`
  - improved time folds `4/4`
  - improved years `7/7`
  - clipped winners `0`
  - saved losses `23`
  - hold-ratchet exits `32`
- Equivalence finding:
  - strongest ORB hold ratchet result matches the prior structural ORB re-entry branch on:
    - `pnl_delta`
    - `improved_time_folds`
    - `improved_years`
    - `saved_losses`
    - `clipped_winners`
- Verdict:
  - `ALPACA_EQUIVALENT_TO_PRIOR_ORB_REENTRY`
- Interpretation:
  - do not treat ORB hold ratchet as a genuinely new post-`v26` mechanism
  - do not build a Blueshift evaluator for this branch
  - do not open another QC proxy from this branch

### Structural ORB pullback-reclaim entry
- File:
  - `results/qc_regime_prototypes/local_orb_v26_pullback_reclaim_entry_alpaca.json`
- Script:
  - `analyze_local_orb_v26_pullback_reclaim_entry.py`
- Mechanism:
  - no immediate breakout entry
  - wait for breakout progress
  - wait for pullback back toward the ORB boundary
  - require reclaim confirmation before entering
- Result:
  - verdict: `ALPACA_LOCAL_REJECTED`
  - best least-bad variant:
    - `pullback_reclaim_after_0.500x_depth_0.10x_confirm_2bar`
  - pnl delta: `-115.96`
  - improved time folds: `2/4`
  - improved years: `4/7`
  - common entry timestamps vs baseline were effectively `0` across variants
- Interpretation:
  - this branch genuinely changed the ORB entry architecture instead of layering another nearby exit patch
  - even with that structural change, it still failed materially on the longer Alpaca lane
  - this strengthens the conclusion that `v26-profit-lock` is near the structural optimum of the current ORB framework
  - do not launch or QC-proxy this branch without radically stronger new evidence

### Structural failed-breakout reversal entry
- File:
  - `results/qc_regime_prototypes/local_orb_v26_failed_breakout_reversal_alpaca.json`
- Script:
  - `analyze_local_orb_v26_failed_breakout_reversal.py`
- Mechanism:
  - no breakout-continuation entry
  - wait for an initial breakout beyond the ORB boundary
  - require the breakout to fail back inside the ORB
  - confirm the failure for `1/2` bars
  - then enter in the opposite direction
- Result:
  - verdict: `ALPACA_LOCAL_REJECTED`
  - best least-bad variant:
    - `failed_reversal_after_0.750x_depth_0.10x_confirm_2bar`
  - pnl delta: `-156.77`
  - improved time folds: `1/4`
  - improved years: `4/7`
  - common trades: `0`
  - failed breakout reversal entries: `123`
- Interpretation:
  - this branch is materially different from breakout continuation, ORB re-entry, ORB hold-ratchet reformulations, and pullback-reclaim continuation entries
  - even with that structural reversal architecture, it still failed materially on the longer Alpaca lane
  - this further strengthens the conclusion that `v26-profit-lock` is near the structural optimum of the current ORB framework
  - do not launch or QC-proxy this branch without radically stronger new evidence

### Opening regime classifier
- File:
  - `results/qc_regime_prototypes/local_orb_v26_opening_regime_classifier_alpaca.json`
- Script:
  - `analyze_local_orb_v26_opening_regime_classifier.py`
- Mechanism:
  - do not enter on the first breakout
  - classify the first breakout as either continuation or failure inside a short decision window
  - if continuation confirms, enter with the breakout
  - if failure confirms, enter in the opposite direction
  - if neither confirms in time, skip the session
- Result:
  - verdict: `ALPACA_LOCAL_NEAR_MISS`
  - best variant:
    - `opening_classifier_cont_0.250x_fail_0.10x_confirm_1bar_window_3bar`
  - pnl delta: `+4.0736`
  - improved time folds: `2/4`
  - improved years: `4/7`
  - common trades: `400`
  - continuation entries: `975`
  - reversal entries: `3`
- Interpretation:
  - this is the first structurally different post-v26 branch to stay positive on the Alpaca lane
  - but it is still not launch-ready
  - the best variant behaves mostly like a delayed continuation filter, not a truly balanced continuation / reversal classifier
  - do not QC-proxy or launch this branch yet without stronger evidence that reversal participation is real

### Blueshift research lane
- Workflow file:
  - `BLUESHIFT_RESEARCH_WORKFLOW.md`
- Shared runtime:
  - `blueshift/blueshift_library/orb_v26_runtime.py`
- Baseline entry:
  - `blueshift/v26_profit_lock_blueshift.py`
- Research evaluator entry:
  - `blueshift/v26_orb_reentry_evaluator_blueshift.py`
- Purpose:
  - use Blueshift for multi-file research prototypes and evaluator scaffolds
  - keep QuantConnect as the sole 10-year promotion authority

---

## Closed Post-v26 Branches

Do not relaunch from these lines without materially stronger new evidence:
- `results/qc_regime_prototypes/local_orb_v26_stagnation_exit.json`
- `results/qc_regime_prototypes/local_orb_v26_stall_giveback_exit.json`
- `results/qc_regime_prototypes/local_orb_v26_fast_failure_abort.json`
- `results/qc_regime_prototypes/local_orb_v26_low_progress_timeout.json`
- `results/qc_regime_prototypes/local_orb_v26_stagnation_exit_alpaca.json`
- `results/qc_regime_prototypes/local_orb_v26_stall_giveback_exit_alpaca.json`
- `results/qc_regime_prototypes/local_orb_v26_fast_failure_abort_alpaca.json`
- `results/qc_regime_prototypes/local_orb_v26_low_progress_timeout_alpaca.json`
- `results/qc_regime_prototypes/alpaca_v26_exit_recheck_summary.json`
- `results/qc_regime_prototypes/v26_weakness_map.json`
- `results/qc_regime_prototypes/local_orb_v26_mid_trade_ratchet.json`
- `lean/QQQ_ORB_DeepBacktest/QQQ_V26_MidRatchet_ProxyAnalyzer_WebIDE.py`
- `results/qc_regime_prototypes/local_orb_v26_orb_reentry_profitlock_gated_alpaca.json`
- `results/qc_regime_prototypes/local_orb_v26_orb_hold_ratchet_alpaca.json`
- `results/qc_regime_prototypes/local_orb_v26_pullback_reclaim_entry_alpaca.json`
- `results/qc_regime_prototypes/local_orb_v26_failed_breakout_reversal_alpaca.json`
- `results/qc_regime_prototypes/local_orb_v26_adaptive_trail_alpaca.json`
- `analyze_local_orb_v26_adaptive_trail.py`
- `results/qc_regime_prototypes/local_orb_v26_orb_trail_alpaca.json`
- `analyze_local_orb_v26_orb_trail.py`
- `lean/QQQ_ORB_DeepBacktest/QQQ_V26_OrbTrail_ProxyAnalyzer_WebIDE.py`
- `QuantConnect results/2017-2026/Geeky Fluorescent Yellow Alligator.json` (QC_REJECTED)
- `results/qc_regime_prototypes/local_orb_v26_orb_range_quality_gate_alpaca.json` (rejected)
- `analyze_local_orb_v26_orb_range_gate.py`
- `results/qc_regime_prototypes/local_orb_v26_risk_normalized_sizing_alpaca.json` (leverage artifact)
- `analyze_local_orb_v26_risk_normalized_sizing.py`

Historical hard closes still apply:
- `v20-tight-trail-early`
- `v23-wide-trail`
- `v24-no-trail`

---

## Strategic Takeaway

Current state:
- official baseline is `v26-profit-lock`
- repo QC files should stay on baseline
- future ideas must beat `v26`, not `v25`
- there is no active candidate
- there is no pending QC evaluator rerun
- Alpaca normalized data is still the preferred local research lane when a script supports custom CSV input
- Blueshift is now the preferred platform when a research branch needs a larger multi-file scaffold
- QC promotion remains the sole 10-year authority

Important interpretation:
- structural ORB re-entry cleared the strongest local bar after `v26`
- but the first real QC proxy came back mixed, not clean
- the branch should not be relaunched blindly
- the first branch-specific refinement also failed cleanly
- the ORB hold ratchet reformulation also collapses onto the same branch
- even the structurally different ORB pullback / retest / reclaim entry branch failed materially on Alpaca
- even the structurally different failed-breakout reversal architecture also failed materially on Alpaca
- the opening regime classifier is the first structurally different branch to stay positive, but it currently collapses mostly into delayed continuation
- the next step should avoid nearby breakout / fakeout redesigns unless they materially increase true reversal participation
- adaptive post-profit-lock trail tightening also failed cleanly: dynamically narrowing the trailing stop clips confirmed winners too aggressively
- ORB-range trailing stop was the last viable-looking post-v26 branch
- it has now been closed by QC rejection
- nine post-v26 mechanisms across exit, entry-quality, and sizing axes are now exhausted
- the current ORB framework appears to be at or near the structural optimum under the v26 baseline, and even structural entry-architecture changes now look weak

---

## Workspace Notes

- Latest workspace bundle: `Geeky Fluorescent Yellow Alligator`
- Latest accepted baseline rerun still remains:
  - `Square Blue Termite`
- normalized Alpaca research files are available:
  - `alpaca/normalized/qqq_5m_alpaca.csv`
  - `alpaca/normalized/qqq_1d_alpaca.csv`
- accepted baseline raw trades are not currently present in workspace
- workspace trades currently available:
  - `Geeky Fluorescent Yellow Alligator_trades.csv`
  - `Retrospective Red Orange Whale_trades.csv`

---

## Next Step

If a newer QC bundle appears:
1. analyze it first
2. reject stale / wrong-version / analyzer-only bundles as usual
3. only promote if it clearly beats `v26-profit-lock` on both headline and stability metrics

If no newer QC bundle appears:
1. keep repo QC files on `v26-profit-lock`
2. use normalized Alpaca files as the preferred local research lane whenever the script supports custom CSV input
3. do not launch `v27` from `Retrospective Red Orange Whale`
4. do not launch `v27` from `Geeky Fluorescent Yellow Alligator`
5. if the next research branch is large or multi-file, prototype it in Blueshift first:
   - `blueshift/v26_profit_lock_blueshift.py`
   - `blueshift/v26_orb_reentry_evaluator_blueshift.py`
   - `blueshift/blueshift_library/orb_v26_runtime.py`
6. use the ORB re-entry postmortem:
   - `analyze_qc_orb_reentry_postmortem.py`
   - `results/qc_regime_prototypes/qc_orb_reentry_postmortem.json`
7. do not reopen the ORB re-entry branch with simple profit-lock gating
8. do not relaunch ORB hold ratchet as a separate branch when its strongest result is equivalent to prior ORB re-entry
9. do not reopen ORB-range trailing stop without materially stronger cross-platform evidence
10. do not launch from `results/qc_regime_prototypes/local_orb_v26_pullback_reclaim_entry_alpaca.json`
11. do not launch from `results/qc_regime_prototypes/local_orb_v26_failed_breakout_reversal_alpaca.json`
12. treat `results/qc_regime_prototypes/local_orb_v26_opening_regime_classifier_alpaca.json` as a local near miss, not a launch-ready candidate
13. do not QC-proxy the opening classifier until reversal participation is materially real
14. keep the ORB baseline untouched while the ICT lane is in paired-data calibration mode
15. use `build_ict_paired_survivor_profile_params(...)` as the new paired-data starting point rather than resetting to the fully relaxed bundle
16. use `build_ict_paired_survivor_plus_session_array_params(...)` as the current best robust extension profile instead of resetting to the fully relaxed bundle
17. do not treat `premium / discount + session-array` as the default extension yet; it is positive but thinner than the robust extension
18. do not re-add `AMD` yet; it still collapses the paired lane to `0 trades`
19. keep the slow-recovery `NY-only` frontier unchanged after the session-array calibration:
   - `broader_imbalance`, `broader_structural`, and `dual_broader` are survivor-only
   - `session_array_off_control` and `shifted_later` are weaker
20. do not prioritize nearby session-array window retries as the next frontier
21. keep the slow-recovery `NY-only` frontier unchanged after the score calibration:
   - `min_score_5`, `min_score_7`, `min_score_5_lower_ote`, and `min_score_7_higher_fvg` are survivor-only
   - `min_score_8` and `min_score_9` are thinner and weaker
22. do not prioritize nearby score-threshold retries as the next frontier
23. keep the slow-recovery `NY-only` frontier unchanged after the reclaim calibration:
   - `reclaim_0p05` is survivor-only
   - `reclaim_0p10` through `reclaim_0p30` are progressively thinner and weaker
24. do not prioritize nearby reclaim-strength retries as the next frontier
25. promote the stronger structure-aware frontier over the older slow-recovery base:
   - `structure_12` and `structure_16` are co-winners at `7 trades / +0.1826% / PF Infinity`
   - use `structure_lookback = 12` as the repo-approved tie-break helper
26. do not prioritize nearby structure-lookback retries as the next frontier unless the broader frontier changes materially
27. keep nearby liquidity-pool lookback retries on the stronger structure-aware frontier classified as survivor-only:
   - `30`, `40`, `60`, `80`, and `100` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
28. do not prioritize nearby liquidity-pool lookback retries as the next frontier
29. keep nearby order-block body-quality retries on the stronger structure-aware frontier classified as survivor-only:
   - `0.20`, `0.25`, `0.35`, `0.40`, and `0.50` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
   - `ob_entries` remain `0` across all variants
30. do not prioritize nearby order-block body-quality retries as the next frontier
31. keep nearby order-block lookback retries on the stronger structure-aware frontier classified as survivor-only:
   - `8`, `12`, `20`, `24`, and `30` all preserve the exact same `7 trades / +0.1826% / PF Infinity`
   - `ob_entries` remain `0` across all variants
32. do not prioritize nearby order-block lookback retries as the next frontier
33. promote the stronger structure-aware frontier again after the new FVG-gap calibration:
   - `fvg_gap_0p0006` improves the lane from `7 trades / +0.1826% / PF Infinity` to `7 trades / +0.1831% / PF Infinity`
   - `fvg_gap_0p0008` only preserves the old base
   - `0.0012`, `0.0015`, and `0.0020` weaken the lane
34. do not prioritize nearby FVG-gap retries as the next frontier unless the broader frontier changes materially
35. keep nearby ATR-period retries on the stronger structure-aware frontier classified as plateaued:
   - `10`, `12`, `16`, `18`, and `20` all preserve the exact same `7 trades / +0.1831% / PF Infinity`
36. do not prioritize nearby ATR-period retries as the next frontier unless the broader frontier changes materially




