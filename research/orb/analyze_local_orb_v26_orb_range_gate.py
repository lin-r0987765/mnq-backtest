#!/usr/bin/env python3
"""
Branch 8: ORB Range Quality Gate  
Hypothesis: Skip entries when the ORB range is below a rolling percentile threshold.

Rationale:
- `quick_failure_loss` is the #1 drag archetype (36% of all drag, -$6822, 45 trades)  
- These trades have mean MFE of only $13 — the breakout never develops
- Narrow ORB ranges may produce false breakouts (noise)
- By requiring a minimum ORB range quality, we skip the noisiest sessions

This is NOT an entry filter (date/regime-based). It is an ORB formation quality gate
based on the structural characteristics of the current session's opening range.

Tested grid:
- min_orb_pct: [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
  (minimum ORB range as % of price to allow entry)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# ─── CONFIG ─────────────────────────────────────────────────────
INTRADAY_CSV = "alpaca/normalized/qqq_5m_alpaca.csv"
DAILY_CSV    = "alpaca/normalized/qqq_1d_alpaca.csv"
BASELINE_REF = "results/qc_regime_prototypes/alpaca_v26_reference_analysis.json"
OUTPUT_FILE  = "results/qc_regime_prototypes/local_orb_v26_orb_range_quality_gate_alpaca.json"

ORB_BARS          = 4          # 20 min ORB
PROFIT_RATIO      = 3.5
TRAILING_PCT      = 0.013
BE_TRIGGER_MULT   = 1.25
BE_ACTIVE_MINUTES = 180
PROFIT_LOCK_TRIGGER_MULT = 1.5
PROFIT_LOCK_LEVEL_MULT   = 0.25
SIZING_PCT        = 0.25
STARTING_EQUITY   = 100_000.0

# Grid: minimum ORB range as % of price
MIN_ORB_PCT_GRID = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]

# ─── LOAD DATA ──────────────────────────────────────────────────
print("[1/5] Loading data...")
df5 = pd.read_csv(INTRADAY_CSV, parse_dates=["Datetime"])
df1d = pd.read_csv(DAILY_CSV, parse_dates=["Datetime"])

df5["timestamp"] = pd.to_datetime(df5["Datetime"], utc=True)
df1d["timestamp"] = pd.to_datetime(df1d["Datetime"], utc=True)

# Normalize column names
for df in [df5, df1d]:
    df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
df5 = df5.sort_values("timestamp").reset_index(drop=True)
df1d = df1d.sort_values("timestamp").reset_index(drop=True)
df1d["date"] = df1d["timestamp"].dt.date

# ─── REGIME FILTER (v26: prev_day_up AND mom3 positive) ────────
print("[2/5] Computing regime filter...")
df1d["prev_close"] = df1d["close"].shift(1)
df1d["prev_day_up"] = df1d["close"].shift(1) > df1d["open"].shift(1)
df1d["sma3"] = df1d["close"].rolling(3).mean()
df1d["mom3_positive"] = df1d["close"].shift(1) > df1d["sma3"].shift(1)
df1d["regime_pass"] = df1d["prev_day_up"] & df1d["mom3_positive"]
regime_dates = set(df1d[df1d["regime_pass"]]["date"].tolist())

# ─── SESSION GROUPING ───────────────────────────────────────────
print("[3/5] Grouping sessions...")
df5["date"] = df5["timestamp"].dt.date

# US market hours: 9:30-16:00 ET = 13:30-20:00 UTC (approx)
df5["hour_utc"] = df5["timestamp"].dt.hour
df5 = df5[(df5["hour_utc"] >= 13) & (df5["hour_utc"] < 21)].copy()

sessions = {}
for date, group in df5.groupby("date"):
    bars = group.sort_values("timestamp").reset_index(drop=True)
    if len(bars) < ORB_BARS + 2:
        continue
    sessions[date] = bars


# ─── SIMULATION ─────────────────────────────────────────────────
def simulate_v26_with_orb_gate(min_orb_pct):
    """
    Simulate v26-profit-lock with an additional ORB range quality gate.
    Skip entries when ORB range / price < min_orb_pct.
    """
    equity = STARTING_EQUITY
    trades = []
    skipped_by_gate = 0
    skipped_by_regime = 0
    
    for date in sorted(sessions.keys()):
        if date not in regime_dates:
            skipped_by_regime += 1
            continue
            
        bars = sessions[date]
        
        # Compute ORB
        orb_bars = bars.iloc[:ORB_BARS]
        orb_high = orb_bars["high"].max()
        orb_low = orb_bars["low"].min()
        orb_range = orb_high - orb_low
        
        if orb_range <= 0:
            continue
        
        mid_price = (orb_high + orb_low) / 2.0
        orb_range_pct = orb_range / mid_price
        
        # ---- ORB RANGE QUALITY GATE ----
        if orb_range_pct < min_orb_pct:
            skipped_by_gate += 1
            continue
        
        # Look for breakout entry
        post_orb = bars.iloc[ORB_BARS:]
        entry_price = None
        entry_idx = None
        
        for i, row in post_orb.iterrows():
            if row["high"] > orb_high:
                entry_price = orb_high
                entry_idx = i
                break
        
        if entry_price is None:
            continue
        
        # Position sizing
        qty = int((equity * SIZING_PCT) / entry_price)
        if qty <= 0:
            continue
        
        # Stops and targets
        stop_price = orb_low
        tp_price = entry_price + PROFIT_RATIO * orb_range
        trail_stop = 0.0
        best_price = entry_price
        be_activated = False
        be_activation_bar = None
        profit_lock_active = False
        profit_lock_stop = 0.0
        
        be_trigger = entry_price + BE_TRIGGER_MULT * orb_range
        profit_lock_trigger = entry_price + PROFIT_LOCK_TRIGGER_MULT * orb_range
        
        exit_price = None
        exit_tag = "EOD"
        bars_in_trade = 0
        mfe = 0.0
        mae = 0.0
        
        remaining = bars.loc[entry_idx + 1:]
        
        for j, bar in remaining.iterrows():
            bars_in_trade += 1
            
            # Update tracking
            if bar["high"] > best_price:
                best_price = bar["high"]
            
            bar_mfe = (bar["high"] - entry_price) * qty
            bar_mae = (bar["low"] - entry_price) * qty
            mfe = max(mfe, bar_mfe)
            mae = min(mae, bar_mae)
            
            # Trail stop
            candidate_trail = best_price * (1 - TRAILING_PCT)
            if candidate_trail > trail_stop:
                trail_stop = candidate_trail
            
            # Breakeven activation
            if not be_activated and bar["high"] >= be_trigger:
                be_activated = True
                be_activation_bar = bars_in_trade
            
            # Breakeven expiry
            if be_activated and not profit_lock_active:
                minutes_since = bars_in_trade * 5
                if be_activation_bar:
                    minutes_since = (bars_in_trade - be_activation_bar) * 5
                if minutes_since >= BE_ACTIVE_MINUTES:
                    be_activated = False
            
            # Profit lock activation
            if not profit_lock_active and bar["high"] >= profit_lock_trigger:
                profit_lock_active = True
                profit_lock_stop = entry_price + PROFIT_LOCK_LEVEL_MULT * orb_range
            
            # Update profit lock stop
            if profit_lock_active:
                candidate = entry_price + PROFIT_LOCK_LEVEL_MULT * orb_range
                if candidate > profit_lock_stop:
                    profit_lock_stop = candidate
            
            # Determine effective stop
            effective_stop = stop_price
            if be_activated:
                effective_stop = max(effective_stop, entry_price)
            if profit_lock_active:
                effective_stop = max(effective_stop, profit_lock_stop)
            effective_stop = max(effective_stop, trail_stop)
            
            # Check exits
            if bar["low"] <= effective_stop:
                exit_price = effective_stop
                if profit_lock_active and effective_stop >= profit_lock_stop:
                    exit_tag = "profit_lock_stop"
                elif be_activated and effective_stop >= entry_price:
                    exit_tag = "be_stop"
                elif effective_stop >= trail_stop and trail_stop > stop_price:
                    exit_tag = "trail_stop"
                else:
                    exit_tag = "orb_stop"
                break
            
            if bar["high"] >= tp_price:
                exit_price = tp_price
                exit_tag = "tp"
                break
        
        if exit_price is None:
            exit_price = remaining.iloc[-1]["close"] if len(remaining) > 0 else entry_price
            exit_tag = "eod_flatten"
        
        pnl = (exit_price - entry_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price
        equity += pnl
        
        trades.append({
            "date": str(date),
            "entry": entry_price,
            "exit": exit_price,
            "qty": qty,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 6),
            "exit_tag": exit_tag,
            "orb_range": round(orb_range, 4),
            "orb_range_pct": round(orb_range_pct, 6),
            "mfe": round(mfe, 2),
            "mae": round(mae, 2),
            "duration_bars": bars_in_trade,
        })
    
    return trades, skipped_by_gate, skipped_by_regime


# ─── RUN GRID ───────────────────────────────────────────────────
print("[4/5] Running grid sweep...")

# First: run baseline (min_orb_pct = 0, no gate)
baseline_trades, _, _ = simulate_v26_with_orb_gate(0.0)
baseline_pnl = sum(t["pnl"] for t in baseline_trades)
baseline_wins = sum(1 for t in baseline_trades if t["pnl"] > 0)
baseline_losses = sum(1 for t in baseline_trades if t["pnl"] <= 0)
baseline_gross_profit = sum(t["pnl"] for t in baseline_trades if t["pnl"] > 0)
baseline_gross_loss = abs(sum(t["pnl"] for t in baseline_trades if t["pnl"] <= 0))
baseline_pf = baseline_gross_profit / baseline_gross_loss if baseline_gross_loss > 0 else 999
baseline_wr = baseline_wins / len(baseline_trades) if baseline_trades else 0

print(f"\n  BASELINE (no gate): {len(baseline_trades)} trades, pnl={baseline_pnl:+.2f}, "
      f"PF={baseline_pf:.4f}, WR={baseline_wr:.1%}")

# Walk-forward folds (same as standard local research)
all_dates = sorted(set(t["date"] for t in baseline_trades))
n = len(all_dates)
fold_size = n // 4
folds = [
    set(all_dates[:fold_size]),
    set(all_dates[fold_size:2*fold_size]),
    set(all_dates[2*fold_size:3*fold_size]),
    set(all_dates[3*fold_size:]),
]

results = []

for min_orb in MIN_ORB_PCT_GRID:
    trades, skipped_gate, skipped_regime = simulate_v26_with_orb_gate(min_orb)
    total_pnl = sum(t["pnl"] for t in trades)
    delta = total_pnl - baseline_pnl
    
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    wr = wins / len(trades) if trades else 0
    
    # Walk-forward analysis
    fold_deltas = []
    for fold_dates in folds:
        fold_bl_pnl = sum(t["pnl"] for t in baseline_trades if t["date"] in fold_dates)
        fold_var_pnl = sum(t["pnl"] for t in trades if t["date"] in fold_dates)
        fold_deltas.append(fold_var_pnl - fold_bl_pnl)
    
    improved_folds = sum(1 for d in fold_deltas if d > 0)
    
    # Annual analysis
    year_results = {}
    for t in trades:
        yr = t["date"][:4]
        if yr not in year_results:
            year_results[yr] = 0
        year_results[yr] += t["pnl"]
    
    bl_year_results = {}
    for t in baseline_trades:
        yr = t["date"][:4]
        if yr not in bl_year_results:
            bl_year_results[yr] = 0
        bl_year_results[yr] += t["pnl"]
    
    improved_years = sum(1 for yr in year_results 
                        if year_results.get(yr, 0) > bl_year_results.get(yr, 0))
    total_years = len(set(list(year_results.keys()) + list(bl_year_results.keys())))
    
    # Trades skipped analysis
    skipped_trades = len(baseline_trades) - len(trades)
    skipped_pnl = baseline_pnl - total_pnl + delta  # net effect
    
    # What ORB ranges were skipped?
    skipped_orb_ranges = [t["orb_range_pct"] for t in baseline_trades 
                          if t["orb_range_pct"] < min_orb]
    skipped_trade_pnls = [t["pnl"] for t in baseline_trades 
                          if t["orb_range_pct"] < min_orb]
    skipped_losses_avoided = sum(1 for p in skipped_trade_pnls if p <= 0)
    skipped_wins_lost = sum(1 for p in skipped_trade_pnls if p > 0)
    
    result = {
        "variant": f"orb_gate_{min_orb:.3f}",
        "min_orb_pct": min_orb,
        "trades": len(trades),
        "baseline_trades": len(baseline_trades),
        "skipped_trades": skipped_trades,
        "skipped_by_gate": skipped_gate,
        "total_pnl": round(total_pnl, 2),
        "baseline_pnl": round(baseline_pnl, 2),
        "delta": round(delta, 2),
        "profit_factor": round(pf, 4),
        "baseline_pf": round(baseline_pf, 4),
        "win_rate": round(wr, 4),
        "baseline_wr": round(baseline_wr, 4),
        "improved_folds": f"{improved_folds}/4",
        "fold_deltas": [round(d, 2) for d in fold_deltas],
        "improved_years": f"{improved_years}/{total_years}",
        "year_pnl": {yr: round(v, 2) for yr, v in sorted(year_results.items())},
        "baseline_year_pnl": {yr: round(v, 2) for yr, v in sorted(bl_year_results.items())},
        "skipped_losses_avoided": skipped_losses_avoided,
        "skipped_wins_lost": skipped_wins_lost,
    }
    results.append(result)
    
    marker = "✓" if delta > 0 and improved_folds >= 3 else ""
    print(f"  {result['variant']:20s}  trades={len(trades):3d}  "
          f"delta={delta:>+8.2f}  PF={pf:.4f}  WR={wr:.1%}  "
          f"folds={improved_folds}/4  yrs={improved_years}/{total_years}  "
          f"skip={skipped_trades}(L={skipped_losses_avoided}/W={skipped_wins_lost}) {marker}")

# ─── CHOOSE BEST ────────────────────────────────────────────────
print("\n[5/5] Selecting best variant...")

# Sort by: improved_folds desc, then delta desc
results.sort(key=lambda r: (int(r["improved_folds"].split("/")[0]), r["delta"]), reverse=True)
best = results[0]

verdict = "ALPACA_LOCAL_REJECTED"
if best["delta"] > 0 and int(best["improved_folds"].split("/")[0]) >= 3:
    verdict = "READY_FOR_QC_PROXY"
elif best["delta"] > 0 and int(best["improved_folds"].split("/")[0]) >= 2:
    verdict = "LOCAL_NEAR_MISS"

output = {
    "research_scope": "v26_orb_range_quality_gate",
    "hypothesis": "Skip entries when ORB range / price < threshold (narrow ORB = noise breakout)",
    "axis": "entry_quality_gate (NOT regime filter)",
    "target_archetype": "quick_failure_loss (36% of all drag, -$6822)",
    "baseline_version": "v26-profit-lock",
    "data_source": "alpaca_normalized",
    "data_span": "2020-07-27 -> 2026-04-07",
    "baseline_trades": len(baseline_trades),
    "baseline_pnl": round(baseline_pnl, 2),
    "baseline_pf": round(baseline_pf, 4),
    "grid_tested": len(results),
    "best_variant": best["variant"],
    "best_delta": best["delta"],
    "best_improved_folds": best["improved_folds"],
    "best_improved_years": best["improved_years"],
    "verdict": verdict,
    "all_variants": results,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Best: {best['variant']} | delta={best['delta']:+.2f} | "
      f"folds={best['improved_folds']} | verdict={verdict}")
print(f"  Saved to: {OUTPUT_FILE}")
