#!/usr/bin/env python3
"""
Branch 8b: ORB-Range-Normalized Position Sizing
Hypothesis: Size positions inversely proportional to ORB range to keep per-trade
            dollar risk roughly constant.

Rationale:
- Current v26: fixed 25% of equity per trade regardless of ORB width
- Wide ORB → large stop distance → large dollar loss on stop-out  
- Narrow ORB → small stop distance → small dollar loss but also small gains
- By sizing = target_risk_dollars / (ORB_range * K), we normalize the dollar risk

This targets ALL loss archetypes simultaneously by reducing outsized losses
on wide-ORB days (the "hard_stop_loss" and "late_washout" categories have
the largest per-trade losses).

Also tested: day-of-week analysis (diagnostic only).

Grid:
- target_risk_pct: [0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
  (max % of equity to risk per trade, which sizes position by ORB range)
- max_alloc_pct: 0.35 (cap max allocation to avoid over-concentration)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# ─── CONFIG ─────────────────────────────────────────────────────
INTRADAY_CSV = "alpaca/normalized/qqq_5m_alpaca.csv"
DAILY_CSV    = "alpaca/normalized/qqq_1d_alpaca.csv"
OUTPUT_FILE  = "results/qc_regime_prototypes/local_orb_v26_risk_normalized_sizing_alpaca.json"

ORB_BARS          = 4
PROFIT_RATIO      = 3.5
TRAILING_PCT      = 0.013
BE_TRIGGER_MULT   = 1.25
BE_ACTIVE_MINUTES = 180
PROFIT_LOCK_TRIGGER_MULT = 1.5
PROFIT_LOCK_LEVEL_MULT   = 0.25
BASELINE_SIZING_PCT = 0.25
STARTING_EQUITY   = 100_000.0

TARGET_RISK_PCT_GRID = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
MAX_ALLOC_PCT = 0.35  # cap

# ─── LOAD DATA ──────────────────────────────────────────────────
print("[1/5] Loading data...")
df5 = pd.read_csv(INTRADAY_CSV, parse_dates=["Datetime"])
df1d = pd.read_csv(DAILY_CSV, parse_dates=["Datetime"])

df5["timestamp"] = pd.to_datetime(df5["Datetime"], utc=True)
df1d["timestamp"] = pd.to_datetime(df1d["Datetime"], utc=True)

for df in [df5, df1d]:
    df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)

df5 = df5.sort_values("timestamp").reset_index(drop=True)
df1d = df1d.sort_values("timestamp").reset_index(drop=True)
df1d["date"] = df1d["timestamp"].dt.date

# ─── REGIME FILTER ──────────────────────────────────────────────
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
df5["hour_utc"] = df5["timestamp"].dt.hour
df5 = df5[(df5["hour_utc"] >= 13) & (df5["hour_utc"] < 21)].copy()

sessions = {}
for date, group in df5.groupby("date"):
    bars = group.sort_values("timestamp").reset_index(drop=True)
    if len(bars) < ORB_BARS + 2:
        continue
    sessions[date] = bars


# ─── SIMULATION ─────────────────────────────────────────────────
def simulate_v26(sizing_mode="fixed", target_risk_pct=None):
    """
    sizing_mode:
      "fixed" = baseline 25% allocation
      "risk_normalized" = size based on ORB range to target_risk_pct of equity
    """
    equity = STARTING_EQUITY
    trades = []
    
    for date in sorted(sessions.keys()):
        if date not in regime_dates:
            continue
            
        bars = sessions[date]
        orb_bars_data = bars.iloc[:ORB_BARS]
        orb_high = orb_bars_data["high"].max()
        orb_low = orb_bars_data["low"].min()
        orb_range = orb_high - orb_low
        
        if orb_range <= 0:
            continue
        
        # Look for breakout
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
        
        # ---- POSITION SIZING ----
        if sizing_mode == "fixed":
            qty = int((equity * BASELINE_SIZING_PCT) / entry_price)
        elif sizing_mode == "risk_normalized":
            # Stop distance = entry - orb_low = orb_range (roughly)
            stop_distance = orb_range
            # target_risk in dollars
            risk_dollars = equity * target_risk_pct
            # qty such that qty * stop_distance = risk_dollars
            qty = int(risk_dollars / stop_distance)
            # Cap at max allocation
            max_qty = int((equity * MAX_ALLOC_PCT) / entry_price)
            qty = min(qty, max_qty)
        
        if qty <= 0:
            continue
        
        alloc_pct = (qty * entry_price) / equity
        
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
            
            if bar["high"] > best_price:
                best_price = bar["high"]
            
            bar_mfe = (bar["high"] - entry_price) * qty
            bar_mae = (bar["low"] - entry_price) * qty
            mfe = max(mfe, bar_mfe)
            mae = min(mae, bar_mae)
            
            candidate_trail = best_price * (1 - TRAILING_PCT)
            if candidate_trail > trail_stop:
                trail_stop = candidate_trail
            
            if not be_activated and bar["high"] >= be_trigger:
                be_activated = True
                be_activation_bar = bars_in_trade
            
            if be_activated and not profit_lock_active:
                if be_activation_bar:
                    minutes_since = (bars_in_trade - be_activation_bar) * 5
                    if minutes_since >= BE_ACTIVE_MINUTES:
                        be_activated = False
            
            if not profit_lock_active and bar["high"] >= profit_lock_trigger:
                profit_lock_active = True
                profit_lock_stop = entry_price + PROFIT_LOCK_LEVEL_MULT * orb_range
            
            if profit_lock_active:
                candidate = entry_price + PROFIT_LOCK_LEVEL_MULT * orb_range
                if candidate > profit_lock_stop:
                    profit_lock_stop = candidate
            
            effective_stop = stop_price
            if be_activated:
                effective_stop = max(effective_stop, entry_price)
            if profit_lock_active:
                effective_stop = max(effective_stop, profit_lock_stop)
            effective_stop = max(effective_stop, trail_stop)
            
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
        equity += pnl
        
        import datetime as dt_mod
        if isinstance(date, dt_mod.date):
            dow = date.weekday()  # 0=Mon, 4=Fri
        else:
            dow = -1
        
        trades.append({
            "date": str(date),
            "entry": entry_price,
            "exit": exit_price,
            "qty": qty,
            "pnl": round(pnl, 2),
            "exit_tag": exit_tag,
            "orb_range": round(orb_range, 4),
            "orb_range_pct": round(orb_range / entry_price, 6),
            "alloc_pct": round(alloc_pct, 4),
            "mfe": round(mfe, 2),
            "mae": round(mae, 2),
            "duration_bars": bars_in_trade,
            "dow": dow,
        })
    
    return trades


# ─── RUN ────────────────────────────────────────────────────────
print("[4/5] Running grid sweep...")

# Baseline
baseline_trades = simulate_v26("fixed")
baseline_pnl = sum(t["pnl"] for t in baseline_trades)
bl_gp = sum(t["pnl"] for t in baseline_trades if t["pnl"] > 0)
bl_gl = abs(sum(t["pnl"] for t in baseline_trades if t["pnl"] <= 0))
baseline_pf = bl_gp / bl_gl if bl_gl > 0 else 999
baseline_wr = sum(1 for t in baseline_trades if t["pnl"] > 0) / len(baseline_trades)

print(f"\n  BASELINE: {len(baseline_trades)} trades, pnl={baseline_pnl:+.2f}, PF={baseline_pf:.4f}, WR={baseline_wr:.1%}")

# Day-of-week diagnostic
print("\n  --- DAY-OF-WEEK DIAGNOSTIC ---")
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
for d in range(5):
    dt = [t for t in baseline_trades if t["dow"] == d]
    if dt:
        dpnl = sum(t["pnl"] for t in dt)
        dwins = sum(1 for t in dt if t["pnl"] > 0)
        dwr = dwins / len(dt)
        print(f"  {dow_names[d]}: {len(dt):3d} trades, pnl={dpnl:>+10.2f}, WR={dwr:.1%}")

# Walk-forward folds
all_dates = sorted(set(t["date"] for t in baseline_trades))
n = len(all_dates)
fold_size = n // 4
folds = [
    set(all_dates[:fold_size]),
    set(all_dates[fold_size:2*fold_size]),
    set(all_dates[2*fold_size:3*fold_size]),
    set(all_dates[3*fold_size:]),
]

# Grid sweep
results = []
for trp in TARGET_RISK_PCT_GRID:
    trades = simulate_v26("risk_normalized", trp)
    total_pnl = sum(t["pnl"] for t in trades)
    delta = total_pnl - baseline_pnl
    
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gp / gl if gl > 0 else 999
    wr = wins / len(trades) if trades else 0
    
    # Allocation stats
    allocs = [t["alloc_pct"] for t in trades]
    avg_alloc = np.mean(allocs) if allocs else 0
    min_alloc = min(allocs) if allocs else 0
    max_alloc = max(allocs) if allocs else 0
    
    # Walk-forward
    fold_deltas = []
    for fold_dates in folds:
        fb = sum(t["pnl"] for t in baseline_trades if t["date"] in fold_dates)
        fv = sum(t["pnl"] for t in trades if t["date"] in fold_dates)
        fold_deltas.append(fv - fb)
    improved_folds = sum(1 for d in fold_deltas if d > 0)
    
    # Annual
    year_pnl = {}
    for t in trades:
        yr = t["date"][:4]
        year_pnl[yr] = year_pnl.get(yr, 0) + t["pnl"]
    bl_year_pnl = {}
    for t in baseline_trades:
        yr = t["date"][:4]
        bl_year_pnl[yr] = bl_year_pnl.get(yr, 0) + t["pnl"]
    improved_years = sum(1 for yr in year_pnl if year_pnl.get(yr, 0) > bl_year_pnl.get(yr, 0))
    total_years = len(set(list(year_pnl.keys()) + list(bl_year_pnl.keys())))
    
    result = {
        "variant": f"risk_norm_{trp:.3f}",
        "target_risk_pct": trp,
        "trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "delta": round(delta, 2),
        "profit_factor": round(pf, 4),
        "win_rate": round(wr, 4),
        "improved_folds": f"{improved_folds}/4",
        "fold_deltas": [round(d, 2) for d in fold_deltas],
        "improved_years": f"{improved_years}/{total_years}",
        "avg_alloc_pct": round(avg_alloc, 4),
        "min_alloc_pct": round(min_alloc, 4),
        "max_alloc_pct": round(max_alloc, 4),
        "year_pnl": {yr: round(v, 2) for yr, v in sorted(year_pnl.items())},
    }
    results.append(result)
    
    marker = "*PASS*" if delta > 0 and improved_folds >= 3 else ""
    print(f"  {result['variant']:20s}  trades={len(trades):3d}  "
          f"delta={delta:>+10.2f}  PF={pf:.4f}  WR={wr:.1%}  "
          f"folds={improved_folds}/4  yrs={improved_years}/{total_years}  "
          f"alloc=[{min_alloc:.1%}-{avg_alloc:.1%}-{max_alloc:.1%}] {marker}")

# ─── VERDICT ────────────────────────────────────────────────────
print("\n[5/5] Selecting best variant...")
results.sort(key=lambda r: (int(r["improved_folds"].split("/")[0]), r["delta"]), reverse=True)
best = results[0]

verdict = "ALPACA_LOCAL_REJECTED"
if best["delta"] > 0 and int(best["improved_folds"].split("/")[0]) >= 3:
    verdict = "READY_FOR_QC_PROXY"
elif best["delta"] > 0 and int(best["improved_folds"].split("/")[0]) >= 2:
    verdict = "LOCAL_NEAR_MISS"

output = {
    "research_scope": "v26_risk_normalized_sizing",
    "hypothesis": "Size positions inversely proportional to ORB range for constant per-trade dollar risk",
    "axis": "position_sizing (genuinely new axis - NOT exit modification)",
    "baseline_version": "v26-profit-lock",
    "data_source": "alpaca_normalized",
    "baseline_trades": len(baseline_trades),
    "baseline_pnl": round(baseline_pnl, 2),
    "baseline_pf": round(baseline_pf, 4),
    "grid_tested": len(results),
    "best_variant": best["variant"],
    "best_delta": best["delta"],
    "best_improved_folds": best["improved_folds"],
    "verdict": verdict,
    "dow_diagnostic": {dow_names[d]: {
        "trades": len([t for t in baseline_trades if t["dow"] == d]),
        "pnl": round(sum(t["pnl"] for t in baseline_trades if t["dow"] == d), 2),
        "win_rate": round(sum(1 for t in baseline_trades if t["dow"] == d and t["pnl"] > 0) / max(1, len([t for t in baseline_trades if t["dow"] == d])), 4),
    } for d in range(5)},
    "all_variants": results,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Best: {best['variant']} | delta={best['delta']:+.2f} | folds={best['improved_folds']} | verdict={verdict}")
print(f"  Saved to: {OUTPUT_FILE}")
