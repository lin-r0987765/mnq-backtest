"""Phase 1: Diagnostic analysis of v18 exit characteristics.

Analyzes the 405 QC trades from Alert Asparagus Panda to understand:
1. Exit type distribution (stop vs TP vs EOD flatten)
2. Return distribution per exit type
3. Holding time distribution
4. Win/loss characteristics by exit type
5. Identifies exit inefficiencies and prioritizes which exit variants to test
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from research.orb.analyze_v18_calmness_bridge_family import load_qc_trades
from daily_session_alignment import NEW_YORK_TZ


def classify_exit(row: pd.Series) -> str:
    """Classify exit type from the trade's exit tag or derived signals."""
    # Try to classify from exit price vs entry price and other signals
    # QC trades typically have Direction and PnL
    # We'll derive from the trade data what we can
    
    # Check columns available
    entry_price = float(row.get("Entry Price", 0))
    exit_price = float(row.get("Exit Price", 0))
    direction = row.get("Direction", "")
    net_pnl = float(row.get("net_pnl", 0))
    quantity = abs(float(row.get("Quantity", 0)))
    
    # For now, we'll classify based on return magnitude patterns
    # In a real analysis, we'd use trade tags from the algorithm
    if quantity == 0 or entry_price == 0:
        return "unknown"
    
    if "Long" in str(direction):
        return_pct = (exit_price - entry_price) / entry_price
    else:
        return_pct = (entry_price - exit_price) / entry_price
    
    return return_pct


def analyze_exit_diagnostics(result_dir: Path) -> dict:
    """Full diagnostic analysis of v18 exit characteristics."""
    bundle, trades = load_qc_trades(result_dir)
    
    # Basic info
    n_trades = len(trades)
    
    # Parse entry/exit times
    trades["entry_time"] = pd.to_datetime(trades["Entry Time"], utc=True).dt.tz_convert(NEW_YORK_TZ)
    trades["exit_time"] = pd.to_datetime(trades["Exit Time"], utc=True).dt.tz_convert(NEW_YORK_TZ)
    trades["entry_price"] = pd.to_numeric(trades["Entry Price"], errors="coerce")
    trades["exit_price"] = pd.to_numeric(trades["Exit Price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["Quantity"], errors="coerce").abs()
    
    # Compute return per trade
    is_long = trades["Direction"].str.contains("Long", case=False, na=False)
    trades["return_pct"] = np.where(
        is_long,
        (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"] * 100.0,
        (trades["entry_price"] - trades["exit_price"]) / trades["entry_price"] * 100.0,
    )
    
    # Holding time in minutes
    trades["holding_minutes"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 60.0
    
    # Exit time of day
    trades["exit_hour"] = trades["exit_time"].dt.hour
    trades["exit_minute"] = trades["exit_time"].dt.minute
    trades["exit_time_hhmm"] = trades["exit_hour"] * 100 + trades["exit_minute"]
    
    # Classify exit type heuristically:
    # - EOD Flatten: exit within 15 min of 16:00 (3:45 PM - 4:00 PM ET)
    # - Take Profit: large positive return (likely hit TP)  
    # - Stop Loss: negative return near trailing stop level
    def classify(row):
        exit_h = row["exit_hour"]
        exit_m = row["exit_minute"]
        ret = row["return_pct"]
        holding = row["holding_minutes"]
        
        # EOD flatten: exit between 15:45-16:00
        if exit_h == 15 and exit_m >= 45:
            return "eod_flatten"
        if exit_h >= 16:
            return "eod_flatten"
        
        # Very short holding + negative return = stopped out quickly
        # Positive return = stopped out with gain or TP
        if ret > 0:
            return "winner_exit"
        else:
            return "loser_exit"
    
    trades["exit_class"] = trades.apply(classify, axis=1)
    
    # Refine winner classification: separate TP from trailing stop exits
    # TP would show returns around specific thresholds related to PROFIT_RATIO
    # We can't determine exactly, but very large returns are likely TP
    
    # PnL statistics by exit class
    exit_stats = {}
    for cls in trades["exit_class"].unique():
        subset = trades[trades["exit_class"] == cls]
        pnl_values = subset["net_pnl"].astype(float)
        ret_values = subset["return_pct"]
        hold_values = subset["holding_minutes"]
        
        exit_stats[cls] = {
            "count": int(len(subset)),
            "total_pnl": round(float(pnl_values.sum()), 2),
            "avg_pnl": round(float(pnl_values.mean()), 2),
            "median_pnl": round(float(pnl_values.median()), 2),
            "avg_return_pct": round(float(ret_values.mean()), 4),
            "median_return_pct": round(float(ret_values.median()), 4),
            "avg_holding_min": round(float(hold_values.mean()), 1),
            "median_holding_min": round(float(hold_values.median()), 1),
            "win_rate": round(float((pnl_values > 0).mean() * 100), 1),
        }
    
    # Return distribution (percentiles)
    all_returns = trades["return_pct"]
    return_distribution = {
        "p5": round(float(all_returns.quantile(0.05)), 4),
        "p10": round(float(all_returns.quantile(0.10)), 4),
        "p25": round(float(all_returns.quantile(0.25)), 4),
        "p50": round(float(all_returns.quantile(0.50)), 4),
        "p75": round(float(all_returns.quantile(0.75)), 4),
        "p90": round(float(all_returns.quantile(0.90)), 4),
        "p95": round(float(all_returns.quantile(0.95)), 4),
        "mean": round(float(all_returns.mean()), 4),
        "std": round(float(all_returns.std()), 4),
        "min": round(float(all_returns.min()), 4),
        "max": round(float(all_returns.max()), 4),
    }
    
    # Holding time distribution
    all_holding = trades["holding_minutes"]
    holding_distribution = {
        "p5": round(float(all_holding.quantile(0.05)), 1),
        "p25": round(float(all_holding.quantile(0.25)), 1),
        "p50": round(float(all_holding.quantile(0.50)), 1),
        "p75": round(float(all_holding.quantile(0.75)), 1),
        "p95": round(float(all_holding.quantile(0.95)), 1),
        "mean": round(float(all_holding.mean()), 1),
        "max": round(float(all_holding.max()), 1),
    }
    
    # Win/loss analysis
    winners = trades[trades["net_pnl"].astype(float) > 0]
    losers = trades[trades["net_pnl"].astype(float) <= 0]
    
    win_loss = {
        "total_trades": int(n_trades),
        "winners": int(len(winners)),
        "losers": int(len(losers)),
        "win_rate_pct": round(float(len(winners) / n_trades * 100), 1),
        "avg_winner_pnl": round(float(winners["net_pnl"].astype(float).mean()), 2) if len(winners) > 0 else 0,
        "avg_loser_pnl": round(float(losers["net_pnl"].astype(float).mean()), 2) if len(losers) > 0 else 0,
        "avg_winner_return_pct": round(float(winners["return_pct"].mean()), 4) if len(winners) > 0 else 0,
        "avg_loser_return_pct": round(float(losers["return_pct"].mean()), 4) if len(losers) > 0 else 0,
        "avg_winner_holding_min": round(float(winners["holding_minutes"].mean()), 1) if len(winners) > 0 else 0,
        "avg_loser_holding_min": round(float(losers["holding_minutes"].mean()), 1) if len(losers) > 0 else 0,
        "total_winner_pnl": round(float(winners["net_pnl"].astype(float).sum()), 2) if len(winners) > 0 else 0,
        "total_loser_pnl": round(float(losers["net_pnl"].astype(float).sum()), 2) if len(losers) > 0 else 0,
        "profit_factor": round(
            float(abs(winners["net_pnl"].astype(float).sum() / losers["net_pnl"].astype(float).sum())), 3
        ) if len(losers) > 0 and losers["net_pnl"].astype(float).sum() != 0 else float("inf"),
    }
    
    # Exit time distribution (when do trades exit during the day?)
    exit_hour_counts = trades["exit_hour"].value_counts().sort_index().to_dict()
    exit_hour_pnl = trades.groupby("exit_hour")["net_pnl"].apply(
        lambda x: round(float(x.astype(float).sum()), 2)
    ).to_dict()
    
    # Return buckets: how many trades fall in each return range
    bins = [-10, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
    labels = [f"{bins[i]:.1f}_to_{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    trades["return_bucket"] = pd.cut(trades["return_pct"], bins=bins, labels=labels, include_lowest=True)
    return_bucket_counts = trades["return_bucket"].value_counts().sort_index().to_dict()
    return_bucket_counts = {str(k): int(v) for k, v in return_bucket_counts.items()}
    
    # Direction breakdown
    longs = trades[is_long]
    shorts = trades[~is_long]
    direction_stats = {
        "long_trades": int(len(longs)),
        "short_trades": int(len(shorts)),
        "long_total_pnl": round(float(longs["net_pnl"].astype(float).sum()), 2) if len(longs) > 0 else 0,
        "short_total_pnl": round(float(shorts["net_pnl"].astype(float).sum()), 2) if len(shorts) > 0 else 0,
        "long_win_rate": round(float((longs["net_pnl"].astype(float) > 0).mean() * 100), 1) if len(longs) > 0 else 0,
        "short_win_rate": round(float((shorts["net_pnl"].astype(float) > 0).mean() * 100), 1) if len(shorts) > 0 else 0,
    }
    
    # Key exit inefficiency indicators
    # 1. How many winners were "small winners" (< 0.3% return) — might be exited too early
    small_winners = winners[winners["return_pct"] < 0.3]
    # 2. How many losers were originally profitable (exit at loss after being up) — "round-trippers"
    # Can't determine without path data, but large negative returns on long holds suggest this
    long_hold_losers = losers[losers["holding_minutes"] > 120]  # losing trades held > 2 hours
    
    inefficiency_indicators = {
        "small_winners_lt_0p3pct": int(len(small_winners)),
        "small_winners_pct_of_all_winners": round(float(len(small_winners) / len(winners) * 100), 1) if len(winners) > 0 else 0,
        "long_hold_losers_gt_2h": int(len(long_hold_losers)),
        "long_hold_losers_avg_pnl": round(float(long_hold_losers["net_pnl"].astype(float).mean()), 2) if len(long_hold_losers) > 0 else 0,
        "eod_flatten_count": int((trades["exit_class"] == "eod_flatten").sum()),
        "eod_flatten_pct": round(float((trades["exit_class"] == "eod_flatten").mean() * 100), 1),
    }
    
    result = {
        "research_scope": "v18_exit_diagnostic",
        "source_bundle": bundle.json_path.stem,
        "total_trades": int(n_trades),
        "exit_class_stats": exit_stats,
        "win_loss_analysis": win_loss,
        "return_distribution": return_distribution,
        "holding_time_distribution": holding_distribution,
        "return_bucket_counts": return_bucket_counts,
        "exit_hour_counts": {str(k): int(v) for k, v in exit_hour_counts.items()},
        "exit_hour_pnl": {str(k): v for k, v in exit_hour_pnl.items()},
        "direction_stats": direction_stats,
        "inefficiency_indicators": inefficiency_indicators,
        "interpretation": "",
    }
    
    # Generate interpretation
    interp_parts = []
    interp_parts.append(f"Total {n_trades} trades analyzed.")
    interp_parts.append(f"Win rate: {win_loss['win_rate_pct']}%.")
    interp_parts.append(f"Profit factor: {win_loss['profit_factor']}.")
    interp_parts.append(f"Average winner: ${win_loss['avg_winner_pnl']:.2f} ({win_loss['avg_winner_return_pct']:.3f}%).")
    interp_parts.append(f"Average loser: ${win_loss['avg_loser_pnl']:.2f} ({win_loss['avg_loser_return_pct']:.3f}%).")
    interp_parts.append(f"EOD flatten rate: {inefficiency_indicators['eod_flatten_pct']:.1f}%.")
    interp_parts.append(f"Small winners (<0.3%): {inefficiency_indicators['small_winners_lt_0p3pct']} ({inefficiency_indicators['small_winners_pct_of_all_winners']:.1f}% of winners).")
    interp_parts.append(f"Long-hold losers (>2h): {inefficiency_indicators['long_hold_losers_gt_2h']}.")
    
    result["interpretation"] = " ".join(interp_parts)
    
    return result


def main():
    result_dir = Path("QuantConnect results/2017-2026")
    result = analyze_exit_diagnostics(result_dir)
    
    output_path = Path("results/qc_regime_prototypes/v18_exit_diagnostic.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Print key findings
    print("=== V18 EXIT DIAGNOSTIC ===")
    print(f"Total trades: {result['total_trades']}")
    print()
    
    print("--- Win/Loss Analysis ---")
    wl = result["win_loss_analysis"]
    print(f"  Winners: {wl['winners']} ({wl['win_rate_pct']}%)")
    print(f"  Losers:  {wl['losers']}")
    print(f"  Profit Factor: {wl['profit_factor']}")
    print(f"  Avg winner: ${wl['avg_winner_pnl']} ({wl['avg_winner_return_pct']}%)")
    print(f"  Avg loser:  ${wl['avg_loser_pnl']} ({wl['avg_loser_return_pct']}%)")
    print(f"  Total winners PnL: ${wl['total_winner_pnl']}")
    print(f"  Total losers PnL:  ${wl['total_loser_pnl']}")
    print()
    
    print("--- Exit Class Stats ---")
    for cls, stats in result["exit_class_stats"].items():
        print(f"  {cls}: {stats['count']} trades, total ${stats['total_pnl']}, avg ${stats['avg_pnl']}, win_rate {stats['win_rate']}%")
    print()
    
    print("--- Return Distribution ---")
    rd = result["return_distribution"]
    print(f"  p5={rd['p5']}% p25={rd['p25']}% p50={rd['p50']}% p75={rd['p75']}% p95={rd['p95']}%")
    print(f"  mean={rd['mean']}% std={rd['std']}% min={rd['min']}% max={rd['max']}%")
    print()
    
    print("--- Holding Time ---")
    ht = result["holding_time_distribution"]
    print(f"  p25={ht['p25']}min p50={ht['p50']}min p75={ht['p75']}min p95={ht['p95']}min max={ht['max']}min")
    print()
    
    print("--- Inefficiency Indicators ---")
    ii = result["inefficiency_indicators"]
    print(f"  Small winners (<0.3%): {ii['small_winners_lt_0p3pct']} ({ii['small_winners_pct_of_all_winners']}% of winners)")
    print(f"  Long-hold losers (>2h): {ii['long_hold_losers_gt_2h']}, avg PnL ${ii['long_hold_losers_avg_pnl']}")
    print(f"  EOD flatten: {ii['eod_flatten_count']} ({ii['eod_flatten_pct']}%)")
    print()
    
    print("--- Direction ---")
    ds = result["direction_stats"]
    print(f"  Longs: {ds['long_trades']}, PnL ${ds['long_total_pnl']}, win rate {ds['long_win_rate']}%")
    print(f"  Shorts: {ds['short_trades']}, PnL ${ds['short_total_pnl']}, win rate {ds['short_win_rate']}%")
    print()
    
    print("--- Return Buckets ---")
    for bucket, count in sorted(result["return_bucket_counts"].items()):
        print(f"  {bucket}: {count}")
    print()
    
    print("--- Exit Hour PnL ---")
    for hour, pnl in sorted(result["exit_hour_pnl"].items()):
        count = result["exit_hour_counts"].get(hour, 0)
        print(f"  Hour {hour}: {count} trades, ${pnl}")


if __name__ == "__main__":
    main()
