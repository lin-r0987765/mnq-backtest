#!/usr/bin/env python3
"""
Conditional breakeven analysis on QC trades.

This script explores whether the breakeven thesis works better for specific
subsets of trades, which could lead to a more targeted (conditional) trigger.

Key questions:
1. What is the distribution of MFE relative to RangeWidth?
   (ORB range width is not in trades CSV, but we can use MFE as an absolute proxy)
2. How does trade duration relate to the MFE/MAE/Drawdown pattern?
3. Can we identify a conditional filter that makes breakeven more reliably positive?

For example: "Apply breakeven only to trades that enter in the first 30 minutes
after ORB" or "Apply breakeven only when ORB range width > X" etc.

Since we don't have ORB range width in the trades CSV, we approximate it from
entry price and the breakout confirm percentage.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from research.qc.analyze_qc_webide_result import resolve_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "qc_breakeven_conditional.json"


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    fees: float
    mae: float
    mfe: float
    drawdown: float
    is_win: bool
    duration_minutes: float
    year: int
    month: int


def parse_money(value: str) -> float:
    return float(value.replace("$", "").replace(",", "").strip())


def load_trades(path: Path) -> list[Trade]:
    trades: list[Trade] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_time = datetime.strptime(
                row["Entry Time"].replace("T", " ").replace("Z", ""),
                "%Y-%m-%d %H:%M:%S",
            )
            exit_time = datetime.strptime(
                row["Exit Time"].replace("T", " ").replace("Z", ""),
                "%Y-%m-%d %H:%M:%S",
            )
            pnl = parse_money(row["P&L"])
            fees = parse_money(row["Fees"])
            mfe = parse_money(row["MFE"])
            mae = parse_money(row["MAE"])
            drawdown = parse_money(row["Drawdown"])
            duration = (exit_time - entry_time).total_seconds() / 60.0

            trades.append(
                Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    direction=row["Direction"].strip(),
                    entry_price=parse_money(row["Entry Price"]),
                    exit_price=parse_money(row["Exit Price"]),
                    quantity=float(row["Quantity"]),
                    pnl=pnl,
                    fees=fees,
                    mae=mae,
                    mfe=mfe,
                    drawdown=drawdown,
                    is_win=pnl > 0,
                    duration_minutes=duration,
                    year=entry_time.year,
                    month=entry_time.month,
                )
            )
    return trades


def breakeven_would_save(trade: Trade, mfe_threshold: float) -> bool:
    """Loss that had MFE > threshold — breakeven stop would have rescued it."""
    return (not trade.is_win) and trade.mfe > mfe_threshold


def breakeven_would_clip(trade: Trade, mfe_threshold: float) -> bool:
    """Winner with MFE > threshold and drawdown >= MFE — breakeven stop could clip it."""
    return trade.is_win and trade.mfe > mfe_threshold and trade.drawdown >= trade.mfe


def breakeven_net(trades: list[Trade], mfe_threshold: float) -> dict:
    """Net breakeven impact on a set of trades."""
    saving = [t for t in trades if breakeven_would_save(t, mfe_threshold)]
    clipping = [t for t in trades if breakeven_would_clip(t, mfe_threshold)]
    salvage = sum((-t.fees) - t.pnl for t in saving)  # move from loss to ~-fees
    harm = sum(t.pnl - (-t.fees) for t in clipping)    # move from win to ~-fees
    net = salvage - harm
    return {
        "trades_total": len(trades),
        "saving_count": len(saving),
        "clipping_count": len(clipping),
        "salvage": round(salvage, 2),
        "harm": round(harm, 2),
        "net": round(net, 2),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * pct))
    return s[max(0, min(idx, len(s) - 1))]


def resolve_trades_path(trades_path_arg: str | None, result_dir_arg: str) -> Path:
    if trades_path_arg:
        return Path(trades_path_arg)
    return resolve_bundle(Path(result_dir_arg)).trades_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Conditional breakeven analysis on QC trades.")
    parser.add_argument("--trades-path", default=None, help="Optional QC trades CSV path.")
    parser.add_argument(
        "--result-dir",
        default=str(PROJECT_ROOT / "QuantConnect results" / "2017-2026"),
        help="Result directory used to resolve the latest bundle when --trades-path is omitted.",
    )
    args = parser.parse_args()

    trades_path = resolve_trades_path(args.trades_path, args.result_dir)
    trades = load_trades(trades_path)
    print(f"Loaded {len(trades)} trades")

    # Overall stats
    all_mfe = [t.mfe for t in trades]
    all_duration = [t.duration_minutes for t in trades]
    print(f"MFE: median={percentile(all_mfe, 0.5):.2f}, p25={percentile(all_mfe, 0.25):.2f}, p75={percentile(all_mfe, 0.75):.2f}")
    print(f"Duration: median={percentile(all_duration, 0.5):.0f}min, p25={percentile(all_duration, 0.25):.0f}min, p75={percentile(all_duration, 0.75):.0f}min")

    # Distribution of MFE/drawdown ratio
    mfe_drawdown_ratios = []
    for t in trades:
        if t.mfe > 0:
            mfe_drawdown_ratios.append(t.drawdown / t.mfe)

    anchor_threshold = 50

    # === Analysis 1: Duration-based conditional ===
    duration_cuts = [60, 120, 180, 240, 300]  # minutes
    duration_analysis = {}
    for dur_cut in duration_cuts:
        short_trades = [t for t in trades if t.duration_minutes <= dur_cut]
        long_trades = [t for t in trades if t.duration_minutes > dur_cut]
        duration_analysis[f"<={dur_cut}min"] = {
            "count": len(short_trades),
            "be_impact": breakeven_net(short_trades, anchor_threshold),
        }
        duration_analysis[f">{dur_cut}min"] = {
            "count": len(long_trades),
            "be_impact": breakeven_net(long_trades, anchor_threshold),
        }

    # === Analysis 2: Entry hour-based conditional ===
    # QC uses UTC times; market open 14:30 UTC (9:30 ET)
    entry_hour_analysis = {}
    early_trades = [t for t in trades if t.entry_time.hour < 16]  # before 11 AM ET
    late_trades = [t for t in trades if t.entry_time.hour >= 16]   # after 11 AM ET
    entry_hour_analysis["early_<16UTC"] = {
        "count": len(early_trades),
        "be_impact": breakeven_net(early_trades, anchor_threshold),
    }
    entry_hour_analysis["late_>=16UTC"] = {
        "count": len(late_trades),
        "be_impact": breakeven_net(late_trades, anchor_threshold),
    }

    # More granular: by entry hour
    by_hour = defaultdict(list)
    for t in trades:
        by_hour[t.entry_time.hour].append(t)
    hourly_be = {}
    for hour in sorted(by_hour):
        h_trades = by_hour[hour]
        be = breakeven_net(h_trades, anchor_threshold)
        hourly_be[f"hour_{hour:02d}"] = {
            "count": len(h_trades),
            "be_impact": be,
        }

    # === Analysis 3: Direction-based conditional ===
    long_trades = [t for t in trades if t.direction.lower() in {"buy", "long"}]
    short_trades = [t for t in trades if t.direction.lower() in {"sell", "short"}]
    direction_analysis = {
        "long": {
            "count": len(long_trades),
            "be_impact": breakeven_net(long_trades, anchor_threshold),
        },
        "short": {
            "count": len(short_trades),
            "be_impact": breakeven_net(short_trades, anchor_threshold),
        },
    }

    # === Analysis 4: MFE/MAE asymmetry ===
    # Trades where MFE > 2*MAE (strong directional move)
    strong_dir = [t for t in trades if t.mfe > 2 * t.mae and t.mae > 0]
    weak_dir = [t for t in trades if t.mfe <= 2 * t.mae or t.mae <= 0]
    mfe_mae_analysis = {
        "strong_directional_mfe>2mae": {
            "count": len(strong_dir),
            "be_impact": breakeven_net(strong_dir, anchor_threshold),
        },
        "weak_directional": {
            "count": len(weak_dir),
            "be_impact": breakeven_net(weak_dir, anchor_threshold),
        },
    }

    # === Analysis 5: Year-conditional (stability check) ===
    by_year = defaultdict(list)
    for t in trades:
        by_year[t.year].append(t)
    year_be = {}
    for year in sorted(by_year):
        y_trades = by_year[year]
        be = breakeven_net(y_trades, anchor_threshold)
        year_be[str(year)] = {
            "count": len(y_trades),
            "be_impact": be,
        }

    # === Analysis 6: Drawdown/MFE ratio distribution ===
    # What fraction of trades with MFE>threshold actually retrace to entry?
    for threshold in [25, 50, 75, 100]:
        candidates = [t for t in trades if t.mfe > threshold]
        if not candidates:
            continue
        losses_eligible = [t for t in candidates if not t.is_win]
        winners_risky = [t for t in candidates if t.is_win and t.drawdown >= t.mfe]
        print(f"\nMFE>{threshold}: {len(candidates)} candidates, "
              f"{len(losses_eligible)} losing, {len(winners_risky)} risky winners "
              f"({len(winners_risky)/len(candidates)*100:.1f}% clip risk)")

    # Find best conditional filter
    print("\n=== Duration conditional at MFE>50 ===")
    for key, val in duration_analysis.items():
        be = val["be_impact"]
        print(f"  {key}: count={val['count']}, saving={be['saving_count']}, "
              f"clipping={be['clipping_count']}, net={be['net']:+.2f}")

    print("\n=== Entry hour conditional at MFE>50 ===")
    for key, val in hourly_be.items():
        be = val["be_impact"]
        if be["saving_count"] + be["clipping_count"] > 0:
            print(f"  {key}: count={val['count']}, saving={be['saving_count']}, "
                  f"clipping={be['clipping_count']}, net={be['net']:+.2f}")

    print("\n=== Direction conditional at MFE>50 ===")
    for key, val in direction_analysis.items():
        be = val["be_impact"]
        print(f"  {key}: count={val['count']}, saving={be['saving_count']}, "
              f"clipping={be['clipping_count']}, net={be['net']:+.2f}")

    print("\n=== Year stability at MFE>50 ===")
    for year, val in year_be.items():
        be = val["be_impact"]
        print(f"  {year}: count={val['count']}, saving={be['saving_count']}, "
              f"clipping={be['clipping_count']}, net={be['net']:+.2f}")

    payload = {
        "research_scope": "qc_breakeven_conditional_analysis",
        "source_file": str(trades_path.relative_to(PROJECT_ROOT)),
        "anchor_mfe_threshold": anchor_threshold,
        "overall_baseline": breakeven_net(trades, anchor_threshold),
        "duration_conditional": duration_analysis,
        "entry_hour_conditional": entry_hour_analysis,
        "hourly_breakdown": hourly_be,
        "direction_conditional": direction_analysis,
        "mfe_mae_asymmetry": mfe_mae_analysis,
        "year_stability": year_be,
        "mfe_distribution": {
            "p10": round(percentile(all_mfe, 0.10), 2),
            "p25": round(percentile(all_mfe, 0.25), 2),
            "p50": round(percentile(all_mfe, 0.50), 2),
            "p75": round(percentile(all_mfe, 0.75), 2),
            "p90": round(percentile(all_mfe, 0.90), 2),
        },
        "duration_distribution": {
            "p10": round(percentile(all_duration, 0.10), 2),
            "p25": round(percentile(all_duration, 0.25), 2),
            "p50": round(percentile(all_duration, 0.50), 2),
            "p75": round(percentile(all_duration, 0.75), 2),
            "p90": round(percentile(all_duration, 0.90), 2),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved conditional breakeven analysis to {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
