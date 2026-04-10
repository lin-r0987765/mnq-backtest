#!/usr/bin/env python3
"""
Bootstrap validation for the breakeven-stop hypothesis on accepted QC trades.

This script does not change strategy code. It estimates the upper-bound salvage
pool among losing trades that were profitable before reversing, and then uses
year-block bootstrap resampling to assess how stable that pool is.

Important limitation:
- This is an optimistic upper bound because it assumes qualifying losing trades
  could have been exited at breakeven minus fees.
- It does not model the possible profit giveback on winning trades.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from research.qc.analyze_qc_webide_result import resolve_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "qc_breakeven_bootstrap.json"

THRESHOLDS = [0, 25, 50, 75, 100, 125, 150]
BOOTSTRAP_SAMPLES = 10000
SEED = 42


@dataclass
class Trade:
    year: int
    pnl: float
    fees: float
    mfe: float
    mae: float
    is_win: bool


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
            pnl = parse_money(row["P&L"])
            fees = parse_money(row["Fees"])
            mfe = parse_money(row["MFE"])
            mae = parse_money(row["MAE"])
            trades.append(
                Trade(
                    year=entry_time.year,
                    pnl=pnl,
                    fees=fees,
                    mfe=mfe,
                    mae=mae,
                    is_win=pnl > 0,
                )
            )
    return trades


def uplift_for_trade(trade: Trade) -> float:
    # Breakeven exit still pays fees, so the optimistic recovered PnL is -fees.
    return (-trade.fees) - trade.pnl


def summarize_threshold(trades: list[Trade], threshold: float, baseline_net_pnl: float) -> dict:
    qualifying = [t for t in trades if (not t.is_win) and t.mfe > threshold]
    uplift_pool = sum(uplift_for_trade(t) for t in qualifying)
    per_year_pool: dict[int, float] = defaultdict(float)
    per_year_count: dict[int, int] = defaultdict(int)
    for trade in qualifying:
        per_year_pool[trade.year] += uplift_for_trade(trade)
        per_year_count[trade.year] += 1

    return {
        "threshold_mfe_dollars": threshold,
        "qualifying_loss_count": len(qualifying),
        "qualifying_loss_pct_of_all_losses": round(
            len(qualifying) / max(1, sum(1 for t in trades if not t.is_win)) * 100.0, 2
        ),
        "uplift_upper_bound": round(uplift_pool, 2),
        "uplift_upper_bound_pct_of_current_net_pnl": round(
            uplift_pool / baseline_net_pnl * 100.0 if baseline_net_pnl != 0 else 0.0, 2
        ),
        "avg_qualifying_mfe": round(mean([t.mfe for t in qualifying]), 2) if qualifying else 0.0,
        "avg_qualifying_loss": round(mean([t.pnl for t in qualifying]), 2) if qualifying else 0.0,
        "positive_years": sum(1 for value in per_year_pool.values() if value > 0),
        "year_breakdown": {
            str(year): {
                "qualifying_losses": per_year_count[year],
                "uplift_upper_bound": round(per_year_pool[year], 2),
            }
            for year in sorted(per_year_pool)
        },
        "required_capture_for_500": round(500.0 / uplift_pool * 100.0, 2) if uplift_pool > 0 else None,
        "required_capture_for_1000": round(1000.0 / uplift_pool * 100.0, 2) if uplift_pool > 0 else None,
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * pct))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def bootstrap_by_year(trades: list[Trade], thresholds: list[int], samples: int, seed: int) -> dict[int, dict]:
    rng = random.Random(seed)
    by_year: dict[int, list[Trade]] = defaultdict(list)
    for trade in trades:
        by_year[trade.year].append(trade)

    years = sorted(by_year)
    aggregates: dict[int, list[float]] = {threshold: [] for threshold in thresholds}
    counts: dict[int, list[int]] = {threshold: [] for threshold in thresholds}

    for _ in range(samples):
        sampled_years = [rng.choice(years) for _ in years]
        sample_trades: list[Trade] = []
        for year in sampled_years:
            sample_trades.extend(by_year[year])

        for threshold in thresholds:
            qualifying = [t for t in sample_trades if (not t.is_win) and t.mfe > threshold]
            aggregates[threshold].append(sum(uplift_for_trade(t) for t in qualifying))
            counts[threshold].append(len(qualifying))

    summary: dict[int, dict] = {}
    for threshold in thresholds:
        uplift_values = sorted(aggregates[threshold])
        count_values = sorted(counts[threshold])
        positive_prob = sum(1 for value in uplift_values if value > 0) / max(1, len(uplift_values))
        summary[threshold] = {
            "samples": samples,
            "positive_uplift_probability": round(positive_prob * 100.0, 2),
            "uplift_p05": round(percentile(uplift_values, 0.05), 2),
            "uplift_p50": round(percentile(uplift_values, 0.50), 2),
            "uplift_p95": round(percentile(uplift_values, 0.95), 2),
            "qualifying_count_p05": int(percentile(count_values, 0.05)),
            "qualifying_count_p50": int(percentile(count_values, 0.50)),
            "qualifying_count_p95": int(percentile(count_values, 0.95)),
        }
    return summary


def resolve_trades_path(trades_path_arg: str | None, result_dir_arg: str) -> Path:
    if trades_path_arg:
        return Path(trades_path_arg)
    return resolve_bundle(Path(result_dir_arg)).trades_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap validation for QC breakeven-stop hypothesis.")
    parser.add_argument("--trades-path", default=None, help="Optional QC trades CSV path.")
    parser.add_argument(
        "--result-dir",
        default=str(PROJECT_ROOT / "QuantConnect results" / "2017-2026"),
        help="Result directory used to resolve the latest bundle when --trades-path is omitted.",
    )
    args = parser.parse_args()

    trades_path = resolve_trades_path(args.trades_path, args.result_dir)
    trades = load_trades(trades_path)
    total_pnl = sum(trade.pnl for trade in trades)
    total_losses = sum(1 for trade in trades if not trade.is_win)
    total_wins = sum(1 for trade in trades if trade.is_win)

    actual = {
        threshold: summarize_threshold(trades, threshold, total_pnl)
        for threshold in THRESHOLDS
    }
    bootstrap = bootstrap_by_year(trades, THRESHOLDS, BOOTSTRAP_SAMPLES, SEED)

    anchor = actual[50]
    payload = {
        "research_scope": "qc_trade_bootstrap_breakeven_upper_bound",
        "source_file": str(trades_path.relative_to(PROJECT_ROOT)),
        "assumption": "Qualifying losing trades could be exited at breakeven minus fees after reaching the MFE threshold.",
        "limitation": "Upper-bound only; does not model winner profit giveback.",
        "trade_summary": {
            "trades": len(trades),
            "wins": total_wins,
            "losses": total_losses,
            "net_pnl": round(total_pnl, 2),
            "years": sorted({trade.year for trade in trades}),
        },
        "threshold_results": [
            {
                **actual[threshold],
                "bootstrap": bootstrap[threshold],
            }
            for threshold in THRESHOLDS
        ],
        "anchor_threshold": {
            "threshold_mfe_dollars": 50,
            "actual": anchor,
            "bootstrap": bootstrap[50],
        },
        "conclusion": {
            "breakeven_remains_primary_research_thread": True,
            "upper_bound_signal_is_stable": bootstrap[50]["positive_uplift_probability"] > 95.0,
            "launch_ready": False,
            "note": (
                "The upper-bound salvage pool is robust, but this still does not justify v20 because "
                "winner-side harm remains unknown and the local proxy only improved 1/4 folds."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved bootstrap analysis to {OUTPUT_PATH}")
    for threshold in [25, 50, 75, 100]:
        row = actual[threshold]
        boot = bootstrap[threshold]
        print(
            f"MFE>{threshold}: losses={row['qualifying_loss_count']}, uplift={row['uplift_upper_bound']:+.2f}, "
            f"bootstrap median={boot['uplift_p50']:+.2f}, p05={boot['uplift_p05']:+.2f}, "
            f"p95={boot['uplift_p95']:+.2f}, prob>0={boot['positive_uplift_probability']:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
