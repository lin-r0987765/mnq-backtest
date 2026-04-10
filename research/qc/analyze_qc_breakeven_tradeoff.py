#!/usr/bin/env python3
"""
Tradeoff analysis for the breakeven-stop hypothesis on accepted QC trades.

This script extends the loss-side salvage view by estimating a proxy for
winner-side giveback using the trade CSV's `MFE` and `Drawdown` columns.

Assumption:
- If a winning trade has `MFE > threshold` and `Drawdown >= MFE`, then after
  reaching that profit peak it likely retraced back to the entry area, meaning
  a breakeven stop could plausibly have exited it near `-fees`.

Limitations:
- This is still an approximation based on trade-level aggregates, not bar-path
  reconstruction.
- It remains an upper-bound style estimate, but it is stricter than looking at
  loss-side salvage alone because it subtracts a proxy for clipped winners.
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

from research.qc.analyze_qc_webide_result import resolve_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "qc_breakeven_tradeoff.json"

THRESHOLDS = [25, 50, 75, 100]
BOOTSTRAP_SAMPLES = 10000
SEED = 42


@dataclass
class Trade:
    year: int
    pnl: float
    fees: float
    mfe: float
    drawdown: float
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
            drawdown = parse_money(row["Drawdown"])
            trades.append(
                Trade(
                    year=entry_time.year,
                    pnl=pnl,
                    fees=fees,
                    mfe=mfe,
                    drawdown=drawdown,
                    is_win=pnl > 0,
                )
            )
    return trades


def breakeven_exit_pnl(trade: Trade) -> float:
    return -trade.fees


def loss_salvage(trade: Trade) -> float:
    return breakeven_exit_pnl(trade) - trade.pnl


def winner_harm(trade: Trade) -> float:
    return trade.pnl - breakeven_exit_pnl(trade)


def winner_could_be_clipped(trade: Trade, threshold: float) -> bool:
    return trade.is_win and trade.mfe > threshold and trade.drawdown >= trade.mfe


def loss_could_be_saved(trade: Trade, threshold: float) -> bool:
    return (not trade.is_win) and trade.mfe > threshold


def summarize_threshold(trades: list[Trade], threshold: float) -> dict:
    saving_losses = [trade for trade in trades if loss_could_be_saved(trade, threshold)]
    clipped_winners = [trade for trade in trades if winner_could_be_clipped(trade, threshold)]

    per_year: dict[int, dict[str, float]] = defaultdict(lambda: {"salvage": 0.0, "harm": 0.0})
    for trade in saving_losses:
        per_year[trade.year]["salvage"] += loss_salvage(trade)
    for trade in clipped_winners:
        per_year[trade.year]["harm"] += winner_harm(trade)

    year_breakdown = {}
    positive_net_years = 0
    for year in sorted(per_year):
        salvage = per_year[year]["salvage"]
        harm = per_year[year]["harm"]
        net = salvage - harm
        if net > 0:
            positive_net_years += 1
        year_breakdown[str(year)] = {
            "loss_salvage_upper_bound": round(salvage, 2),
            "winner_harm_upper_bound": round(harm, 2),
            "net_upper_bound_after_harm": round(net, 2),
        }

    loss_pool = sum(loss_salvage(trade) for trade in saving_losses)
    harm_pool = sum(winner_harm(trade) for trade in clipped_winners)
    net_pool = loss_pool - harm_pool

    return {
        "threshold_mfe_dollars": threshold,
        "saving_loss_count": len(saving_losses),
        "clipped_winner_count": len(clipped_winners),
        "loss_salvage_upper_bound": round(loss_pool, 2),
        "winner_harm_upper_bound": round(harm_pool, 2),
        "net_upper_bound_after_harm": round(net_pool, 2),
        "positive_net_years": positive_net_years,
        "year_breakdown": year_breakdown,
        "required_capture_for_500": round(500.0 / net_pool * 100.0, 2) if net_pool > 0 else None,
        "required_capture_for_1000": round(1000.0 / net_pool * 100.0, 2) if net_pool > 0 else None,
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * pct))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def bootstrap_thresholds(trades: list[Trade], thresholds: list[int], samples: int, seed: int) -> dict[int, dict]:
    rng = random.Random(seed)
    by_year: dict[int, list[Trade]] = defaultdict(list)
    for trade in trades:
        by_year[trade.year].append(trade)
    years = sorted(by_year)

    outputs: dict[int, list[float]] = {threshold: [] for threshold in thresholds}
    for _ in range(samples):
        sample: list[Trade] = []
        for _ in years:
            sample.extend(by_year[rng.choice(years)])
        for threshold in thresholds:
            salvage = sum(
                loss_salvage(trade) for trade in sample if loss_could_be_saved(trade, threshold)
            )
            harm = sum(
                winner_harm(trade) for trade in sample if winner_could_be_clipped(trade, threshold)
            )
            outputs[threshold].append(salvage - harm)

    summary: dict[int, dict] = {}
    for threshold in thresholds:
        values = sorted(outputs[threshold])
        summary[threshold] = {
            "samples": samples,
            "net_positive_probability": round(
                sum(1 for value in values if value > 0) / len(values) * 100.0, 2
            ),
            "net_p05": round(percentile(values, 0.05), 2),
            "net_p50": round(percentile(values, 0.50), 2),
            "net_p95": round(percentile(values, 0.95), 2),
        }
    return summary


def resolve_trades_path(trades_path_arg: str | None, result_dir_arg: str) -> Path:
    if trades_path_arg:
        return Path(trades_path_arg)
    return resolve_bundle(Path(result_dir_arg)).trades_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tradeoff analysis for QC breakeven-stop hypothesis.")
    parser.add_argument("--trades-path", default=None, help="Optional QC trades CSV path.")
    parser.add_argument(
        "--result-dir",
        default=str(PROJECT_ROOT / "QuantConnect results" / "2017-2026"),
        help="Result directory used to resolve the latest bundle when --trades-path is omitted.",
    )
    args = parser.parse_args()

    trades_path = resolve_trades_path(args.trades_path, args.result_dir)
    trades = load_trades(trades_path)
    baseline_net_pnl = round(sum(trade.pnl for trade in trades), 2)
    results = {threshold: summarize_threshold(trades, threshold) for threshold in THRESHOLDS}
    bootstrap = bootstrap_thresholds(trades, THRESHOLDS, BOOTSTRAP_SAMPLES, SEED)

    payload = {
        "research_scope": "qc_trade_breakeven_tradeoff_proxy",
        "source_file": str(trades_path.relative_to(PROJECT_ROOT)),
        "assumption": "Winning trades with MFE > threshold and Drawdown >= MFE likely revisited the entry area after the profit peak, so a breakeven stop could plausibly have clipped them near -fees.",
        "limitation": "Still approximate because the full price path is unavailable; however, this is stricter than loss-only salvage because it subtracts a proxy for clipped winners.",
        "trade_summary": {
            "trades": len(trades),
            "net_pnl": baseline_net_pnl,
            "years": sorted({trade.year for trade in trades}),
        },
        "threshold_results": [
            {**results[threshold], "bootstrap": bootstrap[threshold]}
            for threshold in THRESHOLDS
        ],
        "anchor_threshold": {
            "threshold_mfe_dollars": 50,
            "actual": results[50],
            "bootstrap": bootstrap[50],
        },
        "conclusion": {
            "breakeven_tradeoff_still_supportive": results[50]["net_upper_bound_after_harm"] > 0,
            "anchor_threshold_robust": bootstrap[50]["net_positive_probability"] > 95.0,
            "launch_ready": False,
            "note": "Winner-side giveback is now partially bounded, but this is still a trade-level proxy rather than a full bar-path simulation.",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved tradeoff analysis to {OUTPUT_PATH}")
    for threshold in THRESHOLDS:
        row = results[threshold]
        boot = bootstrap[threshold]
        print(
            f"MFE>{threshold}: salvage={row['loss_salvage_upper_bound']:+.2f}, "
            f"harm={row['winner_harm_upper_bound']:+.2f}, net={row['net_upper_bound_after_harm']:+.2f}, "
            f"net_p05={boot['net_p05']:+.2f}, net_p50={boot['net_p50']:+.2f}, "
            f"net_p95={boot['net_p95']:+.2f}, prob>0={boot['net_positive_probability']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
