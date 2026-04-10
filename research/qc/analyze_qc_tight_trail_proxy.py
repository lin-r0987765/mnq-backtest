#!/usr/bin/env python3
"""
Proxy analysis for tightening the QC trailing stop during the early trade window.

This script does not replay full bar paths. Instead, it uses QC trade summary
fields (P&L, MFE, Drawdown, duration) to estimate the incremental effect of
tightening the baseline 1.3% trailing stop to a smaller percentage for the
first N minutes of a trade.

Assumption:
- If max drawdown from peak never exceeds the trailing threshold, the trade is
  unchanged.
- If drawdown exceeds the trailing threshold, the trade would have exited near:
      max(MFE - threshold, -threshold)
  in gross P&L dollars.

This is explicitly a trade-level proxy, not a full path replay.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from research.qc.analyze_qc_breakeven_conditional import (
    PROJECT_ROOT,
    Trade,
    load_trades,
    resolve_trades_path,
)


OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "qc_tight_trail_proxy.json"
ANALYSIS_VERSION = "v1_incremental_proxy_vs_1p3_trail"
BASELINE_TRAIL_PCT = 0.013
TIGHT_TRAILS = [0.007, 0.008, 0.010]
TIME_GATES = [120, 180, 240]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def trail_threshold_dollars(trade: Trade, trail_pct: float) -> float:
    return trail_pct * trade.entry_price * abs(trade.quantity)


def proxy_exit_pnl(trade: Trade, trail_pct: float) -> float:
    threshold = trail_threshold_dollars(trade, trail_pct)
    if trade.drawdown < threshold:
        return trade.pnl
    return max(trade.mfe - threshold, -threshold)


def evaluate_candidate(trades: list[Trade], tight_trail_pct: float, time_gate_min: int) -> dict:
    total_delta = 0.0
    saved_losses = 0
    clipped_winners = 0
    improved_other = 0
    affected_trades = 0
    threshold_base_values: list[float] = []
    threshold_tight_values: list[float] = []
    by_year: dict[int, float] = defaultdict(float)

    for trade in trades:
        delta = 0.0
        if trade.duration_minutes <= time_gate_min:
            base_proxy = proxy_exit_pnl(trade, BASELINE_TRAIL_PCT)
            tight_proxy = proxy_exit_pnl(trade, tight_trail_pct)
            delta = tight_proxy - base_proxy
            threshold_base_values.append(trail_threshold_dollars(trade, BASELINE_TRAIL_PCT))
            threshold_tight_values.append(trail_threshold_dollars(trade, tight_trail_pct))
            if abs(delta) > 1e-9:
                affected_trades += 1
            if delta > 0 and trade.pnl <= 0:
                saved_losses += 1
            elif delta < 0 and trade.pnl > 0:
                clipped_winners += 1
            elif delta > 0:
                improved_other += 1

        total_delta += delta
        by_year[trade.year] += delta

    year_deltas = {str(year): round(value, 2) for year, value in sorted(by_year.items())}
    positive_years = sum(1 for value in by_year.values() if value > 0)
    negative_years = sum(1 for value in by_year.values() if value < 0)

    return {
        "tight_trail_pct": tight_trail_pct,
        "time_gate_min": time_gate_min,
        "incremental_delta": round(total_delta, 2),
        "saved_losses": saved_losses,
        "clipped_winners": clipped_winners,
        "improved_other": improved_other,
        "affected_trades": affected_trades,
        "positive_years": positive_years,
        "negative_years": negative_years,
        "min_year_delta": round(min(by_year.values()), 2) if by_year else 0.0,
        "year_deltas": year_deltas,
        "threshold_distribution_dollars": {
            "baseline_1p3_pct": {
                "p10": round(percentile(threshold_base_values, 0.10), 2),
                "p25": round(percentile(threshold_base_values, 0.25), 2),
                "p50": round(percentile(threshold_base_values, 0.50), 2),
                "p75": round(percentile(threshold_base_values, 0.75), 2),
                "p90": round(percentile(threshold_base_values, 0.90), 2),
            },
            "tight_pct": {
                "p10": round(percentile(threshold_tight_values, 0.10), 2),
                "p25": round(percentile(threshold_tight_values, 0.25), 2),
                "p50": round(percentile(threshold_tight_values, 0.50), 2),
                "p75": round(percentile(threshold_tight_values, 0.75), 2),
                "p90": round(percentile(threshold_tight_values, 0.90), 2),
            },
        },
    }


def candidate_sort_key(candidate: dict) -> tuple:
    return (
        candidate["clipped_winners"] == 0,
        candidate["incremental_delta"],
        candidate["positive_years"],
        -candidate["negative_years"],
        candidate["saved_losses"],
    )


def summarize_candidate(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    return {
        "tight_trail_pct": candidate["tight_trail_pct"],
        "time_gate_min": candidate["time_gate_min"],
        "incremental_delta": candidate["incremental_delta"],
        "saved_losses": candidate["saved_losses"],
        "clipped_winners": candidate["clipped_winners"],
        "affected_trades": candidate["affected_trades"],
        "positive_years": candidate["positive_years"],
        "negative_years": candidate["negative_years"],
        "min_year_delta": candidate["min_year_delta"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Proxy QC tight-trail analysis.")
    parser.add_argument("--trades-path", default=None, help="Optional QC trades CSV path.")
    parser.add_argument(
        "--result-dir",
        default=str(PROJECT_ROOT / "QuantConnect results" / "2017-2026"),
        help="Result directory used to resolve the latest bundle when --trades-path is omitted.",
    )
    args = parser.parse_args()

    trades_path = resolve_trades_path(args.trades_path, args.result_dir)
    trades = load_trades(trades_path)
    print(f"Loaded {len(trades)} trades from {trades_path.name}")
    print(f"Baseline trail: {BASELINE_TRAIL_PCT:.3%}")
    print(f"Tight trails: {TIGHT_TRAILS}")
    print(f"Time gates: {TIME_GATES}")

    rows = []
    for trail in TIGHT_TRAILS:
        for gate in TIME_GATES:
            row = evaluate_candidate(trades, trail, gate)
            rows.append(row)
            print(
                f"tight={trail:.3%} gate={gate}min: "
                f"delta={row['incremental_delta']:+.2f}, "
                f"saved={row['saved_losses']}, clipped={row['clipped_winners']}, "
                f"years+={row['positive_years']}/9, min_year={row['min_year_delta']:+.2f}"
            )

    best_overall = max(rows, key=candidate_sort_key, default=None)
    best_180 = max([row for row in rows if row["time_gate_min"] == 180], key=candidate_sort_key, default=None)
    best_zero_clip = max(
        [row for row in rows if row["incremental_delta"] > 0 and row["clipped_winners"] == 0],
        key=candidate_sort_key,
        default=None,
    )

    payload = {
        "research_scope": "qc_tight_trail_proxy",
        "analysis_version": ANALYSIS_VERSION,
        "source_file": str(trades_path.relative_to(PROJECT_ROOT)),
        "baseline_trail_pct": BASELINE_TRAIL_PCT,
        "tight_trails": TIGHT_TRAILS,
        "time_gates": TIME_GATES,
        "rows": rows,
        "candidate_summary": {
            "best_overall": summarize_candidate(best_overall),
            "best_180min_candidate": summarize_candidate(best_180),
            "best_zero_clip_positive": summarize_candidate(best_zero_clip),
            "interpretation": (
                "This is a trade-level proxy for replacing the baseline 1.3% trailing "
                "stop with a tighter early trailing stop. If the same 0.7%_180min lane "
                "that won locally also wins here with zero clipping, the tight-trail "
                "thread becomes materially stronger."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved proxy analysis to {OUTPUT_PATH}")
    if best_overall:
        print(
            f"Best overall: tight={best_overall['tight_trail_pct']:.3%}, "
            f"gate={best_overall['time_gate_min']}min, delta={best_overall['incremental_delta']:+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
