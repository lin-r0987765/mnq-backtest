#!/usr/bin/env python3
"""
Sweep duration caps and MFE anchors for the QC breakeven thesis.

This extends the conditional analysis by answering a narrower question:
is the early-duration zero-clipping lane robust across different MFE anchors,
or is it only attractive at a single threshold like MFE > $50?

The goal is not to launch a candidate directly. It is to tighten the
conservative QC-only case around duration-gated breakeven.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from research.qc.analyze_qc_breakeven_conditional import (
    PROJECT_ROOT,
    Trade,
    breakeven_net,
    load_trades,
    resolve_trades_path,
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "qc_breakeven_anchor_sweep.json"
)
ANALYSIS_VERSION = "v1_duration_cap_x_mfe_anchor"
DURATION_CAPS = [60, 120, 180, 240, 300]
MFE_THRESHOLDS = [25, 50, 75, 100, 125, 150, 175, 200]


def year_breakdown(trades: list[Trade], mfe_threshold: float) -> dict[str, dict]:
    by_year: dict[int, list[Trade]] = defaultdict(list)
    for trade in trades:
        by_year[trade.year].append(trade)
    rows: dict[str, dict] = {}
    for year in sorted(by_year):
        impact = breakeven_net(by_year[year], mfe_threshold)
        rows[str(year)] = {
            "count": len(by_year[year]),
            **impact,
        }
    return rows


def summarize_candidate(duration_cap: int, mfe_threshold: float, trades: list[Trade]) -> dict:
    impact = breakeven_net(trades, mfe_threshold)
    years = year_breakdown(trades, mfe_threshold)
    year_nets = [row["net"] for row in years.values()]
    positive_years = sum(1 for value in year_nets if value > 0)
    zero_years = sum(1 for value in year_nets if value == 0)
    return {
        "duration_cap_min": duration_cap,
        "mfe_threshold_dollars": mfe_threshold,
        "count": len(trades),
        "impact": impact,
        "positive_years": positive_years,
        "zero_years": zero_years,
        "negative_years": sum(1 for value in year_nets if value < 0),
        "min_year_net": round(min(year_nets), 2) if year_nets else 0.0,
        "year_breakdown": years,
    }


def conservative_sort_key(candidate: dict) -> tuple:
    impact = candidate["impact"]
    return (
        impact["clipping_count"] == 0,
        candidate["negative_years"] == 0,
        candidate["positive_years"],
        impact["net"],
        impact["saving_count"],
        candidate["duration_cap_min"],
        -candidate["mfe_threshold_dollars"],
    )


def balanced_sort_key(candidate: dict) -> tuple:
    impact = candidate["impact"]
    return (
        candidate["negative_years"] == 0,
        impact["net"],
        -impact["clipping_count"],
        impact["saving_count"],
        -candidate["duration_cap_min"],
        -candidate["mfe_threshold_dollars"],
    )


def strip_year_breakdown(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    summary = dict(candidate)
    summary.pop("year_breakdown", None)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep QC breakeven duration caps and MFE anchors.")
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
    print(f"Duration caps: {DURATION_CAPS}")
    print(f"MFE thresholds: {MFE_THRESHOLDS}")

    sweep_rows = []
    for duration_cap in DURATION_CAPS:
        subset = [trade for trade in trades if trade.duration_minutes <= duration_cap]
        print(f"\nDuration <= {duration_cap} min: {len(subset)} trades")
        for mfe_threshold in MFE_THRESHOLDS:
            row = summarize_candidate(duration_cap, mfe_threshold, subset)
            impact = row["impact"]
            print(
                f"  MFE>{mfe_threshold:>3}: net={impact['net']:+8.2f}, "
                f"save={impact['saving_count']:>2}, clip={impact['clipping_count']:>2}, "
                f"years+={row['positive_years']}/9, min_year={row['min_year_net']:+7.2f}"
            )
            sweep_rows.append(row)

    zero_clip_candidates = [
        row for row in sweep_rows if row["impact"]["clipping_count"] == 0 and row["impact"]["net"] > 0
    ]
    all_year_positive_candidates = [
        row for row in sweep_rows if row["negative_years"] == 0 and row["impact"]["net"] > 0
    ]
    duration_180_candidates = [row for row in sweep_rows if row["duration_cap_min"] == 180]
    duration_180_zero_clip = [
        row
        for row in duration_180_candidates
        if row["impact"]["clipping_count"] == 0 and row["impact"]["net"] > 0
    ]

    best_zero_clip = max(zero_clip_candidates, key=conservative_sort_key, default=None)
    best_all_year_positive = max(all_year_positive_candidates, key=balanced_sort_key, default=None)
    best_180 = max(duration_180_candidates, key=conservative_sort_key, default=None)
    best_180_zero_clip = max(duration_180_zero_clip, key=conservative_sort_key, default=None)

    payload = {
        "research_scope": "qc_breakeven_anchor_sweep",
        "analysis_version": ANALYSIS_VERSION,
        "source_file": str(trades_path.relative_to(PROJECT_ROOT)),
        "duration_caps_min": DURATION_CAPS,
        "mfe_thresholds_dollars": MFE_THRESHOLDS,
        "rows": sweep_rows,
        "candidate_summary": {
            "best_zero_clip_candidate": strip_year_breakdown(best_zero_clip),
            "best_all_year_positive_candidate": strip_year_breakdown(best_all_year_positive),
            "best_180min_candidate": strip_year_breakdown(best_180),
            "best_180min_zero_clip_candidate": strip_year_breakdown(best_180_zero_clip),
            "interpretation": (
                "If the <=180 minute lane stays zero-clipping across many anchors, "
                "that strengthens the conservative QC-only case. Higher duration caps "
                "may still be attractive on net, but clipping must be treated as a real risk."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved anchor sweep to {OUTPUT_PATH}")

    if best_zero_clip:
        impact = best_zero_clip["impact"]
        print("\nBest zero-clip candidate:")
        print(
            f"  <= {best_zero_clip['duration_cap_min']} min, "
            f"MFE > ${best_zero_clip['mfe_threshold_dollars']}"
        )
        print(
            f"  net={impact['net']:+.2f}, save={impact['saving_count']}, "
            f"clip={impact['clipping_count']}, years+={best_zero_clip['positive_years']}/9"
        )

    if best_180_zero_clip:
        impact = best_180_zero_clip["impact"]
        print("\nBest 180-minute zero-clip candidate:")
        print(
            f"  MFE > ${best_180_zero_clip['mfe_threshold_dollars']}, "
            f"net={impact['net']:+.2f}, save={impact['saving_count']}, "
            f"clip={impact['clipping_count']}, years+={best_180_zero_clip['positive_years']}/9"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
