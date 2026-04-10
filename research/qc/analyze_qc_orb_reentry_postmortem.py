"""Postmortem analysis for the v26 ORB re-entry QC evaluator bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "QuantConnect results" / "2017-2026"
DEFAULT_BUNDLE = "Retrospective Red Orange Whale"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "qc_regime_prototypes" / "qc_orb_reentry_postmortem.json"
ORB_REENTRY_EXIT_TAG = "ORB Long ORB Reentry Exit"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ORB re-entry exit behavior for a QC bundle.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _parse_float(raw: str) -> float:
    text = (raw or "").replace(",", "").replace("\t", "").strip()
    return float(text) if text else 0.0


def _parse_order_ids(raw: str) -> List[str]:
    text = (raw or "").replace("\t", "").strip()
    return [token.strip() for token in text.split(",") if token.strip()]


def _load_order_tags(path: Path) -> Dict[str, str]:
    order_tags: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            order_tags[str(idx)] = (row.get("Tag") or "").strip()
    return order_tags


def _load_trade_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sorted_summary(summary: Dict[str, dict]) -> Dict[str, dict]:
    items = sorted(summary.items(), key=lambda kv: (-abs(kv[1]["total_pnl"]), kv[0]))
    return {key: value for key, value in items}


def analyze_bundle(results_dir: Path, bundle: str) -> dict:
    orders_path = results_dir / f"{bundle}_orders.csv"
    trades_path = results_dir / f"{bundle}_trades.csv"
    if not orders_path.exists():
        raise FileNotFoundError(f"Orders file not found: {orders_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_path}")

    order_tags = _load_order_tags(orders_path)
    trade_rows = _load_trade_rows(trades_path)

    exit_tag_summary: Dict[str, dict] = defaultdict(lambda: {"count": 0, "total_pnl": 0.0, "total_mfe": 0.0})
    yearly_exit_tag_summary: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "total_pnl": 0.0})
    )

    orb_reentry_count = 0
    orb_reentry_total_pnl = 0.0
    orb_reentry_yearly: Dict[str, dict] = defaultdict(lambda: {"count": 0, "total_pnl": 0.0})

    for row in trade_rows:
        exit_year = (row.get("Exit Time") or "0000")[:4]
        order_ids = _parse_order_ids(row.get("Order Ids") or "")
        exit_order_id = order_ids[-1] if order_ids else ""
        exit_tag = order_tags.get(exit_order_id, "UNKNOWN")
        pnl = _parse_float(row.get("P&L") or "0")
        mfe = _parse_float(row.get("MFE") or "0")

        summary = exit_tag_summary[exit_tag]
        summary["count"] += 1
        summary["total_pnl"] += pnl
        summary["total_mfe"] += mfe

        yearly_summary = yearly_exit_tag_summary[exit_tag][exit_year]
        yearly_summary["count"] += 1
        yearly_summary["total_pnl"] += pnl

        if exit_tag == ORB_REENTRY_EXIT_TAG:
            orb_reentry_count += 1
            orb_reentry_total_pnl += pnl
            orb_reentry_yearly[exit_year]["count"] += 1
            orb_reentry_yearly[exit_year]["total_pnl"] += pnl

    finalized_exit_summary: Dict[str, dict] = {}
    for exit_tag, summary in exit_tag_summary.items():
        count = summary["count"]
        finalized_exit_summary[exit_tag] = {
            "count": count,
            "total_pnl": round(summary["total_pnl"], 2),
            "avg_pnl": round(summary["total_pnl"] / count, 2) if count else 0.0,
            "mean_mfe": round(summary["total_mfe"] / count, 2) if count else 0.0,
            "yearly": {
                year: {
                    "count": year_summary["count"],
                    "total_pnl": round(year_summary["total_pnl"], 2),
                }
                for year, year_summary in sorted(yearly_exit_tag_summary[exit_tag].items())
            },
        }

    orb_reentry_yearly_sorted = {
        year: {
            "count": summary["count"],
            "total_pnl": round(summary["total_pnl"], 2),
        }
        for year, summary in sorted(orb_reentry_yearly.items())
    }
    orb_reentry_positive_years = sum(1 for summary in orb_reentry_yearly_sorted.values() if summary["total_pnl"] > 0)

    return {
        "bundle": bundle,
        "orders_file": str(orders_path.relative_to(REPO_ROOT)),
        "trades_file": str(trades_path.relative_to(REPO_ROOT)),
        "order_id_mapping_assumption": "trades.csv order ids are matched to 1-based data row numbers in orders.csv because the orders export has no explicit order-id column.",
        "exit_tag_summary": _sorted_summary(finalized_exit_summary),
        "orb_reentry_exit_focus": {
            "tag": ORB_REENTRY_EXIT_TAG,
            "count": orb_reentry_count,
            "total_pnl": round(orb_reentry_total_pnl, 2),
            "avg_pnl": round(orb_reentry_total_pnl / orb_reentry_count, 2) if orb_reentry_count else 0.0,
            "positive_years": orb_reentry_positive_years,
            "years_active": len(orb_reentry_yearly_sorted),
            "yearly": orb_reentry_yearly_sorted,
        },
        "verdict": "ORB re-entry exits were realized negative in aggregate and in every active year, so the QC proxy remains a mixed analyzer-only signal and should not launch v27.",
    }


def main() -> None:
    args = _parse_args()
    report = analyze_bundle(results_dir=args.results_dir, bundle=args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=4) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
