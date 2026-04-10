#!/usr/bin/env python3
"""
Build a diagnostic weakness map for the accepted v26 baseline.

Goal:
- stop treating post-v26 iteration as a blind threshold search
- classify accepted QC trades into failure / success archetypes
- quantify which shortcomings still matter most
- identify which mechanism family is still underexplored
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "qc_regime_prototypes"
OUTPUT_PATH = RESULTS_DIR / "v26_weakness_map.json"
ANALYSIS_VERSION = "v1_v26_weakness_map"

QC_RESULTS_DIR = PROJECT_ROOT / "QuantConnect results" / "2017-2026"
QC_TRADES_PATH = QC_RESULTS_DIR / "Square Blue Termite_trades.csv"
QC_ORDERS_PATH = QC_RESULTS_DIR / "Square Blue Termite_orders.csv"

CATEGORY_ORDER = [
    "quick_failure_loss",
    "hard_stop_loss",
    "low_progress_drift",
    "late_washout",
    "small_flat_positive",
    "healthy_capture",
    "other",
]

CATEGORY_DEFINITIONS = {
    "quick_failure_loss": (
        "P&L <= 0, duration <= 120 minutes, and MFE < $50. "
        "Represents breakouts that fail quickly without developing enough progress."
    ),
    "hard_stop_loss": (
        "P&L <= 0 and exit tag contains Stop, excluding the earlier more-specific categories. "
        "Represents trades that made some progress but still reverted into a stop-based loss."
    ),
    "low_progress_drift": (
        "duration >= 180 minutes, MFE < $50, and final P&L < $50. "
        "Represents long-duration dead trades that never build enough edge."
    ),
    "late_washout": (
        "duration >= 240 minutes, MFE >= $75, and final P&L < $25. "
        "Represents trades that had meaningful progress but gave it back late."
    ),
    "small_flat_positive": (
        "0 < P&L < $50 and duration >= 240 minutes. "
        "Represents tiny retained winners after a long hold."
    ),
    "healthy_capture": (
        "P&L >= $100 or exit tag contains TP. "
        "Represents trades where the system captured a meaningful winner."
    ),
    "other": (
        "Residual mixed bucket that does not meet any of the above definitions."
    ),
}

CATEGORY_INTERPRETATION = {
    "quick_failure_loss": {
        "theme": "early failed breakout",
        "research_status": "already partially explored via fast-failure abort and still rejected locally",
        "next_step_posture": "deprioritize nearby early-abort grids unless the mechanism is materially different",
    },
    "hard_stop_loss": {
        "theme": "moderate progress reversal before durable protection",
        "research_status": "still underexplored after v26",
        "next_step_posture": (
            "highest-priority unresolved weakness; if iteration continues, prefer a new mid-trade "
            "protection ratchet between BE activation and full profit-lock activation"
        ),
    },
    "low_progress_drift": {
        "theme": "long-duration dead trade / drift",
        "research_status": "explicitly tested via low-progress timeout and rejected locally",
        "next_step_posture": "do not reopen nearby late low-progress timeout thresholds",
    },
    "late_washout": {
        "theme": "late-session giveback / EOD deterioration",
        "research_status": "partially improved by v26 but nearby local washout refinements already rejected",
        "next_step_posture": "do not keep scanning nearby stall / giveback grids without a materially new structure",
    },
    "small_flat_positive": {
        "theme": "under-monetized late winners",
        "research_status": "monitor only",
        "next_step_posture": "only revisit if a new mechanism can improve retention without clipping healthy captures",
    },
    "healthy_capture": {
        "theme": "working state",
        "research_status": "preserve",
        "next_step_posture": "do not damage this bucket when testing future mechanisms",
    },
    "other": {
        "theme": "mixed residual state",
        "research_status": "secondary priority",
        "next_step_posture": "leave for later unless a clearer archetype emerges",
    },
}


def parse_exit_tag_map(orders: pd.DataFrame) -> dict[str, str]:
    tags = orders["Tag"].fillna("").astype(str).tolist()
    return {str(index + 1): tag for index, tag in enumerate(tags)}


def parse_exit_tag(order_ids: object, tag_map: dict[str, str]) -> str:
    parts = [part.strip() for part in str(order_ids).replace("\t", "").split(",") if part.strip()]
    if not parts:
        return ""
    return tag_map.get(parts[-1], "")


def classify_trade(row: pd.Series) -> str:
    pnl = float(row["P&L"])
    mfe = float(row["MFE"])
    duration_min = float(row["duration_min"])
    exit_tag = str(row["exit_tag"])

    if pnl <= 0.0 and duration_min <= 120.0 and mfe < 50.0:
        return "quick_failure_loss"
    if duration_min >= 240.0 and mfe >= 75.0 and pnl < 25.0:
        return "late_washout"
    if duration_min >= 180.0 and mfe < 50.0 and pnl < 50.0:
        return "low_progress_drift"
    if pnl <= 0.0 and "Stop" in exit_tag:
        return "hard_stop_loss"
    if 0.0 < pnl < 50.0 and duration_min >= 240.0:
        return "small_flat_positive"
    if pnl >= 100.0 or "TP" in exit_tag:
        return "healthy_capture"
    return "other"


def summarize_category(
    name: str,
    trades: pd.DataFrame,
    *,
    total_trades: int,
    total_negative_drag: float,
) -> dict:
    group = trades[trades["category"] == name].copy()
    negative_drag = float(-group.loc[group["P&L"] < 0.0, "P&L"].sum())
    opportunity_gap = float(group["opportunity_gap"].sum())
    top_tags = (
        group["exit_tag"]
        .value_counts()
        .head(3)
        .to_dict()
    )
    year_counts = (
        group.groupby("entry_year")
        .size()
        .to_dict()
    )

    top_examples = []
    if not group.empty:
        for _, row in group.sort_values("P&L").head(3).iterrows():
            top_examples.append(
                {
                    "entry_time": row["Entry Time"].isoformat(),
                    "exit_time": row["Exit Time"].isoformat(),
                    "pnl": round(float(row["P&L"]), 2),
                    "mfe": round(float(row["MFE"]), 2),
                    "duration_min": round(float(row["duration_min"]), 2),
                    "exit_tag": row["exit_tag"],
                }
            )

    interpretation = CATEGORY_INTERPRETATION[name]
    return {
        "category": name,
        "definition": CATEGORY_DEFINITIONS[name],
        "theme": interpretation["theme"],
        "research_status": interpretation["research_status"],
        "next_step_posture": interpretation["next_step_posture"],
        "count": int(len(group)),
        "share_of_trades_pct": round(100.0 * len(group) / total_trades, 2) if total_trades else 0.0,
        "net_pnl": round(float(group["P&L"].sum()), 2),
        "mean_pnl": round(float(group["P&L"].mean()), 2) if not group.empty else 0.0,
        "median_pnl": round(float(group["P&L"].median()), 2) if not group.empty else 0.0,
        "mean_mfe": round(float(group["MFE"].mean()), 2) if not group.empty else 0.0,
        "mean_mae": round(float(group["MAE"].mean()), 2) if not group.empty else 0.0,
        "mean_drawdown": round(float(group["Drawdown"].mean()), 2) if not group.empty else 0.0,
        "mean_duration_min": round(float(group["duration_min"].mean()), 2) if not group.empty else 0.0,
        "negative_drag": round(negative_drag, 2),
        "negative_drag_share_pct": round(100.0 * negative_drag / total_negative_drag, 2)
        if total_negative_drag > 0.0
        else 0.0,
        "opportunity_gap_total": round(opportunity_gap, 2),
        "opportunity_gap_per_trade": round(opportunity_gap / len(group), 2) if len(group) else 0.0,
        "top_exit_tags": top_tags,
        "year_counts": {str(year): int(count) for year, count in year_counts.items()},
        "worst_examples": top_examples,
    }


def build_priority_board(category_summaries: list[dict]) -> list[dict]:
    summaries = {row["category"]: row for row in category_summaries}
    return [
        {
            "priority": 1,
            "category": "hard_stop_loss",
            "why_now": (
                "This is the largest underexplored unresolved drag bucket after v26. "
                "It still loses meaningful money, average MFE is non-trivial, and most exits are stop-based."
            ),
            "suggested_mechanism_direction": (
                "Test a mid-trade protection ratchet that activates after modest progress, "
                "earlier than the current 1.50x profit-lock trigger but later than the BE gate."
            ),
            "evidence": {
                "count": summaries["hard_stop_loss"]["count"],
                "net_pnl": summaries["hard_stop_loss"]["net_pnl"],
                "mean_mfe": summaries["hard_stop_loss"]["mean_mfe"],
                "opportunity_gap_total": summaries["hard_stop_loss"]["opportunity_gap_total"],
            },
        },
        {
            "priority": 2,
            "category": "late_washout",
            "why_now": (
                "Still a large opportunity-loss bucket, but nearby late-session refinement families "
                "have already been tested and rejected locally."
            ),
            "suggested_mechanism_direction": (
                "Only revisit if the next mechanism is structurally different from stagnation-timeout, "
                "stall-plus-giveback, and low-progress timeout."
            ),
            "evidence": {
                "count": summaries["late_washout"]["count"],
                "net_pnl": summaries["late_washout"]["net_pnl"],
                "opportunity_gap_total": summaries["late_washout"]["opportunity_gap_total"],
            },
        },
        {
            "priority": 3,
            "category": "quick_failure_loss",
            "why_now": (
                "Largest realized drag by dollars, but the fast-failure branch already failed under local evidence."
            ),
            "suggested_mechanism_direction": (
                "Deprioritize nearby early-abort thresholds unless a materially different entry-quality or "
                "microstructure-aware mechanism appears."
            ),
            "evidence": {
                "count": summaries["quick_failure_loss"]["count"],
                "net_pnl": summaries["quick_failure_loss"]["net_pnl"],
                "negative_drag_share_pct": summaries["quick_failure_loss"]["negative_drag_share_pct"],
            },
        },
    ]


def main() -> int:
    print("Loading accepted v26 QC trades and orders...")
    trades = pd.read_csv(QC_TRADES_PATH)
    orders = pd.read_csv(QC_ORDERS_PATH)

    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True)
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True)
    trades["entry_year"] = trades["Entry Time"].dt.year
    trades["duration_min"] = (
        trades["Exit Time"] - trades["Entry Time"]
    ).dt.total_seconds() / 60.0

    for column in ["P&L", "MAE", "MFE", "Drawdown"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce").fillna(0.0)

    tag_map = parse_exit_tag_map(orders)
    trades["exit_tag"] = trades["Order Ids"].apply(lambda value: parse_exit_tag(value, tag_map))
    trades["category"] = trades.apply(classify_trade, axis=1)
    trades["opportunity_gap"] = (trades["MFE"] - trades["P&L"].clip(lower=0.0)).clip(lower=0.0)

    total_trades = int(len(trades))
    total_negative_drag = float(-trades.loc[trades["P&L"] < 0.0, "P&L"].sum())
    category_summaries = [
        summarize_category(
            name,
            trades,
            total_trades=total_trades,
            total_negative_drag=total_negative_drag,
        )
        for name in CATEGORY_ORDER
    ]

    largest_realized_drag = max(category_summaries, key=lambda row: row["negative_drag"])
    largest_opportunity_gap = max(category_summaries, key=lambda row: row["opportunity_gap_total"])
    highest_count_non_capture = max(
        [row for row in category_summaries if row["category"] != "healthy_capture"],
        key=lambda row: row["count"],
    )

    priority_board = build_priority_board(category_summaries)

    payload = {
        "research_scope": "v26_weakness_map",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v26-profit-lock",
        "accepted_reference_bundle": "Square Blue Termite",
        "data": {
            "trades_file": str(QC_TRADES_PATH.relative_to(PROJECT_ROOT)),
            "orders_file": str(QC_ORDERS_PATH.relative_to(PROJECT_ROOT)),
            "trades": total_trades,
            "orders": int(len(orders)),
            "start": trades["Entry Time"].min().isoformat(),
            "end": trades["Exit Time"].max().isoformat(),
        },
        "method": (
            "Classify accepted v26 QC trades into mutually exclusive archetypes so post-v26 "
            "iteration can target real bottlenecks instead of blindly scanning more nearby thresholds."
        ),
        "category_order": CATEGORY_ORDER,
        "category_stats": category_summaries,
        "frontier": {
            "largest_realized_drag": {
                "category": largest_realized_drag["category"],
                "net_negative_drag": largest_realized_drag["negative_drag"],
                "negative_drag_share_pct": largest_realized_drag["negative_drag_share_pct"],
            },
            "largest_opportunity_gap": {
                "category": largest_opportunity_gap["category"],
                "opportunity_gap_total": largest_opportunity_gap["opportunity_gap_total"],
                "opportunity_gap_per_trade": largest_opportunity_gap["opportunity_gap_per_trade"],
            },
            "highest_count_non_capture": {
                "category": highest_count_non_capture["category"],
                "count": highest_count_non_capture["count"],
                "share_of_trades_pct": highest_count_non_capture["share_of_trades_pct"],
            },
        },
        "priority_board": priority_board,
        "structural_conclusion": {
            "main_findings": [
                "The largest realized drag bucket is quick_failure_loss, but that branch was already pressure-tested via fast-failure abort and failed locally.",
                "The most actionable unresolved bucket is hard_stop_loss: meaningful count, meaningful dollar drag, and non-trivial average MFE before stop-out.",
                "Late-session giveback remains real, but nearby late-session refinement branches have already been explored and rejected from local-only evidence.",
                "Low-progress drift is a real archetype but is now a closed research branch rather than an open frontier.",
            ],
            "next_step_rule": (
                "Do not launch a new QC candidate from this map alone. Use it to choose the next genuinely new mechanism family."
            ),
            "preferred_next_mechanism": (
                "Prefer a mid-trade protection ratchet for moderate-progress stop-outs, "
                "rather than more late-session washout thresholds or more low-progress scratch exits."
            ),
        },
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved weakness map to {OUTPUT_PATH}")
    print(json.dumps(payload["frontier"], indent=2))
    print(json.dumps(priority_board, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
