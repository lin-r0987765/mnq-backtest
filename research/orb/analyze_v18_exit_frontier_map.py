#!/usr/bin/env python3
"""
Consolidate the remaining exit-side research frontier for v18.

This script exists to stop repeated low-value exploration after:
- entry-filter research was structurally exhausted
- v23-wide-trail was rejected in real QC
- v24-no-trail was rejected in real QC
- late-session time-decay trailing-stop research also failed locally

It summarizes which exit-side branches are dead, which are disqualified, and
which single unbanned path still has enough combined evidence to justify the
next real iteration step once an accepted baseline rerun bundle is restored.
"""
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "qc_regime_prototypes"
QC_RESULTS_DIR = PROJECT_ROOT / "QuantConnect results" / "2017-2026"
OUTPUT_PATH = RESULTS_DIR / "v18_exit_frontier_map.json"
ANALYSIS_VERSION = "v1_timegated_breakeven_frontier_consolidation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def round_or_none(value, digits: int = 4):
    if value is None:
        return None
    return round(float(value), digits)


def find_anchor_row(rows: list[dict], *, duration_cap_min: int, mfe_threshold_dollars: int) -> dict:
    for row in rows:
        if (
            int(row["duration_cap_min"]) == duration_cap_min
            and int(row["mfe_threshold_dollars"]) == mfe_threshold_dollars
        ):
            return row
    raise ValueError(
        f"Anchor row not found for duration_cap_min={duration_cap_min}, "
        f"mfe_threshold_dollars={mfe_threshold_dollars}"
    )


def find_tradeoff_row(rows: list[dict], *, threshold_mfe_dollars: int) -> dict:
    for row in rows:
        if int(row["threshold_mfe_dollars"]) == threshold_mfe_dollars:
            return row
    raise ValueError(f"Tradeoff row not found for threshold_mfe_dollars={threshold_mfe_dollars}")


def count_existing(paths: list[Path]) -> list[str]:
    return [str(path.name) for path in paths if path.exists()]


def detect_accepted_v18_rerun_trades() -> list[str]:
    found: list[tuple[float, str]] = []
    for log_path in QC_RESULTS_DIR.glob("*_logs.txt"):
        try:
            text = log_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "version=v18-prev-day-mom3" not in text:
            continue
        if "security_type=equity" not in text or "symbol=QQQ" not in text:
            continue
        trades_path = log_path.with_name(log_path.name.replace("_logs.txt", "_trades.csv"))
        if not trades_path.exists():
            continue
        found.append((trades_path.stat().st_mtime, trades_path.name))
    found.sort(reverse=True)
    return [name for _, name in found]


def build_frontier() -> dict:
    timegated = load_json(RESULTS_DIR / "local_orb_timegated_breakeven.json")
    pathsim = load_json(RESULTS_DIR / "local_orb_breakeven_pathsim.json")
    anchor = load_json(RESULTS_DIR / "qc_breakeven_anchor_sweep.json")
    tradeoff = load_json(RESULTS_DIR / "qc_breakeven_tradeoff.json")
    time_decay = load_json(RESULTS_DIR / "local_orb_time_decay_trail.json")
    tight_trail_local = load_json(RESULTS_DIR / "local_orb_tight_trail.json")
    tight_trail_qc = load_json(RESULTS_DIR / "qc_tight_trail_proxy.json")

    historical_known_candidates = [
        QC_RESULTS_DIR / "Alert Asparagus Panda_trades.csv",
        QC_RESULTS_DIR / "Calculating Tan Bat_trades.csv",
        QC_RESULTS_DIR / "Hipster Asparagus Wolf_trades.csv",
        QC_RESULTS_DIR / "Energetic Brown Alpaca_trades.csv",
    ]
    available_accepted_baseline_trades = detect_accepted_v18_rerun_trades()
    if not available_accepted_baseline_trades:
        available_accepted_baseline_trades = count_existing(historical_known_candidates)
    accepted_available = bool(available_accepted_baseline_trades)

    conservative_local = timegated["candidate_summary"]["best_180min_candidate"]
    conservative_qc = find_anchor_row(
        anchor["rows"], duration_cap_min=180, mfe_threshold_dollars=25
    )
    stress_local = timegated["candidate_summary"]["best_total_positive"]
    stress_qc = find_anchor_row(anchor["rows"], duration_cap_min=240, mfe_threshold_dollars=25)
    pathsim_best = pathsim["variants"][4]  # BE=1.00x in the existing artifact ordering
    global_tradeoff = find_tradeoff_row(
        tradeoff["threshold_results"], threshold_mfe_dollars=50
    )
    decay_best = time_decay["candidate_summary"]["best_overall"]
    tight_local = tight_trail_local["candidate_summary"]["best_180min_candidate"]
    tight_qc = tight_trail_qc["candidate_summary"]["best_180min_candidate"]

    frontier = {
        "research_scope": "v18_exit_frontier_map",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v18-prev-day-mom3",
        "accepted_baseline_raw_trades_available": bool(available_accepted_baseline_trades),
        "accepted_baseline_raw_trades_found": available_accepted_baseline_trades,
        "branches": [
            {
                "branch": "conservative_timegated_breakeven",
                "status": (
                    "READY_FOR_QC_CANDIDATE"
                    if accepted_available
                    else "FRONTIER_LEADER_PENDING_BASELINE_RESTORE"
                ),
                "why_it_is_live": (
                    "Only unbanned exit-side path with positive local path-level evidence "
                    "and historical QC-only zero-clipping support."
                ),
                "recommended_local_proxy": {
                    "label": conservative_local["label"],
                    "pnl_delta": round_or_none(conservative_local["pnl_delta"]),
                    "improved_folds": int(conservative_local["improved_folds"]),
                    "clipped_winners": int(conservative_local["clipped_winners"]),
                    "saved_losses": int(conservative_local["saved_losses"]),
                },
                "historical_qc_support": {
                    "lane": "<=180min + MFE>$25",
                    "net_upper_bound": round_or_none(conservative_qc["impact"]["net"], 2),
                    "saved_losses": int(conservative_qc["impact"]["saving_count"]),
                    "clipped_winners": int(conservative_qc["impact"]["clipping_count"]),
                    "positive_years": int(conservative_qc["positive_years"]),
                    "negative_years": int(conservative_qc["negative_years"]),
                },
                "historical_qc_tradeoff_support": {
                    "threshold_mfe_dollars": int(global_tradeoff["threshold_mfe_dollars"]),
                    "net_upper_bound_after_harm": round_or_none(
                        global_tradeoff["net_upper_bound_after_harm"], 2
                    ),
                    "positive_net_years": int(global_tradeoff["positive_net_years"]),
                    "bootstrap_p05": round_or_none(global_tradeoff["bootstrap"]["net_p05"], 2),
                },
            },
            {
                "branch": "stress_timegated_breakeven",
                "status": "STRESS_TEST_ONLY_NOT_LIVE",
                "why_it_is_not_live": (
                    "Higher local uplift exists, but clipping reappears and safety is weaker "
                    "than the conservative 180-minute lane."
                ),
                "recommended_local_proxy": {
                    "label": stress_local["label"],
                    "pnl_delta": round_or_none(stress_local["pnl_delta"]),
                    "improved_folds": int(stress_local["improved_folds"]),
                    "clipped_winners": int(stress_local["clipped_winners"]),
                    "saved_losses": int(stress_local["saved_losses"]),
                },
                "historical_qc_support": {
                    "lane": "<=240min + MFE>$25",
                    "net_upper_bound": round_or_none(stress_qc["impact"]["net"], 2),
                    "saved_losses": int(stress_qc["impact"]["saving_count"]),
                    "clipped_winners": int(stress_qc["impact"]["clipping_count"]),
                },
            },
            {
                "branch": "plain_breakeven_without_time_gate",
                "status": "WEAKER_THAN_TIMEGATED",
                "why_it_is_not_frontier": (
                    "Positive locally, but weaker than the gated version and lacks the same "
                    "clean duration-based clipping control."
                ),
                "best_local_variant": {
                    "label": pathsim_best["label"],
                    "pnl_delta": round_or_none(
                        pathsim_best["metrics"]["total_pnl"] - pathsim["baseline"]["metrics"]["total_pnl"]
                    ),
                    "be_stop_exits": int(pathsim_best["metrics"]["be_stop_exits"]),
                    "improved_folds": int(pathsim_best["improved_vs_baseline_folds"]),
                },
            },
            {
                "branch": "late_session_time_decay_trail",
                "status": "REJECTED_LOCAL_ONLY",
                "why_it_is_dead": (
                    "Already tested as a genuinely new exit mechanism, but no positive full-sample "
                    "candidate was found."
                ),
                "best_local_variant": {
                    "label": decay_best["label"],
                    "pnl_delta": round_or_none(decay_best["pnl_delta"]),
                    "clipped_winners": int(decay_best["clipped_winners"]),
                    "improved_folds": int(decay_best["improved_folds"]),
                },
            },
            {
                "branch": "early_tight_trail",
                "status": "DISQUALIFIED_HISTORICAL_BRANCH",
                "why_it_is_not_actionable": (
                    "Historically strong in both local and QC proxy research, but explicitly banned "
                    "because it was already launched as v20-tight-trail-early."
                ),
                "historical_local_strength": {
                    "label": tight_local["label"],
                    "pnl_delta": round_or_none(tight_local["pnl_delta"]),
                    "clipped_winners": int(tight_local["clipped_winners"]),
                },
                "historical_qc_strength": {
                    "incremental_delta": round_or_none(tight_qc["incremental_delta"], 2),
                    "clipped_winners": int(tight_qc["clipped_winners"]),
                    "positive_years": int(tight_qc["positive_years"]),
                },
            },
            {
                "branch": "wide_or_removed_trail",
                "status": "REJECTED_REAL_QC",
                "why_it_is_dead": (
                    "Both v23-wide-trail and v24-no-trail failed real QuantConnect promotion."
                ),
            },
        ],
        "frontier_summary": {
            "only_viable_unbanned_path": "conservative_timegated_breakeven",
            "recommended_local_proxy": conservative_local["label"],
            "recommended_qc_proxy_lane": "<=180min + MFE>$25",
            "hard_blocker": (
                None
                if accepted_available
                else "Accepted v18 baseline raw trades are missing from the current workspace, so "
                "fresh QC-proxy revalidation cannot yet be rerun."
            ),
            "next_real_iteration_method": (
                [
                    "Use the restored accepted v18 rerun bundle to rerun QC-proxy validation.",
                    "Because the refreshed QC-proxy result still aligns with the historical zero-clipping lane, launch the next QC candidate from BE=1.25x_gate=180min.",
                    "Wait for the real QuantConnect rerun and promote only if it clears the full promotion gate."
                ]
                if accepted_available
                else [
                    "Restore an accepted v18 baseline rerun bundle into QuantConnect results/2017-2026.",
                    "Rerun QC-proxy validation for the conservative time-gated breakeven lane on the restored accepted trades.",
                    "If the restored QC-proxy result still aligns with the historical zero-clipping lane, launch the next QC candidate from BE=1.25x_gate=180min."
                ]
            ),
            "stop_doing": [
                "Do not open new entry-filter families.",
                "Do not relaunch v23-wide-trail or v24-no-trail.",
                "Do not relaunch from the tested local_orb_time_decay_trail grid.",
                "Do not revive v20-tight-trail-early.",
                "Do not spend more rounds on random new mechanisms until the conservative time-gated breakeven lane is resolved."
            ],
        },
    }
    return frontier


def main() -> None:
    frontier = build_frontier()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(frontier, indent=2), encoding="utf-8")
    print(f"Wrote exit frontier summary to {OUTPUT_PATH}")
    print(
        "Recommended next method: "
        f"{frontier['frontier_summary']['only_viable_unbanned_path']}"
    )


if __name__ == "__main__":
    main()
