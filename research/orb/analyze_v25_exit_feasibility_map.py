#!/usr/bin/env python3
"""
Consolidate post-v25 exit-side feasibility after the first three research branches.

Goal:
- stop blind local-only threshold exploration
- compare the residual branches on a common frame
- decide whether the bottleneck is "wrong mechanism" or "insufficient evidence"
"""
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "qc_regime_prototypes"
OUTPUT_PATH = RESULTS_DIR / "v25_exit_feasibility_map.json"
ANALYSIS_VERSION = "v1_post_v25_exit_feasibility"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_branch(name: str, payload: dict) -> dict:
    best_positive = payload["candidate_summary"].get("best_positive")
    best_zero_clip = payload["candidate_summary"].get("best_zero_clip_positive")
    best_balanced = payload["candidate_summary"].get("best_balanced")

    def fold_score(row: dict | None) -> int | None:
        if row is None:
            return None
        return int(row["improved_folds"])

    best_local_total_delta = best_positive["pnl_delta"] if best_positive else None
    best_zero_clip_delta = best_zero_clip["pnl_delta"] if best_zero_clip else None

    strongest = best_positive or best_zero_clip or best_balanced
    if strongest is None:
        verdict = "NO_POSITIVE_SIGNAL"
    elif int(strongest["improved_folds"]) >= 3:
        verdict = "LAUNCHABLE_LOCAL_SIGNAL"
    elif int(strongest["improved_folds"]) >= 1:
        verdict = "LOCAL_NEAR_MISS"
    else:
        verdict = "LOCAL_REJECTED"

    return {
        "branch": name,
        "research_scope": payload["research_scope"],
        "analysis_version": payload.get("analysis_version"),
        "best_positive": best_positive,
        "best_zero_clip_positive": best_zero_clip,
        "best_balanced": best_balanced,
        "best_positive_improved_folds": fold_score(best_positive),
        "best_zero_clip_improved_folds": fold_score(best_zero_clip),
        "best_local_total_delta": best_local_total_delta,
        "best_zero_clip_delta": best_zero_clip_delta,
        "verdict": verdict,
    }


def main() -> int:
    profit_lock = load_json(RESULTS_DIR / "local_orb_v25_profit_lock.json")
    peak_giveback = load_json(RESULTS_DIR / "local_orb_v25_peak_giveback.json")
    partial_scaleout = load_json(RESULTS_DIR / "local_orb_v25_partial_scaleout.json")

    branches = [
        summarize_branch("persistent_profit_lock", profit_lock),
        summarize_branch("late_peak_giveback", peak_giveback),
        summarize_branch("partial_scaleout", partial_scaleout),
    ]

    strongest_local_total = max(
        [b for b in branches if b["best_local_total_delta"] is not None],
        key=lambda b: b["best_local_total_delta"],
        default=None,
    )
    strongest_zero_clip = max(
        [b for b in branches if b["best_zero_clip_delta"] is not None],
        key=lambda b: b["best_zero_clip_delta"],
        default=None,
    )
    strongest_fold_support = max(
        branches,
        key=lambda b: max(
            x for x in [
                b["best_positive_improved_folds"],
                b["best_zero_clip_improved_folds"],
                (b["best_balanced"] or {}).get("improved_folds") if b["best_balanced"] else None,
                -1,
            ] if x is not None
        ),
    )

    payload = {
        "research_scope": "v25_exit_feasibility_map",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v25-timegated-be",
        "branches": branches,
        "frontier": {
            "strongest_local_total_signal": {
                "branch": strongest_local_total["branch"],
                "label": strongest_local_total["best_positive"]["label"],
                "pnl_delta": strongest_local_total["best_positive"]["pnl_delta"],
                "improved_folds": strongest_local_total["best_positive"]["improved_folds"],
            } if strongest_local_total and strongest_local_total["best_positive"] else None,
            "strongest_zero_clip_signal": {
                "branch": strongest_zero_clip["branch"],
                "label": strongest_zero_clip["best_zero_clip_positive"]["label"],
                "pnl_delta": strongest_zero_clip["best_zero_clip_positive"]["pnl_delta"],
                "improved_folds": strongest_zero_clip["best_zero_clip_positive"]["improved_folds"],
            } if strongest_zero_clip and strongest_zero_clip["best_zero_clip_positive"] else None,
            "strongest_fold_support": {
                "branch": strongest_fold_support["branch"],
                "best_positive_improved_folds": strongest_fold_support["best_positive_improved_folds"],
                "best_zero_clip_improved_folds": strongest_fold_support["best_zero_clip_improved_folds"],
            },
        },
        "structural_conclusion": {
            "local_proxy_trade_count": 41,
            "core_observation": (
                "Post-v25 exit branches can still produce small full-sample uplifts locally, "
                "but none of the first three materially new branches improved even 2/4 walk-forward folds."
            ),
            "interpretation": (
                "The bottleneck now looks less like 'we have not tried enough threshold grids' "
                "and more like 'the remaining residual edge is too subtle for the current short local path sample'."
            ),
            "recommended_research_posture": (
                "Do not launch any local-only post-v25 exit candidate yet. If iteration continues, "
                "either gather stronger path-level evidence or build a QC-native evaluator for the next materially new branch."
            ),
            "do_not_repeat": [
                "naive persistent profit-lock thresholds",
                "naive late peak-giveback caps",
                "another local-only threshold grid with the same evidence weakness",
            ],
        },
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved results to {OUTPUT_PATH}")
    print(json.dumps(payload["frontier"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
