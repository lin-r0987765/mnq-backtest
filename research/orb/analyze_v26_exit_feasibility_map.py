#!/usr/bin/env python3
"""
Consolidate post-v26 exit-side feasibility after the first two hybrid branches.

Goal:
- compare the current post-v26 ideas on a single frame
- distinguish stronger local signal from weaker refinements
- stop re-testing nearby late-session washout branches without new evidence
"""
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "qc_regime_prototypes"
OUTPUT_PATH = RESULTS_DIR / "v26_exit_feasibility_map.json"
ANALYSIS_VERSION = "v1_post_v26_exit_feasibility"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_branch(name: str, payload: dict) -> dict:
    summary = payload["candidate_summary"]
    best_positive = summary.get("best_positive")
    best_zero_clip = summary.get("best_zero_clip_positive")
    best_balanced = summary.get("best_balanced")

    def fold_score(row: dict | None) -> int | None:
        if row is None:
            return None
        return int(row["improved_folds"])

    strongest = best_balanced or best_zero_clip or best_positive
    if strongest is None:
        verdict = "LOCAL_REJECTED"
    elif int(strongest["improved_folds"]) >= 3:
        verdict = "READY_FOR_QC_PROXY"
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
        "best_local_total_delta": best_positive["pnl_delta"] if best_positive else None,
        "best_zero_clip_delta": best_zero_clip["pnl_delta"] if best_zero_clip else None,
        "verdict": verdict,
    }


def main() -> int:
    stagnation_timeout = load_json(RESULTS_DIR / "local_orb_v26_stagnation_exit.json")
    stall_giveback = load_json(RESULTS_DIR / "local_orb_v26_stall_giveback_exit.json")

    branches = [
        summarize_branch("post_lock_stagnation_timeout", stagnation_timeout),
        summarize_branch("post_lock_stall_plus_giveback", stall_giveback),
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
            x
            for x in [
                b["best_positive_improved_folds"],
                b["best_zero_clip_improved_folds"],
                (b["best_balanced"] or {}).get("improved_folds") if b["best_balanced"] else None,
                -1,
            ]
            if x is not None
        ),
    )

    payload = {
        "research_scope": "v26_exit_feasibility_map",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v26-profit-lock",
        "branches": branches,
        "frontier": {
            "strongest_local_total_signal": {
                "branch": strongest_local_total["branch"],
                "label": strongest_local_total["best_positive"]["label"],
                "pnl_delta": strongest_local_total["best_positive"]["pnl_delta"],
                "improved_folds": strongest_local_total["best_positive"]["improved_folds"],
            }
            if strongest_local_total and strongest_local_total["best_positive"]
            else None,
            "strongest_zero_clip_signal": {
                "branch": strongest_zero_clip["branch"],
                "label": strongest_zero_clip["best_zero_clip_positive"]["label"],
                "pnl_delta": strongest_zero_clip["best_zero_clip_positive"]["pnl_delta"],
                "improved_folds": strongest_zero_clip["best_zero_clip_positive"]["improved_folds"],
            }
            if strongest_zero_clip and strongest_zero_clip["best_zero_clip_positive"]
            else None,
            "strongest_fold_support": {
                "branch": strongest_fold_support["branch"],
                "best_positive_improved_folds": strongest_fold_support["best_positive_improved_folds"],
                "best_zero_clip_improved_folds": strongest_fold_support["best_zero_clip_improved_folds"],
            },
        },
        "structural_conclusion": {
            "core_observation": (
                "The first post-v26 branch (`pure stagnation-timeout`) remains stronger than the more selective "
                "`stall-plus-giveback` refinement, but neither branch improved more than 1/4 walk-forward folds."
            ),
            "interpretation": (
                "That suggests the remaining late-session washout edge is real but still too subtle for local-only "
                "post-lock exit gating. Adding more selectivity reduced clipping, but also reduced the already-thin signal."
            ),
            "recommended_research_posture": (
                "Do not launch any local-only post-v26 exit candidate yet. If iteration continues, prefer a QC-native "
                "evaluator or a genuinely different mechanism instead of more nearby stall/giveback threshold grids."
            ),
            "do_not_repeat": [
                "pure post-lock stagnation-timeout thresholds from local-only evidence",
                "post-lock stall-plus-giveback thresholds from local-only evidence",
                "another nearby late-session washout threshold grid without stronger evidence",
            ],
        },
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved results to {OUTPUT_PATH}")
    print(json.dumps(payload["frontier"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
