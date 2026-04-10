from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGETS = [
    "mom4_positive",
    "mom5_positive",
    "close_above_sma8",
    "close_above_sma9",
    "close_above_sma7",
]


def score_candidate(name: str, payload: dict[str, object]) -> dict[str, object]:
    kept = payload["kept"]
    excluded = payload["excluded"]
    walk = payload["walkforward"]["summary"]
    loo = payload["leave_one_month_out"]

    score = 0.0
    score += float(kept["net_pnl"]) - float(excluded["net_pnl"])
    score += float(walk["delta_vs_baseline"]) * 2.0
    score += float(walk["improved_vs_baseline_folds"]) * 10.0
    score += float(walk["positive_kept_folds"]) * 4.0
    score += float(walk["excluded_negative_folds"]) * 4.0
    score += float(loo["min_net_pnl"]) * 0.5

    return {
        "label": name,
        "score": round(score, 2),
        "kept_net_pnl": kept["net_pnl"],
        "excluded_net_pnl": excluded["net_pnl"],
        "walkforward_delta_vs_baseline": walk["delta_vs_baseline"],
        "improved_vs_baseline_folds": walk["improved_vs_baseline_folds"],
        "positive_kept_folds": walk["positive_kept_folds"],
        "excluded_negative_folds": walk["excluded_negative_folds"],
        "leave_one_month_out_all_positive": loo["all_positive"],
        "leave_one_month_out_min_net_pnl": loo["min_net_pnl"],
        "verdict": payload["verdict"],
    }


def evaluate(scan_json: Path) -> dict[str, object]:
    obj = json.loads(scan_json.read_text(encoding="utf-8"))
    candidates = obj["candidates"]

    rows = [score_candidate(name, candidates[name]) for name in TARGETS if name in candidates]
    ranked = sorted(
        rows,
        key=lambda row: (
            row["verdict"] == "LOCAL_RESEARCH_LEADER",
            row["leave_one_month_out_all_positive"],
            row["score"],
        ),
        reverse=True,
    )

    return {
        "research_scope": "local_v18_priority_tiebreak",
        "analysis_version": "v1_target_priority_after_adjacent_scan",
        "source_scan_json": scan_json.name,
        "ranked_candidates": ranked,
        "priority_summary": {
            "next_qc_proxy_priority": ranked[0]["label"] if ranked else None,
            "secondary_priority": ranked[1]["label"] if len(ranked) > 1 else None,
            "tertiary_priority": ranked[2]["label"] if len(ranked) > 2 else None,
            "interpretation": (
                "This tiebreak is local-only. It exists to order the next QC-proxy targets once accepted baseline "
                "raw trades are restored, not to launch a new QC candidate directly."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank local-only v18 exploratory leads for the next QC proxy pass.")
    parser.add_argument(
        "--scan-json",
        default="results/qc_regime_prototypes/local_orb_v18_adjacent_trend_scan.json",
    )
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    result = evaluate(Path(args.scan_json))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_v18_priority_tiebreak.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
