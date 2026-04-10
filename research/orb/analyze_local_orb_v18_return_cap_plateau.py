from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.orb.analyze_local_orb_v18_return_cap import evaluate


DEFAULT_RETURN_CAPS = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025]


def round_cap(cap: float) -> float:
    return round(float(cap), 6)


def build_rankings(result: dict[str, object]) -> list[dict[str, object]]:
    baseline_net = float(result["baseline"]["net_pnl"])
    rows: list[dict[str, object]] = []

    for label, candidate in result["candidates"].items():
        kept = candidate["kept"]
        excluded = candidate["excluded"]
        walk = candidate["walkforward"]["summary"]
        loo = candidate["leave_one_month_out"]

        row = {
            "label": label,
            "return_cap": round_cap(candidate["return_cap"]),
            "verdict": candidate["verdict"],
            "kept_trades": int(kept["trades"]),
            "kept_net_pnl": round(float(kept["net_pnl"]), 2),
            "excluded_net_pnl": round(float(excluded["net_pnl"]), 2),
            "delta_vs_baseline_full": round(float(kept["net_pnl"] - baseline_net), 2),
            "walkforward_delta": round(float(walk["delta_vs_baseline"]), 2),
            "improved_vs_baseline_folds": int(walk["improved_vs_baseline_folds"]),
            "positive_kept_folds": int(walk["positive_kept_folds"]),
            "excluded_negative_folds": int(walk["excluded_negative_folds"]),
            "leave_one_month_out_all_positive": bool(loo["all_positive"]),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["walkforward_delta"],
            row["delta_vs_baseline_full"],
            row["kept_net_pnl"],
        ),
        reverse=True,
    )
    return rows


def is_plateau_eligible(row: dict[str, object]) -> bool:
    return (
        row["kept_net_pnl"] > 0
        and row["excluded_net_pnl"] < 0
        and row["delta_vs_baseline_full"] > 0
        and row["walkforward_delta"] > 0
        and row["improved_vs_baseline_folds"] >= 2
        and row["positive_kept_folds"] >= 3
        and row["leave_one_month_out_all_positive"]
    )


def extract_plateau(rankings: list[dict[str, object]], *, step: float) -> dict[str, object]:
    eligible = sorted((row for row in rankings if is_plateau_eligible(row)), key=lambda row: row["return_cap"])
    if not eligible:
        return {
            "eligible_caps": [],
            "best_cap": None,
            "stable_plateau_caps": [],
            "stable_plateau_range": None,
            "plateau_label": "NO_STABLE_PLATEAU",
        }

    best = max(
        eligible,
        key=lambda row: (
            row["walkforward_delta"],
            row["delta_vs_baseline_full"],
            row["kept_net_pnl"],
        ),
    )

    best_idx = next(index for index, row in enumerate(eligible) if row["return_cap"] == best["return_cap"])
    left = best_idx
    right = best_idx

    while left > 0 and abs(eligible[left]["return_cap"] - eligible[left - 1]["return_cap"] - step) < 1e-9:
        left -= 1
    while right < len(eligible) - 1 and abs(eligible[right + 1]["return_cap"] - eligible[right]["return_cap"] - step) < 1e-9:
        right += 1

    plateau = eligible[left : right + 1]
    return {
        "eligible_caps": [row["return_cap"] for row in eligible],
        "best_cap": best["return_cap"],
        "best_label": best["label"],
        "stable_plateau_caps": [row["return_cap"] for row in plateau],
        "stable_plateau_range": {
            "lower": plateau[0]["return_cap"],
            "upper": plateau[-1]["return_cap"],
        },
        "plateau_label": "STABLE_LOCAL_PLATEAU" if len(plateau) >= 2 else "SINGLE_POINT_LOCAL_WINNER",
    }


def summarize(result: dict[str, object], *, step: float) -> dict[str, object]:
    rankings = build_rankings(result)
    plateau = extract_plateau(rankings, step=step)
    top_rows = rankings[:5]

    interpretation = (
        "This follow-up tests whether the prev-day-return-cap idea is a single-point optimum or a stable local plateau. "
        "A stable plateau is stronger evidence than a lone best threshold, but it is still local-only and cannot launch a "
        "new QC candidate while accepted baseline QC raw trades remain unavailable."
    )

    return {
        "research_scope": "local_v18_return_cap_plateau",
        "analysis_version": "v1_plateau_confirmation",
        "step_size": round_cap(step),
        "base_result_file": "results/qc_regime_prototypes/local_orb_v18_return_cap.json",
        "baseline": result["baseline"],
        "rankings": rankings,
        "plateau_summary": plateau,
        "top_candidates": top_rows,
        "interpretation": interpretation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirm whether v18 return-cap winners form a stable local plateau.")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--return-caps", default="0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    return_caps = [float(item.strip()) for item in args.return_caps.split(",") if item.strip()]
    if len(return_caps) < 2:
        raise ValueError("Need at least two return caps to evaluate plateau stability.")

    ordered_caps = sorted(return_caps)
    step = min(round_cap(b - a) for a, b in zip(ordered_caps, ordered_caps[1:]))

    result = evaluate(
        intraday_csv=Path(args.intraday_csv),
        daily_csv=Path(args.daily_csv),
        walk_forward_json=Path(args.walk_forward_json),
        return_caps=ordered_caps,
    )
    summary = summarize(result, step=step)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_orb_v18_return_cap_plateau.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
