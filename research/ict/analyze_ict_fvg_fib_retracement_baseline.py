from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestEngine
from src.data.fetcher import load_ohlcv_csv
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_fvg_fib_retracement_research_profile_params,
)


DEFAULT_INTRADAY = PROJECT_ROOT / "regularized" / "QQQ" / "qqq_5m_regular_et.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_fvg_fib_retracement_baseline.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "fvg_fib_050_079": {"overrides": {}},
        "fvg_only_control": {
            "overrides": {
                "require_ote_zone": False,
            }
        },
        "fvg_fib_062_079": {
            "overrides": {
                "ote_fib_low": 0.618,
                "ote_fib_high": 0.79,
            }
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        ranked.append(
            {
                "label": label,
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
                "fvg_entries": int(payload["metadata"].get("fvg_entries", 0)),
                "ote_required_filtered_shifts": int(
                    payload["metadata"].get("ote_required_filtered_shifts", 0)
                ),
                "fvg_required_filtered_shifts": int(
                    payload["metadata"].get("fvg_required_filtered_shifts", 0)
                ),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["total_trades"] > 0,
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["total_trades"],
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Research the simplified FVG + Fibonacci retracement lane. "
            "This replay only allows FVG delivery arrays and can hard-require "
            "the FVG to sit inside a fib retracement window."
        )
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    engine = BacktestEngine(
        initial_cash=10_000.0,
        fees_pct=0.0005,
        position_size_mode="capital_pct",
        capital_usage_pct=1.0,
        min_shares=0,
    )

    variants: dict[str, dict[str, Any]] = {}
    for label, spec in _variant_specs().items():
        params = build_ict_fvg_fib_retracement_research_profile_params(
            enable_smt=False,
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, intraday_df)
        signals = strategy.generate_signals(intraday_df)
        variants[label] = {
            "params": {
                "require_fvg_delivery": bool(params["require_fvg_delivery"]),
                "require_ote_zone": bool(params["require_ote_zone"]),
                "ote_fib_low": float(params["ote_fib_low"]),
                "ote_fib_high": float(params["ote_fib_high"]),
                "min_score_to_trade": float(params["min_score_to_trade"]),
                "structure_reference_mode": str(params["structure_reference_mode"]),
                "swing_threshold": int(params["swing_threshold"]),
                "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            },
            "metrics": result.metrics,
            "metadata": {
                "bullish_sweeps": int(signals.metadata.get("bullish_sweeps", 0)),
                "bearish_sweeps": int(signals.metadata.get("bearish_sweeps", 0)),
                "bullish_shifts": int(signals.metadata.get("bullish_shifts", 0)),
                "bearish_shifts": int(signals.metadata.get("bearish_shifts", 0)),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
                "fvg_required_filtered_shifts": int(
                    signals.metadata.get("fvg_required_filtered_shifts", 0)
                ),
                "ote_required_filtered_shifts": int(
                    signals.metadata.get("ote_required_filtered_shifts", 0)
                ),
                "rr_filtered_entries": int(signals.metadata.get("rr_filtered_entries", 0)),
            },
        }

    ranked = _rank_variants(variants)
    best = ranked[0] if ranked else None
    baseline_label = "fvg_fib_050_079"
    baseline = variants[baseline_label]

    interpretation = (
        "The simplified FVG + fib retracement study isolates whether hard-requiring "
        "the delivery FVG to sit inside the requested retracement window helps or "
        "hurts relative to a pure FVG-only control."
    )

    output = {
        "analysis": "ict_fvg_fib_retracement_baseline",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "period": (
            f"{intraday_df.index.min()} -> {intraday_df.index.max()}"
            if not intraday_df.empty
            else "empty"
        ),
        "rows_primary": len(intraday_df),
        "risk_standard": {
            "initial_cash": 10_000.0,
            "fees_pct": 0.0005,
            "position_size_mode": "capital_pct",
            "capital_usage_pct": 1.0,
            "min_shares": 0,
            "reward_risk_gate": ">= 1.5:1",
        },
        "variants": variants,
        "variant_ranking": ranked,
        "baseline_label": baseline_label,
        "baseline_metrics": baseline["metrics"],
        "best_variant": best,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
