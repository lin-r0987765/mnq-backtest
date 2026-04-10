from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_ict_frontier_funnel import _build_funnel_summary
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_reentry_repair.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "active_lite_frontier_base": {"max_reentries_per_setup": 0},
        "reentry_1": {"max_reentries_per_setup": 1},
        "reentry_2": {"max_reentries_per_setup": 2},
    }


def _run_variant(merged, *, label: str, params: dict[str, Any], engine: BacktestEngine) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    return {
        "label": label,
        "params": {
            "use_smt_filter": bool(params["use_smt_filter"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "max_reentries_per_setup": int(params["max_reentries_per_setup"]),
        },
        "metrics": result.metrics,
        "funnel": _build_funnel_summary(signals.metadata),
        "metadata": {
            "reentry_enabled": bool(signals.metadata.get("reentry_enabled", False)),
            "reentry_entries": int(signals.metadata.get("reentry_entries", 0)),
            "reentry_stop_rearms": int(signals.metadata.get("reentry_stop_rearms", 0)),
            "reentry_exhausted_setups": int(signals.metadata.get("reentry_exhausted_setups", 0)),
            "bullish_entries": int(signals.metadata.get("bullish_entries", 0)),
            "bearish_entries": int(signals.metadata.get("bearish_entries", 0)),
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]], base_trades: int, base_return: float) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        ranked.append(
            {
                "label": label,
                "max_reentries_per_setup": int(payload["params"]["max_reentries_per_setup"]),
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "trade_gain_vs_base": int(metrics["total_trades"]) - base_trades,
                "return_gain_vs_base": float(metrics["total_return_pct"]) - base_return,
                "reentry_entries": int(payload["metadata"]["reentry_entries"]),
                "reentry_stop_rearms": int(payload["metadata"]["reentry_stop_rearms"]),
                "reentry_exhausted_setups": int(payload["metadata"]["reentry_exhausted_setups"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["total_return_pct"] > 0,
            row["total_trades"],
            row["total_return_pct"],
            row["profit_factor"],
            -row["max_reentries_per_setup"],
        ),
        reverse=True,
    )
    return ranked


def _verdict(
    *,
    base_label: str,
    base_trades: int,
    base_return: float,
    best_non_base: dict[str, Any] | None,
    best_positive_density_variant: dict[str, Any] | None,
    best_robust_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    if best_robust_variant is not None:
        return (
            "ROBUST_LITE_REENTRY_EXTENSION_IDENTIFIED",
            "Allowing same-zone re-entry improves both trade count and return on the active lite frontier.",
        )
    if best_non_base is None:
        return (
            "LITE_REENTRY_NO_VARIANTS",
            "No re-entry variants were produced on the active lite frontier.",
        )
    if (
        best_non_base["total_trades"] == base_trades
        and abs(best_non_base["total_return_pct"] - base_return) < 1e-9
        and best_non_base["label"] != base_label
    ):
        return (
            "LITE_REENTRY_PLATEAU_ON_ACTIVE_FRONTIER",
            "Same-zone re-entry rearms stop-outs on the active lite frontier, but realized trades and return stay unchanged.",
        )
    if best_positive_density_variant is not None and best_positive_density_variant["total_trades"] > base_trades:
        return (
            "LITE_REENTRY_DENSITY_EXTENSION_ONLY",
            "Same-zone re-entry improves density, but the added activity is not strong enough to replace the active lite frontier.",
        )
    if best_non_base["total_return_pct"] > 0:
        return (
            "LITE_REENTRY_SURVIVOR_BUT_NOT_EXTENSION",
            "Same-zone re-entry stays positive, but it does not improve the active lite frontier enough to promote.",
        )
    return (
        "LITE_REENTRY_REJECTED",
        "Same-zone re-entry does not produce a promotable improvement on the active lite frontier.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay same-zone re-entry variants on the active lite ICT frontier."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    variants: dict[str, dict[str, Any]] = {}
    for label, overrides in _variant_specs().items():
        params = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
        variants[label] = _run_variant(merged, label=label, params=params, engine=engine)

    base_label = "active_lite_frontier_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_return = float(variants[base_label]["metrics"]["total_return_pct"])
    ranked = _rank_variants(variants, base_trades=base_trades, base_return=base_return)
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
    best_positive_density_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] > base_trades
            and row["total_return_pct"] > 0
        ),
        None,
    )
    best_robust_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] >= base_trades
            and row["total_return_pct"] > base_return
        ),
        None,
    )
    verdict, interpretation = _verdict(
        base_label=base_label,
        base_trades=base_trades,
        base_return=base_return,
        best_non_base=best_non_base,
        best_positive_density_variant=best_positive_density_variant,
        best_robust_variant=best_robust_variant,
    )

    output = {
        "analysis": "ict_lite_reentry_repair",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "variant_ranking": ranked,
        "base_label": base_label,
        "base_trades": base_trades,
        "base_total_return_pct": base_return,
        "best_non_base_variant": best_non_base,
        "best_positive_density_variant": best_positive_density_variant,
        "best_robust_variant": best_robust_variant,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
