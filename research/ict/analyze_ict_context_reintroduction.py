from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_research_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_context_reintroduction.json"
)


def _relaxed_overrides(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "use_kill_zones": False,
        "use_daily_bias_filter": False,
        "use_premium_discount_filter": False,
        "use_external_liquidity_filter": False,
        "use_amd_filter": False,
        "use_macro_timing_windows": False,
        "use_prev_session_anchor_filter": False,
        "use_session_array_refinement": False,
    }
    if extra:
        overrides.update(extra)
    return overrides


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "full_stack_smt": {"enable_smt": True, "overrides": {}},
        "context_relaxed_bundle": {"enable_smt": True, "overrides": _relaxed_overrides()},
        "reintro_daily_bias": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_daily_bias_filter": True}),
        },
        "reintro_premium_discount": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_premium_discount_filter": True}),
        },
        "reintro_prev_session_anchor": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_prev_session_anchor_filter": True}),
        },
        "reintro_macro_timing": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_macro_timing_windows": True}),
        },
        "reintro_session_array": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_session_array_refinement": True}),
        },
        "reintro_amd": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_amd_filter": True}),
        },
        "reintro_kill_zones": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_kill_zones": True}),
        },
        "reintro_external_liquidity": {
            "enable_smt": True,
            "overrides": _relaxed_overrides({"use_external_liquidity_filter": True}),
        },
        "reintro_context_core": {
            "enable_smt": True,
            "overrides": _relaxed_overrides(
                {
                    "use_daily_bias_filter": True,
                    "use_premium_discount_filter": True,
                    "use_prev_session_anchor_filter": True,
                }
            ),
        },
        "reintro_timing_bundle": {
            "enable_smt": True,
            "overrides": _relaxed_overrides(
                {
                    "use_kill_zones": True,
                    "use_macro_timing_windows": True,
                    "use_session_array_refinement": True,
                }
            ),
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]], relaxed_trades: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        trades = int(metrics["total_trades"])
        retention = (trades / relaxed_trades) if relaxed_trades > 0 else 0.0
        ranked.append(
            {
                "label": label,
                "total_trades": trades,
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "trade_retention_vs_relaxed": retention,
                "daily_bias_filtered_setups": int(payload["metadata"].get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(payload["metadata"].get("premium_discount_filtered_setups", 0)),
                "amd_filtered_setups": int(payload["metadata"].get("amd_filtered_setups", 0)),
                "prev_session_anchor_filtered_setups": int(payload["metadata"].get("prev_session_anchor_filtered_setups", 0)),
                "macro_timing_filtered_sweeps": int(payload["metadata"].get("macro_timing_filtered_sweeps", 0)),
                "session_array_filtered_shifts": int(payload["metadata"].get("session_array_filtered_shifts", 0)),
                "external_liquidity_filtered_sweeps": int(payload["metadata"].get("external_liquidity_filtered_sweeps", 0)),
                "kill_zone_filtered_sweeps": int(payload["metadata"].get("kill_zone_filtered_sweeps", 0)),
            }
        )

    ranked.sort(
        key=lambda row: (
            row["label"] != "full_stack_smt",
            row["total_trades"] > 0,
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["trade_retention_vs_relaxed"],
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reintroduce ICT context filters one cluster at a time on broader paired QQQ+SPY data."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(
        peer_symbol=args.peer_symbol,
        peer_csv=args.peer_csv,
        period=args.period,
    )
    merged = merge_peer_columns(intraday_df, peer_df)

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    variants: dict[str, dict[str, Any]] = {}
    for label, spec in _variant_specs().items():
        params = build_ict_research_profile_params(
            enable_smt=bool(spec["enable_smt"]),
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "use_smt_filter": bool(params["use_smt_filter"]),
                **spec["overrides"],
            },
            "metrics": result.metrics,
            "metadata": {
                "daily_bias_filtered_setups": int(signals.metadata.get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(signals.metadata.get("premium_discount_filtered_setups", 0)),
                "amd_filtered_setups": int(signals.metadata.get("amd_filtered_setups", 0)),
                "prev_session_anchor_filtered_setups": int(
                    signals.metadata.get("prev_session_anchor_filtered_setups", 0)
                ),
                "macro_timing_filtered_sweeps": int(signals.metadata.get("macro_timing_filtered_sweeps", 0)),
                "session_array_filtered_shifts": int(
                    signals.metadata.get("session_array_filtered_shifts", 0)
                ),
                "external_liquidity_filtered_sweeps": int(
                    signals.metadata.get("external_liquidity_filtered_sweeps", 0)
                ),
                "kill_zone_filtered_sweeps": int(signals.metadata.get("kill_zone_filtered_sweeps", 0)),
                "smt_confirmed_sweeps": int(signals.metadata.get("smt_confirmed_sweeps", 0)),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
            },
        }

    relaxed_trades = int(variants["context_relaxed_bundle"]["metrics"]["total_trades"])
    ranked = _rank_variants(variants, relaxed_trades=relaxed_trades)
    best = ranked[0] if ranked else None

    best_non_relaxed = next(
        (row for row in ranked if row["label"] not in {"context_relaxed_bundle", "full_stack_smt"}),
        None,
    )
    robust_trade_floor = max(3, int(round(relaxed_trades * 0.5)))
    best_robust_variant = next(
        (
            row
            for row in ranked
            if row["label"] not in {"context_relaxed_bundle", "full_stack_smt"}
            and row["total_trades"] >= robust_trade_floor
            and row["trade_retention_vs_relaxed"] >= 0.5
        ),
        None,
    )

    verdict = "NO_ACTIVE_REINTRODUCTION_FOUND"
    interpretation = (
        "No context filter or filter cluster could be reintroduced without collapsing the broader paired ICT lane back toward inactivity. The next step should focus on sweep/SMT geometry calibration rather than adding filters back."
    )
    if best_robust_variant and best_robust_variant["total_return_pct"] > 0:
        verdict = "ROBUST_CONTEXT_REINTRODUCTION_SURVIVOR_IDENTIFIED"
        interpretation = (
            "At least one context filter can be reintroduced on the broader paired lane without sacrificing activity. The next step should focus on the robust survivors before restoring the heavier parts of the full stack."
        )
    elif best_non_relaxed and best_non_relaxed["total_trades"] > 0:
        if best_non_relaxed["total_return_pct"] > 0:
            verdict = "ONLY_SPARSE_CONTEXT_REINTRODUCTION_SURVIVOR_IDENTIFIED"
            interpretation = (
                "A positive reintroduction exists, but only as a sparse survivor with very low trade retention. The next step should prioritize robust filter survivors or geometry calibration, not a direct jump back toward the full stack."
            )
        else:
            verdict = "REINTRODUCTION_REMAINS_ACTIVE_BUT_QUALITY_DEGRADES"
            interpretation = (
                "Context filters can be reintroduced without driving trades to zero, but the surviving profiles degrade return quality. The next step should focus on geometry/SMT calibration before restoring more context."
            )

    output = {
        "analysis": "ict_context_reintroduction",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "variant_ranking": ranked,
        "best_variant": best,
        "best_non_relaxed_variant": best_non_relaxed,
        "best_robust_variant": best_robust_variant,
        "relaxed_reference_label": "context_relaxed_bundle",
        "relaxed_reference_trades": relaxed_trades,
        "robust_trade_floor": robust_trade_floor,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
