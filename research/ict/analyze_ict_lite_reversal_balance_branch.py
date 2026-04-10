from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_qualified_continuation_density_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_profile_params,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_lite_reversal_balance_branch.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "active_lite_quality_base": {
            "builder": "quality_base",
            "overrides": {},
        },
        "qualified_reversal_balance": {
            "builder": "qualified_reversal_balance",
            "overrides": {},
        },
        "qualified_continuation_density": {
            "builder": "qualified_continuation_density",
            "overrides": {},
        },
    }


def _build_params(builder_label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if builder_label == "qualified_reversal_balance":
        return build_ict_lite_reversal_qualified_reversal_balance_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
    if builder_label == "qualified_continuation_density":
        return build_ict_lite_reversal_qualified_continuation_density_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
    return build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
        enable_smt=True,
        overrides=dict(overrides),
    )


def _run_variant(merged, *, label: str, params: dict[str, Any], engine: BacktestEngine) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    return {
        "label": label,
        "params": {
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "slow_recovery_enabled": bool(params["slow_recovery_enabled"]),
            "slow_recovery_bars": int(params["slow_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "take_profit_rr": float(params["take_profit_rr"]),
            "enable_continuation_entry": bool(params["enable_continuation_entry"]),
            "smt_threshold": float(params["smt_threshold"]),
        },
        "metrics": result.metrics,
        "metadata": {
            key: int(signals.metadata.get(key, 0))
            for key in (
                "continuation_entries",
                "continuation_zone_refreshes",
                "slow_recovery_entries",
                "fast_recovery_entries",
                "fvg_delay_filtered_retests",
                "fvg_depth_filtered_retests",
                "bullish_entries",
                "bearish_entries",
            )
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        metadata = payload["metadata"]
        ranked.append(
            {
                "label": label,
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
                "structure_lookback": int(payload["params"]["structure_lookback"]),
                "slow_recovery_bars": int(payload["params"]["slow_recovery_bars"]),
                "fvg_min_gap_pct": float(payload["params"]["fvg_min_gap_pct"]),
                "fvg_revisit_depth_ratio": float(payload["params"]["fvg_revisit_depth_ratio"]),
                "fvg_revisit_min_delay_bars": int(payload["params"]["fvg_revisit_min_delay_bars"]),
                "take_profit_rr": float(payload["params"]["take_profit_rr"]),
                "enable_continuation_entry": bool(payload["params"]["enable_continuation_entry"]),
                "continuation_entries": int(metadata["continuation_entries"]),
                "slow_recovery_entries": int(metadata["slow_recovery_entries"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["total_trades"],
        ),
        reverse=True,
    )
    return ranked


def _verdict(
    *,
    quality_base: dict[str, Any],
    reversal_balance: dict[str, Any],
    continuation_density: dict[str, Any],
) -> tuple[str, str]:
    if (
        reversal_balance["total_return_pct"] > 0
        and reversal_balance["total_trades"] > quality_base["total_trades"]
        and reversal_balance["total_return_pct"] > continuation_density["total_return_pct"]
        and reversal_balance["profit_factor"] > continuation_density["profit_factor"]
    ):
        return (
            "QUALIFIED_REVERSAL_BALANCE_BRANCH_IDENTIFIED",
            "A pure reversal-FVG branch materially improves trade count over the sparse quality lane while keeping much stronger return quality than the denser continuation lane.",
        )
    if reversal_balance["total_return_pct"] > 0 and reversal_balance["total_trades"] > quality_base["total_trades"]:
        return (
            "QUALIFIED_REVERSAL_BALANCE_SURVIVOR_ONLY",
            "A pure reversal-FVG branch improves trade count over the sparse quality lane, but it does not yet outperform the denser continuation lane on return quality.",
        )
    return (
        "QUALIFIED_REVERSAL_BALANCE_REJECTED",
        "The pure reversal-FVG balance branch fails to improve the sparse quality lane in a meaningful way.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the sparse quality ICT lane, a new pure reversal balance lane, and the current dense continuation lane."
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
    engine = BacktestEngine(
        initial_cash=100_000.0,
        fees_pct=0.0005,
        position_size_mode="capital_pct",
        capital_usage_pct=1.0,
        min_shares=40,
    )

    variants: dict[str, dict[str, Any]] = {}
    for label, spec in _variant_specs().items():
        params = _build_params(spec["builder"], spec["overrides"])
        variants[label] = _run_variant(merged, label=label, params=params, engine=engine)

    ranked = _rank_variants(variants)
    quality_base = next(row for row in ranked if row["label"] == "active_lite_quality_base")
    reversal_balance = next(row for row in ranked if row["label"] == "qualified_reversal_balance")
    continuation_density = next(row for row in ranked if row["label"] == "qualified_continuation_density")
    verdict, interpretation = _verdict(
        quality_base=quality_base,
        reversal_balance=reversal_balance,
        continuation_density=continuation_density,
    )

    output = {
        "analysis": "ict_lite_reversal_balance_branch",
        "profile": "qualified_reversal_balance_compare",
        "risk_standard": {
            "initial_cash_usd": 100000,
            "capital_usage_pct": 1.0,
            "minimum_order_shares": 40,
            "reward_risk_floor": 1.5,
        },
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "variant_ranking": ranked,
        "quality_base": quality_base,
        "qualified_reversal_balance": reversal_balance,
        "qualified_continuation_density": continuation_density,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
