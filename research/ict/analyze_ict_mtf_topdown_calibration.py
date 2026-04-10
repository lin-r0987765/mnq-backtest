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
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_mtf_topdown_continuation_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_profile_params,
    build_ict_mtf_topdown_continuation_quality_candidate_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params,
    build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_timing_candidate_profile_params,
)
DEFAULT_INTRADAY = PROJECT_ROOT / "regularized" / "QQQ" / "qqq_1m_regular_et.csv"
if not DEFAULT_INTRADAY.exists():
    DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_1m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "regularized" / "SPY" / "spy_1m_regular_et.csv"
if not DEFAULT_PEER_CSV.exists():
    DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_1m_alpaca.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "qc_regime_prototypes"
RESEARCH_STANDARD = {
    "initial_cash": 10_000.0,
    "fees_pct": 0.0005,
    "position_size_mode": "capital_pct",
    "capital_usage_pct": 1.0,
    "min_shares": 0,
}


def _build_profile(profile: str) -> tuple[str, dict[str, Any]]:
    key = str(profile).strip().lower()
    if key == "execution_candidate":
        return (
            "mtf_topdown_continuation_execution_candidate",
            build_ict_mtf_topdown_continuation_execution_candidate_profile_params(enable_smt=False),
        )
    if key == "setup_execution_candidate":
        return (
            "mtf_topdown_continuation_setup_execution_candidate",
            build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params(enable_smt=False),
        )
    if key == "timing_candidate":
        return (
            "mtf_topdown_continuation_timing_candidate",
            build_ict_mtf_topdown_continuation_timing_candidate_profile_params(enable_smt=False),
        )
    if key == "regularized_long_only_candidate":
        return (
            "mtf_topdown_continuation_regularized_long_only_candidate",
            build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params(
                enable_smt=False
            ),
        )
    if key == "regularized_long_only_am_candidate":
        return (
            "mtf_topdown_continuation_regularized_long_only_am_candidate",
            build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params(
                enable_smt=False
            ),
        )
    if key == "quality_candidate":
        return (
            "mtf_topdown_continuation_quality_candidate",
            build_ict_mtf_topdown_continuation_quality_candidate_profile_params(enable_smt=False),
        )
    return (
        "mtf_topdown_continuation_baseline",
        build_ict_mtf_topdown_continuation_profile_params(enable_smt=False),
    )


def _variant_specs(study: str) -> dict[str, dict[str, Any]]:
    if study == "bias":
        return {
            "baseline": {},
            "bias_lookback_8": {"mtf_bias_lookback": 8},
            "bias_lookback_16": {"mtf_bias_lookback": 16},
            "neutral_1h_stricter": {"mtf_neutral_hourly_min_score": 8.0},
            "neutral_1h_hard_block": {"mtf_neutral_hourly_allows_high_quality": False},
        }
    if study == "setup":
        return {
            "baseline": {},
            "setup_lookback_6": {"mtf_setup_structure_lookback": 6},
            "setup_lookback_12": {"mtf_setup_structure_lookback": 12},
            "setup_gap_0004": {"mtf_setup_fvg_min_gap_pct": 0.0004},
            "setup_gap_0008": {"mtf_setup_fvg_min_gap_pct": 0.0008},
            "setup_body_030": {"mtf_setup_displacement_body_min_pct": 0.30},
            "setup_body_040": {"mtf_setup_displacement_body_min_pct": 0.40},
        }
    if study == "confirmation":
        return {
            "baseline": {},
            "confirm_lookback_2": {"mtf_confirmation_structure_lookback": 2},
            "confirm_lookback_4": {"mtf_confirmation_structure_lookback": 4},
            "confirm_close_055": {"mtf_confirmation_close_ratio": 0.55},
            "confirm_close_070": {"mtf_confirmation_close_ratio": 0.70},
            "confirm_body_030": {"mtf_confirmation_body_min_pct": 0.30},
            "confirm_body_040": {"mtf_confirmation_body_min_pct": 0.40},
        }
    if study == "trigger":
        return {
            "baseline": {},
            "trigger_lookback_3": {"mtf_trigger_structure_lookback": 3},
            "trigger_lookback_7": {"mtf_trigger_structure_lookback": 7},
            "trigger_close_055": {"mtf_trigger_close_ratio": 0.55},
            "trigger_close_070": {"mtf_trigger_close_ratio": 0.70},
            "trigger_body_025": {"mtf_trigger_body_min_pct": 0.25},
            "trigger_body_040": {"mtf_trigger_body_min_pct": 0.40},
            "trigger_expiry_15": {"mtf_trigger_expiry_bars": 15},
            "trigger_expiry_45": {"mtf_trigger_expiry_bars": 45},
        }
    if study == "timing":
        return {
            "baseline": {},
            "timing_exclude_13et": {"mtf_allowed_entry_hours": (9, 10, 11, 12, 14, 15)},
            "timing_tue_to_fri": {"mtf_allowed_entry_weekdays": (1, 2, 3, 4)},
            "timing_wed_to_fri": {"mtf_allowed_entry_weekdays": (2, 3, 4)},
            "timing_tue_to_fri_exclude_13et": {
                "mtf_allowed_entry_weekdays": (1, 2, 3, 4),
                "mtf_allowed_entry_hours": (9, 10, 11, 12, 14, 15),
            },
            "timing_wed_to_fri_exclude_13et": {
                "mtf_allowed_entry_weekdays": (2, 3, 4),
                "mtf_allowed_entry_hours": (9, 10, 11, 12, 14, 15),
            },
        }
    raise ValueError(f"Unsupported study: {study}")


def _data_window(df) -> dict[str, Any]:
    if df.empty:
        return {"start": None, "end": None, "trading_days": 0}
    return {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "trading_days": int(len(set(df.index.date))),
    }


def _period_label(df) -> str:
    window = _data_window(df)
    if window["start"] is None or window["end"] is None:
        return "empty"
    return f'{window["start"]} -> {window["end"]}'


def _resolve_output_path(requested_output: str | None, *, study: str, profile_name: str) -> Path:
    if requested_output:
        return Path(requested_output)
    suffix = "" if profile_name == "mtf_topdown_continuation_baseline" else f"_{profile_name}"
    return DEFAULT_OUTPUT_DIR / f"ict_mtf_topdown_{study}_calibration{suffix}.json"


def _run_variant(
    merged,
    *,
    base_params: dict[str, Any],
    label: str,
    overrides: dict[str, Any],
    engine: BacktestEngine,
) -> dict[str, Any]:
    params = dict(base_params)
    params.update(overrides)
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    return {
        "label": label,
        "overrides": overrides,
        "metrics": result.metrics,
        "metadata": {
            key: int(signals.metadata.get(key, 0))
            for key in (
                "mtf_daily_blocked",
                "mtf_4h_blocked",
                "mtf_1h_blocked",
                "mtf_15m_setups",
                "mtf_5m_confirms",
                "mtf_1m_triggers",
                "mtf_setup_missing_zone",
                "mtf_setup_score_filtered",
                "mtf_setup_expired",
                "mtf_confirm_expired",
                "mtf_weekday_blocked",
                "mtf_hour_blocked",
                "rr_filtered_entries",
                "long_entries",
                "short_entries",
            )
        },
    }


def _rank(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in payloads.items():
        metrics = payload["metrics"]
        metadata = payload["metadata"]
        ranked.append(
            {
                "label": label,
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
                "mtf_15m_setups": int(metadata["mtf_15m_setups"]),
                "mtf_5m_confirms": int(metadata["mtf_5m_confirms"]),
                "mtf_1m_triggers": int(metadata["mtf_1m_triggers"]),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the ICT MTF top-down continuation branch by study layer."
    )
    parser.add_argument(
        "--study",
        choices=["bias", "setup", "confirmation", "trigger", "timing"],
        required=True,
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument(
        "--profile",
        choices=[
            "baseline",
            "quality_candidate",
            "execution_candidate",
            "setup_execution_candidate",
            "timing_candidate",
            "regularized_long_only_candidate",
            "regularized_long_only_am_candidate",
        ],
        default="baseline",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(**RESEARCH_STANDARD)
    profile_name, base_params = _build_profile(args.profile)

    variants: dict[str, dict[str, Any]] = {}
    for label, overrides in _variant_specs(args.study).items():
        variants[label] = _run_variant(
            merged,
            base_params=base_params,
            label=label,
            overrides=overrides,
            engine=engine,
        )

    output = {
        "analysis": "ict_mtf_topdown_calibration",
        "profile": profile_name,
        "study": args.study,
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": _period_label(intraday_df),
        "peer_fetch_period": args.period if not args.peer_csv else None,
        "data_window": _data_window(intraday_df),
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "ranking": _rank(variants),
    }

    output_path = _resolve_output_path(args.output, study=args.study, profile_name=profile_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
