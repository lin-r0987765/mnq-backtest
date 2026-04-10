from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ict.analyze_ict_lite_reversal_baseline import RESEARCH_STANDARD, _engine_config_payload
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.base import BaseStrategy, StrategyResult
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_core_400_short_structure_refined_recovery_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_pending4_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_profile_params,
    build_ict_core_400_short_stat_bias_candidate_profile_params,
    build_ict_core_400_short_structure_bias_candidate_profile_params,
    build_ict_core_400_short_structure_bias_lb6_candidate_profile_params,
    build_ict_core_400_short_structure_refined_candidate_profile_params,
    build_ict_core_400_short_structure_refined_density_candidate_profile_params,
    build_ict_core_400_short_structure_refined_recovery_sl135_candidate_profile_params,
    build_ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_pending3_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_depth04_pending3_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_pending3_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure11_pending3_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_capacity_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_long_refined_timing_candidate_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_profile_params,
)


_REGULARIZED_ROOT = PROJECT_ROOT / "regularized"
DEFAULT_INTRADAY = (
    _REGULARIZED_ROOT / "QQQ" / "qqq_5m_regular_et.csv"
    if (_REGULARIZED_ROOT / "QQQ" / "qqq_5m_regular_et.csv").exists()
    else PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
)
DEFAULT_PEER_CSV = (
    _REGULARIZED_ROOT / "SPY" / "spy_5m_regular_et.csv"
    if (_REGULARIZED_ROOT / "SPY" / "spy_5m_regular_et.csv").exists()
    else PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_asymmetric_mixed_lane.json"


class _CombinedSignalsStrategy(BaseStrategy):
    name = "ICT_Asymmetric_Mixed_Lane"

    def __init__(self, long_signals: StrategyResult, short_signals: StrategyResult, metadata: dict[str, Any]):
        super().__init__(metadata)
        self._long_signals = long_signals
        self._short_signals = short_signals
        self._metadata = metadata

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        del df
        return StrategyResult(
            entries_long=self._long_signals.entries_long,
            exits_long=self._long_signals.exits_long,
            entries_short=self._short_signals.entries_short,
            exits_short=self._short_signals.exits_short,
            metadata=dict(self._metadata),
        )


def _run_strategy(df: pd.DataFrame, params: dict[str, Any]) -> tuple[ICTEntryModelStrategy, StrategyResult]:
    strategy = ICTEntryModelStrategy(params=params)
    return strategy, strategy.generate_signals(df)


def _run_profile(
    merged: pd.DataFrame,
    *,
    label: str,
    long_params: dict[str, Any],
    short_params: dict[str, Any],
    engine: BacktestEngine,
) -> dict[str, Any]:
    long_strategy, long_signals = _run_strategy(merged, long_params)
    short_strategy, short_signals = _run_strategy(merged, short_params)

    conflict_mask = long_signals.entries_long & short_signals.entries_short
    entries_long = long_signals.entries_long & ~conflict_mask
    entries_short = short_signals.entries_short & ~conflict_mask

    combined_metadata = {
        "mixed_lane_conflicts": int(conflict_mask.sum()),
        "long_lane_entries": int(long_signals.entries_long.sum()),
        "short_lane_entries": int(short_signals.entries_short.sum()),
    }
    combined_metadata.update({f"long_{k}": v for k, v in long_signals.metadata.items() if isinstance(v, (int, float, bool, str))})
    combined_metadata.update({f"short_{k}": v for k, v in short_signals.metadata.items() if isinstance(v, (int, float, bool, str))})

    combined_strategy = _CombinedSignalsStrategy(
        StrategyResult(
            entries_long=entries_long,
            exits_long=long_signals.exits_long,
            entries_short=pd.Series(False, index=merged.index),
            exits_short=pd.Series(False, index=merged.index),
            metadata={},
        ),
        StrategyResult(
            entries_long=pd.Series(False, index=merged.index),
            exits_long=pd.Series(False, index=merged.index),
            entries_short=entries_short,
            exits_short=short_signals.exits_short,
            metadata={},
        ),
        combined_metadata,
    )
    result = engine.run(combined_strategy, merged)
    return {
        "label": label,
        "metrics": result.metrics,
        "params": {
            "long_profile": long_strategy.get_params(),
            "short_profile": short_strategy.get_params(),
        },
        "metadata": combined_metadata,
    }


def _baseline_long_params() -> dict[str, Any]:
    return build_ict_lite_reversal_qualified_reversal_balance_profile_params(
        enable_smt=False,
        overrides={"allow_short_entries": False},
    )


def _refined_long_params() -> dict[str, Any]:
    return build_ict_lite_reversal_qualified_reversal_balance_long_refined_candidate_profile_params(
        enable_smt=False
    )


def _refined_long_timing_params() -> dict[str, Any]:
    return build_ict_lite_reversal_qualified_reversal_balance_long_refined_timing_candidate_profile_params(
        enable_smt=False
    )


def _refined_long_sweep035_params() -> dict[str, Any]:
    return build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_candidate_profile_params(
        enable_smt=False
    )


def _refined_long_sweep035_capacity_params() -> dict[str, Any]:
    return (
        build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_capacity_candidate_profile_params(
            enable_smt=False
        )
    )


def _refined_long_sweep035_structure11_pending3_params() -> dict[str, Any]:
    return (
        build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure11_pending3_candidate_profile_params(
            enable_smt=False
        )
    )


def _refined_long_sweep035_structure10_pending3_params() -> dict[str, Any]:
    return (
        build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_pending3_candidate_profile_params(
            enable_smt=False
        )
    )


def _refined_long_sweep035_structure10_depth04_pending3_params() -> dict[str, Any]:
    return (
        build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_depth04_pending3_candidate_profile_params(
            enable_smt=False
        )
    )


def _refined_long_sweep035_structure10_gap020_depth04_pending3_params() -> dict[str, Any]:
    return (
        build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_pending3_candidate_profile_params(
            enable_smt=False
        )
    )


def _refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_params() -> dict[str, Any]:
    return (
        build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_candidate_profile_params(
            enable_smt=False
        )
    )


def _refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_params() -> dict[str, Any]:
    return (
        build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_candidate_profile_params(
            enable_smt=False
        )
    )


def _profile_specs() -> list[dict[str, Any]]:
    baseline_long = _baseline_long_params()
    refined_long = _refined_long_params()
    refined_long_timing = _refined_long_timing_params()
    refined_long_sweep035 = _refined_long_sweep035_params()
    refined_long_sweep035_capacity = _refined_long_sweep035_capacity_params()
    refined_long_sweep035_structure11_pending3 = _refined_long_sweep035_structure11_pending3_params()
    refined_long_sweep035_structure10_pending3 = _refined_long_sweep035_structure10_pending3_params()
    refined_long_sweep035_structure10_depth04_pending3 = _refined_long_sweep035_structure10_depth04_pending3_params()
    refined_long_sweep035_structure10_gap020_depth04_pending3 = (
        _refined_long_sweep035_structure10_gap020_depth04_pending3_params()
    )
    refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3 = (
        _refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_params()
    )
    refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3 = (
        _refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_params()
    )
    return [
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_plus_short_structure_refined_capacity_threshold0265_pending4_sl135_dailybiaslb8",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_plus_short_structure_refined_recovery_sl135_dailybiaslb8",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3,
            "short_params": build_ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_plus_short_structure_refined_capacity_threshold0265_pending4_sl135",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_plus_short_structure_refined_recovery_sl135",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3,
            "short_params": build_ict_core_400_short_structure_refined_recovery_sl135_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_plus_short_structure_refined_capacity_threshold0265_pending4",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_plus_short_structure_refined_recovery",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_pending3_plus_short_structure_refined_capacity_threshold0265_pending4",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_pending3,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_gap020_depth04_pending3_plus_short_structure_refined_recovery",
            "long_params": refined_long_sweep035_structure10_gap020_depth04_pending3,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_depth04_pending3_plus_short_structure_refined_capacity_threshold0265_pending4",
            "long_params": refined_long_sweep035_structure10_depth04_pending3,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_depth04_pending3_plus_short_structure_refined_recovery",
            "long_params": refined_long_sweep035_structure10_depth04_pending3,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_pending3_plus_short_structure_refined_capacity_threshold0265_pending4",
            "long_params": refined_long_sweep035_structure10_pending3,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_structure10_pending3_plus_short_structure_refined_recovery",
            "long_params": refined_long_sweep035_structure10_pending3,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_structure11_pending3_plus_short_structure_refined_capacity_threshold0265_pending4",
            "long_params": refined_long_sweep035_structure11_pending3,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_structure11_pending3_plus_short_structure_refined_recovery",
            "long_params": refined_long_sweep035_structure11_pending3,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_capacity_plus_short_structure_refined_capacity_threshold0265_pending4",
            "long_params": refined_long_sweep035_capacity,
            "short_params": (
                build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params()
            ),
        },
        {
            "label": "rev_long_refined_sweep035_capacity_plus_short_structure_refined_capacity_pending4",
            "long_params": refined_long_sweep035_capacity,
            "short_params": build_ict_core_400_short_structure_refined_capacity_pending4_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_capacity_plus_short_structure_refined_capacity",
            "long_params": refined_long_sweep035_capacity,
            "short_params": build_ict_core_400_short_structure_refined_capacity_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_plus_short_structure_refined_recovery",
            "long_params": refined_long_sweep035,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_sweep035_plus_short_structure_refined_capacity",
            "long_params": refined_long_sweep035,
            "short_params": build_ict_core_400_short_structure_refined_capacity_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_timing_plus_short_structure_refined_recovery",
            "long_params": refined_long_timing,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_timing_plus_short_structure_refined_capacity",
            "long_params": refined_long_timing,
            "short_params": build_ict_core_400_short_structure_refined_capacity_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_plus_short_structure_refined_recovery",
            "long_params": refined_long,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_refined_plus_short_structure_refined_capacity",
            "long_params": refined_long,
            "short_params": build_ict_core_400_short_structure_refined_capacity_candidate_profile_params(),
        },
        {
            "label": "rev_long_plus_short_structure_refined_recovery",
            "long_params": baseline_long,
            "short_params": build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(),
        },
        {
            "label": "rev_long_plus_short_structure_refined_capacity",
            "long_params": baseline_long,
            "short_params": build_ict_core_400_short_structure_refined_capacity_candidate_profile_params(),
        },
        {
            "label": "rev_long_plus_short_structure_refined",
            "long_params": baseline_long,
            "short_params": build_ict_core_400_short_structure_refined_candidate_profile_params(),
        },
        {
            "label": "rev_long_plus_short_structure_refined_density",
            "long_params": baseline_long,
            "short_params": build_ict_core_400_short_structure_refined_density_candidate_profile_params(),
        },
        {
            "label": "rev_long_plus_short_structure_bias_lb6",
            "long_params": baseline_long,
            "short_params": build_ict_core_400_short_structure_bias_lb6_candidate_profile_params(),
        },
        {
            "label": "rev_long_plus_short_structure_bias",
            "long_params": baseline_long,
            "short_params": build_ict_core_400_short_structure_bias_candidate_profile_params(),
        },
        {
            "label": "rev_long_plus_short_stat_bias",
            "long_params": baseline_long,
            "short_params": build_ict_core_400_short_stat_bias_candidate_profile_params(),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare asymmetric long/short ICT lane combinations.")
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="full_local")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(**RESEARCH_STANDARD)
    profiles = [
        _run_profile(
            merged,
            label=spec["label"],
            long_params=spec["long_params"],
            short_params=spec["short_params"],
            engine=engine,
        )
        for spec in _profile_specs()
    ]
    profiles.sort(
        key=lambda row: (
            row["metrics"]["total_return_pct"] > 0,
            row["metrics"]["total_return_pct"],
            row["metrics"]["profit_factor"],
            row["metrics"]["total_trades"],
        ),
        reverse=True,
    )

    output = {
        "analysis": "ict_asymmetric_mixed_lane",
        "risk_standard": {
            "reward_risk_gate": ">= 1.5:1",
            "engine": _engine_config_payload(engine),
        },
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "profiles": profiles,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
