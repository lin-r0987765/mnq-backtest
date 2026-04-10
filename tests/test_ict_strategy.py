from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.data.fetcher import resample_ohlcv
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_core_400_baseline_profile_params,
    build_ict_core_400_short_structure_refined_recovery_candidate_profile_params,
    build_ict_core_400_short_structure_refined_recovery_sl135_candidate_profile_params,
    build_ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_pending4_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_profile_params,
    build_ict_core_400_short_structure_refined_candidate_profile_params,
    build_ict_core_400_short_structure_refined_density_candidate_profile_params,
    build_ict_core_400_short_stat_bias_candidate_profile_params,
    build_ict_core_400_short_structure_bias_candidate_profile_params,
    build_ict_core_400_short_structure_bias_lb6_candidate_profile_params,
    build_ict_core_400_short_only_profile_params,
    build_ict_complete_soft_premium_profile_params,
    build_ict_complete_soft_prev_session_profile_params,
    build_ict_complete_soft_session_array_profile_params,
    build_ict_fvg_fib_retracement_research_profile_params,
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
    _compute_daily_bias,
    _compute_higher_timeframe_structure_bias,
    _compute_intraday_regime,
    _latest_confirmed_swing_level,
    _quality_score_bonus,
    StrategyResult,
    _detect_fvg_zone,
    _project_reward_risk_ratio,
    _same_trading_day,
    _structure_confirmation_score,
    build_ict_mtf_topdown_continuation_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params,
    build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_timing_candidate_profile_params,
    build_ict_research_profile_params,
)


def _frame(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-07 14:30:00+00:00", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=index)


def _frame_1m(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-07 14:30:00+00:00", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=index)


def _frame_with_index(
    timestamps: list[str], rows: list[tuple[float, float, float, float, float]]
) -> pd.DataFrame:
    index = pd.to_datetime(timestamps, utc=True)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=index)


class ICTEntryModelStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "trade_sessions": False,
            "liq_sweep_lookback": 4,
            "structure_lookback": 4,
            "liq_sweep_recovery_bars": 3,
            "ob_lookback": 4,
            "fvg_max_age": 4,
        }

    def test_generates_bullish_entry_from_fvg_retest(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )

        result = ICTEntryModelStrategy(params=self.params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(result.metadata["bullish_sweeps"], 1)
        self.assertEqual(result.metadata["fvg_entries"], 1)

    def test_allow_long_entries_false_blocks_bullish_entry(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )

        params = {**self.params, "allow_long_entries": False}
        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.entries_short.sum()), 0)

    def test_fvg_fib_retracement_research_profile_enforces_fvg_and_ote(self) -> None:
        params = build_ict_fvg_fib_retracement_research_profile_params()

        self.assertTrue(bool(params["require_fvg_delivery"]))
        self.assertTrue(bool(params["require_ote_zone"]))
        self.assertEqual(float(params["ote_fib_low"]), 0.5)
        self.assertEqual(float(params["ote_fib_high"]), 0.79)
        self.assertEqual(float(params["score_ote_zone"]), 0.0)
        self.assertEqual(float(params["min_score_to_trade"]), 5.0)

    def test_slow_recovery_branch_can_rescue_a_sweep_that_fast_recovery_loses(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        fast_only_params = {
            **self.params,
            "liq_sweep_recovery_bars": 1,
        }
        dual_speed_params = {
            **fast_only_params,
            "slow_recovery_enabled": True,
            "slow_recovery_bars": 3,
        }

        fast_only_result = ICTEntryModelStrategy(params=fast_only_params).generate_signals(df)
        dual_speed_result = ICTEntryModelStrategy(params=dual_speed_params).generate_signals(df)

        self.assertEqual(int(fast_only_result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(fast_only_result.metadata["sweep_expired_before_shift"]), 1)
        self.assertEqual(int(dual_speed_result.entries_long.sum()), 1)
        self.assertTrue(bool(dual_speed_result.entries_long.iloc[7]))
        self.assertEqual(int(dual_speed_result.metadata["slow_recovery_shifts"]), 1)
        self.assertEqual(int(dual_speed_result.metadata["slow_recovery_entries"]), 1)
        self.assertEqual(int(dual_speed_result.metadata["fast_recovery_entries"]), 0)

    def test_reward_risk_gate_blocks_low_target_entry_but_no_gate_allows_it(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        no_gate_params = {
            **self.params,
            "take_profit_rr": 1.0,
            "min_reward_risk_ratio": 0.0,
        }
        gated_params = {
            **self.params,
            "take_profit_rr": 1.0,
            "min_reward_risk_ratio": 1.5,
        }

        no_gate_result = ICTEntryModelStrategy(params=no_gate_params).generate_signals(df)
        gated_result = ICTEntryModelStrategy(params=gated_params).generate_signals(df)

        self.assertEqual(int(no_gate_result.entries_long.sum()), 1)
        self.assertEqual(int(gated_result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(gated_result.metadata["rr_filtered_entries"]), 1)

    def test_require_fvg_delivery_blocks_order_block_fallback(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "require_fvg_delivery": True,
            "require_ote_zone": False,
            "min_score_to_trade": 5,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=(5, 101.5, 102.0),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_required_filtered_shifts"]), 1)
        self.assertEqual(int(result.metadata["ob_entries"]), 0)

    def test_require_ote_zone_blocks_fvg_outside_retracement_window(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "require_fvg_delivery": True,
            "require_ote_zone": True,
            "min_score_to_trade": 5,
        }

        with patch(
            "src.strategies.ict_entry_model._zone_in_ote",
            return_value=False,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["ote_required_filtered_shifts"]), 1)

    def test_require_ote_zone_allows_fvg_inside_retracement_window(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "require_fvg_delivery": True,
            "require_ote_zone": True,
            "min_score_to_trade": 5,
        }

        with patch(
            "src.strategies.ict_entry_model._zone_in_ote",
            return_value=True,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))

    def test_project_reward_risk_ratio_returns_zero_when_risk_is_zero(self) -> None:
        self.assertEqual(_project_reward_risk_ratio(100.0, 100.0, 104.0), 0.0)
        self.assertAlmostEqual(_project_reward_risk_ratio(100.0, 98.0, 108.0), 4.0)

    def test_sweep_reclaim_ratio_allows_long_when_close_reclaims_deep_enough(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {**self.params, "liq_sweep_reclaim_ratio": 0.15}

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))

    def test_sweep_reclaim_ratio_blocks_long_when_reclaim_is_too_shallow(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {**self.params, "liq_sweep_reclaim_ratio": 0.25}

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)

    def test_choch_score_wiring_can_unlock_a_high_confluence_threshold(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params_without_choch = {
            **self.params,
            "min_score_to_trade": 8,
            "score_choch": 0,
        }
        params_with_choch = {
            **self.params,
            "min_score_to_trade": 8,
            "score_choch": 3,
        }

        result_without_choch = ICTEntryModelStrategy(params=params_without_choch).generate_signals(df)
        result_with_choch = ICTEntryModelStrategy(params=params_with_choch).generate_signals(df)

        self.assertEqual(int(result_without_choch.entries_long.sum()), 0)
        self.assertEqual(int(result_with_choch.entries_long.sum()), 1)
        self.assertTrue(bool(result_with_choch.entries_long.iloc[7]))

    def test_structure_confirmation_score_does_not_double_count_bos_and_choch(self) -> None:
        self.assertEqual(_structure_confirmation_score(2, 3), 3)
        self.assertEqual(_structure_confirmation_score(4, 1), 4)

    def test_quality_score_bonus_rewards_geometry_that_exceeds_baselines(self) -> None:
        bonus = _quality_score_bonus(
            sweep_depth=0.20,
            sweep_reference_level=100.0,
            liq_sweep_threshold=0.001,
            displacement_body_ratio=0.40,
            displacement_body_min_pct=0.10,
            fvg_gap_size=0.20,
            price=100.0,
            fvg_min_gap_pct=0.001,
            score_sweep_depth_quality=1.0,
            score_displacement_quality=1.0,
            score_fvg_gap_quality=1.0,
        )
        self.assertGreater(bonus, 0.0)

    def test_quality_score_bonus_stays_zero_when_geometry_only_meets_floor(self) -> None:
        bonus = _quality_score_bonus(
            sweep_depth=0.10,
            sweep_reference_level=100.0,
            liq_sweep_threshold=0.001,
            displacement_body_ratio=0.10,
            displacement_body_min_pct=0.10,
            fvg_gap_size=0.10,
            price=100.0,
            fvg_min_gap_pct=0.001,
            score_sweep_depth_quality=1.0,
            score_displacement_quality=1.0,
            score_fvg_gap_quality=1.0,
        )
        self.assertEqual(bonus, 0.0)

    def test_fvg_detector_prefers_most_recent_zone(self) -> None:
        df = _frame(
            [
                (100.0, 100.2, 99.8, 100.0, 1000),
                (100.1, 100.4, 99.9, 100.3, 1000),
                (100.3, 101.0, 100.8, 100.9, 1000),
                (100.9, 101.2, 100.9, 101.1, 1000),
                (101.1, 101.8, 101.5, 101.7, 1000),
            ]
        )

        zone = _detect_fvg_zone(df, 0, 4, bullish=True, min_gap_pct=0.001)

        self.assertIsNotNone(zone)
        self.assertEqual(int(zone[0]), 4)
        self.assertAlmostEqual(float(zone[1]), 101.0)
        self.assertAlmostEqual(float(zone[2]), 101.5)

    def test_displacement_body_filter_blocks_weak_structure_shift(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (101.75, 103.0, 101.7, 101.85, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "displacement_body_min_pct": 0.2,
            "liq_sweep_recovery_bars": 1,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["displacement_filtered_shifts"]), 1)

    def test_displacement_body_filter_allows_strong_structure_shift(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "displacement_body_min_pct": 0.6,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["displacement_filtered_shifts"]), 0)

    def test_displacement_range_filter_blocks_underexpanded_structure_shift(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.7, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "displacement_range_atr_mult": 1.6,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["displacement_range_filtered_shifts"]), 1)

    def test_displacement_range_filter_allows_expanded_structure_shift(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 102.3, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "displacement_range_atr_mult": 0.7,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["displacement_range_filtered_shifts"]), 0)

    def test_structure_shift_close_buffer_blocks_weak_close_through(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 102.0, 99.9, 101.9, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "structure_shift_close_buffer_ratio": 0.05,
            "liq_sweep_recovery_bars": 1,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["structure_buffer_filtered_shifts"]), 1)

    def test_structure_shift_close_buffer_allows_strong_close_through(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 102.0, 99.9, 101.9, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "structure_shift_close_buffer_ratio": 0.04,
            "liq_sweep_recovery_bars": 1,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["structure_buffer_filtered_shifts"]), 0)

    def test_fvg_revisit_depth_blocks_shallow_gap_touch(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.35, 101.85, 102.0, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_depth_filtered_retests"]), 1)

    def test_fvg_revisit_delay_blocks_immediate_retest(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.35, 101.85, 102.0, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_revisit_min_delay_bars": 3,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_delay_filtered_retests"]), 1)

    def test_fvg_revisit_delay_allows_later_retest(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.5, 102.9, 102.2, 102.7, 1500),
                (102.6, 103.0, 102.3, 102.8, 1500),
                (102.3, 102.4, 101.7, 101.9, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_revisit_min_delay_bars": 3,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["fvg_delay_filtered_retests"]), 0)

    def test_fvg_origin_lag_blocks_gap_that_forms_too_early_before_shift(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_origin_max_lag_bars": 1,
        }

        # Mock an older FVG origin so this test isolates lag gating itself.
        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(4, 101.6, 101.9),
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_breaker_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_ifvg_zone",
            return_value=None,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_origin_lag_filtered_shifts"]), 1)

    def test_fvg_origin_lag_allows_gap_when_lag_budget_is_two(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_origin_max_lag_bars": 2,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(4, 101.6, 101.9),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["fvg_origin_lag_filtered_shifts"]), 0)

    def test_fvg_origin_body_filter_blocks_weak_bullish_fvg_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_origin_body_min_pct": 0.81,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_breaker_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_ifvg_zone",
            return_value=None,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_origin_body_filtered_shifts"]), 1)

    def test_fvg_origin_body_filter_allows_strong_bullish_fvg_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_origin_body_min_pct": 0.79,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["fvg_origin_body_filtered_shifts"]), 0)

    def test_fvg_origin_close_position_filter_allows_high_closing_bullish_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_close_position_min_pct": 0.85,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["fvg_origin_close_filtered_shifts"]), 0)

    def test_fvg_origin_close_position_filter_blocks_shallow_bullish_origin_close(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_close_position_min_pct": 0.95,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_breaker_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_ifvg_zone",
            return_value=None,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_origin_close_filtered_shifts"]), 1)

    def test_fvg_origin_opposite_wick_filter_allows_clean_bullish_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_opposite_wick_max_pct": 0.11,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["fvg_origin_wick_filtered_shifts"]), 0)

    def test_fvg_origin_opposite_wick_filter_blocks_bullish_origin_with_deep_lower_wick(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_opposite_wick_max_pct": 0.09,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_breaker_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_ifvg_zone",
            return_value=None,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_origin_wick_filtered_shifts"]), 1)

    def test_fvg_origin_range_atr_filter_allows_expansive_bullish_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_range_atr_mult": 1.44,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["fvg_origin_range_filtered_shifts"]), 0)

    def test_fvg_origin_range_atr_filter_blocks_underexpanded_bullish_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_range_atr_mult": 1.46,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_breaker_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_ifvg_zone",
            return_value=None,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_origin_range_filtered_shifts"]), 1)

    def test_fvg_origin_body_atr_filter_allows_impulsive_bullish_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_body_atr_mult": 1.15,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["fvg_origin_body_atr_filtered_shifts"]), 0)

    def test_fvg_origin_body_atr_filter_blocks_underimpulsive_bullish_origin(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_origin_body_atr_mult": 1.16,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(6, 101.6, 101.9),
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_breaker_block_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_ifvg_zone",
            return_value=None,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_origin_body_atr_filtered_shifts"]), 1)

    def test_fvg_origin_lag_can_fall_back_to_order_block(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_origin_max_lag_bars": 1,
        }

        with patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(4, 101.6, 101.9),
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=(6, 101.6, 101.9),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["ob_entries"]), 1)
        self.assertGreaterEqual(int(result.metadata["fvg_origin_lag_filtered_shifts"]), 1)

    def test_same_trading_day_uses_new_york_calendar(self) -> None:
        self.assertTrue(
            _same_trading_day(
                pd.Timestamp("2026-01-07 23:55:00+00:00"),
                pd.Timestamp("2026-01-08 00:05:00+00:00"),
            )
        )
        self.assertFalse(
            _same_trading_day(
                pd.Timestamp("2026-01-08 04:55:00+00:00"),
                pd.Timestamp("2026-01-08 05:05:00+00:00"),
            )
        )

    def test_fvg_touch_cap_blocks_second_touch_after_delayed_first_touch(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.35, 101.85, 102.0, 1500),
                (102.3, 102.4, 101.7, 101.9, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_revisit_min_delay_bars": 3,
            "fvg_max_retest_touches": 1,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_touch_filtered_retests"]), 1)

    def test_fvg_touch_cap_does_not_override_delay_gate_when_cap_is_three(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.35, 101.85, 102.0, 1500),
                (102.3, 102.4, 101.7, 101.9, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_revisit_min_delay_bars": 3,
            "fvg_max_retest_touches": 3,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_delay_filtered_retests"]), 1)
        self.assertEqual(int(result.metadata["fvg_touch_filtered_retests"]), 0)

    def test_fvg_revisit_depth_allows_consequent_encroachment_touch(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.7, 101.9, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["fvg_depth_filtered_retests"]), 0)

    def test_fvg_close_recovery_blocks_weak_rejection_inside_gap(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.7, 101.6, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_rejection_close_ratio": 0.5,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_close_filtered_retests"]), 1)

    def test_fvg_close_recovery_allows_close_back_to_consequent_encroachment(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.7, 101.95, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_rejection_close_ratio": 0.5,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["fvg_close_filtered_retests"]), 0)

    def test_fvg_rejection_wick_blocks_weak_rejection_wick_inside_gap(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (101.79, 102.4, 101.74, 101.95, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_rejection_wick_ratio": 0.2,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_wick_filtered_retests"]), 1)

    def test_fvg_rejection_wick_allows_strong_rejection_from_gap(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.7, 101.95, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_rejection_wick_ratio": 0.5,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["fvg_wick_filtered_retests"]), 0)

    def test_fvg_rejection_body_blocks_weak_directional_reaction(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (101.90, 102.4, 101.7, 101.95, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_rejection_body_min_pct": 0.2,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["fvg_body_filtered_retests"]), 1)

    def test_fvg_rejection_body_allows_strong_directional_reaction(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (101.75, 102.4, 101.7, 101.95, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_revisit_depth_ratio": 0.5,
            "fvg_rejection_body_min_pct": 0.2,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["fvg_body_filtered_retests"]), 0)

    def test_uses_order_block_when_no_fvg_is_available(self) -> None:
        df = _frame(
            [
                (100.0, 100.5, 99.6, 100.1, 1000),
                (100.1, 100.7, 99.7, 100.4, 1000),
                (100.4, 100.8, 99.8, 100.5, 1000),
                (100.5, 100.9, 99.9, 100.6, 1000),
                (100.7, 102.2, 100.2, 100.4, 1200),
                (100.0, 100.9, 99.9, 100.7, 1300),
                (100.6, 100.7, 99.0, 99.1, 1600),
                (99.8, 100.8, 99.7, 100.2, 1500),
                (100.1, 100.2, 98.8, 99.0, 1400),
            ]
        )

        result = ICTEntryModelStrategy(params=self.params).generate_signals(df)

        self.assertEqual(int(result.entries_short.sum()), 1)
        self.assertTrue(bool(result.entries_short.iloc[7]))
        self.assertEqual(result.metadata["bearish_sweeps"], 1)
        self.assertEqual(result.metadata["ob_entries"], 1)

    def test_allow_short_entries_false_blocks_bearish_entry(self) -> None:
        df = _frame(
            [
                (100.0, 100.5, 99.6, 100.1, 1000),
                (100.1, 100.7, 99.7, 100.4, 1000),
                (100.4, 100.8, 99.8, 100.5, 1000),
                (100.5, 100.9, 99.9, 100.6, 1000),
                (100.7, 102.2, 100.2, 100.4, 1200),
                (100.0, 100.9, 99.9, 100.7, 1300),
                (100.6, 100.7, 99.0, 99.1, 1600),
                (99.8, 100.8, 99.7, 100.2, 1500),
                (100.1, 100.2, 98.8, 99.0, 1400),
            ]
        )

        params = {**self.params, "allow_short_entries": False}
        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_short.sum()), 0)
        self.assertEqual(int(result.entries_long.sum()), 0)

    def test_uses_breaker_block_when_no_fvg_or_ob_is_available(self) -> None:
        df = _frame(
            [
                (100.0, 100.6, 99.7, 100.2, 1000),
                (100.2, 100.9, 100.0, 100.5, 1000),
                (100.5, 101.1, 100.3, 100.8, 1000),
                (100.8, 101.3, 100.6, 101.0, 1000),
                (101.0, 101.2, 98.8, 100.0, 1200),
                (100.4, 100.5, 99.7, 99.8, 900),
                (99.9, 101.7, 99.8, 101.5, 1400),
                (100.3, 100.4, 99.9, 100.2, 1300),
                (100.2, 101.8, 100.0, 101.6, 1100),
            ]
        )
        params = {
            **self.params,
            "ob_body_min_pct": 0.95,
            "breaker_lookback": 4,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(result.metadata["breaker_entries"], 1)
        self.assertEqual(result.metadata["ob_entries"], 0)

    def test_uses_ifvg_when_other_delivery_arrays_are_unavailable(self) -> None:
        df = _frame(
            [
                (100.0, 100.6, 99.7, 100.2, 1000),
                (100.2, 100.9, 100.0, 100.5, 1000),
                (100.5, 101.1, 100.3, 100.8, 1000),
                (100.8, 101.3, 100.6, 101.0, 1000),
                (101.0, 101.2, 98.8, 100.0, 1200),
                (100.1, 100.8, 99.9, 100.6, 1000),
                (98.6, 98.7, 98.0, 98.3, 900),
                (99.0, 100.0, 98.9, 99.8, 1100),
                (99.8, 101.8, 98.5, 101.6, 1500),
                (98.85, 99.1, 98.75, 99.0, 1200),
                (99.1, 100.2, 98.9, 100.0, 1100),
            ]
        )
        params = {
            **self.params,
            "liq_sweep_recovery_bars": 5,
            "ob_body_min_pct": 0.95,
            "breaker_lookback": 1,
            "ifvg_lookback": 6,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[9]))
        self.assertEqual(result.metadata["ifvg_entries"], 1)
        self.assertEqual(result.metadata["fvg_entries"], 0)
        self.assertEqual(result.metadata["ob_entries"], 0)
        self.assertEqual(result.metadata["breaker_entries"], 0)

    def test_external_liquidity_filter_allows_long_when_sweep_targets_external_low(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "use_external_liquidity_filter": True,
            "external_liquidity_lookback": 4,
            "external_liquidity_tolerance": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["external_targeted_sweeps"]), 1)
        self.assertEqual(int(result.metadata["external_liquidity_filtered_sweeps"]), 0)

    def test_external_liquidity_filter_blocks_internal_only_long_sweep(self) -> None:
        df = _frame(
            [
                (100.0, 100.6, 95.0, 96.0, 1000),
                (96.0, 97.0, 95.2, 96.5, 1000),
                (99.5, 100.5, 99.8, 100.2, 1000),
                (100.2, 101.2, 100.0, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 99.2, 100.4, 1200),
                (100.5, 102.0, 100.3, 101.9, 1400),
                (101.9, 103.1, 101.9, 102.9, 1600),
                (102.1, 102.2, 101.7, 101.9, 1500),
                (101.8, 103.0, 101.6, 102.8, 1300),
            ]
        )
        params = {
            **self.params,
            "liq_sweep_recovery_bars": 5,
            "use_external_liquidity_filter": True,
            "external_liquidity_lookback": 6,
            "external_liquidity_tolerance": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["external_liquidity_filtered_sweeps"]), 1)
        self.assertEqual(int(result.metadata["external_targeted_sweeps"]), 0)

    def test_smt_filter_allows_long_when_peer_does_not_sweep_low(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        ).assign(
            PeerHigh=[100.5, 100.8, 101.0, 101.1, 100.9, 101.2, 101.6, 101.5, 101.8],
            PeerLow=[99.6, 99.8, 100.0, 100.2, 99.9, 100.0, 100.8, 100.7, 101.0],
        )
        params = {
            **self.params,
            "use_smt_filter": True,
            "smt_lookback": 4,
            "smt_threshold": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["smt_confirmed_sweeps"]), 1)
        self.assertEqual(int(result.metadata["smt_filtered_sweeps"]), 0)

    def test_smt_filter_blocks_long_when_peer_also_sweeps_low(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        ).assign(
            PeerHigh=[100.5, 100.8, 101.0, 101.1, 100.9, 101.2, 101.6, 101.5, 101.8],
            PeerLow=[99.6, 99.8, 100.0, 100.2, 98.9, 100.0, 100.8, 100.7, 101.0],
        )
        params = {
            **self.params,
            "use_smt_filter": True,
            "smt_lookback": 4,
            "smt_threshold": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["smt_filtered_sweeps"]), 1)
        self.assertEqual(int(result.metadata["smt_confirmed_sweeps"]), 0)

    def test_smt_filter_requires_real_peer_columns(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "use_smt_filter": True,
            "smt_lookback": 4,
            "smt_threshold": 0.001,
        }

        with self.assertRaisesRegex(ValueError, "PeerHigh and PeerLow"):
            ICTEntryModelStrategy(params=params).generate_signals(df)

    def test_smt_filter_blocks_when_peer_columns_exist_but_bar_is_missing(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        ).assign(
            PeerHigh=[100.5, 100.8, 101.0, 101.1, None, 101.2, 101.6, 101.5, 101.8],
            PeerLow=[99.6, 99.8, 100.0, 100.2, None, 100.0, 100.8, 100.7, 101.0],
        )
        params = {
            **self.params,
            "use_smt_filter": True,
            "smt_lookback": 4,
            "smt_threshold": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["smt_missing_peer_bars"]), 1)
        self.assertEqual(int(result.metadata["smt_confirmed_sweeps"]), 0)

    def test_smt_confirmed_swing_mode_allows_long_when_peer_holds_confirmed_swing_low(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        ).assign(
            PeerHigh=[100.5, 100.8, 101.0, 101.1, 101.0, 101.2, 101.6, 101.5, 101.8],
            PeerLow=[100.4, 100.1, 99.8, 100.2, 99.95, 100.0, 100.8, 100.7, 101.0],
        )
        params = {
            **self.params,
            "use_smt_filter": True,
            "smt_mode": "confirmed_swing",
            "smt_lookback": 6,
            "smt_swing_threshold": 1,
            "smt_threshold": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(str(result.metadata["smt_mode"]), "confirmed_swing")
        self.assertEqual(int(result.metadata["smt_confirmed_sweeps"]), 1)
        self.assertEqual(int(result.metadata["smt_filtered_sweeps"]), 0)

    def test_smt_confirmed_swing_mode_blocks_long_when_peer_breaks_confirmed_swing_low(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        ).assign(
            PeerHigh=[100.5, 100.8, 101.0, 101.1, 101.0, 101.2, 101.6, 101.5, 101.8],
            PeerLow=[100.4, 100.1, 99.8, 100.2, 99.6, 100.0, 100.8, 100.7, 101.0],
        )
        params = {
            **self.params,
            "use_smt_filter": True,
            "smt_mode": "confirmed_swing",
            "smt_lookback": 6,
            "smt_swing_threshold": 1,
            "smt_threshold": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["smt_filtered_sweeps"]), 1)
        self.assertEqual(int(result.metadata["smt_confirmed_sweeps"]), 0)

    def test_smt_confirmed_swing_mode_tracks_missing_peer_structure(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        ).assign(
            PeerHigh=[100.5, 100.8, 101.0, 101.1, 101.0, 101.2, 101.6, 101.5, 101.8],
            PeerLow=[99.6, 99.7, 99.8, 99.9, 100.0, 100.1, 100.8, 100.9, 101.0],
        )
        params = {
            **self.params,
            "use_smt_filter": True,
            "smt_mode": "confirmed_swing",
            "smt_lookback": 6,
            "smt_swing_threshold": 1,
            "smt_threshold": 0.001,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["smt_missing_peer_structure"]), 1)
        self.assertEqual(int(result.metadata["smt_confirmed_sweeps"]), 0)

    def test_amd_filter_allows_long_after_bullish_manipulation_reclaim(self) -> None:
        df = _frame(
            [
                (100.0, 100.4, 99.9, 100.1, 1000),
                (100.1, 100.6, 100.0, 100.4, 1000),
                (100.4, 100.8, 100.2, 100.6, 1000),
                (100.5, 100.7, 99.6, 100.5, 1200),
                (100.6, 101.0, 100.4, 100.8, 1000),
                (100.8, 101.2, 100.6, 101.0, 1000),
                (101.0, 101.3, 100.8, 101.2, 1000),
                (101.2, 101.3, 99.4, 100.2, 1400),
                (100.4, 103.0, 101.4, 102.8, 1600),
                (101.4, 101.6, 101.2, 101.4, 1200),
                (101.5, 102.5, 101.4, 102.0, 1100),
            ]
        )
        params = {
            **self.params,
            "use_amd_filter": True,
            "amd_accumulation_bars": 3,
            "amd_manipulation_threshold": 0.001,
            "amd_require_midpoint_reclaim": True,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[9]))
        self.assertEqual(int(result.metadata["amd_filtered_setups"]), 0)
        self.assertEqual(int(result.metadata["amd_bullish_sessions"]), 1)

    def test_amd_filter_blocks_long_without_bullish_session_path(self) -> None:
        df = _frame(
            [
                (100.0, 100.4, 99.9, 100.1, 1000),
                (100.1, 100.6, 100.0, 100.4, 1000),
                (100.4, 100.8, 100.2, 100.6, 1000),
                (100.5, 100.7, 99.81, 100.5, 1200),
                (100.6, 101.0, 100.4, 100.8, 1000),
                (100.8, 101.2, 100.6, 101.0, 1000),
                (101.0, 101.3, 100.8, 101.2, 1000),
                (101.2, 101.3, 99.4, 100.2, 1400),
                (100.4, 103.0, 101.4, 102.8, 1600),
                (101.4, 101.6, 101.2, 101.4, 1200),
                (101.5, 102.5, 101.4, 102.0, 1100),
            ]
        )
        params = {
            **self.params,
            "use_amd_filter": True,
            "amd_accumulation_bars": 3,
            "amd_manipulation_threshold": 0.001,
            "amd_require_midpoint_reclaim": True,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["amd_filtered_setups"]), 1)
        self.assertEqual(int(result.metadata["amd_bullish_sessions"]), 0)

    def test_macro_timing_allows_long_inside_window(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "use_macro_timing_windows": True,
            "macro_timezone": "UTC",
            "macro_windows": ((14, 45, 15, 5),),
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["macro_timing_filtered_sweeps"]), 0)

    def test_macro_timing_blocks_long_outside_window(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "use_macro_timing_windows": True,
            "macro_timezone": "UTC",
            "macro_windows": ((15, 30, 15, 40),),
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["macro_timing_filtered_sweeps"]), 1)

    def test_prev_session_anchor_allows_long_with_open_buy_side_target(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.2, 95.0, 97.0, 1000),
            (97.0, 98.0, 96.0, 97.5, 1000),
            (97.5, 99.0, 97.0, 98.0, 1000),
            (98.0, 101.0, 97.0, 100.0, 1000),
            (100.0, 105.0, 99.5, 103.0, 1000),
            (103.0, 104.5, 102.0, 104.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_prev_session_anchor_filter": True,
            "prev_session_anchor_tolerance": 0.05,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[13]))
        self.assertEqual(int(result.metadata["prev_session_anchor_filtered_setups"]), 0)
        self.assertGreater(int(result.metadata["prev_session_anchor_long_bars"]), 0)

    def test_prev_session_anchor_blocks_long_after_previous_high_is_already_taken(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.2, 95.0, 97.0, 1000),
            (97.0, 98.0, 96.0, 97.5, 1000),
            (97.5, 99.0, 97.0, 98.0, 1000),
            (98.0, 101.0, 97.0, 100.0, 1000),
            (100.0, 105.0, 99.5, 103.0, 1000),
            (103.0, 104.5, 102.0, 104.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 104.9, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_prev_session_anchor_filter": True,
            "prev_session_anchor_tolerance": 0.05,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["prev_session_anchor_filtered_setups"]), 1)

    def test_prev_session_anchor_soft_mode_allows_long_but_tracks_mismatch(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.2, 95.0, 97.0, 1000),
            (97.0, 98.0, 96.0, 97.5, 1000),
            (97.5, 99.0, 97.0, 98.0, 1000),
            (98.0, 101.0, 97.0, 100.0, 1000),
            (100.0, 105.0, 99.5, 103.0, 1000),
            (103.0, 104.5, 102.0, 104.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_prev_session_anchor_filter": True,
            "prev_session_anchor_tolerance": 0.05,
            "prev_session_anchor_filter_mode": "soft",
            "prev_session_anchor_mismatch_score_penalty": 0.0,
        }

        mismatch_bias = pd.Series([-1] * len(df), index=df.index, dtype=int)
        mismatch_stats = {"long_bars": int(len(df)), "short_bars": 0}
        with patch(
            "src.strategies.ict_entry_model._compute_prev_session_anchor_bias",
            return_value=(mismatch_bias, mismatch_stats),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[13]))
        self.assertEqual(int(result.metadata["prev_session_anchor_filtered_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["prev_session_anchor_mismatch_setups"]), 1)
        self.assertGreaterEqual(int(result.metadata["prev_session_anchor_softened_setups"]), 1)
        self.assertEqual(int(result.metadata["prev_session_anchor_score_penalty_shifts"]), 0)

    def test_prev_session_anchor_soft_penalty_can_fail_score_gate(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.2, 95.0, 97.0, 1000),
            (97.0, 98.0, 96.0, 97.5, 1000),
            (97.5, 99.0, 97.0, 98.0, 1000),
            (98.0, 101.0, 97.0, 100.0, 1000),
            (100.0, 105.0, 99.5, 103.0, 1000),
            (103.0, 104.5, 102.0, 104.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_prev_session_anchor_filter": True,
            "prev_session_anchor_tolerance": 0.05,
            "prev_session_anchor_filter_mode": "soft",
            "prev_session_anchor_mismatch_score_penalty": 3.0,
        }

        mismatch_bias = pd.Series([-1] * len(df), index=df.index, dtype=int)
        mismatch_stats = {"long_bars": int(len(df)), "short_bars": 0}
        with patch(
            "src.strategies.ict_entry_model._compute_prev_session_anchor_bias",
            return_value=(mismatch_bias, mismatch_stats),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["prev_session_anchor_mismatch_setups"]), 1)
        self.assertGreaterEqual(int(result.metadata["prev_session_anchor_softened_setups"]), 1)
        self.assertGreaterEqual(int(result.metadata["prev_session_anchor_score_penalty_shifts"]), 1)
        self.assertGreaterEqual(int(result.metadata["score_filtered_shifts"]), 1)

    def test_reentry_can_retry_the_same_armed_long_setup_after_stop_out(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 102.0, 98.0, 98.2, 1500),
                (101.8, 102.1, 101.4, 101.7, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "fvg_max_age": 6,
            "max_reentries_per_setup": 1,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual([i for i, flag in enumerate(result.entries_long.tolist()) if flag], [7, 9])
        self.assertEqual([i for i, flag in enumerate(result.exits_long.tolist()) if flag], [8, 10])
        self.assertEqual(int(result.metadata["reentry_entries"]), 1)
        self.assertEqual(int(result.metadata["reentry_stop_rearms"]), 1)
        self.assertEqual(int(result.metadata["reentry_exhausted_setups"]), 0)

    def test_continuation_refresh_can_upgrade_an_armed_long_setup_to_a_newer_fvg(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.8, 103.4, 102.6, 103.2, 1200),
                (103.0, 103.1, 102.5, 102.7, 1100),
                (102.7, 103.6, 102.6, 103.5, 1500),
                (103.2, 103.3, 102.7, 103.0, 1200),
                (102.9, 103.8, 102.8, 103.7, 1300),
            ]
        )
        params = {
            **self.params,
            "enable_continuation_entry": True,
            "fvg_max_age": 6,
            "fvg_revisit_min_delay_bars": 1,
        }

        original_detect_fvg_zone = _detect_fvg_zone

        def _patched_detect_fvg_zone(
            frame,
            sweep_index: int,
            current_idx: int,
            *,
            bullish: bool,
            min_gap_pct: float,
        ):
            if bullish and sweep_index == 4 and current_idx >= 6:
                if current_idx >= 8:
                    return (7, 102.6, 103.0)
                return (5, 101.7, 101.9)
            return original_detect_fvg_zone(
                frame,
                sweep_index,
                current_idx,
                bullish=bullish,
                min_gap_pct=min_gap_pct,
            )

        with patch("src.strategies.ict_entry_model._detect_fvg_zone", side_effect=_patched_detect_fvg_zone):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual([i for i, flag in enumerate(result.entries_long.tolist()) if flag], [9])
        self.assertEqual(int(result.metadata["continuation_zone_refreshes"]), 1)
        self.assertEqual(int(result.metadata["continuation_entries"]), 1)

    def test_session_array_refinement_allows_breaker_in_structural_window(self) -> None:
        df = _frame(
            [
                (100.0, 100.6, 99.7, 100.2, 1000),
                (100.2, 100.9, 100.0, 100.5, 1000),
                (100.5, 101.1, 100.3, 100.8, 1000),
                (100.8, 101.3, 100.6, 101.0, 1000),
                (101.0, 101.2, 98.8, 100.0, 1200),
                (100.4, 100.5, 99.7, 99.8, 900),
                (99.9, 101.7, 99.8, 101.5, 1400),
                (100.3, 100.4, 99.9, 100.2, 1300),
                (100.2, 101.8, 100.0, 101.6, 1100),
            ]
        )
        params = {
            **self.params,
            "ob_body_min_pct": 0.95,
            "breaker_lookback": 4,
            "use_session_array_refinement": True,
            "dealing_array_timezone": "UTC",
            "imbalance_array_windows": (),
            "structural_array_windows": ((15, 0, 15, 15),),
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["session_array_filtered_shifts"]), 0)
        self.assertEqual(int(result.metadata["breaker_entries"]), 1)

    def test_session_array_refinement_blocks_breaker_in_imbalance_window(self) -> None:
        df = _frame(
            [
                (100.0, 100.6, 99.7, 100.2, 1000),
                (100.2, 100.9, 100.0, 100.5, 1000),
                (100.5, 101.1, 100.3, 100.8, 1000),
                (100.8, 101.3, 100.6, 101.0, 1000),
                (101.0, 101.2, 98.8, 100.0, 1200),
                (100.4, 100.5, 99.7, 99.8, 900),
                (99.9, 101.7, 99.8, 101.5, 1400),
                (100.3, 100.4, 99.9, 100.2, 1300),
                (100.2, 101.8, 100.0, 101.6, 1100),
            ]
        )
        params = {
            **self.params,
            "ob_body_min_pct": 0.95,
            "breaker_lookback": 4,
            "use_session_array_refinement": True,
            "dealing_array_timezone": "UTC",
            "imbalance_array_windows": ((15, 0, 15, 15),),
            "structural_array_windows": (),
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["session_array_filtered_shifts"]), 1)

    def test_session_array_soft_mode_allows_breaker_but_tracks_softened_shift(self) -> None:
        df = _frame(
            [
                (100.0, 100.6, 99.7, 100.2, 1000),
                (100.2, 100.9, 100.0, 100.5, 1000),
                (100.5, 101.1, 100.3, 100.8, 1000),
                (100.8, 101.3, 100.6, 101.0, 1000),
                (101.0, 101.2, 98.8, 100.0, 1200),
                (100.4, 100.5, 99.7, 99.8, 900),
                (99.9, 101.7, 99.8, 101.5, 1400),
                (100.3, 100.4, 99.9, 100.2, 1300),
                (100.2, 101.8, 100.0, 101.6, 1100),
            ]
        )
        params = {
            **self.params,
            "use_session_array_refinement": True,
            "session_array_filter_mode": "soft",
            "session_array_mismatch_score_penalty": 0.0,
            "dealing_array_timezone": "UTC",
            "imbalance_array_windows": ((15, 0, 15, 15),),
            "structural_array_windows": (),
            "ob_body_min_pct": 0.95,
            "breaker_lookback": 4,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["session_array_filtered_shifts"]), 0)
        self.assertGreaterEqual(int(result.metadata["session_array_softened_shifts"]), 1)
        self.assertEqual(int(result.metadata["session_array_score_penalty_shifts"]), 0)
        self.assertEqual(int(result.metadata["breaker_entries"]), 1)

    def test_session_array_soft_penalty_can_fail_score_gate(self) -> None:
        df = _frame(
            [
                (100.0, 100.6, 99.7, 100.2, 1000),
                (100.2, 100.9, 100.0, 100.5, 1000),
                (100.5, 101.1, 100.3, 100.8, 1000),
                (100.8, 101.3, 100.6, 101.0, 1000),
                (101.0, 101.2, 98.8, 100.0, 1200),
                (100.4, 100.5, 99.7, 99.8, 900),
                (99.9, 101.7, 99.8, 101.5, 1400),
                (100.3, 100.4, 99.9, 100.2, 1300),
                (100.2, 101.8, 100.0, 101.6, 1100),
            ]
        )
        params = {
            **self.params,
            "use_session_array_refinement": True,
            "session_array_filter_mode": "soft",
            "session_array_mismatch_score_penalty": 3.0,
            "dealing_array_timezone": "UTC",
            "imbalance_array_windows": ((15, 0, 15, 15),),
            "structural_array_windows": (),
            "ob_body_min_pct": 0.95,
            "breaker_lookback": 4,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["session_array_filtered_shifts"]), 0)
        self.assertGreaterEqual(int(result.metadata["session_array_softened_shifts"]), 1)
        self.assertGreaterEqual(int(result.metadata["session_array_score_penalty_shifts"]), 1)
        self.assertGreaterEqual(int(result.metadata["score_filtered_shifts"]), 1)

    def test_research_profile_builder_enables_current_ict_stack_without_forcing_smt(self) -> None:
        params = build_ict_research_profile_params()

        self.assertTrue(bool(params["use_kill_zones"]))
        self.assertTrue(bool(params["use_daily_bias_filter"]))
        self.assertTrue(bool(params["use_premium_discount_filter"]))
        self.assertTrue(bool(params["use_external_liquidity_filter"]))
        self.assertTrue(bool(params["use_amd_filter"]))
        self.assertTrue(bool(params["use_macro_timing_windows"]))
        self.assertTrue(bool(params["use_prev_session_anchor_filter"]))
        self.assertTrue(bool(params["use_session_array_refinement"]))
        self.assertFalse(bool(params["use_smt_filter"]))

    def test_no_entry_without_liquidity_sweep(self) -> None:
        df = _frame(
            [
                (100.0, 100.4, 99.8, 100.2, 1000),
                (100.2, 100.6, 100.0, 100.4, 1000),
                (100.4, 100.8, 100.2, 100.6, 1000),
                (100.6, 101.0, 100.4, 100.8, 1000),
                (100.8, 101.2, 100.6, 101.0, 1000),
                (101.0, 101.4, 100.8, 101.2, 1000),
                (101.2, 101.6, 101.0, 101.4, 1000),
                (101.4, 101.8, 101.2, 101.6, 1000),
            ]
        )

        result = ICTEntryModelStrategy(params=self.params).generate_signals(df)

        self.assertFalse(bool(result.entries_long.any()))
        self.assertFalse(bool(result.entries_short.any()))

    def test_kill_zone_blocks_entry_outside_window(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "use_kill_zones": True,
            "kill_zone_timezone": "UTC",
            "london_kill_start": 9,
            "london_kill_end": 10,
            "ny_am_kill_start": 10,
            "ny_am_kill_end": 11,
            "ny_pm_kill_start": 11,
            "ny_pm_kill_end": 12,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["kill_zone_filtered_sweeps"]), 1)

    def test_kill_zone_allows_entry_inside_window(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.3, 102.4, 101.5, 101.8, 1500),
                (101.9, 103.2, 101.7, 103.0, 1300),
            ]
        )
        params = {
            **self.params,
            "use_kill_zones": True,
            "kill_zone_timezone": "UTC",
            "london_kill_start": 14,
            "london_kill_end": 16,
            "ny_am_kill_start": 20,
            "ny_am_kill_end": 21,
            "ny_pm_kill_start": 22,
            "ny_pm_kill_end": 23,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[7]))
        self.assertEqual(int(result.metadata["kill_zone_filtered_sweeps"]), 0)

    def test_daily_bias_allows_long_when_prior_day_is_bullish(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.2, 95.0, 97.0, 1000),
            (97.0, 98.0, 96.0, 97.5, 1000),
            (97.5, 99.0, 97.0, 98.0, 1000),
            (98.0, 101.0, 97.0, 100.0, 1000),
            (100.0, 104.0, 99.5, 103.0, 1000),
            (103.0, 105.0, 102.0, 104.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_daily_bias_filter": True,
            "daily_bias_lookback": 2,
            "daily_bias_bull_threshold": 0.6,
            "daily_bias_bear_threshold": 0.4,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[13]))
        self.assertEqual(int(result.metadata["daily_bias_filtered_setups"]), 0)

    def test_daily_bias_blocks_long_when_prior_day_is_bearish(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 105.0, 99.0, 104.0, 1000),
            (104.0, 104.5, 101.0, 103.0, 1000),
            (103.0, 103.5, 100.0, 102.0, 1000),
            (102.0, 103.0, 97.0, 99.0, 1000),
            (99.0, 100.0, 95.5, 97.0, 1000),
            (97.0, 98.0, 95.0, 96.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_daily_bias_filter": True,
            "daily_bias_lookback": 2,
            "daily_bias_bull_threshold": 0.6,
            "daily_bias_bear_threshold": 0.4,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["daily_bias_filtered_setups"]), 1)

    def test_daily_bias_structure_mode_carries_last_broken_daily_swing_bias(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-08 14:30:00+00:00",
            "2026-01-08 14:35:00+00:00",
            "2026-01-08 14:40:00+00:00",
            "2026-01-09 14:30:00+00:00",
            "2026-01-09 14:35:00+00:00",
            "2026-01-09 14:40:00+00:00",
            "2026-01-12 14:30:00+00:00",
            "2026-01-12 14:35:00+00:00",
            "2026-01-12 14:40:00+00:00",
        ]
        rows = [
            (100.0, 101.0, 99.0, 100.5, 1000),
            (100.5, 102.0, 100.0, 101.5, 1000),
            (101.5, 103.0, 101.0, 102.5, 1000),
            (102.5, 108.0, 102.0, 107.0, 1000),
            (107.0, 110.0, 106.0, 109.0, 1000),
            (109.0, 110.0, 107.0, 108.5, 1000),
            (108.5, 106.0, 104.0, 105.0, 1000),
            (105.0, 105.5, 103.5, 104.5, 1000),
            (104.5, 105.0, 103.0, 104.0, 1000),
            (104.0, 107.0, 103.5, 106.0, 1000),
            (106.0, 108.0, 105.5, 107.0, 1000),
            (107.0, 108.0, 106.0, 107.5, 1000),
            (107.5, 109.0, 107.0, 108.5, 1000),
            (108.5, 112.0, 108.0, 111.5, 1000),
            (111.5, 113.0, 111.0, 112.5, 1000),
            (112.5, 113.0, 112.0, 112.8, 1000),
            (112.8, 113.5, 112.3, 113.0, 1000),
            (113.0, 114.0, 112.8, 113.5, 1000),
        ]
        df = _frame_with_index(timestamps, rows)

        bias = _compute_daily_bias(
            df,
            enabled=True,
            mode="structure",
            lookback=4,
            swing_threshold=1,
            bull_threshold=0.6,
            bear_threshold=0.4,
            trading_timezone="UTC",
        )

        self.assertTrue((bias.iloc[15:18] == 1).all())

    def test_intraday_regime_marks_high_trend_bars_when_adx_rises(self) -> None:
        df = _frame(
            [
                (100.0, 100.4, 99.8, 100.1, 1000),
                (100.1, 100.7, 99.9, 100.6, 1000),
                (100.6, 101.3, 100.5, 101.2, 1000),
                (101.2, 102.0, 101.0, 101.9, 1000),
                (101.9, 103.0, 101.8, 102.9, 1000),
                (102.9, 104.1, 102.8, 103.8, 1000),
                (103.8, 105.0, 103.7, 104.9, 1000),
                (104.9, 106.2, 104.8, 106.0, 1000),
            ]
        )
        regime, stats = _compute_intraday_regime(
            df,
            enabled=True,
            atr_period=3,
            atr_pct_window=3,
            atr_high_mult=1.05,
            adx_period=3,
            adx_trend_threshold=20.0,
        )
        self.assertGreater(int(regime.sum()), 0)
        self.assertGreater(int(stats["high_regime_bars"]), 0)

    def test_higher_timeframe_structure_bias_carries_hourly_break_bias(self) -> None:
        timestamps = [
            "2026-01-07 14:00:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 16:00:00+00:00",
            "2026-01-07 17:00:00+00:00",
            "2026-01-07 18:00:00+00:00",
        ]
        rows = [
            (100.0, 101.0, 99.0, 100.0, 1000),
            (100.0, 105.0, 99.5, 104.0, 1000),
            (104.0, 103.0, 98.0, 102.0, 1000),
            (102.0, 106.0, 101.5, 106.0, 1000),
            (106.0, 107.0, 105.5, 106.5, 1000),
        ]
        df = _frame_with_index(timestamps, rows)
        bias = _compute_higher_timeframe_structure_bias(
            df,
            enabled=True,
            timeframe="1H",
            lookback=3,
            swing_threshold=1,
        )
        self.assertEqual(int(bias.iloc[-1]), 1)

    def test_premium_discount_allows_long_in_discount(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 101.0, 95.0, 97.0, 1000),
            (97.0, 103.0, 96.0, 102.0, 1000),
            (102.0, 104.0, 101.0, 103.0, 1000),
            (103.0, 107.0, 99.0, 104.0, 1000),
            (104.0, 106.0, 100.0, 103.0, 1000),
            (103.0, 105.0, 101.0, 102.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_premium_discount_filter": True,
            "premium_discount_lookback": 2,
            "premium_discount_neutral_band": 0.05,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertTrue(bool(result.entries_long.iloc[13]))
        self.assertEqual(int(result.metadata["premium_discount_filtered_setups"]), 0)

    def test_premium_discount_blocks_long_in_premium(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.5, 95.0, 96.0, 1000),
            (96.0, 101.0, 95.5, 100.5, 1000),
            (100.5, 103.0, 99.5, 102.0, 1000),
            (102.0, 104.0, 98.0, 103.0, 1000),
            (103.0, 104.0, 99.0, 102.0, 1000),
            (102.0, 103.0, 100.0, 101.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_premium_discount_filter": True,
            "premium_discount_lookback": 2,
            "premium_discount_neutral_band": 0.05,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["premium_discount_filtered_setups"]), 1)

    def test_premium_discount_soft_mode_allows_long_but_tracks_mismatch(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.5, 95.0, 96.0, 1000),
            (96.0, 101.0, 95.5, 100.5, 1000),
            (100.5, 103.0, 99.5, 102.0, 1000),
            (102.0, 104.0, 98.0, 103.0, 1000),
            (103.0, 104.0, 99.0, 102.0, 1000),
            (102.0, 103.0, 100.0, 101.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_premium_discount_filter": True,
            "premium_discount_lookback": 2,
            "premium_discount_neutral_band": 0.05,
            "premium_discount_filter_mode": "soft",
            "premium_discount_mismatch_score_penalty": 0.0,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["premium_discount_filtered_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["premium_discount_mismatch_setups"]), 1)
        self.assertGreaterEqual(int(result.metadata["premium_discount_softened_setups"]), 1)
        self.assertEqual(int(result.metadata["premium_discount_score_penalty_shifts"]), 0)

    def test_premium_discount_soft_penalty_can_fail_score_gate(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.5, 95.0, 96.0, 1000),
            (96.0, 101.0, 95.5, 100.5, 1000),
            (100.5, 103.0, 99.5, 102.0, 1000),
            (102.0, 104.0, 98.0, 103.0, 1000),
            (103.0, 104.0, 99.0, 102.0, 1000),
            (102.0, 103.0, 100.0, 101.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_premium_discount_filter": True,
            "premium_discount_lookback": 2,
            "premium_discount_neutral_band": 0.05,
            "premium_discount_filter_mode": "soft",
            "premium_discount_mismatch_score_penalty": 3.0,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["premium_discount_filtered_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["premium_discount_softened_setups"]), 1)
        self.assertGreaterEqual(int(result.metadata["premium_discount_score_penalty_shifts"]), 1)
        self.assertGreaterEqual(int(result.metadata["score_filtered_shifts"]), 1)

    def test_context_filters_count_independently_on_the_same_rejected_sweep(self) -> None:
        timestamps = [
            "2026-01-05 14:30:00+00:00",
            "2026-01-05 14:35:00+00:00",
            "2026-01-05 14:40:00+00:00",
            "2026-01-06 14:30:00+00:00",
            "2026-01-06 14:35:00+00:00",
            "2026-01-06 14:40:00+00:00",
            "2026-01-07 14:30:00+00:00",
            "2026-01-07 14:35:00+00:00",
            "2026-01-07 14:40:00+00:00",
            "2026-01-07 14:45:00+00:00",
            "2026-01-07 14:50:00+00:00",
            "2026-01-07 14:55:00+00:00",
            "2026-01-07 15:00:00+00:00",
            "2026-01-07 15:05:00+00:00",
            "2026-01-07 15:10:00+00:00",
        ]
        rows = [
            (100.0, 100.5, 95.0, 96.0, 1000),
            (96.0, 101.0, 95.5, 100.5, 1000),
            (100.5, 103.0, 99.5, 102.0, 1000),
            (102.0, 103.0, 97.0, 99.0, 1000),
            (99.0, 100.0, 95.5, 97.0, 1000),
            (97.0, 98.0, 95.0, 96.0, 1000),
            (100.0, 101.0, 99.8, 100.5, 1000),
            (100.5, 101.2, 100.3, 101.0, 1000),
            (101.0, 101.5, 100.8, 101.4, 1000),
            (101.4, 101.8, 101.1, 101.6, 1000),
            (101.5, 101.6, 98.8, 100.0, 1200),
            (100.1, 101.9, 99.9, 101.7, 1400),
            (101.9, 103.0, 101.9, 102.8, 1600),
            (102.3, 102.4, 101.5, 101.8, 1500),
            (101.9, 103.2, 101.7, 103.0, 1300),
        ]
        df = _frame_with_index(timestamps, rows)
        params = {
            **self.params,
            "use_daily_bias_filter": True,
            "daily_bias_lookback": 2,
            "daily_bias_bull_threshold": 0.6,
            "daily_bias_bear_threshold": 0.4,
            "use_premium_discount_filter": True,
            "premium_discount_lookback": 2,
            "premium_discount_neutral_band": 0.05,
        }

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["daily_bias_filtered_setups"]), 1)
        self.assertGreaterEqual(int(result.metadata["premium_discount_filtered_setups"]), 1)

    def test_latest_confirmed_swing_high_uses_confirmed_pivot(self) -> None:
        series = pd.Series([100.0, 101.0, 103.0, 101.5, 100.5, 102.0, 101.0])
        level = _latest_confirmed_swing_level(
            series,
            idx=6,
            lookback=6,
            threshold=1,
            swing_type="high",
        )
        self.assertEqual(level, 103.0)

    def test_latest_confirmed_swing_low_uses_confirmed_pivot(self) -> None:
        series = pd.Series([103.0, 101.0, 99.0, 100.5, 101.5, 100.0, 101.0])
        level = _latest_confirmed_swing_level(
            series,
            idx=6,
            lookback=6,
            threshold=1,
            swing_type="low",
        )
        self.assertEqual(level, 99.0)

    def test_generate_signals_returns_strategy_result_after_full_loop(self) -> None:
        df = _frame(
            [
                (100.0, 100.4, 99.8, 100.2, 1000),
                (100.2, 100.5, 100.0, 100.3, 1000),
                (100.3, 100.6, 100.1, 100.4, 1000),
                (100.4, 100.7, 100.2, 100.5, 1000),
                (100.5, 100.8, 100.3, 100.6, 1000),
                (100.6, 100.9, 100.4, 100.7, 1000),
            ]
        )

        result = ICTEntryModelStrategy(params=self.params).generate_signals(df)

        self.assertIsInstance(result, StrategyResult)
        self.assertEqual(len(result.entries_long), len(df))
        self.assertEqual(len(result.exits_long), len(df))
        self.assertEqual(len(result.entries_short), len(df))
        self.assertEqual(len(result.exits_short), len(df))
        self.assertIsInstance(result.metadata, dict)

    def test_counts_sweep_blocked_by_existing_pending(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.0, 100.5, 98.0, 99.2, 1100),
                (99.2, 100.2, 99.0, 99.8, 1000),
                (99.8, 100.1, 99.5, 99.9, 1000),
            ]
        )

        result = ICTEntryModelStrategy(params=self.params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["bullish_sweeps"]), 1)
        self.assertGreaterEqual(int(result.metadata["sweep_blocked_by_existing_pending"]), 1)

    def test_allows_multiple_same_direction_pending_setups_when_capacity_is_raised(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.0, 100.5, 98.0, 99.2, 1100),
                (99.2, 100.2, 99.0, 99.8, 1000),
                (99.8, 100.1, 99.5, 99.9, 1000),
            ]
        )
        params = {**self.params, "max_pending_setups_per_direction": 2}

        result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["bullish_sweeps"]), 2)
        self.assertEqual(int(result.metadata["sweep_blocked_by_existing_pending"]), 0)

    def test_counts_sweep_expired_before_shift(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.0, 100.5, 98.0, 99.2, 1100),
                (99.2, 100.2, 99.0, 99.8, 1000),
                (99.8, 100.1, 99.5, 99.9, 1000),
                (99.9, 100.0, 99.7, 99.8, 1000),
            ]
        )

        result = ICTEntryModelStrategy(params=self.params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["sweep_expired_before_shift"]), 1)
        self.assertEqual(int(result.metadata["armed_setup_expired_before_retest"]), 0)

    def test_counts_armed_setup_expired_before_retest(self) -> None:
        df = _frame(
            [
                (100.0, 101.0, 99.8, 100.5, 1000),
                (100.5, 101.2, 100.3, 101.0, 1000),
                (101.0, 101.5, 100.8, 101.4, 1000),
                (101.4, 101.8, 101.1, 101.6, 1000),
                (101.5, 101.6, 98.8, 100.0, 1200),
                (100.1, 101.9, 99.9, 101.7, 1400),
                (101.9, 103.0, 101.9, 102.8, 1600),
                (102.1, 102.5, 102.0, 102.3, 1200),
                (102.3, 102.7, 102.2, 102.6, 1200),
                (102.6, 103.0, 102.4, 102.8, 1200),
                (102.8, 103.1, 102.6, 102.9, 1200),
                (103.0, 103.2, 102.8, 103.1, 1200),
            ]
        )

        result = ICTEntryModelStrategy(params=self.params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["bullish_shifts"]), 1)
        self.assertEqual(int(result.metadata["bullish_retest_candidates"]), 0)
        self.assertGreaterEqual(int(result.metadata["armed_setup_expired_before_retest"]), 1)


class ICTMTFTopDownStrategyTest(unittest.TestCase):
    def test_execution_candidate_profile_uses_repaired_execution_overrides(self) -> None:
        params = build_ict_mtf_topdown_continuation_execution_candidate_profile_params()

        self.assertEqual(int(params["mtf_confirmation_structure_lookback"]), 4)
        self.assertEqual(int(params["mtf_trigger_expiry_bars"]), 15)
        self.assertEqual(float(params["mtf_trigger_close_ratio"]), 0.6)

    def test_setup_execution_candidate_profile_adds_best_setup_refinements(self) -> None:
        params = build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params()

        self.assertEqual(int(params["mtf_confirmation_structure_lookback"]), 4)
        self.assertEqual(int(params["mtf_trigger_expiry_bars"]), 15)
        self.assertEqual(float(params["mtf_setup_displacement_body_min_pct"]), 0.30)
        self.assertEqual(float(params["mtf_setup_fvg_min_gap_pct"]), 0.0008)

    def test_timing_candidate_profile_excludes_the_13_et_hour(self) -> None:
        params = build_ict_mtf_topdown_continuation_timing_candidate_profile_params()

        self.assertEqual(tuple(params["mtf_allowed_entry_hours"]), (9, 10, 11, 12, 14, 15))
        self.assertEqual(float(params["mtf_setup_displacement_body_min_pct"]), 0.30)
        self.assertEqual(float(params["mtf_setup_fvg_min_gap_pct"]), 0.0008)

    def test_regularized_long_only_candidate_disables_shorts_and_uses_stronger_hours(self) -> None:
        params = build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params()

        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertEqual(tuple(params["mtf_allowed_entry_hours"]), (10, 11, 12, 14, 15))
        self.assertEqual(float(params["mtf_setup_fvg_min_gap_pct"]), 0.0010)
        self.assertEqual(int(params["mtf_confirmation_structure_lookback"]), 4)

    def test_regularized_long_only_am_candidate_limits_entries_to_morning_window(self) -> None:
        params = build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params()

        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(tuple(params["mtf_allowed_entry_hours"]), (9, 10, 11, 12))
        self.assertEqual(float(params["mtf_setup_fvg_min_gap_pct"]), 0.0010)
        self.assertEqual(int(params["mtf_trigger_structure_lookback"]), 7)

    def test_core_400_baseline_profile_targets_high_density_without_forcing_smt(self) -> None:
        params = build_ict_core_400_baseline_profile_params()

        self.assertFalse(bool(params["trade_sessions"]))
        self.assertTrue(bool(params["enable_continuation_entry"]))
        self.assertEqual(int(params["structure_lookback"]), 5)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.0003)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 2)
        self.assertFalse(bool(params["use_smt_filter"]))

    def test_core_400_short_only_profile_disables_long_entries(self) -> None:
        params = build_ict_core_400_short_only_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertTrue(bool(params["allow_short_entries"]))
        self.assertFalse(bool(params["trade_sessions"]))

    def test_core_400_short_stat_bias_candidate_enables_daily_bias_and_smt(self) -> None:
        params = build_ict_core_400_short_stat_bias_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertTrue(bool(params["use_daily_bias_filter"]))
        self.assertTrue(bool(params["use_smt_filter"]))
        self.assertEqual(float(params["stop_loss_atr_mult"]), 1.5)
        self.assertEqual(float(params["take_profit_rr"]), 2.5)

    def test_core_400_short_structure_bias_candidate_uses_structure_daily_bias(self) -> None:
        params = build_ict_core_400_short_structure_bias_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertTrue(bool(params["use_daily_bias_filter"]))
        self.assertEqual(str(params["daily_bias_mode"]), "structure")
        self.assertEqual(int(params["daily_bias_lookback"]), 5)
        self.assertEqual(float(params["stop_loss_atr_mult"]), 1.5)

    def test_core_400_short_structure_bias_lb6_candidate_extends_bias_lookback(self) -> None:
        params = build_ict_core_400_short_structure_bias_lb6_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertTrue(bool(params["use_daily_bias_filter"]))
        self.assertEqual(str(params["daily_bias_mode"]), "structure")
        self.assertEqual(int(params["daily_bias_lookback"]), 6)
        self.assertEqual(float(params["stop_loss_atr_mult"]), 1.5)

    def test_core_400_short_structure_refined_candidate_relaxes_sweep_geometry(self) -> None:
        params = build_ict_core_400_short_structure_refined_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["daily_bias_lookback"]), 6)
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000275)

    def test_core_400_short_structure_refined_density_candidate_extends_recovery(self) -> None:
        params = build_ict_core_400_short_structure_refined_density_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["daily_bias_lookback"]), 6)
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00025)
        self.assertEqual(int(params["liq_sweep_recovery_bars"]), 18)
        self.assertEqual(int(params["slow_recovery_bars"]), 24)

    def test_core_400_short_structure_refined_recovery_candidate_uses_looser_sweep_threshold(self) -> None:
        params = build_ict_core_400_short_structure_refined_recovery_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000275)
        self.assertEqual(int(params["liq_sweep_recovery_bars"]), 18)
        self.assertEqual(int(params["slow_recovery_bars"]), 24)

    def test_core_400_short_structure_refined_recovery_sl135_candidate_tightens_stop(self) -> None:
        params = build_ict_core_400_short_structure_refined_recovery_sl135_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000275)
        self.assertEqual(float(params["stop_loss_atr_mult"]), 1.35)

    def test_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_extends_bias_window(self) -> None:
        params = build_ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000275)
        self.assertEqual(float(params["stop_loss_atr_mult"]), 1.35)
        self.assertEqual(int(params["daily_bias_lookback"]), 8)

    def test_core_400_short_structure_refined_capacity_candidate_adds_pending_capacity(self) -> None:
        params = build_ict_core_400_short_structure_refined_capacity_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000275)
        self.assertEqual(int(params["liq_sweep_recovery_bars"]), 18)
        self.assertEqual(int(params["slow_recovery_bars"]), 24)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 3)

    def test_core_400_short_structure_refined_capacity_pending4_candidate_adds_one_more_pending_slot(self) -> None:
        params = build_ict_core_400_short_structure_refined_capacity_pending4_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000275)
        self.assertEqual(int(params["liq_sweep_recovery_bars"]), 18)
        self.assertEqual(int(params["slow_recovery_bars"]), 24)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 4)

    def test_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_softens_sweep_gate(self) -> None:
        params = build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000265)
        self.assertEqual(int(params["liq_sweep_recovery_bars"]), 18)
        self.assertEqual(int(params["slow_recovery_bars"]), 24)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 4)

    def test_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_candidate_tightens_stop(self) -> None:
        params = build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_candidate_profile_params()

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000265)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 4)
        self.assertEqual(float(params["stop_loss_atr_mult"]), 1.35)

    def test_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_extends_bias_window(self) -> None:
        params = (
            build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_profile_params()
        )

        self.assertFalse(bool(params["allow_long_entries"]))
        self.assertEqual(int(params["structure_lookback"]), 6)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.000265)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 4)
        self.assertEqual(float(params["stop_loss_atr_mult"]), 1.35)
        self.assertEqual(int(params["daily_bias_lookback"]), 8)

    def test_qualified_reversal_balance_long_refined_candidate_only_relaxes_long_sweep_gate(self) -> None:
        params = build_ict_lite_reversal_qualified_reversal_balance_long_refined_candidate_profile_params()

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertFalse(bool(params["enable_continuation_entry"]))
        self.assertEqual(int(params["structure_lookback"]), 12)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.0004)

    def test_qualified_reversal_balance_long_refined_timing_candidate_enters_one_bar_earlier(self) -> None:
        params = build_ict_lite_reversal_qualified_reversal_balance_long_refined_timing_candidate_profile_params()

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.0004)
        self.assertEqual(int(params["fvg_revisit_min_delay_bars"]), 3)

    def test_qualified_reversal_balance_long_refined_sweep035_candidate_softens_sweep_gate(self) -> None:
        params = build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_candidate_profile_params()

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(int(params["fvg_revisit_min_delay_bars"]), 3)
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00035)

    def test_qualified_reversal_balance_long_refined_sweep035_capacity_candidate_adds_pending_slot(self) -> None:
        params = build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_capacity_candidate_profile_params()

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00035)
        self.assertEqual(int(params["fvg_revisit_min_delay_bars"]), 3)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 2)

    def test_qualified_reversal_balance_long_refined_sweep035_structure11_pending3_candidate_reopens_long_density(self) -> None:
        params = (
            build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure11_pending3_candidate_profile_params()
        )

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00035)
        self.assertEqual(int(params["fvg_revisit_min_delay_bars"]), 3)
        self.assertEqual(int(params["structure_lookback"]), 11)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 3)

    def test_qualified_reversal_balance_long_refined_sweep035_structure10_pending3_candidate_adds_more_shift_tolerance(self) -> None:
        params = (
            build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_pending3_candidate_profile_params()
        )

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00035)
        self.assertEqual(int(params["fvg_revisit_min_delay_bars"]), 3)
        self.assertEqual(int(params["structure_lookback"]), 10)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 3)

    def test_qualified_reversal_balance_long_refined_sweep035_structure10_depth04_pending3_candidate_relaxes_revisit_depth(self) -> None:
        params = (
            build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_depth04_pending3_candidate_profile_params()
        )

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00035)
        self.assertEqual(int(params["structure_lookback"]), 10)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 3)
        self.assertEqual(float(params["fvg_revisit_depth_ratio"]), 0.4)

    def test_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_pending3_candidate_softens_gap_floor(self) -> None:
        params = (
            build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_pending3_candidate_profile_params()
        )

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00035)
        self.assertEqual(int(params["structure_lookback"]), 10)
        self.assertEqual(int(params["max_pending_setups_per_direction"]), 3)
        self.assertEqual(float(params["fvg_revisit_depth_ratio"]), 0.4)
        self.assertEqual(float(params["fvg_min_gap_pct"]), 0.0002)

    def test_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_candidate_enables_intrabar_shift_tolerance(self) -> None:
        params = (
            build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_candidate_profile_params()
        )

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["liq_sweep_threshold"]), 0.00035)
        self.assertEqual(float(params["fvg_min_gap_pct"]), 0.0002)
        self.assertEqual(float(params["fvg_revisit_depth_ratio"]), 0.4)
        self.assertEqual(float(params["structure_shift_intrabar_tolerance_ratio"]), 0.20)
        self.assertEqual(float(params["structure_shift_intrabar_close_position_min_pct"]), 0.70)

    def test_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_candidate_raises_long_target(self) -> None:
        params = (
            build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_candidate_profile_params()
        )

        self.assertTrue(bool(params["allow_long_entries"]))
        self.assertFalse(bool(params["allow_short_entries"]))
        self.assertEqual(float(params["fvg_min_gap_pct"]), 0.0002)
        self.assertEqual(float(params["structure_shift_intrabar_tolerance_ratio"]), 0.20)
        self.assertEqual(float(params["structure_shift_intrabar_close_position_min_pct"]), 0.70)
        self.assertEqual(float(params["take_profit_rr"]), 4.5)

    def test_complete_soft_premium_profile_preserves_full_stack_but_softens_filter(self) -> None:
        params = build_ict_complete_soft_premium_profile_params(enable_smt=True)

        self.assertTrue(bool(params["use_kill_zones"]))
        self.assertTrue(bool(params["use_daily_bias_filter"]))
        self.assertTrue(bool(params["use_premium_discount_filter"]))
        self.assertEqual(str(params["premium_discount_filter_mode"]), "soft")
        self.assertTrue(bool(params["use_smt_filter"]))

    def test_complete_soft_session_array_profile_preserves_full_stack_but_softens_filter(self) -> None:
        params = build_ict_complete_soft_session_array_profile_params(enable_smt=True)

        self.assertTrue(bool(params["use_session_array_refinement"]))
        self.assertEqual(str(params["session_array_filter_mode"]), "soft")
        self.assertTrue(bool(params["use_macro_timing_windows"]))

    def test_complete_soft_prev_session_profile_preserves_full_stack_but_softens_filter(self) -> None:
        params = build_ict_complete_soft_prev_session_profile_params(enable_smt=True)

        self.assertTrue(bool(params["use_prev_session_anchor_filter"]))
        self.assertEqual(str(params["prev_session_anchor_filter_mode"]), "soft")
        self.assertTrue(bool(params["use_external_liquidity_filter"]))

    def _intraday_frame(
        self,
        *,
        trigger_ready: bool = True,
        trigger_inside_zone: bool = True,
    ) -> pd.DataFrame:
        rows: list[tuple[float, float, float, float, float]] = []
        for idx in range(31):
            if idx < 21:
                open_ = 100.0 + idx * 0.02
                close = open_ + 0.02
                high = close + 0.05
                low = open_ - 0.05
            elif idx == 21 and trigger_ready and trigger_inside_zone:
                open_, high, low, close = 102.1, 102.3, 101.8, 102.0
            elif idx == 22 and trigger_ready and trigger_inside_zone:
                open_, high, low, close = 102.0, 102.9, 101.9, 102.8
            elif idx >= 21 and trigger_ready and not trigger_inside_zone:
                open_, high, low, close = 102.8, 103.4, 102.7, 103.2
            elif idx >= 21:
                open_, high, low, close = 100.2, 100.3, 100.0, 100.1
            else:
                open_, high, low, close = 101.1, 101.3, 100.9, 101.0
            rows.append((open_, high, low, close, 1000.0))
        return _frame_1m(rows)

    def _intraday_frame_with_tail(
        self,
        tail_rows: list[tuple[float, float, float, float, float]],
        *,
        base_high: float = 102.4,
    ) -> pd.DataFrame:
        rows: list[tuple[float, float, float, float, float]] = []
        for _ in range(21):
            rows.append((102.0, base_high, 101.9, 102.1, 1000.0))
        rows.extend(tail_rows)
        while len(rows) < 31:
            rows.append((102.8, 103.0, 102.7, 102.9, 1000.0))
        return _frame_1m(rows[:31])

    def _setup_frame_15m(self) -> pd.DataFrame:
        return _frame_with_index(
            [
                "2026-01-07 14:00:00+00:00",
                "2026-01-07 14:15:00+00:00",
                "2026-01-07 14:30:00+00:00",
                "2026-01-07 14:45:00+00:00",
            ],
            [
                (100.0, 100.9, 99.7, 100.4, 1000.0),
                (100.4, 100.8, 99.2, 100.1, 1000.0),
                (100.1, 101.0, 99.9, 100.7, 1000.0),
                (100.9, 103.0, 101.2, 102.8, 1200.0),
            ],
        )

    def _setup_frame_15m_plain_breakout(self) -> pd.DataFrame:
        return _frame_with_index(
            [
                "2026-01-07 14:00:00+00:00",
                "2026-01-07 14:15:00+00:00",
                "2026-01-07 14:30:00+00:00",
                "2026-01-07 14:45:00+00:00",
            ],
            [
                (100.0, 100.9, 99.7, 100.4, 1000.0),
                (100.4, 101.1, 100.1, 100.9, 1000.0),
                (100.9, 101.4, 100.7, 101.2, 1000.0),
                (101.3, 103.0, 101.2, 102.8, 1200.0),
            ],
        )

    def _confirm_frame_5m(self) -> pd.DataFrame:
        return _frame_with_index(
            [
                "2026-01-07 14:35:00+00:00",
                "2026-01-07 14:40:00+00:00",
                "2026-01-07 14:45:00+00:00",
                "2026-01-07 14:50:00+00:00",
            ],
            [
                (100.0, 101.0, 99.8, 100.8, 1000.0),
                (100.8, 101.0, 100.4, 100.7, 1000.0),
                (100.7, 101.4, 100.5, 101.0, 1000.0),
                (101.0, 102.4, 101.4, 102.3, 1000.0),
            ],
        )

    def _mtf_params(self, **overrides: float | int | bool | str) -> dict[str, object]:
        return build_ict_mtf_topdown_continuation_profile_params(
            overrides={
                "trade_sessions": False,
                "mtf_setup_structure_lookback": 1,
                "mtf_confirmation_structure_lookback": 1,
                "mtf_trigger_structure_lookback": 1,
                "min_score_to_trade": 4,
                "mtf_neutral_hourly_min_score": 4,
                **overrides,
            }
        )

    def test_resample_ohlcv_right_closed_blocks_future_bar_leakage(self) -> None:
        df = _frame_1m(
            [
                (1.0, 1.1, 0.9, 1.0, 100.0),
                (2.0, 2.1, 1.9, 2.0, 100.0),
                (3.0, 3.1, 2.9, 3.0, 100.0),
                (4.0, 4.1, 3.9, 4.0, 100.0),
                (5.0, 5.1, 4.9, 5.0, 100.0),
                (6.0, 6.1, 5.9, 6.0, 100.0),
                (7.0, 7.1, 6.9, 7.0, 100.0),
            ]
        )

        resampled = resample_ohlcv(df, "5m", label="right", closed="right")

        self.assertEqual(float(resampled.loc[pd.Timestamp("2026-01-07 14:35:00+00:00"), "Close"]), 6.0)
        self.assertEqual(float(resampled.loc[pd.Timestamp("2026-01-07 14:40:00+00:00"), "Close"]), 7.0)

    def test_mtf_daily_and_4h_conflict_blocks_setup(self) -> None:
        df = self._intraday_frame()
        params = self._mtf_params()
        bias_long = pd.Series(1, index=df.index, dtype=int)
        bias_short = pd.Series(-1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[bias_long, bias_short, bias_short],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["mtf_daily_blocked"]), 1)

    def test_mtf_neutral_daily_but_aligned_4h_and_1h_allows_long_entry(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params()
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.entries_short.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 1)
        self.assertEqual(int(result.metadata["mtf_5m_confirms"]), 1)
        self.assertEqual(int(result.metadata["mtf_1m_triggers"]), 1)

    def test_mtf_neutral_daily_requires_hourly_alignment(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params()
        daily = pd.Series(0, index=df.index, dtype=int)
        four_hour = pd.Series(1, index=df.index, dtype=int)
        hourly_neutral = pd.Series(0, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, four_hour, hourly_neutral],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["mtf_1h_blocked"]), 1)

    def test_mtf_plain_breakout_without_sweep_does_not_create_setup(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params()
        bias = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[bias, bias, bias],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m_plain_breakout(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["mtf_setup_missing_context"]), 1)

    def test_mtf_requires_15m_zone_before_any_entry(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params()
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=None,
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["mtf_setup_missing_zone"]), 1)

    def test_mtf_uses_order_block_fallback_when_fvg_missing(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params()
        bias = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[bias, bias, bias],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=None,
        ), patch(
            "src.strategies.ict_entry_model._detect_order_block_zone",
            return_value=(2, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["mtf_setup_ob_zones"]), 1)

    def test_mtf_requires_1m_trigger_after_5m_confirmation(self) -> None:
        df = self._intraday_frame(trigger_ready=False)
        params = self._mtf_params()
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 1)
        self.assertEqual(int(result.metadata["mtf_5m_confirms"]), 1)
        self.assertEqual(int(result.metadata["mtf_1m_triggers"]), 0)

    def test_mtf_trigger_requires_zone_contact_even_for_micro_shift(self) -> None:
        df = self._intraday_frame(trigger_ready=True, trigger_inside_zone=False)
        params = self._mtf_params()
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_5m_confirms"]), 1)
        self.assertEqual(int(result.metadata["mtf_1m_triggers"]), 0)

    def test_mtf_touch_activation_allows_1m_trigger_after_zone_edge_touch(self) -> None:
        df = self._intraday_frame_with_tail(
            [
                (102.30, 102.40, 102.10, 102.20, 1000.0),
                (102.60, 102.85, 102.60, 102.75, 1000.0),
            ]
        )
        params = self._mtf_params(
            mtf_trigger_structure_lookback=20,
            mtf_max_stop_distance_atr_mult=0.0,
            mtf_touch_trigger_bars=5,
        )
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertGreaterEqual(int(result.metadata["mtf_touch_activated_triggers"]), 1)

    def test_mtf_fast_retest_entry_can_fill_without_1m_micro_shift(self) -> None:
        df = self._intraday_frame_with_tail(
            [
                (102.20, 102.50, 102.20, 102.40, 1000.0),
            ]
        )
        params = self._mtf_params(
            mtf_trigger_structure_lookback=50,
            mtf_trigger_close_ratio=0.95,
            mtf_trigger_body_min_pct=0.95,
            mtf_fast_retest_entry_enabled=True,
            mtf_fast_retest_displacement_body_min_pct=0.55,
            mtf_fast_retest_displacement_close_ratio=0.7,
            mtf_fast_retest_min_close_ratio=0.55,
        )
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertGreaterEqual(int(result.metadata["mtf_fast_retest_entries"]), 1)

    def test_mtf_trigger_rejection_can_use_wick_and_volume_expansion(self) -> None:
        df = self._intraday_frame_with_tail(
            [
                (102.30, 102.45, 101.55, 102.40, 2500.0),
            ]
        )
        params = self._mtf_params(
            mtf_trigger_structure_lookback=50,
            mtf_trigger_close_ratio=0.9,
            mtf_trigger_body_min_pct=0.1,
            mtf_trigger_rejection_wick_ratio=0.6,
            mtf_trigger_rejection_volume_ratio=1.5,
            mtf_fast_retest_entry_enabled=False,
        )
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["mtf_fast_retest_entries"]), 0)

    def test_mtf_stop_distance_filter_blocks_entries_with_oversized_stop(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params(mtf_max_stop_distance_atr_mult=0.05)
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["mtf_stop_distance_filtered_entries"]), 1)

    def test_mtf_entry_weekday_filter_blocks_entry_outside_allowlist(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params(mtf_allowed_entry_weekdays=(4,))
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_5m_confirms"]), 1)
        self.assertEqual(int(result.metadata["mtf_1m_triggers"]), 0)
        self.assertEqual(int(result.metadata["mtf_weekday_blocked"]), 1)

    def test_mtf_entry_hour_filter_blocks_entry_outside_allowlist(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params(mtf_allowed_entry_hours=(10,))
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_5m_confirms"]), 1)
        self.assertEqual(int(result.metadata["mtf_1m_triggers"]), 0)
        self.assertEqual(int(result.metadata["mtf_hour_blocked"]), 1)

    def test_mtf_entry_timing_allowlists_permit_entry_inside_window(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params(
            mtf_allowed_entry_weekdays=(2,),
            mtf_allowed_entry_hours=(9,),
        )
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 1)
        self.assertEqual(int(result.metadata["mtf_weekday_blocked"]), 0)
        self.assertEqual(int(result.metadata["mtf_hour_blocked"]), 0)

    def test_mtf_neutral_hourly_requires_real_high_quality_score(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params(mtf_neutral_hourly_min_score=100.0)
        aligned = pd.Series(1, index=df.index, dtype=int)
        hourly_neutral = pd.Series(0, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[aligned, aligned, hourly_neutral],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["mtf_1h_blocked"]), 1)

    def test_mtf_quality_weights_respect_zero_configuration(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params(
            score_sweep_depth_quality=0.0,
            score_displacement_quality=0.0,
            score_fvg_gap_quality=0.0,
            min_score_to_trade=6,
        )
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[aligned, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertEqual(int(result.metadata["mtf_15m_setups"]), 0)
        self.assertGreaterEqual(int(result.metadata["mtf_setup_score_filtered"]), 1)

    def test_mtf_rr_gate_blocks_entry_when_reward_is_below_floor(self) -> None:
        df = self._intraday_frame(trigger_ready=True)
        params = self._mtf_params(take_profit_rr=1.0, min_reward_risk_ratio=1.5)
        daily = pd.Series(0, index=df.index, dtype=int)
        aligned = pd.Series(1, index=df.index, dtype=int)

        with patch(
            "src.strategies.ict_entry_model._compute_higher_timeframe_structure_bias",
            side_effect=[daily, aligned, aligned],
        ), patch(
            "src.strategies.ict_entry_model.resample_ohlcv",
            side_effect=[self._setup_frame_15m(), self._confirm_frame_5m()],
        ), patch(
            "src.strategies.ict_entry_model._detect_fvg_zone",
            return_value=(1, 101.5, 102.5),
        ):
            result = ICTEntryModelStrategy(params=params).generate_signals(df)

        self.assertEqual(int(result.entries_long.sum()), 0)
        self.assertGreaterEqual(int(result.metadata["rr_filtered_entries"]), 1)


if __name__ == "__main__":
    unittest.main()
