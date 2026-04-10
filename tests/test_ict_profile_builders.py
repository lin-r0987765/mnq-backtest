from __future__ import annotations

import unittest

from src.strategies.ict_entry_model import (
    build_ict_lite_reversal_profile_params,
    build_ict_lite_reversal_regime_mtf_alignment_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_profile_params,
    build_ict_lite_reversal_qualified_continuation_density_profile_params,
    build_ict_lite_reversal_quick_density_repair_profile_params,
    build_ict_lite_reversal_quick_swing_structure_repair_profile_params,
    build_ict_lite_reversal_dual_speed_recovery_profile_params,
    build_ict_lite_reversal_relaxed_smt_profile_params,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_profile_params,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
    build_ict_paired_survivor_plus_session_array_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_params,
    build_ict_paired_survivor_profile_params,
    build_ict_research_profile_params,
    build_ict_strict_soft_premium_profile_params,
    build_ict_strict_soft_prev_session_profile_params,
    build_ict_strict_soft_session_array_profile_params,
)


class ICTProfileBuildersTest(unittest.TestCase):
    def test_research_profile_keeps_full_context_stack_enabled(self) -> None:
        params = build_ict_research_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_daily_bias_filter"])
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertTrue(params["use_amd_filter"])
        self.assertTrue(params["use_prev_session_anchor_filter"])
        self.assertTrue(params["use_external_liquidity_filter"])

    def test_paired_survivor_profile_keeps_only_robust_survivors(self) -> None:
        params = build_ict_paired_survivor_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_prev_session_anchor_filter"])
        self.assertTrue(params["use_external_liquidity_filter"])
        self.assertFalse(params["use_daily_bias_filter"])
        self.assertFalse(params["use_premium_discount_filter"])
        self.assertFalse(params["use_amd_filter"])
        self.assertFalse(params["use_macro_timing_windows"])
        self.assertFalse(params["use_session_array_refinement"])

    def test_paired_survivor_plus_session_array_keeps_first_robust_extension(self) -> None:
        params = build_ict_paired_survivor_plus_session_array_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_prev_session_anchor_filter"])
        self.assertTrue(params["use_external_liquidity_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertFalse(params["use_daily_bias_filter"])
        self.assertFalse(params["use_premium_discount_filter"])
        self.assertFalse(params["use_amd_filter"])
        self.assertFalse(params["use_macro_timing_windows"])

    def test_paired_survivor_plus_session_array_loose_sweep_keeps_geometry_extension(self) -> None:
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_prev_session_anchor_filter"])
        self.assertTrue(params["use_external_liquidity_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertEqual(params["liq_sweep_threshold"], 0.0008)
        self.assertFalse(params["use_daily_bias_filter"])
        self.assertFalse(params["use_premium_discount_filter"])

    def test_paired_survivor_plus_session_array_loose_sweep_short_smt_keeps_smt_extension(self) -> None:
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_prev_session_anchor_filter"])
        self.assertTrue(params["use_external_liquidity_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertEqual(params["liq_sweep_threshold"], 0.0008)
        self.assertEqual(params["smt_lookback"], 10)
        self.assertFalse(params["use_daily_bias_filter"])
        self.assertFalse(params["use_premium_discount_filter"])

    def test_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_keeps_context_survivor(self) -> None:
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_prev_session_anchor_filter"])
        self.assertTrue(params["use_external_liquidity_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertEqual(params["liq_sweep_threshold"], 0.0008)
        self.assertEqual(params["smt_lookback"], 10)
        self.assertFalse(params["use_daily_bias_filter"])

    def test_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_keeps_session_extension(self) -> None:
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_params(
            enable_smt=True
        )
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertEqual(params["liq_sweep_threshold"], 0.0008)
        self.assertEqual(params["smt_lookback"], 10)
        self.assertEqual(params["london_open"], 0)
        self.assertEqual(params["london_close"], 0)
        self.assertEqual(params["ny_open"], 14)
        self.assertEqual(params["ny_close"], 20)

    def test_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_keeps_geometry_extension(self) -> None:
        params = (
            build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_params(
                enable_smt=True
            )
        )
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertEqual(params["liq_sweep_threshold"], 0.0008)
        self.assertEqual(params["liq_sweep_recovery_bars"], 4)
        self.assertEqual(params["smt_lookback"], 10)
        self.assertEqual(params["london_open"], 0)
        self.assertEqual(params["ny_open"], 14)

    def test_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_keeps_new_frontier(self) -> None:
        params = (
            build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
                enable_smt=True
            )
        )
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertEqual(params["liq_sweep_threshold"], 0.0008)
        self.assertEqual(params["liq_sweep_recovery_bars"], 4)
        self.assertEqual(params["structure_lookback"], 12)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0006)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)
        self.assertEqual(params["fvg_revisit_min_delay_bars"], 3)
        self.assertEqual(params["displacement_body_min_pct"], 0.10)
        self.assertEqual(params["smt_lookback"], 10)
        self.assertEqual(params["london_open"], 0)
        self.assertEqual(params["ny_open"], 14)

    def test_lite_reversal_profile_removes_heaviest_pre_arm_context_filters(self) -> None:
        params = build_ict_lite_reversal_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertFalse(params["use_premium_discount_filter"])
        self.assertFalse(params["use_external_liquidity_filter"])
        self.assertFalse(params["use_prev_session_anchor_filter"])
        self.assertFalse(params["use_session_array_refinement"])
        self.assertFalse(params["use_amd_filter"])
        self.assertFalse(params["use_macro_timing_windows"])
        self.assertFalse(params["use_kill_zones"])
        self.assertTrue(params["trade_sessions"])
        self.assertEqual(params["structure_lookback"], 12)
        self.assertEqual(params["liq_sweep_threshold"], 0.0008)
        self.assertEqual(params["liq_sweep_recovery_bars"], 4)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)
        self.assertEqual(params["fvg_revisit_min_delay_bars"], 3)
        self.assertEqual(params["displacement_body_min_pct"], 0.10)

    def test_lite_reversal_relaxed_smt_profile_keeps_density_extension(self) -> None:
        params = build_ict_lite_reversal_relaxed_smt_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertEqual(params["smt_threshold"], 0.0015)
        self.assertFalse(params["use_premium_discount_filter"])
        self.assertFalse(params["use_external_liquidity_filter"])
        self.assertFalse(params["use_prev_session_anchor_filter"])
        self.assertFalse(params["use_session_array_refinement"])
        self.assertEqual(params["structure_lookback"], 12)
        self.assertEqual(params["displacement_body_min_pct"], 0.10)

    def test_lite_reversal_quick_density_repair_profile_applies_four_fast_changes(self) -> None:
        params = build_ict_lite_reversal_quick_density_repair_profile_params()
        self.assertFalse(params["use_smt_filter"])
        self.assertFalse(params["use_premium_discount_filter"])
        self.assertFalse(params["use_external_liquidity_filter"])
        self.assertFalse(params["use_prev_session_anchor_filter"])
        self.assertFalse(params["use_session_array_refinement"])
        self.assertEqual(params["structure_lookback"], 5)
        self.assertEqual(params["liq_sweep_recovery_bars"], 15)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0003)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.0)

    def test_lite_reversal_quick_swing_structure_repair_profile_uses_confirmed_swings(self) -> None:
        params = build_ict_lite_reversal_quick_swing_structure_repair_profile_params()
        self.assertFalse(params["use_smt_filter"])
        self.assertEqual(params["structure_reference_mode"], "swing")
        self.assertEqual(params["swing_threshold"], 2)
        self.assertEqual(params["structure_lookback"], 5)
        self.assertEqual(params["liq_sweep_recovery_bars"], 15)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0003)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.0)

    def test_strict_soft_premium_profile_turns_hard_filter_into_score_pressure(self) -> None:
        params = build_ict_strict_soft_premium_profile_params()
        self.assertFalse(params["use_smt_filter"])
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertEqual(params["premium_discount_filter_mode"], "soft")
        self.assertEqual(params["premium_discount_mismatch_score_penalty"], 2.0)
        self.assertTrue(params["use_session_array_refinement"])
        self.assertEqual(params["structure_lookback"], 12)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0006)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)

    def test_strict_soft_session_array_profile_turns_delivery_timing_rejects_into_score_pressure(self) -> None:
        params = build_ict_strict_soft_session_array_profile_params()
        self.assertFalse(params["use_smt_filter"])
        self.assertTrue(params["use_session_array_refinement"])
        self.assertEqual(params["session_array_filter_mode"], "soft")
        self.assertEqual(params["session_array_mismatch_score_penalty"], 2.0)
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertEqual(params["structure_lookback"], 12)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0006)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)

    def test_strict_soft_prev_session_profile_turns_anchor_rejects_into_score_pressure(self) -> None:
        params = build_ict_strict_soft_prev_session_profile_params()
        self.assertFalse(params["use_smt_filter"])
        self.assertTrue(params["use_prev_session_anchor_filter"])
        self.assertEqual(params["prev_session_anchor_filter_mode"], "soft")
        self.assertEqual(params["prev_session_anchor_mismatch_score_penalty"], 2.0)
        self.assertTrue(params["use_premium_discount_filter"])
        self.assertEqual(params["structure_lookback"], 12)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0006)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)

    def test_lite_reversal_relaxed_smt_looser_sweep_profile_keeps_geometry_extension(self) -> None:
        params = build_ict_lite_reversal_relaxed_smt_looser_sweep_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertEqual(params["smt_threshold"], 0.0015)
        self.assertEqual(params["liq_sweep_threshold"], 0.0006)
        self.assertFalse(params["use_premium_discount_filter"])
        self.assertFalse(params["use_external_liquidity_filter"])
        self.assertFalse(params["use_prev_session_anchor_filter"])
        self.assertFalse(params["use_session_array_refinement"])
        self.assertEqual(params["structure_lookback"], 12)

    def test_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_keeps_round2_extension(self) -> None:
        params = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertEqual(params["smt_threshold"], 0.0015)
        self.assertEqual(params["liq_sweep_threshold"], 0.0006)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0010)
        self.assertEqual(params["fvg_revisit_min_delay_bars"], 2)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)
        self.assertEqual(params["displacement_body_min_pct"], 0.10)
        self.assertFalse(params["use_premium_discount_filter"])
        self.assertFalse(params["use_external_liquidity_filter"])

    def test_lite_reversal_dual_speed_recovery_profile_enables_slow_branch(self) -> None:
        params = build_ict_lite_reversal_dual_speed_recovery_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["slow_recovery_enabled"])
        self.assertEqual(params["slow_recovery_bars"], 8)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0010)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)
        self.assertFalse(params["enable_continuation_entry"])

    def test_lite_reversal_qualified_continuation_density_profile_targets_positive_high_density_lane(self) -> None:
        params = build_ict_lite_reversal_qualified_continuation_density_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertTrue(params["slow_recovery_enabled"])
        self.assertTrue(params["enable_continuation_entry"])
        self.assertEqual(params["slow_recovery_bars"], 12)
        self.assertEqual(params["structure_lookback"], 8)
        self.assertEqual(params["smt_threshold"], 0.0013)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0003)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.35)
        self.assertEqual(params["fvg_revisit_min_delay_bars"], 4)
        self.assertEqual(params["take_profit_rr"], 3.0)
        self.assertEqual(params["liq_sweep_threshold"], 0.0006)

    def test_lite_reversal_qualified_reversal_balance_profile_keeps_pure_reversal_path(self) -> None:
        params = build_ict_lite_reversal_qualified_reversal_balance_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertFalse(params["enable_continuation_entry"])
        self.assertTrue(params["slow_recovery_enabled"])
        self.assertEqual(params["slow_recovery_bars"], 12)
        self.assertEqual(params["structure_lookback"], 12)
        self.assertEqual(params["smt_threshold"], 0.0010)
        self.assertEqual(params["fvg_min_gap_pct"], 0.0003)
        self.assertEqual(params["fvg_revisit_depth_ratio"], 0.5)
        self.assertEqual(params["fvg_revisit_min_delay_bars"], 4)
        self.assertEqual(params["take_profit_rr"], 3.0)

    def test_lite_reversal_regime_mtf_alignment_profile_enables_dynamic_relaxation_with_1h_guardrail(self) -> None:
        params = build_ict_lite_reversal_regime_mtf_alignment_profile_params(enable_smt=True)
        self.assertTrue(params["use_smt_filter"])
        self.assertFalse(params["enable_continuation_entry"])
        self.assertTrue(params["use_regime_adaptation"])
        self.assertTrue(params["use_higher_timeframe_alignment"])
        self.assertEqual(params["higher_timeframe_alignment_mode"], "hard")
        self.assertEqual(params["higher_timeframe_bias_timeframe"], "1H")
        self.assertEqual(params["regime_high_smt_threshold"], 0.0010)
        self.assertEqual(params["regime_high_fvg_min_gap_pct"], 0.0003)
        self.assertEqual(params["regime_high_fvg_revisit_depth_ratio"], 0.35)
        self.assertEqual(params["regime_high_min_reward_risk_ratio"], 1.25)


if __name__ == "__main__":
    unittest.main()
