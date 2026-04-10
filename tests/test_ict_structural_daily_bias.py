from __future__ import annotations

import unittest

from research.ict.analyze_ict_structural_daily_bias import _rank_variants, _verdict


class ICTStructuralDailyBiasTest(unittest.TestCase):
    def test_rank_variants_prefers_robust_extension(self) -> None:
        variants = {
            "active_lite_frontier_base": {
                "params": {
                    "use_daily_bias_filter": False,
                    "daily_bias_mode": "statistical",
                    "daily_bias_swing_threshold": 1,
                },
                "metrics": {"total_trades": 18, "total_return_pct": 0.43, "profit_factor": 5.0},
                "metadata": {
                    "daily_bias_filtered_setups": 0,
                    "accepted_sweeps": 1200,
                    "shift_candidates": 57,
                },
            },
            "daily_bias_structure_1": {
                "params": {
                    "use_daily_bias_filter": True,
                    "daily_bias_mode": "structure",
                    "daily_bias_swing_threshold": 1,
                },
                "metrics": {"total_trades": 18, "total_return_pct": 0.46, "profit_factor": 5.2},
                "metadata": {
                    "daily_bias_filtered_setups": 6,
                    "accepted_sweeps": 1190,
                    "shift_candidates": 55,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18, base_return=0.43)
        self.assertEqual(ranked[0]["label"], "daily_bias_structure_1")

    def test_verdict_marks_plateau_when_results_are_unchanged(self) -> None:
        verdict, _ = _verdict(
            base_label="active_lite_frontier_base",
            base_trades=18,
            base_return=0.43,
            best_non_base={"label": "daily_bias_structure_1", "total_trades": 18, "total_return_pct": 0.43},
            best_positive_density_variant=None,
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "STRUCTURAL_DAILY_BIAS_PLATEAU_ON_ACTIVE_FRONTIER")

    def test_verdict_marks_density_only_when_trade_count_rises_without_beating_base(self) -> None:
        verdict, _ = _verdict(
            base_label="active_lite_frontier_base",
            base_trades=18,
            base_return=0.43,
            best_non_base={"label": "daily_bias_structure_1", "total_trades": 22, "total_return_pct": 0.39},
            best_positive_density_variant={
                "label": "daily_bias_structure_1",
                "total_trades": 22,
                "total_return_pct": 0.39,
            },
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "STRUCTURAL_DAILY_BIAS_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
