from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_pending_capacity import _rank_variants, _verdict


class ICTLitePendingCapacityTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"max_pending_setups_per_direction": 1},
                "metrics": {"total_trades": 18, "total_return_pct": 0.4353, "profit_factor": 5.0247},
                "metadata": {
                    "bullish_sweeps": 9,
                    "bearish_sweeps": 9,
                    "bullish_shifts": 5,
                    "bearish_shifts": 4,
                    "bullish_retest_candidates": 5,
                    "bearish_retest_candidates": 4,
                    "sweep_blocked_by_existing_pending": 10,
                    "sweep_expired_before_shift": 8,
                    "armed_setup_expired_before_retest": 1,
                },
            },
            "pending_cap_2": {
                "params": {"max_pending_setups_per_direction": 2},
                "metrics": {"total_trades": 20, "total_return_pct": 0.41, "profit_factor": 4.2},
                "metadata": {
                    "bullish_sweeps": 10,
                    "bearish_sweeps": 10,
                    "bullish_shifts": 6,
                    "bearish_shifts": 5,
                    "bullish_retest_candidates": 6,
                    "bearish_retest_candidates": 5,
                    "sweep_blocked_by_existing_pending": 4,
                    "sweep_expired_before_shift": 10,
                    "armed_setup_expired_before_retest": 2,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "pending_cap_2")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "pending_cap_2", "total_trades": 19, "total_return_pct": 0.45},
            best_positive_density_variant={"label": "pending_cap_2", "total_trades": 19, "total_return_pct": 0.45},
            best_robust_variant={"label": "pending_cap_2", "total_trades": 19, "total_return_pct": 0.45},
        )
        self.assertEqual(verdict, "ROBUST_LITE_PENDING_CAPACITY_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "pending_cap_3", "total_trades": 22, "total_return_pct": 0.31},
            best_positive_density_variant={"label": "pending_cap_3", "total_trades": 22, "total_return_pct": 0.31},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_PENDING_CAPACITY_DENSITY_EXTENSION_ONLY")

    def test_verdict_marks_plateau_when_result_is_identical(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "pending_cap_2", "total_trades": 18, "total_return_pct": 0.4353},
            best_positive_density_variant=None,
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_PENDING_CAPACITY_PLATEAU_ON_ACTIVE_FRONTIER")


if __name__ == "__main__":
    unittest.main()
