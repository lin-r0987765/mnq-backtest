from __future__ import annotations

import unittest

from research.ict.analyze_ict_swing_structure_repair import _rank_variants, _verdict


class ICTSwingStructureRepairTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_expectancy(self) -> None:
        variants = {
            "quick_density_rolling": {
                "params": {"structure_reference_mode": "rolling", "swing_threshold": 0},
                "metrics": {"total_trades": 109, "total_return_pct": -0.94, "profit_factor": 0.45, "win_rate_pct": 30.0},
            },
            "quick_swing_2": {
                "params": {"structure_reference_mode": "swing", "swing_threshold": 2},
                "metrics": {"total_trades": 60, "total_return_pct": 0.12, "profit_factor": 1.4, "win_rate_pct": 55.0},
            },
        }
        ranked = _rank_variants(variants)
        self.assertEqual(ranked[0]["label"], "quick_swing_2")

    def test_verdict_marks_positive_density_extension(self) -> None:
        active_lite = {"metrics": {"total_trades": 18}}
        quick_density = {"metrics": {"total_trades": 109, "total_return_pct": -0.94}}
        best_swing_variant = {"metrics": {"total_trades": 40, "total_return_pct": 0.12}}

        verdict, _ = _verdict(
            active_lite=active_lite,
            quick_density=quick_density,
            best_swing_variant=best_swing_variant,
        )

        self.assertEqual(verdict, "SWING_STRUCTURE_REPAIR_POSITIVE_DENSITY_EXTENSION_IDENTIFIED")

    def test_verdict_marks_expectancy_recovery_only(self) -> None:
        active_lite = {"metrics": {"total_trades": 18}}
        quick_density = {"metrics": {"total_trades": 109, "total_return_pct": -0.94}}
        best_swing_variant = {"metrics": {"total_trades": 55, "total_return_pct": -0.10}}

        verdict, _ = _verdict(
            active_lite=active_lite,
            quick_density=quick_density,
            best_swing_variant=best_swing_variant,
        )

        self.assertEqual(verdict, "SWING_STRUCTURE_REPAIR_EXPECTANCY_RECOVERY_BUT_NOT_GATE_CLEAR")


if __name__ == "__main__":
    unittest.main()
