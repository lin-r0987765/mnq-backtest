from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_active_swing_structure import _rank_variants, _verdict


class ICTLiteActiveSwingStructureTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_expectancy(self) -> None:
        variants = {
            "active_lite_rolling_base": {
                "params": {"structure_reference_mode": "rolling", "swing_threshold": 3},
                "metrics": {"total_trades": 18, "total_return_pct": 0.4353, "profit_factor": 5.0247, "win_rate_pct": 72.2},
                "metadata": {"swing_structure_missing_reference": 0},
            },
            "active_lite_swing_2": {
                "params": {"structure_reference_mode": "swing", "swing_threshold": 2},
                "metrics": {"total_trades": 25, "total_return_pct": 0.22, "profit_factor": 2.4, "win_rate_pct": 58.0},
                "metadata": {"swing_structure_missing_reference": 4},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "active_lite_rolling_base")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "active_lite_swing_2", "total_trades": 18, "total_return_pct": 0.50},
            best_positive_density_variant=None,
            best_robust_variant={"label": "active_lite_swing_2", "total_trades": 18, "total_return_pct": 0.50},
        )
        self.assertEqual(verdict, "ROBUST_ACTIVE_LITE_SWING_STRUCTURE_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "active_lite_swing_2", "total_trades": 22, "total_return_pct": 0.20},
            best_positive_density_variant={"label": "active_lite_swing_2", "total_trades": 22, "total_return_pct": 0.20},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "ACTIVE_LITE_SWING_STRUCTURE_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
