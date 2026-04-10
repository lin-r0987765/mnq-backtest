from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_score_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureFvgScoreCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_fvg_score_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("fvg_score_1", specs)
        self.assertIn("fvg_score_6", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1831, "profit_factor": 3.0},
                    "params": {"score_fvg": 2.0},
                    "metadata": {"fvg_entries": 7, "bullish_sweeps": 151, "bearish_sweeps": 122},
                },
                "fvg_score_3": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0},
                    "params": {"score_fvg": 3.0},
                    "metadata": {"fvg_entries": 6, "bullish_sweeps": 150, "bearish_sweeps": 120},
                },
                "fvg_score_6": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"score_fvg": 6.0},
                    "metadata": {"fvg_entries": 8, "bullish_sweeps": 155, "bearish_sweeps": 125},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "fvg_score_3")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
