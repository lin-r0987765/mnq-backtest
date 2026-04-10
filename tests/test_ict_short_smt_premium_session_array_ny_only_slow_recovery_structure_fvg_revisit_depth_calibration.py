from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_revisit_depth_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureFvgRevisitDepthCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_ce_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("fvg_revisit_depth_0p25", specs)
        self.assertIn("fvg_revisit_depth_1p00", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1831, "profit_factor": 3.0},
                    "params": {"fvg_revisit_depth_ratio": 0.0},
                    "metadata": {"fvg_entries": 7, "ifvg_entries": 0, "fvg_depth_filtered_retests": 0, "bullish_sweeps": 151, "bearish_sweeps": 122},
                },
                "fvg_revisit_depth_0p25": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0},
                    "params": {"fvg_revisit_depth_ratio": 0.25},
                    "metadata": {"fvg_entries": 6, "ifvg_entries": 0, "fvg_depth_filtered_retests": 2, "bullish_sweeps": 150, "bearish_sweeps": 120},
                },
                "fvg_revisit_depth_1p00": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"fvg_revisit_depth_ratio": 1.0},
                    "metadata": {"fvg_entries": 8, "ifvg_entries": 0, "fvg_depth_filtered_retests": 3, "bullish_sweeps": 155, "bearish_sweeps": 125},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "fvg_revisit_depth_0p25")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
