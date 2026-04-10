from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_rejection_wick_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureFvgRejectionWickCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_rejection_wick_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("fvg_rejection_wick_0p10", specs)
        self.assertIn("fvg_rejection_wick_0p50", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.2042, "profit_factor": 3.0},
                    "params": {"fvg_rejection_wick_ratio": 0.0},
                    "metadata": {
                        "fvg_entries": 7,
                        "ifvg_entries": 0,
                        "fvg_wick_filtered_retests": 0,
                        "bullish_sweeps": 151,
                        "bearish_sweeps": 122,
                    },
                },
                "fvg_rejection_wick_0p10": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.21, "profit_factor": 2.0},
                    "params": {"fvg_rejection_wick_ratio": 0.1},
                    "metadata": {
                        "fvg_entries": 6,
                        "ifvg_entries": 0,
                        "fvg_wick_filtered_retests": 2,
                        "bullish_sweeps": 150,
                        "bearish_sweeps": 120,
                    },
                },
                "fvg_rejection_wick_0p50": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"fvg_rejection_wick_ratio": 0.5},
                    "metadata": {
                        "fvg_entries": 8,
                        "ifvg_entries": 0,
                        "fvg_wick_filtered_retests": 3,
                        "bullish_sweeps": 155,
                        "bearish_sweeps": 125,
                    },
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "fvg_rejection_wick_0p10")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
