from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_ifvg_lookback_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureIFVGLookbackCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_ifvg_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("ifvg_lookback_8", specs)
        self.assertIn("ifvg_lookback_40", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1826, "profit_factor": 3.0},
                    "params": {"ifvg_lookback": 20},
                    "metadata": {"ifvg_entries": 0},
                },
                "ifvg_lookback_12": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0},
                    "params": {"ifvg_lookback": 12},
                    "metadata": {"ifvg_entries": 1},
                },
                "ifvg_lookback_40": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"ifvg_lookback": 40},
                    "metadata": {"ifvg_entries": 0},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "ifvg_lookback_12")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
