from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_rr_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureRRCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_rr_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("rr_3p0", specs)
        self.assertIn("rr_6p0", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1826, "profit_factor": 3.0, "avg_trade_pct": 0.0261},
                    "params": {"take_profit_rr": 4.0},
                    "metadata": {"fvg_entries": 7},
                },
                "rr_5p0": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0, "avg_trade_pct": 0.0316},
                    "params": {"take_profit_rr": 5.0},
                    "metadata": {"fvg_entries": 6},
                },
                "rr_3p0": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5, "avg_trade_pct": 0.0100},
                    "params": {"take_profit_rr": 3.0},
                    "metadata": {"fvg_entries": 8},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "rr_5p0")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
