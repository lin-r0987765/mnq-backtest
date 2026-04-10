from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_sweep_lookback_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureSweepLookbackCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_lookback_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("sweep_lookback_30", specs)
        self.assertIn("sweep_lookback_90", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1831, "profit_factor": 3.0},
                    "params": {"liq_sweep_lookback": 50},
                    "metadata": {"bullish_sweeps": 151, "bearish_sweeps": 122, "fvg_entries": 7},
                },
                "sweep_lookback_40": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0},
                    "params": {"liq_sweep_lookback": 40},
                    "metadata": {"bullish_sweeps": 170, "bearish_sweeps": 130, "fvg_entries": 6},
                },
                "sweep_lookback_90": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"liq_sweep_lookback": 90},
                    "metadata": {"bullish_sweeps": 100, "bearish_sweeps": 80, "fvg_entries": 8},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "sweep_lookback_40")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
