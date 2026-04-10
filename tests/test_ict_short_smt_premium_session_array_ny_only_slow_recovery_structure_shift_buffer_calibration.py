from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_shift_buffer_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureShiftBufferCalibrationTest(
    unittest.TestCase
):
    def test_variant_specs_include_base_and_shift_buffer_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("shift_buffer_0p05", specs)
        self.assertIn("shift_buffer_0p25", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.2042, "profit_factor": 3.0},
                    "params": {"structure_shift_close_buffer_ratio": 0.0},
                    "metadata": {
                        "fvg_entries": 7,
                        "structure_buffer_filtered_shifts": 0,
                        "bullish_sweeps": 151,
                        "bearish_sweeps": 122,
                    },
                },
                "shift_buffer_0p05": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.21, "profit_factor": 2.0},
                    "params": {"structure_shift_close_buffer_ratio": 0.05},
                    "metadata": {
                        "fvg_entries": 6,
                        "structure_buffer_filtered_shifts": 3,
                        "bullish_sweeps": 150,
                        "bearish_sweeps": 120,
                    },
                },
                "shift_buffer_0p25": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"structure_shift_close_buffer_ratio": 0.25},
                    "metadata": {
                        "fvg_entries": 8,
                        "structure_buffer_filtered_shifts": 5,
                        "bullish_sweeps": 155,
                        "bearish_sweeps": 125,
                    },
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "shift_buffer_0p05")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
