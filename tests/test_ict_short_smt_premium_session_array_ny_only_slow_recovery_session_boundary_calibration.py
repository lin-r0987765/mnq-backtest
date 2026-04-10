from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_session_boundary_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoverySessionBoundaryCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_boundary_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("slow_recovery_base", specs)
        self.assertIn("ny_close_19", specs)
        self.assertIn("ny_open_15", specs)
        self.assertIn("trade_sessions_off_control", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "slow_recovery_base": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.1627, "profit_factor": 3.0},
                    "params": {"trade_sessions": True, "ny_open": 14, "ny_close": 20},
                    "metadata": {},
                },
                "ny_close_19": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.17, "profit_factor": 2.0},
                    "params": {"trade_sessions": True, "ny_open": 14, "ny_close": 19},
                    "metadata": {},
                },
                "trade_sessions_off_control": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"trade_sessions": False, "ny_open": 14, "ny_close": 20},
                    "metadata": {},
                },
            },
            base_trades=6,
        )
        self.assertEqual(ranked[0]["label"], "ny_close_19")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 4 / 6)


if __name__ == "__main__":
    unittest.main()
