from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_session_array_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoverySessionArrayCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_window_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("slow_recovery_base", specs)
        self.assertIn("session_array_off_control", specs)
        self.assertIn("broader_imbalance", specs)
        self.assertIn("shifted_later", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "slow_recovery_base": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.1627, "profit_factor": 3.0},
                    "params": {"use_session_array_refinement": True},
                    "metadata": {},
                },
                "broader_imbalance": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.17, "profit_factor": 2.0},
                    "params": {"use_session_array_refinement": True},
                    "metadata": {},
                },
                "session_array_off_control": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"use_session_array_refinement": False},
                    "metadata": {},
                },
            },
            base_trades=6,
        )
        self.assertEqual(ranked[0]["label"], "broader_imbalance")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 4 / 6)


if __name__ == "__main__":
    unittest.main()
