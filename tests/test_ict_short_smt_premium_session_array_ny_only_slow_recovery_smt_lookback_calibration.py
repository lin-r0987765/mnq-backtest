from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_smt_lookback_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoverySmtLookbackCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_lookback_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("slow_recovery_base", specs)
        self.assertIn("smt_lookback_8", specs)
        self.assertIn("smt_lookback_6", specs)
        self.assertIn("smt_off_control", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "slow_recovery_base": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.1627, "profit_factor": 3.1},
                    "params": {"smt_lookback": 10, "use_smt_filter": True},
                    "metadata": {},
                },
                "smt_lookback_8": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.17, "profit_factor": 2.0},
                    "params": {"smt_lookback": 8, "use_smt_filter": True},
                    "metadata": {},
                },
                "smt_off_control": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.4},
                    "params": {"smt_lookback": 10, "use_smt_filter": False},
                    "metadata": {},
                },
            },
            base_trades=6,
        )
        self.assertEqual(ranked[0]["label"], "smt_lookback_8")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 4 / 6)


if __name__ == "__main__":
    unittest.main()
