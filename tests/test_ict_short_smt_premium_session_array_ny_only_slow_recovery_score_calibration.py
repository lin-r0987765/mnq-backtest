from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_score_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryScoreCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_score_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("slow_recovery_base", specs)
        self.assertIn("min_score_5", specs)
        self.assertIn("min_score_9", specs)
        self.assertIn("min_score_7_higher_fvg", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "slow_recovery_base": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.1627, "profit_factor": 3.0},
                    "params": {"min_score_to_trade": 6.0, "score_fvg": 2.0, "score_ote_zone": 2.0},
                    "metadata": {},
                },
                "min_score_7": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.17, "profit_factor": 2.0},
                    "params": {"min_score_to_trade": 7.0, "score_fvg": 2.0, "score_ote_zone": 2.0},
                    "metadata": {},
                },
                "min_score_9": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"min_score_to_trade": 9.0, "score_fvg": 2.0, "score_ote_zone": 2.0},
                    "metadata": {},
                },
            },
            base_trades=6,
        )
        self.assertEqual(ranked[0]["label"], "min_score_7")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 5 / 6)


if __name__ == "__main__":
    unittest.main()
