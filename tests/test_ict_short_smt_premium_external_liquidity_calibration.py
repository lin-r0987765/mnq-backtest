from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_external_liquidity_calibration import _rank_variants, _variant_specs


class ICTShortSmtPremiumExternalLiquidityCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_external_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("premium_base", specs)
        self.assertIn("external_default", specs)
        self.assertIn("external_shorter_lookback_60", specs)
        self.assertIn("external_shorter_and_looser", specs)

    def test_rank_variants_prefers_positive_robust_external_variant(self) -> None:
        ranked = _rank_variants(
            {
                "premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "external_shorter_and_looser": {
                    "metrics": {"total_trades": 3, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "external_default": {
                    "metrics": {"total_trades": 0, "total_return_pct": 0.0, "profit_factor": 0.0},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "external_shorter_and_looser")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.6)


if __name__ == "__main__":
    unittest.main()
