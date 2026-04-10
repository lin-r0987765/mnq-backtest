from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_context_reintroduction import _rank_variants, _variant_specs


class ICTShortSmtContextReintroductionTest(unittest.TestCase):
    def test_variant_specs_include_short_smt_base_and_reintroductions(self) -> None:
        specs = _variant_specs()
        self.assertIn("short_smt_base", specs)
        self.assertIn("reintro_premium_discount", specs)
        self.assertIn("reintro_daily_bias_and_premium_discount", specs)
        self.assertIn("reintro_context_core", specs)

    def test_rank_variants_prefers_positive_robust_reintroduction(self) -> None:
        ranked = _rank_variants(
            {
                "short_smt_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "reintro_premium_discount": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "reintro_daily_bias": {
                    "metrics": {"total_trades": 2, "total_return_pct": 0.03, "profit_factor": 3.0},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "reintro_premium_discount")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.8)


if __name__ == "__main__":
    unittest.main()
