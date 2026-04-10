from __future__ import annotations

import unittest

from research.ict.analyze_ict_context_reintroduction import _rank_variants, _variant_specs


class ICTContextReintroductionTest(unittest.TestCase):
    def test_variant_specs_include_relaxed_reference_and_reintroduced_filters(self) -> None:
        specs = _variant_specs()
        self.assertIn("context_relaxed_bundle", specs)
        self.assertIn("reintro_daily_bias", specs)
        self.assertIn("reintro_context_core", specs)
        self.assertTrue(bool(specs["reintro_daily_bias"]["enable_smt"]))

    def test_rank_variants_prefers_positive_active_non_full_stack_profile(self) -> None:
        ranked = _rank_variants(
            {
                "full_stack_smt": {
                    "metrics": {"total_trades": 0, "total_return_pct": 0.0, "profit_factor": 0.0},
                    "metadata": {},
                },
                "context_relaxed_bundle": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1, "profit_factor": 1.9},
                    "metadata": {},
                },
                "reintro_daily_bias": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.2, "profit_factor": 1.5},
                    "metadata": {},
                },
            },
            relaxed_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "reintro_daily_bias")


if __name__ == "__main__":
    unittest.main()
