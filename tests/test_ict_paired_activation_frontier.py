from __future__ import annotations

import unittest

from research.ict.analyze_ict_paired_activation_frontier import _rank_variants, _variant_specs


class ICTPairedActivationFrontierTest(unittest.TestCase):
    def test_variant_specs_include_minimal_relaxation_ladder(self) -> None:
        specs = _variant_specs()
        self.assertIn("full_stack_smt", specs)
        self.assertIn("minimal_structure_default", specs)
        self.assertIn("minimal_structure_loose_sweep", specs)
        self.assertFalse(bool(specs["minimal_structure_default"]["enable_smt"]))

    def test_rank_variants_prefers_trading_variant_over_zero_trade_profile(self) -> None:
        ranked = _rank_variants(
            {
                "zero": {
                    "metrics": {"total_trades": 0, "total_return_pct": 0.0, "profit_factor": 0.0},
                    "metadata": {},
                },
                "active": {
                    "metrics": {"total_trades": 2, "total_return_pct": -0.5, "profit_factor": 0.8},
                    "metadata": {},
                },
            }
        )
        self.assertEqual(ranked[0]["label"], "active")


if __name__ == "__main__":
    unittest.main()
