from __future__ import annotations

import unittest

from research.ict.analyze_ict_survivor_smt_recalibration import _rank_variants, _variant_specs


class ICTSurvivorSmtRecalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_control_and_smt_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("loose_sweep_base", specs)
        self.assertIn("no_smt_control", specs)
        self.assertIn("shorter_smt_lookback_10", specs)
        self.assertIn("looser_smt_threshold_0p0015", specs)

    def test_rank_variants_prefers_positive_robust_smt_variant(self) -> None:
        ranked = _rank_variants(
            {
                "loose_sweep_base": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.08, "profit_factor": 2.0},
                    "metadata": {},
                },
                "shorter_smt_lookback_10": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.09, "profit_factor": 2.2},
                    "metadata": {},
                },
                "no_smt_control": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.07, "profit_factor": 1.8},
                    "metadata": {},
                },
            },
            base_trades=6,
        )
        self.assertEqual(ranked[0]["label"], "shorter_smt_lookback_10")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 5 / 6)


if __name__ == "__main__":
    unittest.main()
