from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_frontier_sweep_threshold import _rank_variants, _verdict


class ICTLiteFrontierSweepThresholdTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"liq_sweep_threshold": 0.0006},
                "metrics": {"total_trades": 18, "total_return_pct": 0.4353, "profit_factor": 5.0247},
                "metadata": {
                    "fvg_entries": 13,
                    "ob_entries": 4,
                    "breaker_entries": 1,
                    "ifvg_entries": 0,
                    "bullish_sweeps": 8,
                    "bearish_sweeps": 10,
                },
            },
            "sweep_0p0004": {
                "params": {"liq_sweep_threshold": 0.0004},
                "metrics": {"total_trades": 20, "total_return_pct": 0.30, "profit_factor": 3.5},
                "metadata": {
                    "fvg_entries": 15,
                    "ob_entries": 4,
                    "breaker_entries": 1,
                    "ifvg_entries": 0,
                    "bullish_sweeps": 9,
                    "bearish_sweeps": 11,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "sweep_0p0004")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "sweep_0p0005", "total_trades": 18, "total_return_pct": 0.45},
            best_positive_density_variant=None,
            best_robust_variant={"label": "sweep_0p0005", "total_trades": 18, "total_return_pct": 0.45},
        )
        self.assertEqual(verdict, "ROBUST_LITE_FRONTIER_SWEEP_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "sweep_0p0004", "total_trades": 20, "total_return_pct": 0.30},
            best_positive_density_variant={"label": "sweep_0p0004", "total_trades": 20, "total_return_pct": 0.30},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_FRONTIER_SWEEP_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
