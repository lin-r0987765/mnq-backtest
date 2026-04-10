from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_frontier_revisit_delay import _rank_variants, _verdict


class ICTLiteFrontierRevisitDelayTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"fvg_revisit_min_delay_bars": 2},
                "metrics": {"total_trades": 18, "total_return_pct": 0.4353, "profit_factor": 5.0247},
                "metadata": {"fvg_delay_filtered_retests": 2, "fvg_entries": 12, "ob_entries": 5, "breaker_entries": 1},
            },
            "delay_1": {
                "params": {"fvg_revisit_min_delay_bars": 1},
                "metrics": {"total_trades": 20, "total_return_pct": 0.34, "profit_factor": 3.2},
                "metadata": {"fvg_delay_filtered_retests": 0, "fvg_entries": 14, "ob_entries": 5, "breaker_entries": 1},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "delay_1")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "delay_3", "total_trades": 18, "total_return_pct": 0.44},
            best_positive_density_variant=None,
            best_robust_variant={"label": "delay_3", "total_trades": 18, "total_return_pct": 0.44},
        )
        self.assertEqual(verdict, "ROBUST_LITE_FRONTIER_REVISIT_DELAY_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "delay_1", "total_trades": 20, "total_return_pct": 0.34},
            best_positive_density_variant={"label": "delay_1", "total_trades": 20, "total_return_pct": 0.34},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_FRONTIER_REVISIT_DELAY_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
