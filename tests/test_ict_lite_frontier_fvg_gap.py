from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_frontier_fvg_gap import _rank_variants, _verdict


class ICTLiteFrontierFvgGapTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"fvg_min_gap_pct": 0.0010},
                "metrics": {"total_trades": 18, "total_return_pct": 0.4353, "profit_factor": 5.0247},
                "metadata": {"fvg_entries": 12, "ob_entries": 5, "breaker_entries": 1, "ifvg_entries": 0},
            },
            "fvg_gap_0p0003": {
                "params": {"fvg_min_gap_pct": 0.0003},
                "metrics": {"total_trades": 20, "total_return_pct": 0.3379, "profit_factor": 4.0458},
                "metadata": {"fvg_entries": 15, "ob_entries": 4, "breaker_entries": 1, "ifvg_entries": 0},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "fvg_gap_0p0003")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "fvg_gap_0p0005", "total_trades": 18, "total_return_pct": 0.36},
            best_positive_density_variant=None,
            best_robust_variant={"label": "fvg_gap_0p0005", "total_trades": 18, "total_return_pct": 0.36},
        )
        self.assertEqual(verdict, "ROBUST_LITE_FRONTIER_FVG_GAP_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "fvg_gap_0p0003", "total_trades": 20, "total_return_pct": 0.3379},
            best_positive_density_variant={"label": "fvg_gap_0p0003", "total_trades": 20, "total_return_pct": 0.3379},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_FRONTIER_FVG_GAP_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
