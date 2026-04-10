from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_continuation_repair import _rank_variants, _verdict


class ICTLiteContinuationRepairTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "active_lite_frontier_base": {
                "params": {"enable_continuation_entry": False},
                "metrics": {"total_trades": 18, "total_return_pct": 0.43, "profit_factor": 5.0},
                "metadata": {
                    "continuation_zone_refreshes": 0,
                    "continuation_entries": 0,
                    "delivery_missing_shifts": 20,
                },
            },
            "continuation_on": {
                "params": {"enable_continuation_entry": True},
                "metrics": {"total_trades": 20, "total_return_pct": 0.46, "profit_factor": 5.2},
                "metadata": {
                    "continuation_zone_refreshes": 3,
                    "continuation_entries": 2,
                    "delivery_missing_shifts": 17,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18, base_return=0.43)
        self.assertEqual(ranked[0]["label"], "continuation_on")

    def test_verdict_marks_plateau_when_results_are_unchanged(self) -> None:
        verdict, _ = _verdict(
            base_label="active_lite_frontier_base",
            base_trades=18,
            base_return=0.43,
            best_non_base={"label": "continuation_on", "total_trades": 18, "total_return_pct": 0.43},
            best_positive_density_variant=None,
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_CONTINUATION_PLATEAU_ON_ACTIVE_FRONTIER")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_label="active_lite_frontier_base",
            base_trades=18,
            base_return=0.43,
            best_non_base={"label": "continuation_on", "total_trades": 20, "total_return_pct": 0.46},
            best_positive_density_variant={"label": "continuation_on", "total_trades": 20, "total_return_pct": 0.46},
            best_robust_variant={"label": "continuation_on", "total_trades": 20, "total_return_pct": 0.46},
        )
        self.assertEqual(verdict, "ROBUST_LITE_CONTINUATION_EXTENSION_IDENTIFIED")


if __name__ == "__main__":
    unittest.main()
