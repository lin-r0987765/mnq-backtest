from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_reentry_repair import _rank_variants, _verdict


class ICTLiteReentryRepairTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "active_lite_frontier_base": {
                "params": {"max_reentries_per_setup": 0},
                "metrics": {"total_trades": 18, "total_return_pct": 0.35, "profit_factor": 4.6},
                "metadata": {
                    "reentry_entries": 0,
                    "reentry_stop_rearms": 0,
                    "reentry_exhausted_setups": 0,
                },
            },
            "reentry_1": {
                "params": {"max_reentries_per_setup": 1},
                "metrics": {"total_trades": 20, "total_return_pct": 0.38, "profit_factor": 4.8},
                "metadata": {
                    "reentry_entries": 2,
                    "reentry_stop_rearms": 3,
                    "reentry_exhausted_setups": 0,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18, base_return=0.35)
        self.assertEqual(ranked[0]["label"], "reentry_1")

    def test_verdict_marks_plateau_when_results_are_unchanged(self) -> None:
        verdict, _ = _verdict(
            base_label="active_lite_frontier_base",
            base_trades=18,
            base_return=0.35,
            best_non_base={"label": "reentry_1", "total_trades": 18, "total_return_pct": 0.35},
            best_positive_density_variant=None,
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_REENTRY_PLATEAU_ON_ACTIVE_FRONTIER")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_label="active_lite_frontier_base",
            base_trades=18,
            base_return=0.35,
            best_non_base={"label": "reentry_1", "total_trades": 20, "total_return_pct": 0.38},
            best_positive_density_variant={"label": "reentry_1", "total_trades": 20, "total_return_pct": 0.38},
            best_robust_variant={"label": "reentry_1", "total_trades": 20, "total_return_pct": 0.38},
        )
        self.assertEqual(verdict, "ROBUST_LITE_REENTRY_EXTENSION_IDENTIFIED")


if __name__ == "__main__":
    unittest.main()
