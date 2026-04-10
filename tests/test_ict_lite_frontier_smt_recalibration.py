from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_frontier_smt_recalibration import _rank_variants, _verdict


class ICTLiteFrontierSMTRecalibrationTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"use_smt_filter": True, "smt_lookback": 10, "smt_threshold": 0.0015},
                "metrics": {"total_trades": 18, "total_return_pct": 0.3529, "profit_factor": 4.6792},
                "metadata": {
                    "smt_filtered_sweeps": 191,
                    "bullish_shift_candidates": 25,
                    "bearish_shift_candidates": 32,
                    "fvg_entries": 13,
                    "ob_entries": 4,
                    "breaker_entries": 1,
                },
            },
            "smt_threshold_0p0018": {
                "params": {"use_smt_filter": True, "smt_lookback": 10, "smt_threshold": 0.0018},
                "metrics": {"total_trades": 20, "total_return_pct": 0.31, "profit_factor": 3.2},
                "metadata": {
                    "smt_filtered_sweeps": 150,
                    "bullish_shift_candidates": 28,
                    "bearish_shift_candidates": 35,
                    "fvg_entries": 14,
                    "ob_entries": 5,
                    "breaker_entries": 1,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "smt_threshold_0p0018")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "smt_threshold_0p0018", "total_trades": 19, "total_return_pct": 0.36},
            best_positive_density_variant={"label": "smt_threshold_0p0018", "total_trades": 19, "total_return_pct": 0.36},
            best_robust_variant={"label": "smt_threshold_0p0018", "total_trades": 19, "total_return_pct": 0.36},
        )
        self.assertEqual(verdict, "ROBUST_LITE_FRONTIER_SMT_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "smt_off_control", "total_trades": 24, "total_return_pct": 0.20},
            best_positive_density_variant={"label": "smt_off_control", "total_trades": 24, "total_return_pct": 0.20},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_FRONTIER_SMT_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
