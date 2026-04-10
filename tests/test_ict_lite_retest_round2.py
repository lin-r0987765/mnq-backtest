from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_retest_round2 import _rank_variants, _verdict


class ICTLiteRetestRound2Test(unittest.TestCase):
    def test_rank_variants_prefers_higher_trade_and_positive_return(self) -> None:
        variants = {
            "lite_relaxed_smt_looser_sweep_base": {
                "params": {
                    "fvg_revisit_depth_ratio": 0.5,
                    "fvg_revisit_min_delay_bars": 3,
                    "displacement_body_min_pct": 0.1,
                },
                "metrics": {"total_trades": 18, "total_return_pct": 0.3229, "profit_factor": 4.3},
                "metadata": {"fvg_entries": 13, "ob_entries": 4, "breaker_entries": 1},
            },
            "delay_2": {
                "params": {
                    "fvg_revisit_depth_ratio": 0.5,
                    "fvg_revisit_min_delay_bars": 2,
                    "displacement_body_min_pct": 0.1,
                },
                "metrics": {"total_trades": 20, "total_return_pct": 0.33, "profit_factor": 4.0},
                "metadata": {"fvg_entries": 14, "ob_entries": 4, "breaker_entries": 2},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "delay_2")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3229,
            best_non_base={"label": "delay_2", "total_trades": 20, "total_return_pct": 0.33},
            best_robust_variant={"label": "delay_2", "total_trades": 20, "total_return_pct": 0.33},
        )
        self.assertEqual(verdict, "ROBUST_LITE_RETEST_EXTENSION_IDENTIFIED")

    def test_verdict_marks_survivor_when_no_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3229,
            best_non_base={"label": "depth_0p25", "total_trades": 17, "total_return_pct": 0.31},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_RETEST_SURVIVOR_BUT_NOT_EXTENSION")


if __name__ == "__main__":
    unittest.main()
