from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_regime_mtf_alignment import _rank_variants, _verdict


class ICTLiteRegimeMTFAlignmentTest(unittest.TestCase):
    def test_rank_variants_prefers_stronger_regime_branch(self) -> None:
        variants = {
            "active_lite_quality_base": {
                "metrics": {
                    "total_trades": 18,
                    "total_return_pct": 13.8395,
                    "profit_factor": 6.3079,
                    "win_rate_pct": 72.2222,
                },
                "metadata": {
                    "high_regime_bars": 0,
                    "higher_timeframe_filtered_setups": 0,
                    "higher_timeframe_softened_setups": 0,
                },
            },
            "regime_mtf_hard": {
                "metrics": {
                    "total_trades": 32,
                    "total_return_pct": 12.0,
                    "profit_factor": 2.5,
                    "win_rate_pct": 60.0,
                },
                "metadata": {
                    "high_regime_bars": 4000,
                    "higher_timeframe_filtered_setups": 20,
                    "higher_timeframe_softened_setups": 0,
                },
            },
        }
        ranked = _rank_variants(variants)
        self.assertEqual(ranked[0]["label"], "active_lite_quality_base")

    def test_verdict_identifies_regime_branch_when_it_beats_balance(self) -> None:
        verdict, _ = _verdict(
            quality_base={"total_trades": 18, "total_return_pct": 13.8395, "profit_factor": 6.3079},
            reversal_balance={"total_trades": 40, "total_return_pct": 13.0, "profit_factor": 3.0},
            regime_best={"total_trades": 42, "total_return_pct": 13.2, "profit_factor": 2.7},
        )
        self.assertEqual(verdict, "REGIME_MTF_REVERSAL_BRANCH_IDENTIFIED")

    def test_verdict_marks_survivor_only_when_balance_is_still_better(self) -> None:
        verdict, _ = _verdict(
            quality_base={"total_trades": 18, "total_return_pct": 13.8395, "profit_factor": 6.3079},
            reversal_balance={"total_trades": 40, "total_return_pct": 13.4177, "profit_factor": 3.1879},
            regime_best={"total_trades": 30, "total_return_pct": 12.5, "profit_factor": 2.4},
        )
        self.assertEqual(verdict, "REGIME_MTF_REVERSAL_SURVIVOR_ONLY")


if __name__ == "__main__":
    unittest.main()
