from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_geometry_round1 import _rank_variants, _verdict


class ICTLiteGeometryRound1Test(unittest.TestCase):
    def test_rank_variants_prefers_higher_trade_and_positive_return(self) -> None:
        variants = {
            "lite_relaxed_smt_base": {
                "params": {"structure_lookback": 12, "liq_sweep_threshold": 0.0008, "fvg_min_gap_pct": 0.0006},
                "metrics": {"total_trades": 16, "total_return_pct": 0.2902, "profit_factor": 4.6},
                "metadata": {"fvg_entries": 12, "ob_entries": 3, "breaker_entries": 1},
            },
            "structure_10": {
                "params": {"structure_lookback": 10, "liq_sweep_threshold": 0.0008, "fvg_min_gap_pct": 0.0006},
                "metrics": {"total_trades": 18, "total_return_pct": 0.295, "profit_factor": 4.0},
                "metadata": {"fvg_entries": 13, "ob_entries": 3, "breaker_entries": 2},
            },
        }
        ranked = _rank_variants(variants, base_trades=16)
        self.assertEqual(ranked[0]["label"], "structure_10")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=16,
            base_return=0.2902,
            best_non_base={"label": "structure_10", "total_trades": 18, "total_return_pct": 0.295},
            best_robust_variant={"label": "structure_10", "total_trades": 18, "total_return_pct": 0.295},
        )
        self.assertEqual(verdict, "ROBUST_LITE_GEOMETRY_EXTENSION_IDENTIFIED")

    def test_verdict_marks_survivor_when_no_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=16,
            base_return=0.2902,
            best_non_base={"label": "structure_8", "total_trades": 15, "total_return_pct": 0.285},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_GEOMETRY_SURVIVOR_BUT_NOT_EXTENSION")


if __name__ == "__main__":
    unittest.main()
