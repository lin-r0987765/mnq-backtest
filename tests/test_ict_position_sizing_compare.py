from __future__ import annotations

import unittest

from research.ict.analyze_ict_position_sizing_compare import _rank_variants, _shares_summary, _verdict


class ICTPositionSizingCompareTest(unittest.TestCase):
    def test_shares_summary_extracts_min_max_and_average(self) -> None:
        summary = _shares_summary(
            [
                {"shares": 40},
                {"shares": 80},
                {"shares": 60},
            ]
        )
        self.assertEqual(summary["min_shares"], 40)
        self.assertEqual(summary["max_shares"], 80)
        self.assertEqual(summary["avg_shares"], 60.0)
        self.assertEqual(summary["entry_count"], 3)

    def test_rank_variants_prefers_higher_return(self) -> None:
        ranked = _rank_variants(
            {
                "research_fixed_10": {"metrics": {"total_return_pct": 0.35, "profit_factor": 4.6, "total_trades": 18}},
                "fixed_40_shares": {"metrics": {"total_return_pct": 1.2, "profit_factor": 4.6, "total_trades": 18}},
            }
        )
        self.assertEqual(ranked[0]["label"], "fixed_40_shares")

    def test_verdict_marks_sizing_impact(self) -> None:
        verdict, _ = _verdict(0.35, {"label": "fixed_40_shares", "total_return_pct": 1.2})
        self.assertEqual(verdict, "POSITION_SIZING_IMPACT_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
