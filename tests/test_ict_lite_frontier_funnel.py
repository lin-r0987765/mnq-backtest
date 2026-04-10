from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_frontier_funnel import _build_stage_drop_summary, _verdict


class ICTLiteFrontierFunnelTest(unittest.TestCase):
    def test_build_stage_drop_summary_sorts_largest_drop_first(self) -> None:
        funnel = {
            "stages": {
                "raw_sweep_candidates": 100,
                "accepted_sweeps": 50,
                "shift_candidates": 20,
                "armed_setups": 15,
                "retest_candidates": 10,
                "entries": 8,
            }
        }
        summary = _build_stage_drop_summary(funnel)
        self.assertEqual(summary[0]["transition"], "raw_sweep_candidates->accepted_sweeps")
        self.assertEqual(summary[0]["drop_count"], 50)
        self.assertEqual(summary[-1]["transition"], "retest_candidates->entries")

    def test_verdict_marks_choke_point_when_below_100(self) -> None:
        verdict, interpretation = _verdict(18, {"label": "smt_filtered_sweeps", "count": 301}, 82)
        self.assertEqual(verdict, "LITE_FRONTIER_FUNNEL_IDENTIFIED_NEXT_CHOKE_POINT")
        self.assertIn("82 trades short", interpretation)
        self.assertIn("smt_filtered_sweeps = 301", interpretation)

    def test_verdict_marks_gate_clear_when_over_100(self) -> None:
        verdict, _ = _verdict(120, {"label": "delivery_missing_shifts", "count": 5}, 0)
        self.assertEqual(verdict, "LITE_FRONTIER_FUNNEL_CLEARS_100_TRADE_GATE")


if __name__ == "__main__":
    unittest.main()
