from __future__ import annotations

import unittest

from research.ict.analyze_ict_continuation_lane_compare import _compare_verdict


class ICTContinuationLaneCompareTest(unittest.TestCase):
    def test_verdict_marks_robust_outperformance(self) -> None:
        verdict, _ = _compare_verdict(
            {"total_trades": 18, "total_return_pct": 0.3529},
            {"total_trades": 24, "total_return_pct": 0.41},
        )
        self.assertEqual(verdict, "ROBUST_CONTINUATION_LANE_OUTPERFORMS_REVERSAL")

    def test_verdict_marks_density_candidate(self) -> None:
        verdict, _ = _compare_verdict(
            {"total_trades": 18, "total_return_pct": 0.3529},
            {"total_trades": 24, "total_return_pct": 0.30},
        )
        self.assertEqual(verdict, "CONTINUATION_LANE_DENSITY_CANDIDATE")

    def test_verdict_marks_survivor_when_not_denser(self) -> None:
        verdict, _ = _compare_verdict(
            {"total_trades": 18, "total_return_pct": 0.3529},
            {"total_trades": 16, "total_return_pct": 0.20},
        )
        self.assertEqual(verdict, "CONTINUATION_LANE_SURVIVES_BUT_NOT_DENSER")


if __name__ == "__main__":
    unittest.main()
