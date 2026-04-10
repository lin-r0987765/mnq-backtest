from __future__ import annotations

import unittest

import pandas as pd

from research.ict.analyze_ict_combined_lanes import _merge_lane_signals, _verdict
from src.strategies.base import StrategyResult


class ICTCombinedLanesTest(unittest.TestCase):
    def test_merge_prefers_reversal_on_same_bar(self) -> None:
        index = pd.date_range("2026-01-01", periods=3, freq="5min")
        reversal = StrategyResult(
            entries_long=pd.Series([True, False, False], index=index),
            exits_long=pd.Series([False, True, False], index=index),
            entries_short=pd.Series([False, False, False], index=index),
            exits_short=pd.Series([False, False, False], index=index),
            metadata={},
        )
        continuation = StrategyResult(
            entries_long=pd.Series([False, False, False], index=index),
            exits_long=pd.Series([False, False, False], index=index),
            entries_short=pd.Series([False, False, True], index=index),
            exits_short=pd.Series([False, False, False], index=index),
            metadata={},
        )
        merged = _merge_lane_signals(reversal, continuation)
        self.assertTrue(bool(merged.entries_long.iat[0]))
        self.assertFalse(bool(merged.entries_short.iat[0]))
        self.assertEqual(merged.metadata["reversal_entries"], 1)

    def test_merge_records_lane_conflict(self) -> None:
        index = pd.date_range("2026-01-01", periods=2, freq="5min")
        reversal = StrategyResult(
            entries_long=pd.Series([True, False], index=index),
            exits_long=pd.Series([False, True], index=index),
            entries_short=pd.Series([False, False], index=index),
            exits_short=pd.Series([False, False], index=index),
            metadata={},
        )
        continuation = StrategyResult(
            entries_long=pd.Series([False, False], index=index),
            exits_long=pd.Series([False, False], index=index),
            entries_short=pd.Series([True, False], index=index),
            exits_short=pd.Series([False, True], index=index),
            metadata={},
        )
        merged = _merge_lane_signals(reversal, continuation)
        self.assertEqual(merged.metadata["same_bar_lane_conflicts"], 1)

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            {"total_trades": 18, "total_return_pct": 0.3529},
            {"total_trades": 21, "total_return_pct": 0.31},
        )
        self.assertEqual(verdict, "COMBINED_LANE_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
