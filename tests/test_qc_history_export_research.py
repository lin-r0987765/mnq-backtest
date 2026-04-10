from __future__ import annotations

import unittest

import pandas as pd

from export_qc_qqq_history_research import normalize_qc_history_frame, resample_ohlcv


class QuantConnectHistoryExportTest(unittest.TestCase):
    def test_normalize_qc_history_frame_handles_symbol_multiindex(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                ("QQQ", pd.Timestamp("2026-01-07 14:30:00+00:00")),
                ("QQQ", pd.Timestamp("2026-01-07 14:31:00+00:00")),
            ],
            names=["symbol", "time"],
        )
        history = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.5, 100.5],
                "close": [100.5, 101.5],
                "volume": [1000, 1500],
            },
            index=index,
        )

        normalized = normalize_qc_history_frame(history)

        self.assertEqual(list(normalized.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(normalized.index.name, "Datetime")
        self.assertEqual(len(normalized), 2)
        self.assertEqual(int(normalized.iloc[1]["Volume"]), 1500)

    def test_resample_ohlcv_builds_left_labeled_five_minute_bar(self) -> None:
        index = pd.date_range("2026-01-07 14:30:00+00:00", periods=5, freq="min", tz="UTC")
        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1, 2, 3, 4, 5],
            },
            index=index,
        )
        frame.index.name = "Datetime"

        resampled = resample_ohlcv(frame, "5min")

        self.assertEqual(len(resampled), 1)
        self.assertEqual(resampled.index[0], pd.Timestamp("2026-01-07 14:30:00+00:00"))
        self.assertEqual(float(resampled.iloc[0]["Open"]), 100.0)
        self.assertEqual(float(resampled.iloc[0]["High"]), 105.0)
        self.assertEqual(float(resampled.iloc[0]["Low"]), 99.0)
        self.assertEqual(float(resampled.iloc[0]["Close"]), 104.5)
        self.assertEqual(int(resampled.iloc[0]["Volume"]), 15)

    def test_resample_ohlcv_hour_offset_keeps_half_hour_anchor(self) -> None:
        index = pd.date_range("2026-01-07 14:30:00+00:00", periods=60, freq="min", tz="UTC")
        frame = pd.DataFrame(
            {
                "Open": range(60),
                "High": range(1, 61),
                "Low": range(60),
                "Close": range(1, 61),
                "Volume": [10] * 60,
            },
            index=index,
        )
        frame.index.name = "Datetime"

        resampled = resample_ohlcv(frame, "60min", offset="30min")

        self.assertEqual(len(resampled), 1)
        self.assertEqual(resampled.index[0], pd.Timestamp("2026-01-07 14:30:00+00:00"))
        self.assertEqual(int(resampled.iloc[0]["Volume"]), 600)


if __name__ == "__main__":
    unittest.main()
