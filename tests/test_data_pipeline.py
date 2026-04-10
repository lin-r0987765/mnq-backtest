from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.coverage import build_coverage_report, describe_price_coverage
from src.data.fetcher import merge_peer_columns
from src.data.polygon_provider import polygon_aggs_to_frame


class DataPipelineTest(unittest.TestCase):
    def test_coverage_flags_short_intraday_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
            df = pd.DataFrame(
                {
                    "Open": range(10),
                    "High": range(1, 11),
                    "Low": range(10),
                    "Close": range(1, 11),
                    "Volume": [100] * 10,
                },
                index=index,
            )
            df.to_csv(root / "qqq_5m.csv")

            entry = describe_price_coverage(
                root / "qqq_5m.csv",
                interval="5m",
                target_years=8.0,
                recommended_provider="polygon",
                notes="test",
            )
            self.assertFalse(entry.meets_target)
            self.assertLess(entry.span_years, 8.0)

    def test_build_coverage_report_marks_missing_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index = pd.date_range("2010-01-01", periods=4000, freq="D", tz="UTC")
            df = pd.DataFrame(
                {
                    "Open": range(4000),
                    "High": range(1, 4001),
                    "Low": range(4000),
                    "Close": range(1, 4001),
                    "Volume": [100] * 4000,
                },
                index=index,
            )
            df.to_csv(root / "qqq_1d.csv")
            report = build_coverage_report(root)
            self.assertIn("qqq_5m.csv", report["missing_targets"])
            self.assertIn("qqq_1h.csv", report["missing_targets"])
            self.assertNotIn("qqq_1d.csv", report["missing_targets"])

    def test_polygon_payload_to_frame(self) -> None:
        payload = {
            "results": [
                {"t": 1704115800000, "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 1234},
                {"t": 1704116100000, "o": 100.5, "h": 101.5, "l": 100.0, "c": 101.0, "v": 2345},
            ]
        }
        df = polygon_aggs_to_frame(payload)
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(len(df), 2)
        self.assertEqual(float(df.iloc[0]["Open"]), 100.0)

    def test_merge_peer_columns_aligns_strictly_on_timestamp(self) -> None:
        index = pd.date_range("2026-01-01 14:30:00", periods=3, freq="5min", tz="America/New_York")
        primary = pd.DataFrame(
            {
                "Open": [100.0, 100.2, 100.4],
                "High": [100.5, 100.6, 100.8],
                "Low": [99.8, 100.0, 100.2],
                "Close": [100.1, 100.4, 100.7],
                "Volume": [1000, 1000, 1000],
            },
            index=index,
        )
        peer = pd.DataFrame(
            {
                "Open": [500.0, 500.2],
                "High": [500.5, 500.6],
                "Low": [499.8, 500.0],
                "Close": [500.1, 500.4],
                "Volume": [2000, 2000],
            },
            index=index[:2],
        )

        merged = merge_peer_columns(primary, peer)

        self.assertIn("PeerHigh", merged.columns)
        self.assertIn("PeerLow", merged.columns)
        self.assertEqual(float(merged["PeerHigh"].iloc[0]), 500.5)
        self.assertTrue(pd.isna(merged["PeerHigh"].iloc[2]))


if __name__ == "__main__":
    unittest.main()
