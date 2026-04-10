from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

from research.shared.analyze_single_factor_robustness import build_feature_frame
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


class DailySessionAlignmentTest(unittest.TestCase):
    def test_helper_shifts_features_to_next_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqq_1d.csv"
            dates = pd.to_datetime(
                [
                    "2025-01-02 05:00:00+00:00",
                    "2025-01-03 05:00:00+00:00",
                    "2025-01-06 05:00:00+00:00",
                    "2025-01-07 05:00:00+00:00",
                    "2025-01-08 05:00:00+00:00",
                ]
            )
            pd.DataFrame(
                {
                    "Datetime": dates,
                    "Close": [100.0, 99.0, 101.0, 102.0, 103.0],
                    "High": [101.0, 100.0, 102.0, 103.0, 104.0],
                    "Low": [99.0, 98.0, 100.0, 101.0, 102.0],
                    "Open": [100.0, 100.0, 100.0, 101.0, 102.0],
                    "Volume": [1000, 1000, 1000, 1000, 1000],
                }
            ).to_csv(path, index=False)

            daily = load_daily_market_frame(path)
            daily["flag"] = [False, True, False, True, True]
            aligned, calendar_dates = align_features_to_next_session(daily, ["flag"])

            self.assertEqual(aligned.iloc[0]["source_date"].isoformat(), "2025-01-02")
            self.assertEqual(aligned.iloc[0]["date"].isoformat(), "2025-01-03")
            self.assertEqual(aligned.iloc[1]["source_date"].isoformat(), "2025-01-03")
            self.assertEqual(aligned.iloc[1]["date"].isoformat(), "2025-01-06")
            self.assertEqual(calendar_dates[-1].isoformat(), "2025-01-08")

    def test_build_feature_frame_uses_prior_session_information(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqq_1d.csv"
            dates = pd.to_datetime(
                [
                    "2025-01-02 05:00:00+00:00",
                    "2025-01-03 05:00:00+00:00",
                    "2025-01-06 05:00:00+00:00",
                    "2025-01-07 05:00:00+00:00",
                    "2025-01-08 05:00:00+00:00",
                ]
            )
            pd.DataFrame(
                {
                    "Datetime": dates,
                    "Close": [100.0, 99.0, 101.0, 102.0, 103.0],
                    "High": [101.0, 100.0, 102.0, 103.0, 104.0],
                    "Low": [99.0, 98.0, 100.0, 101.0, 102.0],
                    "Open": [100.0, 100.0, 100.0, 101.0, 102.0],
                    "Volume": [1000, 1000, 1000, 1000, 1000],
                }
            ).to_csv(path, index=False)

            features, _ = build_feature_frame(path)
            by_date = {row["date"].isoformat(): row for row in features.to_dict(orient="records")}

            self.assertFalse(by_date["2025-01-06"]["prev_day_up"])
            self.assertTrue(by_date["2025-01-07"]["prev_day_up"])
            self.assertTrue(by_date["2025-01-08"]["prev_day_up"])
            self.assertTrue(by_date["2025-01-08"]["mom3_positive"])

    def test_proxy_analyzer_file_fits_qc_editor_limit(self) -> None:
        path = Path("lean/QQQ_ORB_DeepBacktest/QQQ_V18_SlowTrend_ProxyAnalyzer_WebIDE.py")
        self.assertLessEqual(path.stat().st_size, 32000)


if __name__ == "__main__":
    unittest.main()
