from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

from research.qc.analyze_qc_regime_prototypes import build_daily_regime_frame, compute_profit_factor


class AnalyzeQcRegimePrototypesTest(unittest.TestCase):
    def test_compute_profit_factor(self) -> None:
        s = pd.Series([3.0, -1.0, 2.0, -2.0])
        self.assertAlmostEqual(compute_profit_factor(s), 5.0 / 3.0, places=6)

    def test_build_daily_regime_frame_has_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qqq_1d.csv"
            dates = pd.date_range("2025-01-01", periods=260, freq="B", tz="UTC")
            pd.DataFrame(
                {
                    "Datetime": dates,
                    "Close": 100 + pd.Series(range(len(dates))) * 0.1,
                    "High": 101 + pd.Series(range(len(dates))) * 0.1,
                    "Low": 99 + pd.Series(range(len(dates))) * 0.1,
                    "Open": 100 + pd.Series(range(len(dates))) * 0.1,
                    "Volume": 1000,
                }
            ).to_csv(path, index=False)
            frame = build_daily_regime_frame(path)
            self.assertIn("trend_up", frame.columns)
            self.assertIn("trend_strength", frame.columns)
            self.assertIn("low_gap", frame.columns)


if __name__ == "__main__":
    unittest.main()
