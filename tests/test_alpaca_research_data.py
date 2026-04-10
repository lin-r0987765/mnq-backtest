from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.orb.analyze_local_orb_v25_profit_lock import load_csv_5m
from prepare_alpaca_research_data import TIMEFRAME_TO_OUTPUT, normalize_alpaca_frame


class AlpacaResearchDataTest(unittest.TestCase):
    def test_supported_timeframe_mapping_contains_spy_compatible_intervals(self) -> None:
        self.assertEqual(TIMEFRAME_TO_OUTPUT["5Min"], "5m")
        self.assertEqual(TIMEFRAME_TO_OUTPUT["1Day"], "1d")

    def test_normalize_alpaca_frame_matches_repo_csv_shape(self) -> None:
        raw = pd.DataFrame(
            {
                "timestamp": ["2026-01-02 14:35:00+00:00", "2026-01-02 14:30:00+00:00"],
                "open": [100.0, 99.0],
                "high": [101.0, 100.0],
                "low": [98.5, 98.0],
                "close": [100.5, 99.5],
                "volume": [2000, 1000],
                "trade_count": [20, 10],
                "vwap": [100.2, 99.3],
            }
        )

        normalized = normalize_alpaca_frame(raw)

        self.assertEqual(
            list(normalized.columns),
            ["Datetime", "Close", "High", "Low", "Open", "Volume"],
        )
        self.assertEqual(normalized.iloc[0]["Datetime"], "2026-01-02 14:30:00+00:00")
        self.assertEqual(float(normalized.iloc[0]["Open"]), 99.0)

    def test_load_csv_5m_accepts_single_header_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "qqq_5m_alpaca.csv"
            frame = pd.DataFrame(
                {
                    "Datetime": ["2026-01-02 14:30:00+00:00", "2026-01-02 14:35:00+00:00"],
                    "Close": [100.5, 101.0],
                    "High": [101.0, 101.5],
                    "Low": [99.5, 100.0],
                    "Open": [100.0, 100.5],
                    "Volume": [1000, 1200],
                }
            )
            frame.to_csv(path, index=False)

            loaded = load_csv_5m(path)

            self.assertEqual(list(loaded.columns), ["Open", "High", "Low", "Close", "Volume"])
            self.assertEqual(str(loaded.index.tz), "America/New_York")
            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
