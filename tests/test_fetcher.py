from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.fetcher import load_ohlcv_csv


class FetcherCsvLoaderTest(unittest.TestCase):
    def test_load_ohlcv_csv_accepts_regularized_lowercase_schema(self) -> None:
        csv_text = "\n".join(
            [
                "timestamp,open,high,low,close,volume,session_date,is_missing",
                "2026-04-08 09:30:00-04:00,100.0,101.0,99.5,100.5,1000,2026-04-08,False",
                "2026-04-08 09:31:00-04:00,100.5,101.2,100.4,101.0,1100,2026-04-08,False",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "qqq_1m_regular_et.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            df = load_ohlcv_csv(csv_path)

        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(len(df), 2)
        self.assertEqual(str(df.index[0]), "2026-04-08 09:30:00-04:00")
        self.assertEqual(float(df.iloc[0]["Open"]), 100.0)
        self.assertEqual(float(df.iloc[1]["Close"]), 101.0)

    def test_load_ohlcv_csv_accepts_regularized_schema_with_timestamp_column(self) -> None:
        csv_text = "\n".join(
            [
                "open,high,low,close,volume,timestamp,session_date",
                "100.0,101.0,99.5,100.5,1000,2026-04-08 09:30:00-04:00,2026-04-08",
                "100.5,101.2,100.4,101.0,1100,2026-04-08 09:35:00-04:00,2026-04-08",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "qqq_5m_regular_et.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            df = load_ohlcv_csv(csv_path)

        self.assertEqual(len(df), 2)
        self.assertEqual(str(df.index[0]), "2026-04-08 09:30:00-04:00")
        self.assertEqual(float(df.iloc[1]["Close"]), 101.0)


if __name__ == "__main__":
    unittest.main()
