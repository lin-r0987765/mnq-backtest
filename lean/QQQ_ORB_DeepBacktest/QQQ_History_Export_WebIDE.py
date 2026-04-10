from AlgorithmImports import *

from datetime import datetime, timedelta
import gzip
import json
import pandas as pd


class QQQHistoryExportWebIDE(QCAlgorithm):
    """
    QuantConnect Web IDE exporter for adjusted QQQ history.

    This algorithm saves yearly gzip CSV chunks into the QuantConnect Object Store
    so they can be downloaded after the backtest completes.
    """

    SYMBOL = "QQQ"
    EXPORT_START = datetime(2017, 4, 3)
    EXPORT_END_INCLUSIVE = datetime(2026, 4, 2)
    EXPORT_5M = True
    EXPORT_1H = True
    EXPORT_1D = True

    def initialize(self) -> None:
        self.set_time_zone(TimeZones.NEW_YORK)
        self.set_start_date(2026, 4, 2)
        self.set_end_date(2026, 4, 2)
        self.set_cash(100000)

        equity = self.add_equity(
            self.SYMBOL,
            Resolution.MINUTE,
            fill_forward=False,
            extended_market_hours=False,
            data_normalization_mode=DataNormalizationMode.ADJUSTED,
        )
        self.symbol = equity.symbol
        self._export_done = False
        self._manifest = {
            "symbol": self.SYMBOL,
            "start": self.EXPORT_START.date().isoformat(),
            "end_inclusive": self.EXPORT_END_INCLUSIVE.date().isoformat(),
            "files": [],
        }

        self.debug(
            "QQQ history export init | "
            f"symbol={self.SYMBOL} | start={self.EXPORT_START.date()} | "
            f"end_inclusive={self.EXPORT_END_INCLUSIVE.date()} | "
            f"project_id={getattr(self, 'project_id', 'unknown')}"
        )

    def on_data(self, data: Slice) -> None:
        if self._export_done:
            return
        self._export_done = True
        self._run_export()

    def on_end_of_algorithm(self) -> None:
        manifest_key = self._object_store_key("manifest_json")
        self.object_store.save(manifest_key, json.dumps(self._manifest, indent=2))
        self.debug(
            "QQQ history export summary | "
            f"files={len(self._manifest['files'])} | manifest_key={manifest_key}"
        )

    def _run_export(self) -> None:
        for year in range(self.EXPORT_START.year, self.EXPORT_END_INCLUSIVE.year + 1):
            year_start = max(self.EXPORT_START, datetime(year, 1, 1))
            year_end_inclusive = min(self.EXPORT_END_INCLUSIVE, datetime(year, 12, 31))
            year_end_exclusive = year_end_inclusive + timedelta(days=1)

            if self.EXPORT_5M or self.EXPORT_1H:
                minute_history = self.history(self.symbol, year_start, year_end_exclusive, Resolution.MINUTE)
                minute_frame = self._normalize_qc_history_frame(minute_history)
            else:
                minute_frame = self._empty_history_frame()

            if self.EXPORT_5M:
                five_minute = self._trim_frame(
                    self._resample_ohlcv(minute_frame, "5min"),
                    year_start,
                    year_end_inclusive,
                )
                self._save_year_chunk(five_minute, interval_label="5m", year=year)

            if self.EXPORT_1H:
                one_hour = self._trim_frame(
                    self._resample_ohlcv(minute_frame, "60min", offset="30min"),
                    year_start,
                    year_end_inclusive,
                )
                self._save_year_chunk(one_hour, interval_label="1h", year=year)

            if self.EXPORT_1D:
                daily_history = self.history(self.symbol, year_start, year_end_exclusive, Resolution.DAILY)
                daily_frame = self._trim_frame(
                    self._normalize_qc_history_frame(daily_history),
                    year_start,
                    year_end_inclusive,
                )
                self._save_year_chunk(daily_frame, interval_label="1d", year=year)

    def _save_year_chunk(self, frame: pd.DataFrame, *, interval_label: str, year: int) -> None:
        if frame.empty:
            self.debug(f"QQQ history export skip | interval={interval_label} | year={year} | rows=0")
            return

        key = self._object_store_key(f"qqq_{interval_label}_{year}_csv_gz")

        export_frame = frame.copy()
        export_frame.index = pd.to_datetime(export_frame.index, utc=True, errors="coerce")
        export_frame.index.name = "Datetime"
        export_frame = export_frame[["Close", "High", "Low", "Open", "Volume"]]
        csv_bytes = export_frame.to_csv().encode("utf-8")
        gzip_bytes = gzip.compress(csv_bytes)
        save_ok = self.object_store.save_bytes(key, gzip_bytes)

        details = {
            "key": key,
            "interval": interval_label,
            "year": year,
            "rows": int(len(export_frame)),
            "start": export_frame.index.min().isoformat(),
            "end": export_frame.index.max().isoformat(),
        }
        self._manifest["files"].append(details)
        self.debug(
            "QQQ history export saved | "
            f"interval={interval_label} | year={year} | rows={details['rows']} | key={key} | save_ok={save_ok}"
        )

    def _object_store_key(self, filename: str) -> str:
        project_id = str(getattr(self, "project_id", "project"))
        return f"{project_id}/qc_history_export_{filename}"

    @staticmethod
    def _empty_history_frame() -> pd.DataFrame:
        frame = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        frame.index = pd.DatetimeIndex([], tz="UTC", name="Datetime")
        return frame

    @staticmethod
    def _infer_time_column(frame: pd.DataFrame) -> str:
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                return column

        for column in frame.columns:
            parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if parsed.notna().any():
                return column

        raise ValueError("Unable to infer the timestamp column from QuantConnect history output.")

    def _normalize_qc_history_frame(self, history: pd.DataFrame) -> pd.DataFrame:
        if history is None or len(history) == 0:
            return self._empty_history_frame()

        frame = history.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.reset_index()
        else:
            frame = frame.sort_index().reset_index()

        time_column = self._infer_time_column(frame)
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
        frame = frame[frame[time_column].notna()].sort_values(time_column).set_index(time_column)
        frame = frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        frame = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[frame.index.notna()]
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame.index.name = "Datetime"

        for column in ["Open", "High", "Low", "Close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0).astype("int64")

        return frame

    @staticmethod
    def _resample_ohlcv(frame: pd.DataFrame, rule: str, offset: str | None = None) -> pd.DataFrame:
        if frame.empty:
            empty = QQQHistoryExportWebIDE._empty_history_frame()
            empty.index.name = "Datetime"
            return empty

        resample_kwargs = {"label": "left", "closed": "left"}
        if offset is not None:
            resample_kwargs["offset"] = offset

        resampled = frame.resample(rule, **resample_kwargs).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
        resampled.index.name = "Datetime"
        resampled["Volume"] = pd.to_numeric(resampled["Volume"], errors="coerce").fillna(0).astype("int64")
        return resampled

    @staticmethod
    def _trim_frame(frame: pd.DataFrame, start: datetime, end_inclusive: datetime) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        start_ts = pd.Timestamp(start, tz="UTC")
        end_exclusive_ts = pd.Timestamp(end_inclusive + timedelta(days=1), tz="UTC")
        return frame[(frame.index >= start_ts) & (frame.index < end_exclusive_ts)].copy()
