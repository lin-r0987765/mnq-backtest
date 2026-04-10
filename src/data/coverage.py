from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TARGET_YEARS = 8.0
DEFAULT_DATA_TARGETS = {
    "qqq_5m.csv": {
        "interval": "5m",
        "target_years": DEFAULT_TARGET_YEARS,
        "recommended_provider": "polygon",
        "notes": "5m 長歷史建議改用可提供多年分鐘級資料的 provider。",
    },
    "qqq_1h.csv": {
        "interval": "1h",
        "target_years": DEFAULT_TARGET_YEARS,
        "recommended_provider": "polygon",
        "notes": "1h 歷史可和 5m 一起由同一 provider 回補。",
    },
    "qqq_1d.csv": {
        "interval": "1d",
        "target_years": DEFAULT_TARGET_YEARS,
        "recommended_provider": "yfinance",
        "notes": "日線通常可先用免費來源補足。",
    },
}


@dataclass
class CoverageEntry:
    file: str
    interval: str
    path: str
    exists: bool
    rows: int
    start_utc: str | None
    end_utc: str | None
    span_days: float
    span_years: float
    target_years: float
    meets_target: bool
    recommended_provider: str
    notes: str


def _read_price_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        df = pd.read_csv(path, index_col=0)

    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()].copy()
    return df


def describe_price_coverage(
    path: Path,
    *,
    interval: str,
    target_years: float,
    recommended_provider: str,
    notes: str,
) -> CoverageEntry:
    if not path.exists():
        return CoverageEntry(
            file=path.name,
            interval=interval,
            path=str(path),
            exists=False,
            rows=0,
            start_utc=None,
            end_utc=None,
            span_days=0.0,
            span_years=0.0,
            target_years=target_years,
            meets_target=False,
            recommended_provider=recommended_provider,
            notes=notes,
        )

    df = _read_price_csv(path)
    rows = len(df)
    if rows == 0:
        return CoverageEntry(
            file=path.name,
            interval=interval,
            path=str(path),
            exists=True,
            rows=0,
            start_utc=None,
            end_utc=None,
            span_days=0.0,
            span_years=0.0,
            target_years=target_years,
            meets_target=False,
            recommended_provider=recommended_provider,
            notes=notes,
        )

    start = df.index.min()
    end = df.index.max()
    span_days = max((end - start).total_seconds() / 86400.0, 0.0)
    span_years = span_days / 365.25

    return CoverageEntry(
        file=path.name,
        interval=interval,
        path=str(path),
        exists=True,
        rows=rows,
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        span_days=round(span_days, 3),
        span_years=round(span_years, 3),
        target_years=target_years,
        meets_target=span_years >= target_years,
        recommended_provider=recommended_provider,
        notes=notes,
    )


def build_coverage_report(
    project_root: str | Path,
    *,
    targets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    active_targets = targets or DEFAULT_DATA_TARGETS
    entries: dict[str, Any] = {}
    missing = []

    for filename, spec in active_targets.items():
        entry = describe_price_coverage(
            root / filename,
            interval=str(spec["interval"]),
            target_years=float(spec.get("target_years", DEFAULT_TARGET_YEARS)),
            recommended_provider=str(spec.get("recommended_provider", "")),
            notes=str(spec.get("notes", "")),
        )
        entries[filename] = asdict(entry)
        if not entry.meets_target:
            missing.append(filename)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_years_default": DEFAULT_TARGET_YEARS,
        "files": entries,
        "all_targets_met": len(missing) == 0,
        "missing_targets": missing,
    }


def write_coverage_report(project_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = build_coverage_report(project_root)
    path = Path(output_path)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
