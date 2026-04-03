"""
Result logger – saves each backtest run to JSON and appends to CSV.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestResult

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_result(result: BacktestResult, results_dir: Path | None = None) -> Path:
    """
    Persist a BacktestResult to:
    - results/backtest_YYYYMMDD_HHMMSS_<strategy>.json
    - results/backtest_log.csv  (append)

    Returns the path to the saved JSON file.
    """
    out_dir = results_dir or _ensure_results_dir()
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"backtest_{ts}_{result.strategy_name}.json"

    payload: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "strategy": result.strategy_name,
        "params": result.params,
        "metrics": result.metrics,
        "equity_curve": [round(v, 2) for v in result.equity_curve],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    _append_csv(result, out_dir)

    return json_path


def _append_csv(result: BacktestResult, out_dir: Path) -> None:
    csv_path = out_dir / "backtest_log.csv"
    file_exists = csv_path.exists()

    row: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "strategy": result.strategy_name,
        **{f"param_{k}": v for k, v in result.params.items()},
        **result.metrics,
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_history(results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all saved JSON results from the results directory."""
    out_dir = results_dir or _ensure_results_dir()
    results = []
    for path in sorted(out_dir.glob("backtest_*.json")):
        with open(path, encoding="utf-8") as f:
            try:
                results.append(json.load(f))
            except json.JSONDecodeError:
                pass
    return results
