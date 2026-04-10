from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


HALF_DAY_MARKER = "half-day detected"
VERSION_PATTERN = re.compile(r"version=(v[0-9][^|\r\n ]+)")


@dataclass
class ResultBundle:
    json_path: Path
    logs_path: Path
    orders_path: Path
    trades_path: Path


def resolve_bundle(result_dir: Path) -> ResultBundle:
    json_path = max(result_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    prefix = json_path.stem
    logs_path = result_dir / f"{prefix}_logs.txt"
    orders_path = result_dir / f"{prefix}_orders.csv"
    trades_path = result_dir / f"{prefix}_trades.csv"
    if not logs_path.exists():
        logs_path = max(result_dir.glob("*_logs.txt"), key=lambda p: p.stat().st_mtime)
    if not orders_path.exists():
        orders_path = max(result_dir.glob("*_orders.csv"), key=lambda p: p.stat().st_mtime)
    if not trades_path.exists():
        trades_path = max(result_dir.glob("*_trades.csv"), key=lambda p: p.stat().st_mtime)
    return ResultBundle(
        json_path=json_path,
        logs_path=logs_path,
        orders_path=orders_path,
        trades_path=trades_path,
    )


def count_same_bar_eod_reentries(orders: pd.DataFrame) -> list[tuple[str, list[str]]]:
    tagged = orders.copy()
    tagged["Tag2"] = tagged["Tag"].fillna("").astype(str).str.strip()
    rows: list[tuple[str, list[str]]] = []
    for timestamp, tags in tagged.groupby("Time")["Tag2"].apply(list).items():
        if "ORB EOD Flatten" in tags and ("ORB Long" in tags or "ORB Short" in tags):
            rows.append((timestamp, tags))
    return rows


def build_report(result_dir: Path) -> dict[str, object]:
    bundle = resolve_bundle(result_dir)
    logs = bundle.logs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    orders = pd.read_csv(bundle.orders_path)
    trades = pd.read_csv(bundle.trades_path)
    stats_obj = json.loads(bundle.json_path.read_text(encoding="utf-8"))
    stats = stats_obj.get("statistics", {}) or stats_obj.get("Statistics", {})
    reentries = count_same_bar_eod_reentries(orders)
    detected_versions = sorted(
        {
            match.group(1)
            for line in logs
            for match in [VERSION_PATTERN.search(line)]
            if match is not None
        }
    )
    return {
        "json_file": bundle.json_path.name,
        "logs_file": bundle.logs_path.name,
        "orders_file": bundle.orders_path.name,
        "trades_file": bundle.trades_path.name,
        "contains_version_marker": bool(detected_versions),
        "detected_versions": detected_versions,
        "contains_halfday_marker": any(HALF_DAY_MARKER in line for line in logs),
        "same_bar_eod_reentry_count": len(reentries),
        "same_bar_eod_reentries": reentries,
        "orders_count": int(len(orders)),
        "trades_count": int(len(trades)),
        "net_profit": stats.get("Net Profit"),
        "sharpe_ratio": stats.get("Sharpe Ratio"),
        "win_rate": stats.get("Win Rate"),
        "drawdown": stats.get("Drawdown"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze QuantConnect Web IDE result exports.")
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="QuantConnect results/2017-2026",
        help="Directory containing QuantConnect json/logs/orders/trades exports.",
    )
    args = parser.parse_args()
    report = build_report(Path(args.result_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
