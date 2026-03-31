"""
Grid search parameter optimiser.

Iterates over a parameter grid, runs the backtest for each combination,
and returns the results ranked by a chosen metric (default: sharpe_ratio).
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

import pandas as pd
from rich.console import Console
from rich.progress import track

from src.backtest.engine import BacktestEngine, BacktestResult
from src.strategies.base import BaseStrategy

console = Console()


def build_param_grid(param_ranges: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Convert {param_name: [val1, val2, ...]} into a flat list of dicts.

    Example
    -------
    >>> build_param_grid({"orb_bars": [2, 3], "profit_ratio": [1.5, 2.0]})
    [{'orb_bars': 2, 'profit_ratio': 1.5},
     {'orb_bars': 2, 'profit_ratio': 2.0},
     {'orb_bars': 3, 'profit_ratio': 1.5},
     {'orb_bars': 3, 'profit_ratio': 2.0}]
    """
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def grid_search(
    strategy_cls: type[BaseStrategy],
    param_ranges: dict[str, list[Any]],
    df: pd.DataFrame,
    engine: BacktestEngine | None = None,
    optimize_metric: str = "sharpe_ratio",
    top_n: int = 5,
) -> list[BacktestResult]:
    """
    Run a full grid search over *param_ranges* for *strategy_cls*.

    Parameters
    ----------
    strategy_cls    : Uninstantiated strategy class (e.g. ORBStrategy)
    param_ranges    : {param_name: [values_to_try]}
    df              : OHLCV DataFrame to backtest on
    engine          : BacktestEngine instance (created with defaults if None)
    optimize_metric : Metric name to rank results by (higher = better)
    top_n           : Return only the top N results

    Returns
    -------
    List of BacktestResult objects sorted by *optimize_metric* descending.
    """
    engine = engine or BacktestEngine()
    param_grid = build_param_grid(param_ranges)

    console.print(
        f"[cyan]Grid search: {len(param_grid)} combinations "
        f"for {strategy_cls.name}[/cyan]"
    )

    results: list[BacktestResult] = []

    for params in track(param_grid, description="Running grid search…"):
        try:
            strategy = strategy_cls(params=params)
            result = engine.run(strategy, df)
            results.append(result)
        except Exception as exc:
            console.print(f"[yellow]Skipped {params}: {exc}[/yellow]")

    if not results:
        return []

    # Sort by the chosen metric (descending)
    results.sort(
        key=lambda r: r.metrics.get(optimize_metric, float("-inf")),
        reverse=True,
    )

    console.print(
        f"[green]Best {optimize_metric}: "
        f"{results[0].metrics.get(optimize_metric, 0):.4f} "
        f"with params {results[0].params}[/green]"
    )

    return results[:top_n]


# ── Pre-defined search spaces ──────────────────────────────────────────────

ORB_PARAM_RANGES: dict[str, list[Any]] = {
    "orb_bars": [2, 3, 4, 6],
    "profit_ratio": [1.5, 2.0, 2.5, 3.0],
    "close_before_min": [10, 15, 20],
}

VWAP_PARAM_RANGES: dict[str, list[Any]] = {
    "k": [1.0, 1.5, 2.0],
    "sl_k_add": [0.3, 0.5, 0.7],
    "std_window": [15, 20, 30],
    "rsi_os": [30, 35, 40],
    "rsi_ob": [60, 65, 70],
}

# Compact grids for quick runs
ORB_QUICK_RANGES: dict[str, list[Any]] = {
    "orb_bars": [2, 3, 4],
    "profit_ratio": [1.5, 2.0, 2.5],
}

VWAP_QUICK_RANGES: dict[str, list[Any]] = {
    "k": [1.0, 1.5, 2.0],
    "sl_k_add": [0.5],
    "rsi_os": [30, 35],
    "rsi_ob": [65, 70],
}
