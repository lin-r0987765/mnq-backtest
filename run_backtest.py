#!/usr/bin/env python3
"""
MNQ Automated Backtest Runner
==============================
Usage:
    python run_backtest.py [--quick] [--symbol NQ=F] [--period 60d]

Steps
-----
1. Download NQ=F 5-minute data (60-day history)
2. Run ORB strategy (default params)
3. Run VWAP Reversion strategy (default params)
4. Quick grid search → best params for each strategy
5. Re-run both strategies with optimised params
6. Save results to results/ (JSON + CSV)
7. Generate equity curve charts
8. Print Rich summary table
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Make sure the project root is on PYTHONPATH ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.data.fetcher import fetch_nq_data
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine, BacktestResult
from src.reporting.logger import save_result
from src.reporting.charts import plot_equity_curve, plot_equity_plotly, plot_comparison
from src.optimizer.grid_search import (
    grid_search,
    ORB_QUICK_RANGES,
    VWAP_QUICK_RANGES,
)

console = Console()
RESULTS_DIR = PROJECT_ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNQ Automated Backtest Runner")
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip full grid search, use compact param ranges"
    )
    parser.add_argument(
        "--symbol", default="NQ=F",
        help="Primary ticker symbol (default: NQ=F)"
    )
    parser.add_argument(
        "--period", default="60d",
        help="History period for yfinance (default: 60d)"
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Skip grid search entirely"
    )
    return parser.parse_args()


def print_header() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]MNQ Quantitative Backtest System[/bold cyan]\n"
            "[dim]Strategies: ORB + VWAP Reversion  |  Data: NQ=F 5-min[/dim]",
            border_style="cyan",
        )
    )
    console.print(
        f"[dim]Started at {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]\n"
    )


def print_metrics_table(results: list[BacktestResult]) -> None:
    """Print a formatted Rich table with all backtest results."""
    table = Table(
        title="Backtest Results Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )

    table.add_column("Strategy", style="cyan", width=22)
    table.add_column("Return %", justify="right", style="bold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Max DD %", justify="right", style="red")
    table.add_column("Win %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Profit Factor", justify="right")

    for r in results:
        m = r.metrics
        ret = m.get("total_return_pct", 0.0)
        ret_style = "green" if ret >= 0 else "red"

        table.add_row(
            r.strategy_name,
            f"[{ret_style}]{ret:+.2f}%[/{ret_style}]",
            f"{m.get('sharpe_ratio', 0):.3f}",
            f"{m.get('sortino_ratio', 0):.3f}",
            f"{m.get('max_drawdown_pct', 0):.2f}%",
            f"{m.get('win_rate_pct', 0):.1f}%",
            str(m.get("total_trades", 0)),
            f"{m.get('profit_factor', 0):.3f}",
        )

    console.print(table)


def run_single(
    strategy_cls,
    params: dict | None,
    df,
    engine: BacktestEngine,
    label: str = "",
) -> BacktestResult | None:
    """Run one strategy and handle errors gracefully."""
    try:
        strategy = strategy_cls(params=params)
        console.print(f"[blue]▶ Running {strategy.name} {label}…[/blue]")
        t0 = time.time()
        result = engine.run(strategy, df)
        elapsed = time.time() - t0
        m = result.metrics
        console.print(
            f"  [green]✓[/green] {strategy.name} | "
            f"Return: [bold]{m.get('total_return_pct', 0):+.2f}%[/bold] | "
            f"Sharpe: {m.get('sharpe_ratio', 0):.3f} | "
            f"Trades: {m.get('total_trades', 0)} | "
            f"({elapsed:.1f}s)"
        )
        return result
    except Exception as exc:
        console.print(f"[red]✗ {strategy_cls.name} failed: {exc}[/red]")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    print_header()

    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. Download data ──────────────────────────────────────────────────
    console.rule("[bold]Step 1 – Data Download[/bold]")
    try:
        df = fetch_nq_data(symbol=args.symbol, period=args.period)
    except Exception as exc:
        console.print(f"[red]Data download failed: {exc}[/red]")
        return 1

    console.print(
        f"  Bars: [bold]{len(df)}[/bold] | "
        f"From: {df.index[0].strftime('%Y-%m-%d')} → "
        f"{df.index[-1].strftime('%Y-%m-%d')}\n"
    )

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=1.0)
    all_results: list[BacktestResult] = []

    # ── 2. Default ORB ────────────────────────────────────────────────────
    console.rule("[bold]Step 2 – ORB (default params)[/bold]")
    orb_default = run_single(ORBStrategy, None, df, engine, "(default)")
    if orb_default:
        all_results.append(orb_default)

    # ── 3. Default VWAP ───────────────────────────────────────────────────
    console.rule("[bold]Step 3 – VWAP Reversion (default params)[/bold]")
    vwap_default = run_single(VWAPReversionStrategy, None, df, engine, "(default)")
    if vwap_default:
        all_results.append(vwap_default)

    # ── 4. Grid search ────────────────────────────────────────────────────
    best_orb_result: BacktestResult | None = orb_default
    best_vwap_result: BacktestResult | None = vwap_default

    if not args.no_grid:
        console.rule("[bold]Step 4 – Grid Search[/bold]")

        console.print("[cyan]ORB grid search…[/cyan]")
        orb_top = grid_search(
            ORBStrategy, ORB_QUICK_RANGES, df, engine,
            optimize_metric="sharpe_ratio", top_n=3
        )
        if orb_top:
            best_orb_result = orb_top[0]

        console.print("\n[cyan]VWAP Reversion grid search…[/cyan]")
        vwap_top = grid_search(
            VWAPReversionStrategy, VWAP_QUICK_RANGES, df, engine,
            optimize_metric="sharpe_ratio", top_n=3
        )
        if vwap_top:
            best_vwap_result = vwap_top[0]

        # ── 5. Re-run with best params ────────────────────────────────────
        console.rule("[bold]Step 5 – Optimised Re-run[/bold]")
        if best_orb_result and best_orb_result is not orb_default:
            orb_opt = run_single(
                ORBStrategy, best_orb_result.params, df, engine, "(optimised)"
            )
            if orb_opt:
                orb_opt.strategy_name = "ORB_Optimised"
                all_results.append(orb_opt)
                best_orb_result = orb_opt

        if best_vwap_result and best_vwap_result is not vwap_default:
            vwap_opt = run_single(
                VWAPReversionStrategy, best_vwap_result.params, df, engine,
                "(optimised)"
            )
            if vwap_opt:
                vwap_opt.strategy_name = "VWAP_Optimised"
                all_results.append(vwap_opt)
                best_vwap_result = vwap_opt

    # ── 6. Save results ───────────────────────────────────────────────────
    console.rule("[bold]Step 6 – Saving Results[/bold]")
    saved_paths: list[Path] = []
    for result in all_results:
        try:
            path = save_result(result, RESULTS_DIR)
            saved_paths.append(path)
            console.print(f"  [green]Saved[/green] → {path.name}")
        except Exception as exc:
            console.print(f"  [yellow]Save failed for {result.strategy_name}: {exc}[/yellow]")

    # ── 7. Charts ─────────────────────────────────────────────────────────
    console.rule("[bold]Step 7 – Generating Charts[/bold]")
    chart_paths: list[Path] = []
    for result in all_results:
        try:
            png = plot_equity_curve(result, RESULTS_DIR)
            chart_paths.append(png)
            console.print(f"  [green]Chart saved[/green] → {png.name}")
            # Optional Plotly HTML
            html = plot_equity_plotly(result, RESULTS_DIR)
            if html:
                console.print(f"  [green]Interactive chart[/green] → {html.name}")
        except Exception as exc:
            console.print(f"  [yellow]Chart error for {result.strategy_name}: {exc}[/yellow]")

    if len(all_results) >= 2:
        try:
            comp = plot_comparison(all_results, RESULTS_DIR)
            console.print(f"  [green]Comparison chart[/green] → {comp.name}")
        except Exception:
            pass

    # ── 8. Summary table ──────────────────────────────────────────────────
    console.rule("[bold]Step 8 – Summary[/bold]")
    if all_results:
        print_metrics_table(all_results)
    else:
        console.print("[red]No results to display.[/red]")

    console.print(
        f"\n[dim]Completed at "
        f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]"
    )
    console.print(
        f"[dim]Results saved to: {RESULTS_DIR}[/dim]\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
