#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Research Backtest Runner
==============================
Usage:
    python run_backtest.py [--quick] [--symbol QQQ] [--period 60d] [--include-vwap] [--include-ict]

Steps
-----
1. Load the repo-preferred intraday dataset (local QQQ CSV first, yfinance fallback)
2. Run ORB strategy (default params)
3. Optionally run VWAP Reversion strategy
4. Quick grid search → best params for active strategies
5. Re-run active strategies with optimised params
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Make sure the project root is on PYTHONPATH ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.data.fetcher import fetch_nq_data, fetch_peer_data, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_research_profile_params,
)
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
    parser = argparse.ArgumentParser(description="Local research backtest runner")
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip full grid search, use compact param ranges"
    )
    parser.add_argument(
        "--symbol", default="QQQ",
        help="Primary ticker symbol for fallback downloads (default: QQQ)"
    )
    parser.add_argument(
        "--period", default="60d",
        help="History period for yfinance (default: 60d)"
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Skip grid search entirely"
    )
    parser.add_argument(
        "--include-vwap", action="store_true",
        help="Enable the optional VWAP module in default flow"
    )
    parser.add_argument(
        "--include-ict", action="store_true",
        help="Enable the experimental ICT entry model module"
    )
    parser.add_argument(
        "--ict-basic", action="store_true",
        help="Run the bare ICT baseline instead of the repo research profile"
    )
    parser.add_argument(
        "--ict-peer-symbol", default=None,
        help="Optional peer symbol for real SMT backtests (example: SPY)"
    )
    parser.add_argument(
        "--ict-peer-csv", default=None,
        help="Optional peer CSV path for real SMT backtests"
    )
    return parser.parse_args()


def print_header(include_vwap: bool, include_ict: bool, peer_label: str | None) -> None:
    modules = ["ORB"]
    if include_vwap:
        modules.append("VWAP Reversion")
    if include_ict:
        ict_label = "ICT Entry Model (research profile)"
        if peer_label:
            ict_label += f" + peer={peer_label}"
        modules.append(ict_label)
    strategy_label = " + ".join(modules)
    console.print(
        Panel.fit(
            "[bold cyan]Local Quantitative Backtest System[/bold cyan]\n"
            f"[dim]Strategies: {strategy_label}  |  Data: repo-preferred intraday feed[/dim]",
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
        console.print(f"[blue]Running {strategy.name} {label}...[/blue]")
        t0 = time.time()
        result = engine.run(strategy, df)
        elapsed = time.time() - t0
        m = result.metrics
        console.print(
            f"  [green]OK[/green] {strategy.name} | "
            f"Return: [bold]{m.get('total_return_pct', 0):+.2f}%[/bold] | "
            f"Sharpe: {m.get('sharpe_ratio', 0):.3f} | "
            f"Trades: {m.get('total_trades', 0)} | "
            f"({elapsed:.1f}s)"
        )
        return result
    except Exception as exc:
        console.print(f"[red]{strategy_cls.name} failed: {exc}[/red]")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    peer_label = args.ict_peer_csv or args.ict_peer_symbol
    print_header(args.include_vwap, args.include_ict, peer_label)

    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. Download data ──────────────────────────────────────────────────
    console.rule("[bold]Step 1 – Data Download[/bold]")
    try:
        df = fetch_nq_data(symbol=args.symbol, period=args.period)
    except Exception as exc:
        console.print(f"[red]Data download failed: {exc}[/red]")
        return 1

    peer_df = None
    if args.include_ict and (args.ict_peer_symbol or args.ict_peer_csv):
        console.print("[cyan]Loading peer-symbol data for ICT SMT...[/cyan]")
        try:
            peer_df = fetch_peer_data(
                peer_symbol=args.ict_peer_symbol,
                peer_csv=args.ict_peer_csv,
                period=args.period,
            )
            df = merge_peer_columns(df, peer_df)
            matched_peer_bars = int(df["PeerHigh"].notna().sum()) if "PeerHigh" in df.columns else 0
            console.print(
                f"[green]Peer data merged[/green] | matched bars: [bold]{matched_peer_bars}[/bold]"
            )
        except Exception as exc:
            console.print(f"[red]Peer data load failed: {exc}[/red]")
            return 1

    console.print(
        f"  Bars: [bold]{len(df)}[/bold] | "
        f"From: {df.index[0].strftime('%Y-%m-%d')} → "
        f"{df.index[-1].strftime('%Y-%m-%d')}\n"
    )

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
    all_results: list[BacktestResult] = []

    # ── 2. Default ORB ────────────────────────────────────────────────────
    console.rule("[bold]Step 2 – ORB (default params)[/bold]")
    orb_default = run_single(ORBStrategy, None, df, engine, "(default)")
    if orb_default:
        all_results.append(orb_default)

    # ── 3. Default VWAP ───────────────────────────────────────────────────
    vwap_default: BacktestResult | None = None
    if args.include_vwap:
        console.rule("[bold]Step 3 – VWAP Reversion (default params)[/bold]")
        vwap_default = run_single(VWAPReversionStrategy, None, df, engine, "(default)")
        if vwap_default:
            all_results.append(vwap_default)
    else:
        console.rule("[bold]Step 3 – VWAP Module[/bold]")
        console.print("[yellow]VWAP module disabled in default flow. Use --include-vwap to enable.[/yellow]")

    # ── 4. Grid search ────────────────────────────────────────────────────
    ict_default: BacktestResult | None = None
    if args.include_ict:
        console.rule("[bold]Step 3B ??ICT Entry Model[/bold]")
        ict_params = None
        ict_label = "(basic defaults)" if args.ict_basic else "(research profile)"
        if not args.ict_basic:
            ict_params = build_ict_research_profile_params(enable_smt=peer_df is not None)
        ict_default = run_single(ICTEntryModelStrategy, ict_params, df, engine, ict_label)
        if ict_default:
            all_results.append(ict_default)
    else:
        console.rule("[bold]Step 3B ??ICT Module[/bold]")
        console.print("[yellow]ICT module disabled in default flow. Use --include-ict to enable.[/yellow]")

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

        if args.include_vwap:
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

        if args.include_vwap and best_vwap_result and best_vwap_result is not vwap_default:
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

    # ── 8. Summary table ──────────�
