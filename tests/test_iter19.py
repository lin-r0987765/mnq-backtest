#!/usr/bin/env python3
"""
Quick test for iteration 19 - VWAP max_trades_per_day: 3
"""
import sys
import os
from pathlib import Path

# Ensure path is set
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Set matplotlib to non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

# Now import the rest
from src.data.fetcher import fetch_nq_data
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine
from rich.console import Console

console = Console()

def main():
    console.print("[bold cyan]Testing Iteration 19: VWAP max_trades_per_day=3[/bold cyan]\n")

    # Load data
    console.print("[bold]Loading data...[/bold]")
    try:
        df = fetch_nq_data()
        console.print(f"[green]Loaded {len(df)} bars[/green]")
        console.print(f"  Data types:\n{df.dtypes}\n")
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        return 1

    # Run VWAP strategy
    console.print("[bold]Running VWAP Reversion Strategy (v19)...[/bold]")
    try:
        engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
        strategy = VWAPReversionStrategy()

        console.print(f"  Strategy params:\n{strategy.params}\n")

        result = engine.run(strategy, df)

        m = result.metrics
        console.print("[green]✓ Strategy ran successfully![/green]\n")

        console.print("[bold]VWAP Reversion Results (v19):[/bold]")
        console.print(f"  Return: {m.get('total_return_pct', 0):+.2f}%")
        console.print(f"  Sharpe: {m.get('sharpe_ratio', 0):.3f}")
        console.print(f"  Win Rate: {m.get('win_rate_pct', 0):.1f}%")
        console.print(f"  Max Drawdown: {m.get('max_drawdown_pct', 0):.2f}%")
        console.print(f"  Total Trades: {m.get('total_trades', 0)}")
        console.print(f"  Profit Factor: {m.get('profit_factor', 0):.3f}\n")

        # Check rollback conditions
        console.print("[bold]Rollback Checks (v18 baseline: Sharpe=10.484, WR=55.6%, MaxDD=-0.035%):[/bold]")

        sharpe_threshold = 8.387  # 80% of 10.484
        wr_threshold = 50.6  # 55.6% - 5pp

        sharpe = m.get('sharpe_ratio', 0)
        wr = m.get('win_rate_pct', 0)
        dd = m.get('max_drawdown_pct', 0)

        sharpe_ok = sharpe >= sharpe_threshold
        wr_ok = wr >= wr_threshold
        dd_ok = dd >= -0.0525  # -0.035% with 50% deterioration margin

        console.print(f"  Sharpe {sharpe:.3f} >= {sharpe_threshold} (80% baseline): {'✓' if sharpe_ok else '✗'}")
        console.print(f"  WR {wr:.1f}% >= {wr_threshold}%: {'✓' if wr_ok else '✗'}")
        console.print(f"  MaxDD {dd:.3f}% >= -0.0525% (50% margin): {'✓' if dd_ok else '✗'}\n")

        if sharpe_ok and wr_ok and dd_ok:
            console.print("[green bold]✓ All checks passed - v19 is valid![/green bold]")
            return 0
        else:
            console.print("[red bold]✗ Rollback conditions triggered![/red bold]")
            return 1

    except Exception as e:
        console.print(f"[red]Strategy failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
