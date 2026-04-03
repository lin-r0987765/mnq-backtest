"""
Chart generation for backtest results.

Produces:
  • Equity curve (matplotlib, saved as PNG)
  • Drawdown chart (matplotlib, saved as PNG)
  • Interactive HTML equity curve (plotly, optional)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from src.backtest.engine import BacktestResult

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curve(
    result: BacktestResult,
    save_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """
    Plot the equity curve and drawdown for *result*.
    Saves to `results/equity_<strategy>_<timestamp>.png` and returns the path.
    """
    out_dir = save_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    equity = np.array(result.equity_curve)
    index = np.arange(len(equity))

    # Drawdown
    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / roll_max * 100

    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"equity_{result.strategy_name}_{ts}.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # ── Equity curve ────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(index, equity, linewidth=1.5, color="#2196F3", label="Equity")
    ax1.axhline(equity[0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.fill_between(index, equity[0], equity, where=equity >= equity[0],
                     alpha=0.15, color="#4CAF50")
    ax1.fill_between(index, equity[0], equity, where=equity < equity[0],
                     alpha=0.15, color="#F44336")

    metrics = result.metrics
    title = (
        f"{result.strategy_name} | "
        f"Return: {metrics.get('total_return_pct', 0):.2f}% | "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | "
        f"Max DD: {metrics.get('max_drawdown_pct', 0):.2f}% | "
        f"Win: {metrics.get('win_rate_pct', 0):.1f}%"
    )
    ax1.set_title(title, fontsize=11, pad=10)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Drawdown ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(index, drawdown, 0, alpha=0.7, color="#F44336", label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Bar number")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return out_path


def plot_equity_plotly(
    result: BacktestResult,
    save_dir: Path | None = None,
) -> Path | None:
    """
    Generate an interactive Plotly HTML equity chart.
    Returns the path or None if plotly is not installed.
    """
    if not _HAS_PLOTLY:
        return None

    out_dir = save_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"equity_{result.strategy_name}_{ts}.html"

    equity = result.equity_curve
    x = list(range(len(equity)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=equity,
        mode="lines",
        name="Equity",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.1)",
    ))

    metrics = result.metrics
    fig.update_layout(
        title=(
            f"{result.strategy_name} | "
            f"Return: {metrics.get('total_return_pct', 0):.2f}% | "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}"
        ),
        xaxis_title="Bar number",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=500,
    )

    fig.write_html(str(out_path))
    return out_path


def plot_comparison(
    results: list[BacktestResult],
    save_dir: Path | None = None,
) -> Path:
    """Plot multiple equity curves on the same chart for comparison."""
    out_dir = save_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"comparison_{ts}.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    for i, result in enumerate(results):
        equity = np.array(result.equity_curve)
        # Normalise to 100
        norm = equity / equity[0] * 100
        label = (
            f"{result.strategy_name} "
            f"({result.metrics.get('total_return_pct', 0):.1f}%)"
        )
        ax.plot(norm, label=label, color=colors[i % len(colors)], linewidth=1.5)

    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Strategy Comparison – Normalised Equity")
    ax.set_ylabel("Normalised Value (base=100)")
    ax.set_xlabel("Bar number")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
