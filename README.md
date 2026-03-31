# MNQ Quantitative Backtest System

Automated intraday backtesting framework for **NQ / MNQ futures** using:

- **Opening Range Breakout (ORB)** strategy
- **VWAP Mean-Reversion** strategy

Data is sourced from Yahoo Finance (`NQ=F`, 5-minute bars, 60-day history).

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/lin-r0987765/mnq-backtest.git
cd mnq-backtest
pip install -r requirements.txt

# Run a full backtest
python run_backtest.py

# Run without grid search (faster)
python run_backtest.py --no-grid

# Use NDX index as proxy if NQ=F is unavailable
python run_backtest.py --symbol "^NDX"
```

---

## Project Structure

```
mnq-backtest/
├── src/
│   ├── data/
│   │   └── fetcher.py          # yfinance data downloader
│   ├── strategies/
│   │   ├── base.py             # Abstract base strategy
│   │   ├── orb.py              # Opening Range Breakout
│   │   └── vwap_reversion.py   # VWAP mean-reversion + RSI filter
│   ├── backtest/
│   │   └── engine.py           # vectorbt wrapper + manual fallback
│   ├── reporting/
│   │   ├── metrics.py          # Sharpe, Sortino, Calmar, etc.
│   │   ├── logger.py           # JSON + CSV result persistence
│   │   └── charts.py           # Matplotlib + Plotly equity charts
│   └── optimizer/
│       └── grid_search.py      # Parameter grid search
├── results/                    # Auto-generated backtest results
├── run_backtest.py             # Main entry point
├── requirements.txt
└── pyproject.toml
```

---

## Strategies

### Opening Range Breakout (ORB)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `orb_bars` | 3 | Number of 5-min bars for opening range (3 = 15 min) |
| `profit_ratio` | 2.0 | Take-profit = range width × ratio |
| `close_before_min` | 15 | Force-close N minutes before session end |

**Logic:**
- Build the opening range from the first N bars after 09:30 ET
- Long if price breaks above range high; stop at range low
- Short if price breaks below range low; stop at range high
- One trade per direction per day; forced flat at 15:45 ET

### VWAP Mean-Reversion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 1.5 | Band multiplier (VWAP ± k × std) |
| `sl_k_add` | 0.5 | Extra k for stop-loss |
| `std_window` | 20 | Rolling std window |
| `rsi_period` | 14 | RSI period |
| `rsi_os` | 35 | Oversold threshold (long entry) |
| `rsi_ob` | 65 | Overbought threshold (short entry) |

**Logic:**
- Session-anchored VWAP reset at 09:30 ET each day
- Long when price drops below `VWAP - k×std` AND RSI < `rsi_os`
- Short when price rises above `VWAP + k×std` AND RSI > `rsi_ob`
- Target: return to VWAP midline

---

## Output

Each run produces:

```
results/
├── backtest_YYYYMMDD_HHMMSS_ORB.json
├── backtest_YYYYMMDD_HHMMSS_VWAP_Reversion.json
├── backtest_log.csv              ← cumulative history
├── equity_ORB_YYYYMMDD_HHMMSS.png
├── equity_VWAP_Reversion_YYYYMMDD_HHMMSS.png
├── equity_ORB_YYYYMMDD_HHMMSS.html   ← interactive Plotly
└── comparison_YYYYMMDD_HHMMSS.png
```

### JSON Schema

```json
{
  "timestamp": "2024-03-29T14:00:00+00:00",
  "strategy": "ORB",
  "params": { "orb_bars": 3, "profit_ratio": 2.0, "close_before_min": 15 },
  "metrics": {
    "total_return_pct": 12.5,
    "sharpe_ratio": 1.34,
    "sortino_ratio": 1.89,
    "calmar_ratio": 1.52,
    "max_drawdown_pct": -8.2,
    "win_rate_pct": 54.3,
    "total_trades": 87,
    "profit_factor": 1.45,
    "avg_trade_pct": 0.14
  },
  "equity_curve": [100000.0, ...]
}
```

---

## Automated Scheduling

The system is configured to run automatically every hour via the **Cowork Scheduler**.
Results are accumulated in `results/backtest_log.csv` over time.

To run manually on a schedule with cron:

```cron
0 * * * * cd /path/to/mnq-backtest && python run_backtest.py --no-grid >> results/cron.log 2>&1
```

---

## Metrics Glossary

| Metric | Description |
|--------|-------------|
| `total_return_pct` | Total return over backtest period (%) |
| `sharpe_ratio` | Risk-adjusted return (annualised, 5-min bars) |
| `sortino_ratio` | Downside-risk adjusted return |
| `calmar_ratio` | Annual return / max drawdown |
| `max_drawdown_pct` | Largest peak-to-trough decline (%) |
| `win_rate_pct` | Percentage of profitable trades |
| `total_trades` | Total number of completed trades |
| `profit_factor` | Gross profit / gross loss |
| `avg_trade_pct` | Average return per trade (%) |

---

## Disclaimer

> This system is for **educational and research purposes only**.
> Past backtest performance does not guarantee future results.
> NQ/MNQ futures trading involves substantial risk of loss.
