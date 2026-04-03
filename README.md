# ICT Trading System - QQQ

Automated trading system based on Inner Circle Trader (ICT) concepts, with backtesting, analysis, and auto-optimization.

## ICT Strategy Components
- **Market Structure**: Break of Structure (BOS) & Change of Character (CHoCH)
- **Order Blocks**: Institutional supply/demand zones
- **Fair Value Gaps (FVG)**: Price imbalances
- **Liquidity Sweeps**: Stop hunt detection
- **Optimal Trade Entry (OTE)**: Fibonacci retracement zones (61.8%-78.6%)

## Confluence Scoring
Each ICT signal adds to a composite score. Trades are only taken when the score exceeds the threshold, ensuring multiple confirmations.

## Usage
```bash
# Basic backtest
python main.py

# Backtest + parameter optimization
python main.py --optimize
```

## Auto-Optimization
The system includes a scheduled task that runs every hour to:
1. Re-run backtests with varied parameters
2. Find optimal parameter combinations
3. Update config automatically
4. Track improvement across iterations

## Files
- `config.py` - All tunable parameters
- `ict_indicators.py` - ICT indicator calculations
- `backtester.py` - Backtesting engine
- `analyzer.py` - Performance analysis & reporting
- `optimizer.py` - Auto-optimization module
- `charts.py` - Chart generation
- `data_fetcher.py` - Data loading (Yahoo Finance / CSV / synthetic)
- `main.py` - Main entry point
