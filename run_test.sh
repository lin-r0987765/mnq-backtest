#!/bin/bash
cd /sessions/quirky-busy-cannon/mnt/claude--mnq-backtest
python3 run_backtest.py --no-grid 2>&1 | tee iteration_19_backtest.log
