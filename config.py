"""
ICT Trading System Configuration
=================================
All tunable parameters for the ICT strategy.
Version tracking for iterative optimization.
"""

VERSION = "1.0.0"
ITERATION = 2

# === Data Settings ===
SYMBOL = "QQQ"
BACKTEST_YEARS = 3
DATA_INTERVAL = "1d"  # daily candles

# === Account Settings ===
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE_PCT = 0.02       # Risk 2% per trade
MAX_POSITIONS = 3
COMMISSION_PER_TRADE = 1.0     # $1 per trade

# === ICT Strategy Parameters ===

# Market Structure
STRUCTURE_LOOKBACK = 20.0# Bars to identify swing highs/lows
SWING_THRESHOLD = 3            # Min bars on each side for swing point

# Order Blocks
OB_LOOKBACK = 15.0# Bars to look back for order blocks
OB_BODY_MIN_PCT = 0.3          # Min body/range ratio for OB candle
OB_MITIGATION_TOUCH = True     # OB invalidated after first touch

# Fair Value Gaps
FVG_MIN_GAP_PCT = 0.001# Min gap size as % of price (0.2%)
FVG_MAX_AGE = 20               # Max bars before FVG expires

# Liquidity Sweeps
LIQ_SWEEP_LOOKBACK = 50        # Bars to identify liquidity pools
LIQ_SWEEP_THRESHOLD = 0.001    # Min % beyond level to count as sweep
LIQ_SWEEP_RECOVERY_BARS = 3    # Bars to recover after sweep

# Optimal Trade Entry (Fibonacci)
OTE_FIB_LOW = 0.618            # OTE zone lower bound
OTE_FIB_HIGH = 0.786           # OTE zone upper bound

# === Risk Management ===
STOP_LOSS_ATR_MULT = 2.0# SL = entry ± ATR * mult
TAKE_PROFIT_RR = 3.5# Risk:Reward ratio for TP
TRAILING_STOP_ATR = 2.0# Trailing stop ATR multiplier
ATR_PERIOD = 18.0# ATR calculation period
MAX_DAILY_LOSS_PCT = 0.05      # Max 5% daily loss → stop trading

# === Signal Scoring ===
# Each ICT confluence adds to score; trade when score >= threshold
SCORE_ORDER_BLOCK = 2
SCORE_FVG = 2
SCORE_LIQUIDITY_SWEEP = 3
SCORE_OTE_ZONE = 2
SCORE_BOS = 2
SCORE_CHOCH = 3
MIN_SCORE_TO_TRADE = 6.0# Min confluence score to enter

# === Session Filters (UTC hours) ===
LONDON_OPEN = 8
LONDON_CLOSE = 16
NY_OPEN = 13
NY_CLOSE = 21
TRADE_SESSIONS = True          # Only trade during sessions

# === Optimization Ranges (for auto-iteration) ===
PARAM_RANGES = {
    "STRUCTURE_LOOKBACK": (10, 40, 5),
    "OB_LOOKBACK": (5, 20, 5),
    "FVG_MIN_GAP_PCT": (0.001, 0.005, 0.001),
    "STOP_LOSS_ATR_MULT": (1.0, 3.0, 0.5),
    "TAKE_PROFIT_RR": (2.0, 5.0, 0.5),
    "MIN_SCORE_TO_TRADE": (3, 7, 1),
    "ATR_PERIOD": (10, 20, 2),
    "TRAILING_STOP_ATR": (1.5, 3.0, 0.5),
}
