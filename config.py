"""
ICT Trading System Configuration
=================================
All tunable parameters for the ICT strategy.
Version tracking for iterative optimization.
"""

VERSION = "3.50.233"
ITERATION = 320

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
STRUCTURE_LOOKBACK = int(20)# Bars to identify swing highs/lows
SWING_THRESHOLD = 3            # Min bars on each side for swing point

# Order Blocks
OB_LOOKBACK = int(15)# Bars to look back for order blocks
BREAKER_LOOKBACK = int(15)# Bars to look back for breaker blocks
IFVG_LOOKBACK = int(20)# Bars to look back for inversion FVGs
OB_BODY_MIN_PCT = 0.3          # Min body/range ratio for OB candle
OB_MITIGATION_TOUCH = True     # OB invalidated after first touch

# Fair Value Gaps
FVG_MIN_GAP_PCT = 0.001  # v3 ?踐?????# Min gap size as % of price (0.2%)
FVG_MAX_AGE = 20               # Max bars before FVG expires
FVG_REVISIT_MIN_DELAY_BARS = 1  # Min bars to wait after the structure-shift arm before a valid revisit entry
MAX_PENDING_SETUPS_PER_DIRECTION = 1  # Max simultaneous pending setups allowed per direction
MAX_REENTRIES_PER_SETUP = 0  # Max stop-out retries allowed per armed setup; 0 disables same-zone re-entry
ENABLE_CONTINUATION_ENTRY = False  # Allow armed FVG/IFVG setups to refresh into newer same-direction FVGs before entry
FVG_ORIGIN_MAX_LAG_BARS = 0    # Max bars between FVG origin and the structure-shift confirmation bar; 0 disables the gate
FVG_ORIGIN_BODY_MIN_PCT = 0.0  # Min body ratio required on the FVG origin candle; 0 disables the gate
FVG_ORIGIN_BODY_ATR_MULT = 0.0  # Min absolute body size of the FVG origin candle as a multiple of ATR; 0 disables the gate
FVG_ORIGIN_CLOSE_POSITION_MIN_PCT = 0.0  # Min close-position ratio on the FVG origin candle; bullish candles must close high in range, bearish candles must close low in range
FVG_ORIGIN_OPPOSITE_WICK_MAX_PCT = 0.0  # Max allowed opposite-side wick ratio on the FVG origin candle; 0 disables the gate
FVG_ORIGIN_RANGE_ATR_MULT = 0.0  # Min full range of the FVG origin candle as a multiple of ATR; 0 disables the gate
FVG_MAX_RETEST_TOUCHES = 0     # Max zone touches allowed before an FVG/IFVG setup is invalidated; 0 disables the cap
DISPLACEMENT_BODY_MIN_PCT = 0.0  # Min body/range ratio on the structure-shift candle
DISPLACEMENT_RANGE_ATR_MULT = 0.0  # Min shift-candle full range as a multiple of ATR to qualify as true displacement
STRUCTURE_SHIFT_CLOSE_BUFFER_RATIO = 0.0  # Min close-through distance beyond the swept structure level as a fraction of shift-candle range
FVG_REVISIT_DEPTH_RATIO = 0.0  # Min revisit depth into the FVG as a fraction of gap height; 0.5 = consequent encroachment
FVG_REJECTION_CLOSE_RATIO = 0.0  # Min close-back recovery inside the FVG on the entry bar; 0.5 = close back to consequent encroachment
FVG_REJECTION_WICK_RATIO = 0.0  # Min rejection wick on the entry bar as a fraction of gap height
FVG_REJECTION_BODY_MIN_PCT = 0.0  # Min directional body/range ratio on the entry bar during FVG rejection

# Liquidity Sweeps
LIQ_SWEEP_LOOKBACK = 50        # Bars to identify liquidity pools
LIQ_SWEEP_THRESHOLD = 0.001    # Min % beyond level to count as sweep
LIQ_SWEEP_RECOVERY_BARS = 3    # Bars to recover after sweep
LIQ_SWEEP_RECLAIM_RATIO = 0.0  # Extra close-back reclaim beyond the swept level as a fraction of sweep depth

# Internal vs External Liquidity
USE_EXTERNAL_LIQUIDITY_FILTER = False
EXTERNAL_LIQUIDITY_LOOKBACK = 80
EXTERNAL_LIQUIDITY_TOLERANCE = 0.001

# SMT Divergence
USE_SMT_FILTER = False
SMT_LOOKBACK = 20
SMT_THRESHOLD = 0.001
ICT_PEER_SYMBOL = "SPY"
ICT_RESEARCH_PROFILE = "ict_full_stack_v1"
ICT_REQUIRE_REAL_PEER_FOR_SMT = True

# AMD / Market-Maker Path Logic
USE_AMD_FILTER = False
AMD_ACCUMULATION_BARS = 3
AMD_MANIPULATION_THRESHOLD = 0.001
AMD_REQUIRE_MIDPOINT_RECLAIM = True

# Macro Timing Windows
USE_MACRO_TIMING_WINDOWS = False
MACRO_TIMEZONE = "America/New_York"
MACRO_WINDOWS = (
    (9, 50, 10, 10),
    (10, 50, 11, 10),
)

# Previous-Session Liquidity Map / Dealing-Range Anchors
USE_PREV_SESSION_ANCHOR_FILTER = False
PREV_SESSION_ANCHOR_TOLERANCE = 0.05
PREV_SESSION_ANCHOR_FILTER_MODE = "hard"
PREV_SESSION_ANCHOR_MISMATCH_SCORE_PENALTY = 0.0

# Session-Specific Dealing-Array Refinement
USE_SESSION_ARRAY_REFINEMENT = False
SESSION_ARRAY_FILTER_MODE = "hard"
SESSION_ARRAY_MISMATCH_SCORE_PENALTY = 0.0
DEALING_ARRAY_TIMEZONE = "America/New_York"
IMBALANCE_ARRAY_WINDOWS = (
    (3, 0, 4, 30),
    (9, 30, 10, 30),
)
STRUCTURAL_ARRAY_WINDOWS = (
    (10, 30, 11, 30),
    (13, 0, 14, 30),
)

# Optimal Trade Entry (Fibonacci)
OTE_FIB_LOW = 0.618            # OTE zone lower bound
OTE_FIB_HIGH = 0.786           # OTE zone upper bound

# === Risk Management ===
STOP_LOSS_ATR_MULT = 2.0    # SL = entry 蝪?ATR * mult
TAKE_PROFIT_RR = 4.0        # Risk:Reward ratio for TP (?????4.0)
MIN_REWARD_RISK_RATIO = 1.5  # Min projected reward:risk ratio required before entry
TRAILING_STOP_ATR = 2.0     # Trailing stop ATR multiplier
ATR_PERIOD = int(14)        # ATR calculation period????鞈?
MAX_DAILY_LOSS_PCT = 0.05      # Max 5% daily loss ??stop trading

# === Signal Scoring ===
# Each ICT confluence adds to score; trade when score >= threshold
SCORE_ORDER_BLOCK = 2
SCORE_BREAKER_BLOCK = 2
SCORE_IFVG = 2
SCORE_FVG = 2
SCORE_LIQUIDITY_SWEEP = 3
SCORE_OTE_ZONE = 2
SCORE_BOS = 2
SCORE_CHOCH = 3
MIN_SCORE_TO_TRADE = 6.0    # Min confluence score to enter

# === Session Filters (UTC hours) ===
LONDON_OPEN = 8
LONDON_CLOSE = 16
NY_OPEN = 13
NY_CLOSE = 21
TRADE_SESSIONS = True          # Only trade during sessions

# === ICT Kill Zone Specialization ===
# Kept off by default until the new specialization is validated on longer data.
USE_KILL_ZONES = False
KILL_ZONE_TIMEZONE = "America/New_York"
LONDON_KILL_START = 3
LONDON_KILL_END = 4
NY_AM_KILL_START = 10
NY_AM_KILL_END = 11
NY_PM_KILL_START = 14
NY_PM_KILL_END = 15

# === ICT Higher-Timeframe Daily Bias ===
# Conservative first pass: derive bias from the previous completed day and its
# close location inside a rolling completed-daily range.
USE_DAILY_BIAS_FILTER = False
DAILY_BIAS_MODE = "statistical"  # "statistical" or "structure"
DAILY_BIAS_LOOKBACK = 3
DAILY_BIAS_SWING_THRESHOLD = 1
DAILY_BIAS_BULL_THRESHOLD = 0.60
DAILY_BIAS_BEAR_THRESHOLD = 0.40

# === ICT Premium / Discount Context ===
# Conservative first pass: derive premium/discount from the current bar's
# position inside a rolling completed-daily range.
USE_PREMIUM_DISCOUNT_FILTER = False
PREMIUM_DISCOUNT_LOOKBACK = 5
PREMIUM_DISCOUNT_NEUTRAL_BAND = 0.05

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


