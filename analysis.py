#!/usr/bin/env python3
"""
QC Baseline Trades Analysis - Research Hypothesis Discovery
Analyzes 2017-2026 trades from Retrospective Tan Salmon strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Load the trades
csv_path = "/sessions/sharp-magical-cannon/mnt/mnq-backtest/QuantConnect results/2017-2026/Retrospective Tan Salmon_trades.csv"
df = pd.read_csv(csv_path)

# Parse timestamps as UTC
df['Entry Time'] = pd.to_datetime(df['Entry Time'], utc=True)
df['Exit Time'] = pd.to_datetime(df['Exit Time'], utc=True)

# Convert to ET/NY time for analysis
et_tz = pytz.timezone('US/Eastern')
df['Entry Time ET'] = df['Entry Time'].dt.tz_convert(et_tz)
df['Exit Time ET'] = df['Exit Time'].dt.tz_convert(et_tz)

# Extract components
df['Entry Hour'] = df['Entry Time ET'].dt.hour
df['Entry Day'] = df['Entry Time ET'].dt.day_name()
df['Entry Month'] = df['Entry Time ET'].dt.month
df['Entry Date'] = df['Entry Time ET'].dt.date
df['IsWin'] = df['IsWin'].astype(int)

# Calculate trade duration
df['Duration'] = df['Exit Time'] - df['Entry Time']
df['Duration Hours'] = df['Duration'].dt.total_seconds() / 3600

# Create duration buckets
def duration_bucket(hours):
    if hours < 1:
        return '< 1 hour'
    elif hours < 3:
        return '1-3 hours'
    elif hours < 5:
        return '3-5 hours'
    else:
        return '5+ hours'

df['Duration Bucket'] = df['Duration Hours'].apply(duration_bucket)

print("=" * 80)
print("QC BASELINE TRADES ANALYSIS (2017-2026)")
print("=" * 80)
print(f"\nTotal trades: {len(df)}")
print(f"Total P&L: ${df['P&L'].sum():,.2f}")
print(f"Win rate: {df['IsWin'].mean()*100:.2f}%")
print(f"Winning trades: {df['IsWin'].sum()}")
print(f"Losing trades: {len(df) - df['IsWin'].sum()}")
print(f"Date range: {df['Entry Date'].min()} to {df['Entry Date'].max()}")

# ============================================================================
# 1. TIME-OF-DAY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("1. TIME-OF-DAY ANALYSIS (ET/NY Time)")
print("=" * 80)

hourly = df.groupby('Entry Hour').agg({
    'P&L': ['count', 'sum', 'mean'],
    'IsWin': ['sum', 'mean']
}).round(2)

hourly.columns = ['Trade Count', 'Total P&L', 'Avg P&L', 'Wins', 'Win Rate']
hourly['Win Rate'] = (hourly['Win Rate'] * 100).round(2)

# Calculate profit factor per hour
profit_factor_hourly = []
for hour in df['Entry Hour'].unique():
    hour_trades = df[df['Entry Hour'] == hour]
    gains = hour_trades[hour_trades['P&L'] > 0]['P&L'].sum()
    losses = abs(hour_trades[hour_trades['P&L'] < 0]['P&L'].sum())
    pf = gains / losses if losses > 0 else np.inf
    profit_factor_hourly.append({'Hour': hour, 'Profit Factor': pf})

pf_df = pd.DataFrame(profit_factor_hourly).set_index('Hour')
hourly = hourly.join(pf_df)
hourly = hourly.sort_index()

print("\n" + hourly.to_string())
print(f"\nBest hour by win rate: {hourly['Win Rate'].idxmax()}h (win rate: {hourly['Win Rate'].max():.1f}%)")
print(f"Best hour by total P&L: {hourly['Total P&L'].idxmax()}h (P&L: ${hourly['Total P&L'].max():,.2f})")
print(f"Worst hour by win rate: {hourly['Win Rate'].idxmin()}h (win rate: {hourly['Win Rate'].min():.1f}%)")

# ============================================================================
# 2. DAY-OF-WEEK ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. DAY-OF-WEEK ANALYSIS")
print("=" * 80)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
daily = df.groupby('Entry Day').agg({
    'P&L': ['count', 'sum', 'mean'],
    'IsWin': ['sum', 'mean']
}).round(2)

daily.columns = ['Trade Count', 'Total P&L', 'Avg P&L', 'Wins', 'Win Rate']
daily['Win Rate'] = (daily['Win Rate'] * 100).round(2)

# Calculate profit factor per day
profit_factor_daily = []
for day in df['Entry Day'].unique():
    day_trades = df[df['Entry Day'] == day]
    gains = day_trades[day_trades['P&L'] > 0]['P&L'].sum()
    losses = abs(day_trades[day_trades['P&L'] < 0]['P&L'].sum())
    pf = gains / losses if losses > 0 else np.inf
    profit_factor_daily.append({'Day': day, 'Profit Factor': pf})

pf_daily_df = pd.DataFrame(profit_factor_daily).set_index('Day')
daily = daily.join(pf_daily_df)

# Reorder by day of week
daily = daily.reindex([d for d in day_order if d in daily.index])

print("\n" + daily.to_string())
print(f"\nBest day by total P&L: {daily['Total P&L'].idxmax()} (P&L: ${daily['Total P&L'].max():,.2f})")
print(f"Worst day by total P&L: {daily['Total P&L'].idxmin()} (P&L: ${daily['Total P&L'].min():,.2f})")

# ============================================================================
# 3. MONTHLY SEASONALITY
# ============================================================================
print("\n" + "=" * 80)
print("3. MONTHLY SEASONALITY")
print("=" * 80)

month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
               6: 'June', 7: 'July', 8: 'August', 9: 'September',
               10: 'October', 11: 'November', 12: 'December'}

monthly = df.groupby('Entry Month').agg({
    'P&L': ['count', 'sum', 'mean'],
    'IsWin': ['sum', 'mean']
}).round(2)

monthly.columns = ['Trade Count', 'Total P&L', 'Avg P&L', 'Wins', 'Win Rate']
monthly['Win Rate'] = (monthly['Win Rate'] * 100).round(2)

# Calculate profit factor per month
profit_factor_monthly = []
for month in df['Entry Month'].unique():
    month_trades = df[df['Entry Month'] == month]
    gains = month_trades[month_trades['P&L'] > 0]['P&L'].sum()
    losses = abs(month_trades[month_trades['P&L'] < 0]['P&L'].sum())
    pf = gains / losses if losses > 0 else np.inf
    profit_factor_monthly.append({'Month': month, 'Profit Factor': pf})

pf_monthly_df = pd.DataFrame(profit_factor_monthly).set_index('Month')
monthly = monthly.join(pf_monthly_df)
monthly.index = monthly.index.map(month_names)

print("\n" + monthly.to_string())
print(f"\nBest month by total P&L: {monthly['Total P&L'].idxmax()} (P&L: ${monthly['Total P&L'].max():,.2f})")
print(f"Worst month by total P&L: {monthly['Total P&L'].idxmin()} (P&L: ${monthly['Total P&L'].min():,.2f})")

# ============================================================================
# 4. TRADE DURATION VS OUTCOME
# ============================================================================
print("\n" + "=" * 80)
print("4. TRADE DURATION VS OUTCOME")
print("=" * 80)

winners = df[df['IsWin'] == 1]
losers = df[df['IsWin'] == 0]

print(f"\nAverage duration (winners): {winners['Duration Hours'].mean():.2f} hours")
print(f"Average duration (losers): {losers['Duration Hours'].mean():.2f} hours")
print(f"Median duration (winners): {winners['Duration Hours'].median():.2f} hours")
print(f"Median duration (losers): {losers['Duration Hours'].median():.2f} hours")

print("\nDuration bucketing:")
duration_analysis = df.groupby('Duration Bucket').agg({
    'P&L': ['count', 'sum', 'mean'],
    'IsWin': ['sum', 'mean']
}).round(2)

duration_analysis.columns = ['Trade Count', 'Total P&L', 'Avg P&L', 'Wins', 'Win Rate']
duration_analysis['Win Rate'] = (duration_analysis['Win Rate'] * 100).round(2)

# Reorder buckets
bucket_order = ['< 1 hour', '1-3 hours', '3-5 hours', '5+ hours']
duration_analysis = duration_analysis.reindex([b for b in bucket_order if b in duration_analysis.index])

print("\n" + duration_analysis.to_string())

# Correlation analysis
correlation = df[['Duration Hours', 'P&L']].corr().iloc[0, 1]
print(f"\nCorrelation between duration and P&L: {correlation:.4f}")

# ============================================================================
# 5. MAE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. MAE (MAX ADVERSE EXCURSION) ANALYSIS")
print("=" * 80)

# For winners
winners_mae = winners['MAE'].describe().round(2)
print(f"\nWinning trades MAE statistics:")
print(f"  Mean MAE: ${winners_mae['mean']:.2f}")
print(f"  Median MAE: ${winners_mae['50%']:.2f}")
print(f"  Std Dev: ${winners_mae['std']:.2f}")
print(f"  Min: ${winners_mae['min']:.2f}")
print(f"  Max: ${winners_mae['max']:.2f}")

# For losers
losers_mae = losers['MAE'].describe().round(2)
print(f"\nLosing trades MAE statistics:")
print(f"  Mean MAE: ${losers_mae['mean']:.2f}")
print(f"  Median MAE: ${losers_mae['50%']:.2f}")
print(f"  Std Dev: ${losers_mae['std']:.2f}")

# Analyze MFE (max favorable excursion)
print(f"\nLosing trades with positive MFE (were in profit at some point):")
losers_with_mfe = losers[losers['MFE'] > 0]
print(f"  Count: {len(losers_with_mfe)} ({len(losers_with_mfe)/len(losers)*100:.1f}% of losses)")
print(f"  Average MFE: ${losers_with_mfe['MFE'].mean():.2f}")
print(f"  Average final loss: ${losers_with_mfe['P&L'].mean():.2f}")

# Losing trades that had MFE > $50 (could-have-been winners if exited early)
losers_with_large_mfe = losers[losers['MFE'] > 50]
print(f"\nLosing trades with MFE > $50 (missed profit opportunities):")
print(f"  Count: {len(losers_with_large_mfe)} ({len(losers_with_large_mfe)/len(losers)*100:.1f}% of losses)")
if len(losers_with_large_mfe) > 0:
    print(f"  Average MFE: ${losers_with_large_mfe['MFE'].mean():.2f}")
    print(f"  Average final loss: ${losers_with_large_mfe['P&L'].mean():.2f}")
    print(f"  Median missed profit: ${(losers_with_large_mfe['MFE'] - losers_with_large_mfe['P&L']).median():.2f}")

# Overall fraction of losses with MFE > 0
fraction_with_mfe = len(losers_with_mfe) / len(losers)
print(f"\nFraction of losses with any positive MFE: {fraction_with_mfe:.1%}")

# ============================================================================
# RESEARCH HYPOTHESES
# ============================================================================
print("\n" + "=" * 80)
print("RESEARCH HYPOTHESES & INSIGHTS")
print("=" * 80)

# Best performing hours
best_hours = hourly.nlargest(3, 'Win Rate')
print(f"\n1. OPTIMAL TRADING HOURS:")
print(f"   Top 3 hours by win rate: {', '.join([str(h) + ':00' for h in best_hours.index.tolist()])}")
print(f"   These hours show significantly higher success rates.")

# Seasonality patterns
best_months = monthly.nlargest(3, 'Total P&L')
print(f"\n2. SEASONAL PATTERNS:")
print(f"   Best performing months: {', '.join(best_months.index.tolist())}")
print(f"   Combined P&L from top 3 months: ${best_months['Total P&L'].sum():,.2f}")

# Day patterns
best_days = daily.nlargest(3, 'Win Rate')
print(f"\n3. DAY-OF-WEEK EFFECTS:")
print(f"   Best days by win rate: {', '.join(best_days.index.tolist())}")

# Duration insights
if len(winners) > 0:
    duration_ratio = winners['Duration Hours'].mean() / losers['Duration Hours'].mean()
    print(f"\n4. DURATION ANALYSIS:")
    print(f"   Winners hold longer by {(duration_ratio - 1)*100:.1f}%")
    print(f"   Consider: Are we exiting winners too early in losers? Should we hold longer?")

# MAE insights
print(f"\n5. EXECUTION QUALITY (MAE Analysis):")
print(f"   Winners average MAE: ${winners['MAE'].mean():.2f}")
print(f"   Losers average MAE: ${losers['MAE'].mean():.2f}")
print(f"   {fraction_with_mfe:.1%} of losses were in profit at some point!")
print(f"   {len(losers_with_large_mfe)} losses ({len(losers_with_large_mfe)/len(losers)*100:.1f}%) had MFE > $50 (major missed exits)")
print(f"   IMPLICATION: Exit rules may be too tight or not responsive to MFE.")

# Profit factor hypothesis
avg_pf = hourly['Profit Factor'].mean()
best_pf_hour = hourly['Profit Factor'].idxmax()
print(f"\n6. PROFIT FACTOR ANALYSIS:")
print(f"   Average profit factor across hours: {avg_pf:.2f}")
print(f"   Best hour for profit factor: {best_pf_hour}:00 (PF: {hourly.loc[best_pf_hour, 'Profit Factor']:.2f})")

print("\n" + "=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
