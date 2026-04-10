#!/usr/bin/env python3
"""
v11 虧損年份 regime 特徵分析
用 v12 trades（因為包含 v11 的所有交易）分析 2017-2019 虧損年份的模式。
目標：找出可區分虧損年 vs 盈利年的 regime 特徵。
"""

import csv, io
from collections import defaultdict
from datetime import datetime

# --- 讀取 v12 trades ---
trades_path = "/sessions/clever-vibrant-hawking/mnt/mnq-backtest/QuantConnect results/2017-2026/Square Green Dragonfly_trades.csv"

trades = []
with open(trades_path, 'r', newline='') as f:
    content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        entry_dt = datetime.fromisoformat(row['Entry Time'].strip().replace('Z', '+00:00'))
        exit_dt = datetime.fromisoformat(row['Exit Time'].strip().replace('Z', '+00:00'))
        pnl = float(row['P&L'].strip())
        fees = float(row['Fees'].strip())
        mfe = float(row['MFE'].strip())
        mae = float(row['MAE'].strip())
        entry_price = float(row['Entry Price'].strip())
        exit_price = float(row['Exit Price'].strip())
        qty = int(row['Quantity'].strip())

        is_eod = exit_dt.strftime('%H:%M') == '19:50'
        duration_min = (exit_dt - entry_dt).total_seconds() / 60

        trades.append({
            'date': entry_dt.strftime('%Y-%m-%d'),
            'year': entry_dt.strftime('%Y'),
            'month': entry_dt.strftime('%Y-%m'),
            'entry_hour': entry_dt.hour + entry_dt.minute / 60,
            'entry_dt': entry_dt,
            'exit_dt': exit_dt,
            'pnl': pnl,
            'fees': fees,
            'net': pnl - fees,
            'mfe': mfe,
            'mae': mae,
            'is_win': int(row['IsWin'].strip()),
            'is_eod': is_eod,
            'exit_type': 'TP' if (pnl > 0 and not is_eod) else ('EOD' if is_eod else 'SL/Trail'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'qty': qty,
            'duration_min': duration_min,
            'pnl_per_share': (pnl - fees) / qty if qty > 0 else 0,
        })

# Only first entries per day (approximate v11)
by_date = defaultdict(list)
for t in trades:
    by_date[t['date']].append(t)
for d in by_date:
    by_date[d].sort(key=lambda x: x['entry_dt'])

v11_trades = [by_date[d][0] for d in sorted(by_date.keys())]

print("=" * 80)
print("v11 虧損年份 Regime 特徵分析")
print("=" * 80)

# --- 按年分組 ---
losing_years = ['2017', '2018', '2019']
winning_years = ['2020', '2021', '2023', '2024', '2025']

lose_trades = [t for t in v11_trades if t['year'] in losing_years]
win_trades = [t for t in v11_trades if t['year'] in winning_years]

def group_stats(trade_list, label):
    if not trade_list:
        return
    n = len(trade_list)
    net = sum(t['net'] for t in trade_list)
    wins = sum(1 for t in trade_list if t['is_win'])
    avg_mfe = sum(t['mfe'] for t in trade_list) / n
    avg_mae = sum(t['mae'] for t in trade_list) / n
    avg_dur = sum(t['duration_min'] for t in trade_list) / n
    avg_entry_hour = sum(t['entry_hour'] for t in trade_list) / n
    tp_count = sum(1 for t in trade_list if t['exit_type'] == 'TP')
    sl_count = sum(1 for t in trade_list if t['exit_type'] == 'SL/Trail')
    eod_count = sum(1 for t in trade_list if t['exit_type'] == 'EOD')

    print(f"\n{label}:")
    print(f"  筆數: {n}, 淨利: ${net:.2f}, 勝率: {wins/n*100:.1f}%")
    print(f"  平均MFE: ${avg_mfe:.2f}, 平均MAE: ${avg_mae:.2f}")
    print(f"  平均持倉: {avg_dur:.0f}min, 平均進場時間(UTC): {avg_entry_hour:.1f}h")
    print(f"  出場分佈: TP={tp_count}({tp_count/n*100:.0f}%), SL/Trail={sl_count}({sl_count/n*100:.0f}%), EOD={eod_count}({eod_count/n*100:.0f}%)")
    print(f"  平均每筆淨利: ${net/n:.2f}")

print("\n## 1. 虧損年 vs 盈利年整體對比")
print("-" * 60)
group_stats(lose_trades, "虧損年 (2017-2019)")
group_stats(win_trades, "盈利年 (2020-2025)")

# --- 按出場類型分析 ---
print(f"\n\n## 2. 出場類型分析 (虧損年 vs 盈利年)")
print("-" * 60)

for exit_type in ['TP', 'SL/Trail', 'EOD']:
    lt = [t for t in lose_trades if t['exit_type'] == exit_type]
    wt = [t for t in win_trades if t['exit_type'] == exit_type]
    print(f"\n{exit_type}:")
    if lt:
        print(f"  虧損年: {len(lt)}筆, 平均淨利=${sum(t['net'] for t in lt)/len(lt):.2f}")
    if wt:
        print(f"  盈利年: {len(wt)}筆, 平均淨利=${sum(t['net'] for t in wt)/len(wt):.2f}")

# --- 按月份分析 ---
print(f"\n\n## 3. 月份季節性分析")
print("-" * 60)

month_stats = defaultdict(lambda: {'count': 0, 'net': 0, 'wins': 0})
for t in v11_trades:
    m = int(t['month'].split('-')[1])
    month_stats[m]['count'] += 1
    month_stats[m]['net'] += t['net']
    month_stats[m]['wins'] += t['is_win']

print(f"\n{'月份':>4} | {'筆數':>4} | {'淨利':>10} | {'勝率':>6} | {'平均':>8}")
print("-" * 50)
for m in range(1, 13):
    s = month_stats[m]
    if s['count'] > 0:
        wr = s['wins'] / s['count'] * 100
        avg = s['net'] / s['count']
        print(f"  {m:>2} | {s['count']:>4} | ${s['net']:>8.2f} | {wr:>5.1f}% | ${avg:>6.2f}")

# --- 按 entry hour 分析 ---
print(f"\n\n## 4. 進場時間分析 (UTC)")
print("-" * 60)

hour_stats = defaultdict(lambda: {'count': 0, 'net': 0, 'wins': 0})
for t in v11_trades:
    h = int(t['entry_hour'])
    hour_stats[h]['count'] += 1
    hour_stats[h]['net'] += t['net']
    hour_stats[h]['wins'] += t['is_win']

print(f"\n{'時間':>6} | {'筆數':>4} | {'淨利':>10} | {'勝率':>6} | {'平均':>8}")
print("-" * 50)
for h in sorted(hour_stats.keys()):
    s = hour_stats[h]
    if s['count'] > 0:
        wr = s['wins'] / s['count'] * 100
        avg = s['net'] / s['count']
        print(f"  {h:>2}:xx | {s['count']:>4} | ${s['net']:>8.2f} | {wr:>5.1f}% | ${avg:>6.2f}")

# --- 持倉時間 vs 結果 ---
print(f"\n\n## 5. 持倉時間 vs 結果")
print("-" * 60)

short_hold = [t for t in v11_trades if t['duration_min'] <= 30]
mid_hold = [t for t in v11_trades if 30 < t['duration_min'] <= 120]
long_hold = [t for t in v11_trades if t['duration_min'] > 120]

for label, group in [("≤30min", short_hold), ("30-120min", mid_hold), (">120min", long_hold)]:
    if group:
        n = len(group)
        net = sum(t['net'] for t in group)
        wins = sum(1 for t in group if t['is_win'])
        print(f"  {label}: {n}筆, 淨利=${net:.2f}, 勝率={wins/n*100:.1f}%, 平均=${net/n:.2f}")

# --- MFE capture ratio ---
print(f"\n\n## 6. MFE 捕獲效率 (虧損年 vs 盈利年)")
print("-" * 60)

for label, group in [("虧損年", lose_trades), ("盈利年", win_trades)]:
    tp_trades = [t for t in group if t['exit_type'] == 'TP']
    if tp_trades:
        # For TP trades, how much of MFE was captured?
        capture = [t['net'] / t['mfe'] if t['mfe'] > 0 else 0 for t in tp_trades]
        avg_capture = sum(capture) / len(capture)
        print(f"  {label} TP trades: {len(tp_trades)}筆, 平均MFE捕獲率={avg_capture*100:.1f}%")

    sl_trades = [t for t in group if t['exit_type'] == 'SL/Trail']
    if sl_trades:
        # For SL trades, MAE vs actual loss
        print(f"  {label} SL trades: {len(sl_trades)}筆, 平均淨利=${sum(t['net'] for t in sl_trades)/len(sl_trades):.2f}, 平均MAE=${sum(t['mae'] for t in sl_trades)/len(sl_trades):.2f}")

# --- 連敗分析 ---
print(f"\n\n## 7. 連敗分析")
print("-" * 60)

streak = 0
max_streak = 0
max_streak_start = None
current_start = None
streaks = []

for t in v11_trades:
    if not t['is_win']:
        if streak == 0:
            current_start = t['date']
        streak += 1
    else:
        if streak > 0:
            streaks.append({'length': streak, 'start': current_start, 'end': t['date']})
        if streak > max_streak:
            max_streak = streak
            max_streak_start = current_start
        streak = 0

if streak > 0:
    streaks.append({'length': streak, 'start': current_start, 'end': v11_trades[-1]['date']})

streaks.sort(key=lambda x: x['length'], reverse=True)
print(f"\n最長連敗: {max_streak} 筆 (開始 {max_streak_start})")
print(f"\n前 5 最長連敗:")
for s in streaks[:5]:
    print(f"  {s['length']}連敗: {s['start']} ~ {s['end']}")

print(f"\n\n{'='*80}")
print("Regime Filter 改善方向總結")
print("=" * 80)
print("""
基於以上分析，可探索的 regime filter 改善方向：

1. 月份/季節性 filter
   - 如果某些月份持續為負，可考慮靜默
   - 需要更多年份數據驗證（目前 8 年可能不夠穩健）

2. 連敗 kill switch
   - 連敗達到 N 筆時暫停交易直到條件重置
   - 但需小心 overfitting

3. 進場時間 filter
   - 是否某些時段（過早/過晚）的進場品質明顯較差

4. 前提：更新 qqq_1d.csv
   - 目前只到 2019-10-02，嚴重限制 regime 研究能力
   - 建議使用者更新日線數據到 2026
""")
