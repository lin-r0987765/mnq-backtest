#!/usr/bin/env python3
"""
v13-regime-reentry 離線驗證
用 v12-dual-entry 的真實交易紀錄，模擬 v13 的雙條件過濾效果。

目標：
1. 找出 v12 中同日有 2 筆交易的天數
2. 判斷第一筆是否為 TP 出場（用 MFE 與 P&L 正向判斷）
3. 用 qqq_1d.csv 計算當日 trend_strength 並判斷是否達到 1.15x 門檻
4. 模擬 v13 過濾後的淨利差異
"""

import csv
import sys
from datetime import datetime, timedelta
from collections import defaultdict

# --- 讀取 v12 trades ---
trades_path = "/sessions/clever-vibrant-hawking/mnt/mnq-backtest/QuantConnect results/2017-2026/Square Green Dragonfly_trades.csv"

trades = []
with open(trades_path, 'r', newline='') as f:
    t_content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    import io as io2
    reader = csv.DictReader(io2.StringIO(t_content))
    for row in reader:
        entry_time = row['Entry Time'].strip()
        exit_time = row['Exit Time'].strip()
        pnl = float(row['P&L'].strip())
        fees = float(row['Fees'].strip())
        mfe = float(row['MFE'].strip())
        mae = float(row['MAE'].strip())
        is_win = int(row['IsWin'].strip())
        entry_price = float(row['Entry Price'].strip())
        exit_price = float(row['Exit Price'].strip())
        qty = int(row['Quantity'].strip())

        # Extract date from entry time
        entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        trade_date = entry_dt.strftime('%Y-%m-%d')

        net_pnl = pnl - fees

        # Determine exit type heuristic:
        # TP exit: P&L > 0 and exit was not at EOD (19:50 UTC = 15:50 ET = market close)
        # SL exit: P&L < 0 and exit early
        # EOD exit: exit at 19:50
        exit_hour_min = exit_dt.strftime('%H:%M')
        is_eod_exit = exit_hour_min == '19:50'

        # TP heuristic: positive P&L and not EOD exit, OR MFE reached high ratio
        is_tp_exit = (pnl > 0) and not is_eod_exit

        trades.append({
            'date': trade_date,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_dt': entry_dt,
            'exit_dt': exit_dt,
            'pnl': pnl,
            'fees': fees,
            'net_pnl': net_pnl,
            'mfe': mfe,
            'mae': mae,
            'is_win': is_win,
            'is_eod_exit': is_eod_exit,
            'is_tp_exit': is_tp_exit,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'qty': qty,
        })

# --- 按日分組 ---
by_date = defaultdict(list)
for t in trades:
    by_date[t['date']].append(t)

# Sort each day's trades by entry time
for d in by_date:
    by_date[d].sort(key=lambda x: x['entry_dt'])

# --- 讀取 qqq_1d.csv 計算 trend_strength ---
daily_path = "/sessions/clever-vibrant-hawking/mnt/mnq-backtest/qqq_1d.csv"
daily_data = {}
with open(daily_path, 'r', newline='') as f:
    content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    import io
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        try:
            dt_str = row['Datetime'].split('+')[0].split(' ')[0]
            close = float(row['Close'])
            high = float(row['High'])
            low = float(row['Low'])
            opn = float(row['Open'])
        except (TypeError, ValueError):
            continue  # skip malformed rows
        daily_data[dt_str] = {
            'close': close,
            'high': high,
            'low': low,
            'open': opn,
            'range': high - low,
            'body': abs(close - opn),
            'direction': 1 if close > opn else -1,
        }

# Compute rolling ATR-like trend strength (20-day avg range)
sorted_dates = sorted(daily_data.keys())
for i, d in enumerate(sorted_dates):
    if i >= 20:
        avg_range = sum(daily_data[sorted_dates[j]]['range'] for j in range(i-20, i)) / 20
        daily_data[d]['avg_range_20'] = avg_range
        daily_data[d]['trend_strength_ratio'] = daily_data[d]['range'] / avg_range if avg_range > 0 else 1.0
    else:
        daily_data[d]['avg_range_20'] = None
        daily_data[d]['trend_strength_ratio'] = None

# --- 分析同日多筆交易 ---
print("=" * 80)
print("v13-regime-reentry 離線驗證報告")
print("=" * 80)

single_entry_days = 0
multi_entry_days = 0
second_entries = []

for d, day_trades in sorted(by_date.items()):
    if len(day_trades) == 1:
        single_entry_days += 1
    elif len(day_trades) >= 2:
        multi_entry_days += 1
        first = day_trades[0]
        for subsequent in day_trades[1:]:
            dd = daily_data.get(d, {})
            tsr = dd.get('trend_strength_ratio')
            second_entries.append({
                'date': d,
                'first_tp': first['is_tp_exit'],
                'first_pnl': first['net_pnl'],
                'second_pnl': subsequent['net_pnl'],
                'second_is_win': subsequent['is_win'],
                'trend_strength_ratio': tsr,
                'first_exit_type': 'TP' if first['is_tp_exit'] else ('EOD' if first['is_eod_exit'] else 'SL/Trail'),
            })

print(f"\n總交易日數: {len(by_date)}")
print(f"單筆進場日: {single_entry_days}")
print(f"多筆進場日: {multi_entry_days}")
print(f"第二筆進場數: {len(second_entries)}")

# --- 分析第二筆進場表現 ---
print(f"\n{'='*80}")
print("第二筆進場表現分析")
print(f"{'='*80}")

total_2nd_pnl = sum(e['second_pnl'] for e in second_entries)
total_2nd_wins = sum(1 for e in second_entries if e['second_is_win'])
print(f"\n所有第二筆進場:")
print(f"  數量: {len(second_entries)}")
print(f"  淨利: ${total_2nd_pnl:.2f}")
print(f"  勝率: {total_2nd_wins/len(second_entries)*100:.1f}%" if second_entries else "  N/A")

# --- 模擬 v13 條件：第一筆 TP + trend_strength_ratio >= 1.15 ---
print(f"\n{'='*80}")
print("v13 條件模擬：第一筆 TP 出場 + trend_strength >= 1.15")
print(f"{'='*80}")

v13_pass = [e for e in second_entries if e['first_tp'] and e.get('trend_strength_ratio') and e['trend_strength_ratio'] >= 1.15]
v13_fail = [e for e in second_entries if not (e['first_tp'] and e.get('trend_strength_ratio') and e['trend_strength_ratio'] >= 1.15)]

v13_pass_pnl = sum(e['second_pnl'] for e in v13_pass)
v13_fail_pnl = sum(e['second_pnl'] for e in v13_fail)
v13_pass_wins = sum(1 for e in v13_pass if e['second_is_win'])
v13_fail_wins = sum(1 for e in v13_fail if e['second_is_win'])

print(f"\n通過 v13 條件的第二筆:")
print(f"  數量: {len(v13_pass)}")
if v13_pass:
    print(f"  淨利: ${v13_pass_pnl:.2f}")
    print(f"  勝率: {v13_pass_wins/len(v13_pass)*100:.1f}%")
    print(f"  平均每筆: ${v13_pass_pnl/len(v13_pass):.2f}")

print(f"\n被 v13 過濾掉的第二筆:")
print(f"  數量: {len(v13_fail)}")
if v13_fail:
    print(f"  淨利: ${v13_fail_pnl:.2f}")
    print(f"  勝率: {v13_fail_wins/len(v13_fail)*100:.1f}%")
    print(f"  平均每筆: ${v13_fail_pnl/len(v13_fail):.2f}")

# --- 分析各門檻的敏感度 ---
print(f"\n{'='*80}")
print("trend_strength_ratio 門檻敏感度分析")
print(f"{'='*80}")

for threshold in [1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]:
    passed = [e for e in second_entries if e['first_tp'] and e.get('trend_strength_ratio') and e['trend_strength_ratio'] >= threshold]
    if passed:
        pnl = sum(e['second_pnl'] for e in passed)
        wins = sum(1 for e in passed if e['second_is_win'])
        wr = wins / len(passed) * 100
        print(f"  threshold={threshold:.2f}: 通過={len(passed):3d}, 淨利=${pnl:8.2f}, 勝率={wr:.1f}%, 平均=${pnl/len(passed):7.2f}")
    else:
        print(f"  threshold={threshold:.2f}: 通過=  0")

# --- 只看 TP 條件（不考慮 trend）---
print(f"\n{'='*80}")
print("單獨條件分析")
print(f"{'='*80}")

tp_only = [e for e in second_entries if e['first_tp']]
non_tp = [e for e in second_entries if not e['first_tp']]

print(f"\n條件 1：第一筆 TP 出場")
print(f"  TP 出場後的第二筆: {len(tp_only)} 筆, 淨利=${sum(e['second_pnl'] for e in tp_only):.2f}")
if tp_only:
    tp_wins = sum(1 for e in tp_only if e['second_is_win'])
    print(f"  勝率: {tp_wins/len(tp_only)*100:.1f}%")
print(f"  非 TP 出場後的第二筆: {len(non_tp)} 筆, 淨利=${sum(e['second_pnl'] for e in non_tp):.2f}")
if non_tp:
    non_tp_wins = sum(1 for e in non_tp if e['second_is_win'])
    print(f"  勝率: {non_tp_wins/len(non_tp)*100:.1f}%")

# --- 按年分析 ---
print(f"\n{'='*80}")
print("按年分析第二筆進場 (全部 vs v13 過濾後)")
print(f"{'='*80}")

years = sorted(set(e['date'][:4] for e in second_entries))
print(f"\n{'年份':>6} | {'全部':>6} | {'淨利':>10} | {'v13通過':>8} | {'v13淨利':>10} | {'被濾掉淨利':>12}")
print("-" * 72)
for y in years:
    year_all = [e for e in second_entries if e['date'][:4] == y]
    year_pass = [e for e in v13_pass if e['date'][:4] == y]
    year_fail = [e for e in v13_fail if e['date'][:4] == y]
    all_pnl = sum(e['second_pnl'] for e in year_all)
    pass_pnl = sum(e['second_pnl'] for e in year_pass)
    fail_pnl = sum(e['second_pnl'] for e in year_fail)
    print(f"{y:>6} | {len(year_all):>6} | ${all_pnl:>8.2f} | {len(year_pass):>8} | ${pass_pnl:>8.2f} | ${fail_pnl:>10.2f}")

# --- 逐筆詳情 ---
print(f"\n{'='*80}")
print("同日多筆進場逐筆詳情")
print(f"{'='*80}")
print(f"\n{'日期':>12} | {'1st出場':>8} | {'1st淨利':>10} | {'2nd淨利':>10} | {'TSR':>6} | {'v13?':>5}")
print("-" * 72)
for e in second_entries:
    tsr_str = f"{e['trend_strength_ratio']:.2f}" if e['trend_strength_ratio'] else 'N/A'
    v13_ok = '✓' if e in v13_pass else '✗'
    print(f"{e['date']:>12} | {e['first_exit_type']:>8} | ${e['first_pnl']:>8.2f} | ${e['second_pnl']:>8.2f} | {tsr_str:>6} | {v13_ok:>5}")

# --- 總結 ---
print(f"\n{'='*80}")
print("總結")
print(f"{'='*80}")
print(f"\nv12 全部第二筆: {len(second_entries)} 筆, 淨利=${total_2nd_pnl:.2f}")
print(f"v13 保留第二筆: {len(v13_pass)} 筆, 淨利=${v13_pass_pnl:.2f}")
print(f"v13 過濾掉的:   {len(v13_fail)} 筆, 淨利=${v13_fail_pnl:.2f}")
delta = v13_pass_pnl - total_2nd_pnl + sum(e['second_pnl'] for e in v13_fail)  # This is 0 by definition
# The real question: v13 keeps v13_pass and removes v13_fail
# So v13 estimated net = v11 net + v13_pass_pnl
v11_net = 1047.18
v13_estimated = v11_net + v13_pass_pnl
print(f"\nv11 基準淨利: ${v11_net:.2f}")
print(f"v13 預估淨利: ${v13_estimated:.2f} (v11 + 保留的第二筆)")
print(f"差異: ${v13_estimated - v11_net:.2f}")

if v13_pass_pnl > 0:
    print(f"\n結論: v13 條件能有效過濾不良第二筆，預估可為 v11 增加 ${v13_pass_pnl:.2f}")
elif v13_pass_pnl < -20:
    print(f"\n結論: 即使通過 v13 條件的第二筆仍然虧損，建議考慮更嚴格的條件或放棄 re-entry")
else:
    print(f"\n結論: v13 條件篩選後第二筆接近損益兩平，re-entry 不具明顯優勢")
