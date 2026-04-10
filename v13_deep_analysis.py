#!/usr/bin/env python3
"""
v13 深層分析：用 v12 交易資料驗證 re-entry 條件的有效性。
重點：TP 條件是否真的能篩選出好的第二筆進場？
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
        is_eod = exit_dt.strftime('%H:%M') == '19:50'
        is_tp = (pnl > 0) and not is_eod

        trades.append({
            'date': entry_dt.strftime('%Y-%m-%d'),
            'year': entry_dt.strftime('%Y'),
            'entry_dt': entry_dt,
            'exit_dt': exit_dt,
            'pnl': pnl,
            'fees': fees,
            'net': pnl - fees,
            'mfe': mfe,
            'mae': mae,
            'is_win': int(row['IsWin'].strip()),
            'is_eod': is_eod,
            'is_tp': is_tp,
            'exit_type': 'TP' if is_tp else ('EOD' if is_eod else 'SL/Trail'),
            'entry_price': float(row['Entry Price'].strip()),
            'qty': int(row['Quantity'].strip()),
        })

# Group by date
by_date = defaultdict(list)
for t in trades:
    by_date[t['date']].append(t)
for d in by_date:
    by_date[d].sort(key=lambda x: x['entry_dt'])

print("=" * 80)
print("v13 Re-entry 深層有效性分析")
print("=" * 80)

# === 核心發現 1：第二筆的 TP 條件倒掛 ===
print("\n## 1. 第二筆進場的觸發條件分析")
print("-" * 60)

tp_2nd = []
non_tp_2nd = []
for d, day_trades in sorted(by_date.items()):
    if len(day_trades) < 2:
        continue
    first = day_trades[0]
    for second in day_trades[1:]:
        if first['is_tp']:
            tp_2nd.append((first, second))
        else:
            non_tp_2nd.append((first, second))

print(f"\n第一筆 TP 出場 → 第二筆:")
print(f"  筆數: {len(tp_2nd)}")
if tp_2nd:
    tp_net = sum(s['net'] for _, s in tp_2nd)
    tp_wins = sum(1 for _, s in tp_2nd if s['is_win'])
    print(f"  淨利: ${tp_net:.2f}")
    print(f"  勝率: {tp_wins/len(tp_2nd)*100:.1f}%")
    print(f"  平均: ${tp_net/len(tp_2nd):.2f}")
    for f, s in tp_2nd:
        print(f"    {s['date']} | 1st淨利=${f['net']:>8.2f} | 2nd淨利=${s['net']:>8.2f} | 2nd出場={s['exit_type']}")

print(f"\n第一筆 SL/Trail/EOD 出場 → 第二筆:")
print(f"  筆數: {len(non_tp_2nd)}")
if non_tp_2nd:
    ntp_net = sum(s['net'] for _, s in non_tp_2nd)
    ntp_wins = sum(1 for _, s in non_tp_2nd if s['is_win'])
    print(f"  淨利: ${ntp_net:.2f}")
    print(f"  勝率: {ntp_wins/len(non_tp_2nd)*100:.1f}%")
    print(f"  平均: ${ntp_net/len(non_tp_2nd):.2f}")
    for f, s in non_tp_2nd:
        print(f"    {s['date']} | 1st淨利=${f['net']:>8.2f} | 1st出場={f['exit_type']} | 2nd淨利=${s['net']:>8.2f} | 2nd出場={s['exit_type']}")

# === 核心發現 2：整體第二筆的價值 ===
print(f"\n\n## 2. 第二筆進場的整體價值")
print("-" * 60)

all_first = []
all_second = []
for d, day_trades in sorted(by_date.items()):
    all_first.append(day_trades[0])
    if len(day_trades) >= 2:
        for s in day_trades[1:]:
            all_second.append(s)

single_day_trades = [t for d, dt in by_date.items() for t in dt if len(dt) == 1]

print(f"\n單筆日交易 (v11 等效):")
print(f"  筆數: {len(single_day_trades)}")
single_net = sum(t['net'] for t in single_day_trades)
single_wins = sum(1 for t in single_day_trades if t['is_win'])
print(f"  淨利: ${single_net:.2f}")
print(f"  勝率: {single_wins/len(single_day_trades)*100:.1f}%")

print(f"\n多筆日-第一筆:")
multi_first = [day_trades[0] for d, day_trades in by_date.items() if len(day_trades) >= 2]
mf_net = sum(t['net'] for t in multi_first)
mf_wins = sum(1 for t in multi_first if t['is_win'])
print(f"  筆數: {len(multi_first)}")
print(f"  淨利: ${mf_net:.2f}")
print(f"  勝率: {mf_wins/len(multi_first)*100:.1f}%")

print(f"\n多筆日-第二筆 (v12 新增):")
second_net = sum(t['net'] for t in all_second)
second_wins = sum(1 for t in all_second if t['is_win'])
print(f"  筆數: {len(all_second)}")
print(f"  淨利: ${second_net:.2f}")
print(f"  勝率: {second_wins/len(all_second)*100:.1f}%")
print(f"  平均: ${second_net/len(all_second):.2f}")

# === 核心發現 3：MFE/MAE 效率分析 ===
print(f"\n\n## 3. MFE/MAE 效率對比")
print("-" * 60)

def mfe_mae_stats(trade_list, label):
    if not trade_list:
        return
    avg_mfe = sum(t['mfe'] for t in trade_list) / len(trade_list)
    avg_mae = sum(t['mae'] for t in trade_list) / len(trade_list)
    ratio = avg_mfe / avg_mae if avg_mae > 0 else float('inf')
    print(f"  {label}: 平均MFE=${avg_mfe:.2f}, 平均MAE=${avg_mae:.2f}, MFE/MAE={ratio:.2f}")

mfe_mae_stats(single_day_trades, "單筆日")
mfe_mae_stats(multi_first, "多筆日-第一筆")
mfe_mae_stats(all_second, "多筆日-第二筆")

# === 核心發現 4：按年第二筆貢獻 ===
print(f"\n\n## 4. 按年第二筆貢獻")
print("-" * 60)
print(f"\n{'年份':>6} | {'2nd筆數':>8} | {'2nd淨利':>10} | {'2nd勝率':>8} | {'邊際價值':>10}")
print("-" * 60)

years = sorted(set(t['year'] for t in all_second))
for y in years:
    yr_2nd = [t for t in all_second if t['year'] == y]
    yr_net = sum(t['net'] for t in yr_2nd)
    yr_wins = sum(1 for t in yr_2nd if t['is_win'])
    yr_wr = yr_wins / len(yr_2nd) * 100
    yr_avg = yr_net / len(yr_2nd)
    print(f"{y:>6} | {len(yr_2nd):>8} | ${yr_net:>8.2f} | {yr_wr:>6.1f}% | ${yr_avg:>8.2f}")

# === 核心發現 5：時間分佈 ===
print(f"\n\n## 5. 第二筆進場時間分佈")
print("-" * 60)
for t in all_second:
    entry_hour = t['entry_dt'].strftime('%H:%M')
    exit_hour = t['exit_dt'].strftime('%H:%M')
    duration_min = (t['exit_dt'] - t['entry_dt']).total_seconds() / 60
    print(f"  {t['date']} | 進場={entry_hour} | 出場={exit_hour} | 持有={duration_min:.0f}min | 淨利=${t['net']:>8.2f}")

# === 結論 ===
print(f"\n\n{'='*80}")
print("關鍵結論")
print("=" * 80)

print(f"""
1. v12 的第二筆進場只有 {len(all_second)} 筆，整體淨利 ${second_net:.2f}
   → 邊際貢獻極小，不值得為此增加策略複雜度

2. v13 的 TP 條件方向性問題：
   - TP 後的第二筆: {len(tp_2nd)} 筆, 淨利=${sum(s['net'] for _,s in tp_2nd):.2f}
   - 非 TP 後的第二筆: {len(non_tp_2nd)} 筆, 淨利=${sum(s['net'] for _,s in non_tp_2nd):.2f}
   → TP 條件反而過濾掉了「表現更好」的第二筆

3. 每日數據 (qqq_1d.csv) 只到 2019-10-02，無法為 2020+ 提供 trend_strength
   → 即使條件設計正確，離線驗證也不完整

4. 建議：
   a) 放棄 re-entry 優化路線 — 15 筆 / 8 年的樣本太小，無法得出穩健結論
   b) v11-single-entry 就是最佳選擇：簡單、穩定、可重現
   c) 研究方向應轉為 regime filter 的精細化（減少虧損年份），而非增加進場次數
   d) 更新 qqq_1d.csv 到最新，以支持未來的 regime 研究
""")
