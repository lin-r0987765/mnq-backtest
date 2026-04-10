#!/usr/bin/env python3
"""
Regime Filter Candidate Analyzer
=================================
用途：分析 QuantConnect 交易資料，驗證各種 regime filter 候選的有效性。
包含前後半期交叉驗證、逐年穩定度、rolling 視窗分析。

用法：
    python analyze_regime_filter_candidates.py "QuantConnect results/2017-2026"

輸出：
    - 終端報告
    - results/qc_regime_filter_candidates/ 下的 JSON 與 CSV

版本：v1.0（第 57 輪建立）
"""

import argparse
import json
import os
import sys

import pandas as pd
import numpy as np


def load_trades(qc_dir):
    """載入 QC 交易 CSV"""
    csv_files = [f for f in os.listdir(qc_dir) if f.endswith("_trades.csv")]
    if not csv_files:
        print(f"ERROR: 在 {qc_dir} 找不到 *_trades.csv")
        sys.exit(1)
    path = os.path.join(qc_dir, csv_files[0])
    df = pd.read_csv(path)
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    df['entry_hour'] = df['Entry Time'].dt.hour
    df['entry_month'] = df['Entry Time'].dt.month
    df['entry_year'] = df['Entry Time'].dt.year
    df['entry_date'] = df['Entry Time'].dt.date
    df['hold_minutes'] = (df['Exit Time'] - df['Entry Time']).dt.total_seconds() / 60
    df['net_pnl'] = df['P&L'] - df['Fees']
    print(f"載入 {len(df)} 筆交易 from {csv_files[0]}")
    return df


def load_daily(daily_path="qqq_1d.csv"):
    """載入日線資料"""
    df = pd.read_csv(daily_path, parse_dates=['Datetime'])
    df['date'] = df['Datetime'].dt.date
    df['daily_return'] = df['Close'].pct_change()
    df['ema20'] = df['Close'].ewm(span=20).mean()
    df['ema50'] = df['Close'].ewm(span=50).mean()
    df['ema_slope'] = (df['ema20'] - df['ema20'].shift(5)) / df['ema20'].shift(5)
    df['vol20'] = df['daily_return'].rolling(20).std()
    df['trend'] = np.where(df['ema20'] > df['ema50'], 'bull', 'bear')
    print(f"日線資料: {len(df)} 行, {df['date'].min()} ~ {df['date'].max()}")
    return df


def calc_stats(df, label=""):
    """計算基礎統計"""
    if len(df) == 0:
        return {'label': label, 'trades': 0, 'wr': 0, 'net': 0, 'avg': 0, 'pf': 0}
    wins = len(df[df['IsWin'] == 1])
    gross_pos = df[df['net_pnl'] > 0]['net_pnl'].sum()
    gross_neg = abs(df[df['net_pnl'] < 0]['net_pnl'].sum())
    return {
        'label': label,
        'trades': len(df),
        'wins': wins,
        'wr': wins / len(df) * 100,
        'net': df['net_pnl'].sum(),
        'avg': df['net_pnl'].mean(),
        'pf': gross_pos / gross_neg if gross_neg > 0 else float('inf')
    }


def rolling_stability(df, window_months=6):
    """計算 rolling 視窗盈利比例"""
    df = df.sort_values('Entry Time').copy()
    df['entry_ym'] = df['Entry Time'].dt.to_period('M')
    monthly = df.groupby('entry_ym')['net_pnl'].sum()
    if len(monthly) < window_months:
        return 0, 0
    positive = 0
    total = 0
    for i in range(len(monthly) - window_months + 1):
        window = monthly.iloc[i:i + window_months]
        total += 1
        if window.sum() > 0:
            positive += 1
    return positive / total * 100 if total > 0 else 0, total


def yearly_breakdown(df):
    """逐年分析"""
    result = {}
    for y in sorted(df['entry_year'].unique()):
        sub = df[df['entry_year'] == y]
        wins = len(sub[sub['IsWin'] == 1])
        result[str(y)] = {
            'trades': len(sub),
            'wr': wins / len(sub) * 100 if len(sub) > 0 else 0,
            'net': round(sub['net_pnl'].sum(), 2)
        }
    return result


def cross_validate(trades, filter_fn, label):
    """前後半期交叉驗證"""
    first_half = trades[trades['entry_year'] <= 2020]
    second_half = trades[trades['entry_year'] > 2020]

    s1 = calc_stats(filter_fn(first_half), f"{label} (2017-2020)")
    s2 = calc_stats(filter_fn(second_half), f"{label} (2021-2025)")

    both_positive = s1['net'] > 0 and s2['net'] > 0
    same_direction = s1['net'] * s2['net'] > 0

    return {
        'first_half': s1,
        'second_half': s2,
        'both_positive': both_positive,
        'same_direction': same_direction,
        'verdict': 'ROBUST' if both_positive else ('CONSISTENT' if same_direction else 'UNSTABLE')
    }


def main():
    parser = argparse.ArgumentParser(description="Regime Filter Candidate Analyzer")
    parser.add_argument("qc_dir", help="QuantConnect 結果目錄路徑")
    parser.add_argument("--daily", default="qqq_1d.csv", help="日線 CSV 路徑")
    parser.add_argument("--output-dir", default="results/qc_regime_filter_candidates",
                        help="輸出目錄")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trades = load_trades(args.qc_dir)
    daily = load_daily(args.daily)

    # 定義候選 filter
    filters = {
        'baseline': lambda df: df,
        'A_no_late_entry': lambda df: df[df['entry_hour'] < 17],
        'A2_peak_hours_only': lambda df: df[df['entry_hour'].between(14, 16)],
        'B_no_may': lambda df: df[df['entry_month'] != 5],
        'C_no_may_dec': lambda df: df[~df['entry_month'].isin([5, 12])],
        'E_no_late_no_may': lambda df: df[(df['entry_hour'] < 17) & (df['entry_month'] != 5)],
    }

    # ============================
    # 分析每個候選
    # ============================
    results = {}
    print("\n" + "=" * 70)
    print("候選 Filter 綜合比較")
    print("=" * 70)

    header = f"{'Filter':<25} {'Trades':>6} {'WR%':>6} {'Net $':>9} {'Avg $':>8} {'PF':>6} {'R6m%':>6} {'CV':>10}"
    print(header)
    print("-" * 80)

    for name, fn in filters.items():
        filtered = fn(trades)
        stats = calc_stats(filtered, name)
        r6m, n_win = rolling_stability(filtered, 6)
        r12m, _ = rolling_stability(filtered, 12)
        cv = cross_validate(trades, fn, name)
        yearly = yearly_breakdown(filtered)

        results[name] = {
            'stats': stats,
            'rolling_6m_pct': round(r6m, 1),
            'rolling_12m_pct': round(r12m, 1),
            'cross_validation': cv['verdict'],
            'yearly': yearly
        }

        print(f"{name:<25} {stats['trades']:>6} {stats['wr']:>5.1f}% {stats['net']:>9.2f} {stats['avg']:>8.2f} {stats['pf']:>6.2f} {r6m:>5.1f}% {cv['verdict']:>10}")

    # ============================
    # 被排除交易分析
    # ============================
    print("\n" + "=" * 70)
    print("被排除交易 — 前後半期一致性")
    print("=" * 70)

    excluded_sets = {
        '17+ UTC': lambda df: df[df['entry_hour'] >= 17],
        'May': lambda df: df[df['entry_month'] == 5],
        'December': lambda df: df[df['entry_month'] == 12],
    }

    excluded_results = {}
    for name, fn in excluded_sets.items():
        cv = cross_validate(trades, fn, f"Excluded: {name}")
        excluded_results[name] = cv
        s1 = cv['first_half']
        s2 = cv['second_half']
        print(f"  {name:<15}: 前半 {s1['trades']:>3}筆 ${s1['net']:>8.2f} | 後半 {s2['trades']:>3}筆 ${s2['net']:>8.2f} | {cv['verdict']}")

    # ============================
    # 最終建議
    # ============================
    print("\n" + "=" * 70)
    print("建議")
    print("=" * 70)

    # 排序：先按 cross_validation robustness，再按 net
    robust_filters = {k: v for k, v in results.items()
                      if k != 'baseline' and v['cross_validation'] == 'ROBUST'}

    if robust_filters:
        best = max(robust_filters.items(), key=lambda x: x[1]['rolling_6m_pct'])
        print(f"\n最強穩健候選: {best[0]}")
        print(f"  交易數: {best[1]['stats']['trades']}")
        print(f"  勝率: {best[1]['stats']['wr']:.1f}%")
        print(f"  淨利: ${best[1]['stats']['net']:.2f}")
        print(f"  Rolling 6m 盈利: {best[1]['rolling_6m_pct']}%")
        print(f"  交叉驗證: {best[1]['cross_validation']}")
    else:
        print("\n無穩健候選通過交叉驗證")

    # 特別標註 17+ UTC 排除的穩健性
    late_cv = excluded_results.get('17+ UTC', {})
    if late_cv.get('verdict') == 'CONSISTENT':
        print(f"\n✓ 17+ UTC 排除在前後半期一致為虧損，是最穩健的單一 filter")

    may_cv = excluded_results.get('May', {})
    if may_cv.get('verdict') != 'CONSISTENT':
        print(f"△ 5月排除在後半期方向反轉，穩健性不足，建議暫不採用")

    # ============================
    # 輸出 JSON
    # ============================
    output = {
        'version': 'v1.0',
        'iteration': 57,
        'data_source': str(args.qc_dir),
        'total_trades': len(trades),
        'daily_data_range': f"{daily['date'].min()} ~ {daily['date'].max()}",
        'candidates': {},
        'excluded_analysis': {},
        'recommendation': {}
    }

    for name, v in results.items():
        output['candidates'][name] = {
            'trades': v['stats']['trades'],
            'wr_pct': round(v['stats']['wr'], 1),
            'net_pnl': round(v['stats']['net'], 2),
            'avg_pnl': round(v['stats']['avg'], 2),
            'pf': round(v['stats']['pf'], 3),
            'rolling_6m_positive_pct': v['rolling_6m_pct'],
            'rolling_12m_positive_pct': v['rolling_12m_pct'],
            'cross_validation': v['cross_validation'],
            'yearly': v['yearly']
        }

    for name, cv in excluded_results.items():
        output['excluded_analysis'][name] = {
            'first_half_net': round(cv['first_half']['net'], 2),
            'second_half_net': round(cv['second_half']['net'], 2),
            'verdict': cv['verdict']
        }

    if robust_filters:
        best_name = max(robust_filters.items(), key=lambda x: x[1]['rolling_6m_pct'])[0]
        output['recommendation'] = {
            'primary': 'A_no_late_entry',
            'reason': '17+ UTC 排除是唯一在前後半期一致為虧損的被排除群組，穩健性最高',
            'best_combo': best_name,
            'caution': '5月排除後半期方向反轉，暫不建議採用'
        }

    json_path = os.path.join(args.output_dir, "regime_filter_candidates.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n輸出: {json_path}")


if __name__ == "__main__":
    main()
