"""分析最新回測交易記錄"""
import json, glob, os

# 找到最新結果
results_dir = "results"
orb_files = sorted(glob.glob(os.path.join(results_dir, "*ORB.json")))
vwap_files = sorted(glob.glob(os.path.join(results_dir, "*VWAP_Reversion.json")))

print("=" * 60)
print("最新回測分析")
print("=" * 60)

for label, flist in [("ORB", orb_files), ("VWAP_Reversion", vwap_files)]:
    if not flist:
        continue
    f = flist[-1]
    d = json.load(open(f))
    m = d["metrics"]
    print(f"\n--- {label} ({os.path.basename(f)}) ---")
    for k, v in m.items():
        print(f"  {k}: {v}")

    trades = d.get("trades", [])
    if trades:
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        print(f"  手動統計 - wins: {len(wins)}, losses: {len(losses)}")
        if wins:
            avg_w = sum(t["pnl"] for t in wins) / len(wins)
            print(f"  avg_win_pnl: {avg_w:.2f}")
        if losses:
            avg_l = sum(t["pnl"] for t in losses) / len(losses)
            print(f"  avg_loss_pnl: {avg_l:.2f}")
        # 分析方向
        long_trades = [t for t in trades if t.get("side") == "long"]
        short_trades = [t for t in trades if t.get("side") == "short"]
        long_wins = [t for t in long_trades if t.get("pnl", 0) > 0]
        short_wins = [t for t in short_trades if t.get("pnl", 0) > 0]
        if long_trades:
            print(f"  多頭: {len(long_trades)} 筆, 勝率 {len(long_wins)/len(long_trades)*100:.1f}%")
        if short_trades:
            print(f"  空頭: {len(short_trades)} 筆, 勝率 {len(short_wins)/len(short_trades)*100:.1f}%")

# 也看優化版
orb_opt_files = sorted(glob.glob(os.path.join(results_dir, "*ORB_Optimised.json")))
vwap_opt_files = sorted(glob.glob(os.path.join(results_dir, "*VWAP_Optimised.json")))

for label, flist in [("ORB_Optimised", orb_opt_files), ("VWAP_Optimised", vwap_opt_files)]:
    if not flist:
        continue
    f = flist[-1]
    d = json.load(open(f))
    m = d["metrics"]
    print(f"\n--- {label} ({os.path.basename(f)}) ---")
    for k, v in m.items():
        print(f"  {k}: {v}")
    trades = d.get("trades", [])
    if trades:
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        print(f"  手動統計 - wins: {len(wins)}, losses: {len(losses)}")
        long_trades = [t for t in trades if t.get("side") == "long"]
        short_trades = [t for t in trades if t.get("side") == "short"]
        long_wins = [t for t in long_trades if t.get("pnl", 0) > 0]
        short_wins = [t for t in short_trades if t.get("pnl", 0) > 0]
        if long_trades:
            print(f"  多頭: {len(long_trades)} 筆, 勝率 {len(long_wins)/len(long_trades)*100:.1f}%")
        if short_trades:
            print(f"  空頭: {len(short_trades)} 筆, 勝率 {len(short_wins)/len(short_trades)*100:.1f}%")
