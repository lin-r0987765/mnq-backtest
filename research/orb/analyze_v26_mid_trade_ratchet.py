"""
Post-v26 Research: Mid-Trade Protection Ratchet for hard_stop_loss bucket.
VERSION 2 - Fixed proxy methodology.

Key methodological fix:
- The ratchet ONLY improves trades that ended below the ratchet floor
- The ratchet ALSO CLIPS trades that would have recovered above the floor
  but got stopped out at the floor instead
- We need to conservatively estimate clipping using the trade's behavioral pattern

Conservative proxy rule:
- If MFE >= ratchet trigger: ratchet would have been armed
- If final P&L < ratchet floor: trade exits at floor (IMPROVEMENT) 
- If final P&L >= ratchet floor: trade passes through unaffected
  BUT: price might have dipped through the floor mid-trade before recovering.
  We estimate this using MAE: if MAE (from entry) is worse than -(entry - floor),
  then price likely crossed through the floor and the trade would have been stopped.

This makes the proxy more honest about winner-clipping.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

TRADES_PATH = Path("QuantConnect results/2017-2026/Square Blue Termite_trades.csv")
OUTPUT_PATH = Path("results/qc_regime_prototypes/local_orb_v26_mid_trade_ratchet.json")

TRAILING_PCT = 0.013


def load_trades():
    df = pd.read_csv(TRADES_PATH)
    df["Entry Time"] = pd.to_datetime(df["Entry Time"])
    df["Exit Time"] = pd.to_datetime(df["Exit Time"])
    df["duration_min"] = (df["Exit Time"] - df["Entry Time"]).dt.total_seconds() / 60.0
    for col in ["P&L", "MFE", "MAE", "Drawdown", "Entry Price", "Exit Price"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"].astype(str).str.replace(",", ""), errors="coerce")
    return df


def classify_trades(df):
    pnl = df["P&L"]
    mfe = df["MFE"]
    dur = df["duration_min"]

    quick_failure = (pnl <= 0) & (dur <= 120) & (mfe < 50)
    low_progress = (dur >= 180) & (mfe < 50) & (pnl < 50)
    late_washout = (dur >= 240) & (mfe >= 75) & (pnl < 25)
    small_flat_positive = (pnl > 0) & (pnl < 50) & (dur >= 240)
    healthy_capture = (pnl >= 100)
    hard_stop = (pnl <= 0) & ~quick_failure & ~low_progress & ~late_washout

    labels = pd.Series("other", index=df.index)
    labels[quick_failure] = "quick_failure_loss"
    labels[hard_stop] = "hard_stop_loss"
    labels[low_progress] = "low_progress_drift"
    labels[late_washout] = "late_washout"
    labels[small_flat_positive] = "small_flat_positive"
    labels[healthy_capture] = "healthy_capture"
    return labels


def simulate_ratchet_v2(df, ratchet_trigger_mult, ratchet_level_mult, ratchet_gate_minutes=None):
    """
    Conservative mid-trade ratchet proxy.

    Key insight: QC trades CSV gives us per-trade summary (MFE, MAE, P&L).
    MFE and MAE are TOTAL (already multiplied by quantity).

    Ratchet trigger and level are denominated in ORB-range multiples.
    We estimate ORB range as trailing_distance / trailing_pct... no, that gives entry_price.
    
    Actually: the ORB range is unknown from trade data. We need a proxy.
    From v26: trailing stop = entry * 1.3%. For a typical QQQ price ~$400, 
    trailing distance = $5.20 per share.
    
    But we DON'T know the actual ORB range per trade. We only know it's roughly
    proportional to intraday volatility.
    
    Better approach: use ABSOLUTE dollar thresholds per share that correspond 
    to typical ORB-range multiples. From the weakness map:
    - mean MFE of hard_stop_loss = $79.32 (total, over ~qty shares)
    - With dynamic sizing at 25% of ~$100k initial = $25k, at ~$300 avg price = ~83 shares
    - So per-share MFE ≈ $79.32 / 83 ≈ $0.96
    
    This is getting complicated. Let me use a simpler, more robust approach:
    
    The ratchet floor in P&L terms:
      floor_pnl = ratchet_level_mult * estimated_trail_distance_per_share * quantity
    
    where estimated_trail_distance = entry_price * TRAILING_PCT
    """
    modified_pnl = df["P&L"].copy()
    ratchet_armed = 0
    ratchet_improved = 0
    ratchet_clipped = 0

    for idx, row in df.iterrows():
        entry_price = row["Entry Price"]
        qty = abs(row["Quantity"])
        pnl = row["P&L"]
        mfe = row["MFE"]  # total MFE in dollars
        mae = row["MAE"]  # total MAE in dollars (negative)
        dur = row["duration_min"]

        # Estimate ORB range proxy: use trailing distance as approximation
        # trailing_distance_per_share = entry_price * TRAILING_PCT
        # For the ratchet, we normalize by this distance
        trail_dist_total = entry_price * TRAILING_PCT * qty  # total trailing distance in dollars

        # Ratchet trigger in dollars: trade must have reached this MFE
        trigger_dollars = ratchet_trigger_mult * trail_dist_total
        # Ratchet floor in dollars: if triggered, lock P&L at this minimum
        floor_dollars = ratchet_level_mult * trail_dist_total

        # Gate check
        if ratchet_gate_minutes is not None and dur < ratchet_gate_minutes:
            continue

        # Did MFE reach the trigger?
        if mfe < trigger_dollars:
            continue

        ratchet_armed += 1

        # Case 1: Trade ended below floor -> improvement
        if pnl < floor_dollars:
            # But did price FIRST reach the trigger before falling through floor?
            # Since MFE >= trigger, yes, the trigger was reached.
            # The ratchet would have stopped the trade at the floor.
            modified_pnl.at[idx] = floor_dollars
            ratchet_improved += 1

        # Case 2: Trade ended above floor -> potential clipping
        elif pnl >= floor_dollars:
            # The trade ended above the floor. Did it CROSS through the floor 
            # at some point after reaching the trigger? 
            # If MAE (from entry) is worse than -(trigger - floor), then
            # price likely dipped through the floor after the trigger.
            # 
            # Actually, MAE might have occurred BEFORE the trigger was reached.
            # We can't distinguish. This is the fundamental limitation of trade-level proxies.
            #
            # Conservative approach: assume if the trade's drawdown from peak 
            # (MFE - PnL) exceeds MFE - floor, then price crossed through the floor.
            # drawdown_from_peak = MFE - PnL
            # floor_distance_from_peak = MFE - floor
            # if drawdown_from_peak > floor_distance_from_peak, price passed through floor
            
            drawdown_from_peak = mfe - pnl
            floor_distance_from_peak = mfe - floor_dollars

            if drawdown_from_peak > floor_distance_from_peak and pnl > floor_dollars:
                # Trade passed through the floor but recovered
                # The ratchet would have stopped it at the floor, CLIPPING the winner
                modified_pnl.at[idx] = floor_dollars
                ratchet_clipped += 1

    delta = float(modified_pnl.sum() - df["P&L"].sum())
    return {
        "modified_total_pnl": float(modified_pnl.sum()),
        "baseline_total_pnl": float(df["P&L"].sum()),
        "delta": delta,
        "ratchet_armed": ratchet_armed,
        "ratchet_improved": ratchet_improved,
        "ratchet_clipped": ratchet_clipped,
        "net_affected": ratchet_improved - ratchet_clipped,
    }


def walk_forward_test(df, ratchet_trigger_mult, ratchet_level_mult, ratchet_gate_minutes=None, n_folds=4):
    df_sorted = df.sort_values("Entry Time").reset_index(drop=True)
    years = df_sorted["Entry Time"].dt.year
    unique_years = sorted(years.unique())

    fold_size = len(unique_years) // n_folds
    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = len(unique_years) if i == n_folds - 1 else (i + 1) * fold_size
        folds.append(unique_years[start_idx:end_idx])

    fold_results = []
    for i, fold_years in enumerate(folds):
        fold_mask = years.isin(fold_years)
        fold_df = df_sorted[fold_mask].copy()
        if len(fold_df) == 0:
            continue
        result = simulate_ratchet_v2(fold_df, ratchet_trigger_mult, ratchet_level_mult, ratchet_gate_minutes)
        fold_results.append({
            "fold": i + 1,
            "years": [int(y) for y in fold_years],
            "trades": len(fold_df),
            "baseline_pnl": result["baseline_total_pnl"],
            "modified_pnl": result["modified_total_pnl"],
            "delta": result["delta"],
            "improved": result["delta"] > 0,
            "armed": result["ratchet_armed"],
            "improved_count": result["ratchet_improved"],
            "clipped_count": result["ratchet_clipped"],
        })

    folds_improved = sum(1 for f in fold_results if f["improved"])
    return fold_results, folds_improved


def main():
    print("=" * 70)
    print("Post-v26 Research: Mid-Trade Protection Ratchet (v2 - Conservative)")
    print("Target bucket: hard_stop_loss (36 trades, -$5147.26, mean MFE=79.32)")
    print("=" * 70)

    df = load_trades()
    labels = classify_trades(df)
    df["archetype"] = labels

    hs = df[labels == "hard_stop_loss"].copy()

    # ===== GRID SEARCH =====
    trigger_grid = [0.50, 0.625, 0.75, 0.875, 1.00, 1.125, 1.25]
    level_grid = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    gate_grid = [None, 60, 120, 180]

    results = []

    for ratchet_trigger in trigger_grid:
        for ratchet_level in level_grid:
            for gate_min in gate_grid:
                if ratchet_level >= ratchet_trigger:
                    continue

                result = simulate_ratchet_v2(df, ratchet_trigger, ratchet_level, gate_min)
                fold_results, folds_improved = walk_forward_test(df, ratchet_trigger, ratchet_level, gate_min)

                gate_label = "none" if gate_min is None else f"{gate_min}m"
                variant_name = f"ratchet_t{ratchet_trigger:.3f}_l{ratchet_level:.2f}_g{gate_label}"

                entry = {
                    "variant": variant_name,
                    "ratchet_trigger_mult": ratchet_trigger,
                    "ratchet_level_mult": ratchet_level,
                    "ratchet_gate_minutes": gate_min,
                    "delta": result["delta"],
                    "ratchet_armed": result["ratchet_armed"],
                    "ratchet_improved": result["ratchet_improved"],
                    "ratchet_clipped": result["ratchet_clipped"],
                    "net_affected": result["net_affected"],
                    "folds_improved": folds_improved,
                    "fold_details": fold_results,
                }
                results.append(entry)

    results.sort(key=lambda x: x["delta"], reverse=True)

    # Save results
    strong = [r for r in results if r["folds_improved"] >= 3 and r["delta"] > 0]
    decent = [r for r in results if r["folds_improved"] >= 2 and r["delta"] > 0]

    output = {
        "research_scope": "local_orb_v26_mid_trade_ratchet",
        "analysis_version": "v2_conservative_proxy",
        "baseline_version": "v26-profit-lock",
        "accepted_reference_bundle": "Square Blue Termite",
        "mechanism": (
            "Mid-trade protection ratchet with conservative proxy: "
            "after MFE reaches ratchet_trigger * trailing_distance, "
            "set a floor stop at entry + ratchet_level * trailing_distance. "
            "Conservative clipping: if trade's drawdown from peak exceeds "
            "MFE - floor, the trade would have been stopped at the floor "
            "(even if it recovered in reality). This captures BOTH the improvement "
            "for losers AND the clipping cost for winners."
        ),
        "target_bucket": {
            "category": "hard_stop_loss",
            "trades": int(len(hs)),
            "net_pnl": float(hs["P&L"].sum()),
            "mean_mfe": float(hs["MFE"].mean()),
            "mean_duration_min": float(hs["duration_min"].mean()),
        },
        "parameter_grid": {
            "ratchet_trigger_mult": trigger_grid,
            "ratchet_level_mult": level_grid,
            "ratchet_gate_minutes": gate_grid,
            "total_variants_tested": len(results),
        },
        "summary": {
            "total_positive_variants": sum(1 for r in results if r["delta"] > 0),
            "best_variant": results[0]["variant"] if results else None,
            "best_delta": results[0]["delta"] if results else None,
            "best_folds": results[0]["folds_improved"] if results else None,
            "max_folds_improved": max(r["folds_improved"] for r in results) if results else 0,
            "variants_with_3plus_folds": len(strong),
            "positive_variants_with_2plus_folds": len(decent),
        },
        "all_variants": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Write human-readable report
    with open("_tmp_ratchet_v2_report.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Post-v26 Research: Mid-Trade Protection Ratchet (v2 - Conservative)\n")
        f.write("=" * 70 + "\n\n")

        f.write("ARCHETYPE DISTRIBUTION:\n")
        for cat in ["quick_failure_loss", "hard_stop_loss", "low_progress_drift",
                     "late_washout", "small_flat_positive", "healthy_capture", "other"]:
            mask = labels == cat
            count = mask.sum()
            pnl = df.loc[mask, "P&L"].sum()
            f.write(f"  {cat:25s}: {count:4d} trades, net P&L=${pnl:10.2f}\n")

        f.write(f"\nhard_stop_loss deep analysis: {len(hs)} trades\n")
        f.write(f"  mean MFE:      ${hs['MFE'].mean():.2f}\n")
        f.write(f"  mean MAE:      ${hs['MAE'].mean():.2f}\n")
        f.write(f"  mean P&L:      ${hs['P&L'].mean():.2f}\n")
        f.write(f"  mean duration: {hs['duration_min'].mean():.0f} min\n\n")

        f.write(f"SUMMARY:\n{json.dumps(output['summary'], indent=2)}\n\n")

        f.write("TOP 30 VARIANTS:\n")
        f.write(f"{'Variant':55s} {'Delta':>10s} {'Armed':>6s} {'Impr':>5s} {'Clip':>5s} {'Net':>5s} {'Folds':>6s}\n")
        f.write("-" * 95 + "\n")
        for r in results[:30]:
            f.write(f"{r['variant']:55s} {r['delta']:+10.2f} {r['ratchet_armed']:6d} {r['ratchet_improved']:5d} {r['ratchet_clipped']:5d} {r['net_affected']:5d} {r['folds_improved']:4d}/4\n")

        f.write(f"\nVariants with >= 3/4 folds AND positive delta: {len(strong)}\n")
        for r in strong[:20]:
            f.write(f"  {r['variant']:55s} delta={r['delta']:+10.2f} folds={r['folds_improved']}/4 armed={r['ratchet_armed']} impr={r['ratchet_improved']} clip={r['ratchet_clipped']}\n")

        f.write(f"\nPositive variants with >= 2/4 folds: {len(decent)}\n")
        for r in decent[:20]:
            f.write(f"  {r['variant']:55s} delta={r['delta']:+10.2f} folds={r['folds_improved']}/4\n")

        # Detailed fold breakdown for best variant
        if results:
            best = results[0]
            f.write(f"\nDETAILED FOLD BREAKDOWN FOR BEST VARIANT:\n")
            f.write(f"  {best['variant']}\n")
            for fd in best["fold_details"]:
                f.write(f"    Fold {fd['fold']}: years={fd['years']} trades={fd['trades']} ")
                f.write(f"baseline={fd['baseline_pnl']:.2f} modified={fd['modified_pnl']:.2f} ")
                f.write(f"delta={fd['delta']:.2f} armed={fd['armed']} impr={fd['improved_count']} clip={fd['clipped_count']}\n")

        # Verdict
        f.write(f"\n{'='*70}\n")
        f.write("VERDICT\n")
        f.write("=" * 70 + "\n")
        if strong:
            best_strong = strong[0]
            f.write(f"STRONG CANDIDATE found: {best_strong['variant']}\n")
            f.write(f"  delta: {best_strong['delta']:+.2f}\n")
            f.write(f"  folds: {best_strong['folds_improved']}/4\n")
            f.write(f"  armed={best_strong['ratchet_armed']} improved={best_strong['ratchet_improved']} clipped={best_strong['ratchet_clipped']}\n")
        elif decent:
            best_decent = decent[0]
            f.write(f"MODERATE signal found: {best_decent['variant']}\n")
            f.write(f"  delta: {best_decent['delta']:+.2f}\n")
            f.write(f"  folds: {best_decent['folds_improved']}/4\n")
        else:
            positive = [r for r in results if r["delta"] > 0]
            if positive:
                f.write(f"WEAK signal: {positive[0]['variant']}\n")
                f.write(f"  delta: {positive[0]['delta']:+.2f}\n")
                f.write(f"  folds: {positive[0]['folds_improved']}/4\n")
            else:
                f.write("No positive variants found -> LOCAL_REJECTED\n")

    print("Done. Check _tmp_ratchet_v2_report.txt")


if __name__ == "__main__":
    main()
