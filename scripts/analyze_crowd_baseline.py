"""
Analyze crowd baseline returns and compare to model strategies.

Answers: "What's the ROI if you just bet with market consensus?"
The standard backtest engine can't answer this because CrowdModel has edge=0,
so Kelly never fires. This script bypasses that by simulating flat-bet strategies.

Strategies tested:
  A. Top-1 Crowd Pick — $10 flat bet on highest-priced bucket
  B. Top-3 Crowd Picks — $10 flat bet on each of 3 highest-priced buckets
  C. Proportional — $10 spread across all buckets proportional to price (always wins)
  D. Top-1 + 5% Edge — What if you had 5% better info than the crowd?
  E-H. Actual models (TailBoost, Consensus, Directional, PriceDynamics) via backtest engine

Usage:
    python scripts/analyze_crowd_baseline.py
"""

import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BACKTEST_DIR = ROOT / "data" / "backtest" / "events"
INDEX_FILE = ROOT / "data" / "backtest" / "backtest_index.json"


def load_events():
    """Load all backtest events with metadata and entry prices."""
    with open(INDEX_FILE) as f:
        index = json.load(f)

    events = []
    for ev in index["events"]:
        slug = ev["event_slug"]
        ev_dir = BACKTEST_DIR / slug

        # Load metadata
        meta_path = ev_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        winning_bucket = meta.get("winning_bucket")
        if not winning_bucket:
            continue

        tier = meta.get("ground_truth_tier", "bronze")
        buckets = meta.get("buckets", [])
        if not buckets:
            continue

        # Get entry prices: try prices.parquet T-24h, fallback to metadata price_yes
        prices_path = ev_dir / "prices.parquet"
        entry_prices = {}

        if prices_path.exists():
            try:
                pdf = pd.read_parquet(prices_path)
                end_date = pd.Timestamp(meta["end_date"], tz="UTC")
                target_time = end_date - pd.Timedelta(hours=24)

                # Find closest snapshot to target_time
                pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)
                pdf["time_diff"] = (pdf["timestamp"] - target_time).abs()

                # Get closest timestamp within 12h window
                mask = pdf["time_diff"] <= pd.Timedelta(hours=12)
                if mask.any():
                    closest = pdf[mask].sort_values("time_diff").groupby("bucket_label").first()
                    entry_prices = closest["price"].to_dict()
            except Exception:
                pass

        # Fallback to metadata prices
        if not entry_prices:
            for b in buckets:
                p = b.get("price_yes", 0.0)
                if p is not None:
                    entry_prices[b["bucket_label"]] = float(p)

        # Skip if we only have resolution prices (all 0 or 1)
        non_extreme = [p for p in entry_prices.values() if 0.02 < p < 0.98]
        if len(non_extreme) < 2:
            continue

        events.append({
            "slug": slug,
            "tier": tier,
            "winning_bucket": winning_bucket,
            "entry_prices": entry_prices,
            "buckets": buckets,
            "n_buckets": len(buckets),
            "market_type": meta.get("market_type", "weekly"),
        })

    return events


def simulate_strategies(events):
    """Run all crowd-based strategies on the events."""
    results = {
        "top1_crowd": [],
        "top3_crowd": [],
        "proportional": [],
        "top1_edge5pct": [],
    }

    for ev in events:
        prices = ev["entry_prices"]
        winner = ev["winning_bucket"]
        tier = ev["tier"]

        # Sort buckets by price descending
        sorted_buckets = sorted(prices.items(), key=lambda x: x[1], reverse=True)

        # --- Strategy A: Top-1 Crowd Pick ---
        top_label, top_price = sorted_buckets[0]
        wager = 10.0
        if top_label == winner and top_price > 0:
            payout = wager / top_price  # binary market payout
            pnl = payout - wager
        else:
            pnl = -wager
        results["top1_crowd"].append({
            "slug": ev["slug"], "tier": tier, "wager": wager, "pnl": pnl,
            "won": top_label == winner, "bucket": top_label, "price": top_price,
        })

        # --- Strategy B: Top-3 Crowd Picks ---
        for label, price in sorted_buckets[:3]:
            wager = 10.0
            if label == winner and price > 0:
                payout = wager / price
                pnl = payout - wager
            else:
                pnl = -wager
            results["top3_crowd"].append({
                "slug": ev["slug"], "tier": tier, "wager": wager, "pnl": pnl,
                "won": label == winner, "bucket": label, "price": price,
            })

        # --- Strategy C: Proportional (spread $10 across all buckets) ---
        total_price = sum(prices.values())
        if total_price > 0:
            # Each bucket gets $10 * (price / total_price)
            total_wager = 0.0
            total_pnl_c = 0.0
            for label, price in prices.items():
                bucket_wager = 10.0 * (price / total_price)
                total_wager += bucket_wager
                if label == winner and price > 0:
                    payout = bucket_wager / price
                    total_pnl_c += (payout - bucket_wager)
                else:
                    total_pnl_c += -bucket_wager
            results["proportional"].append({
                "slug": ev["slug"], "tier": tier, "wager": 10.0, "pnl": total_pnl_c,
                "won": True,  # always wins something
            })

        # --- Strategy D: Top-1 with hypothetical 5% edge ---
        # Simulate: if your model said P = market_price + 0.05, use Kelly
        top_label, top_price = sorted_buckets[0]
        model_prob = min(top_price + 0.05, 0.95)
        edge = model_prob - top_price  # ~0.05
        kelly_f = edge / (1.0 - top_price)
        bankroll = 1000.0
        wager = bankroll * kelly_f * 0.25  # quarter Kelly
        wager = max(5.0, min(wager, 50.0))  # cap at $50
        if top_label == winner and top_price > 0:
            payout = wager / top_price
            pnl = payout - wager
        else:
            pnl = -wager
        results["top1_edge5pct"].append({
            "slug": ev["slug"], "tier": tier, "wager": wager, "pnl": pnl,
            "won": top_label == winner, "bucket": top_label, "price": top_price,
            "kelly_f": kelly_f,
        })

    return results


def compute_metrics(trades, label=""):
    """Compute performance metrics for a list of trades."""
    if not trades:
        return {"label": label, "n_bets": 0}

    n_bets = len(trades)
    n_wins = sum(1 for t in trades if t["won"])
    total_wagered = sum(t["wager"] for t in trades)
    total_pnl = sum(t["pnl"] for t in trades)
    roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0.0
    win_rate = (n_wins / n_bets * 100) if n_bets > 0 else 0.0

    # Per-event P&L for drawdown and Sharpe
    # Group by event slug
    event_pnls = defaultdict(float)
    for t in trades:
        event_pnls[t["slug"]] += t["pnl"]
    pnl_series = list(event_pnls.values())

    # Max drawdown (running cumulative P&L)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_series:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    # Sharpe-like: mean / std of per-event P&L
    mean_pnl = np.mean(pnl_series)
    std_pnl = np.std(pnl_series) if len(pnl_series) > 1 else 0.0
    sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0

    return {
        "label": label,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "total_wagered": total_wagered,
        "total_pnl": total_pnl,
        "roi": roi,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_pnl_per_event": mean_pnl,
    }


def print_table(rows, title=""):
    """Print a formatted comparison table."""
    if title:
        print(f"\n{'='*90}")
        print(f"  {title}")
        print(f"{'='*90}")

    header = f"{'Strategy':<30} {'Bets':>5} {'Wins':>5} {'WinR%':>6} {'Wagered':>9} {'P&L':>8} {'ROI%':>7} {'MaxDD':>7} {'Sharpe':>7}"
    print(header)
    print("-" * len(header))

    for r in rows:
        if r["n_bets"] == 0:
            print(f"{r['label']:<30} {'N/A':>5}")
            continue
        print(
            f"{r['label']:<30} {r['n_bets']:>5} {r['n_wins']:>5} "
            f"{r['win_rate']:>5.1f}% ${r['total_wagered']:>7.0f} "
            f"{'${:>+7.0f}'.format(r['total_pnl'])} {r['roi']:>+6.1f}% "
            f"${r['max_drawdown']:>5.0f} {r['sharpe']:>+6.2f}"
        )


def main():
    print("Loading backtest events...")
    events = load_events()
    print(f"Loaded {len(events)} events with valid entry prices")

    tier_counts = defaultdict(int)
    for ev in events:
        tier_counts[ev["tier"]] += 1
    for t, c in sorted(tier_counts.items()):
        print(f"  {t}: {c} events")

    print("\nSimulating crowd-based strategies...")
    results = simulate_strategies(events)

    # --- Overall results ---
    overall_rows = []
    for strategy_key, label in [
        ("top1_crowd", "A: Top-1 Crowd Pick"),
        ("top3_crowd", "B: Top-3 Crowd Picks"),
        ("proportional", "C: Proportional ($10)"),
        ("top1_edge5pct", "D: Top-1 + 5% Edge (Kelly)"),
    ]:
        overall_rows.append(compute_metrics(results[strategy_key], label))

    print_table(overall_rows, "OVERALL — ALL TIERS")

    # --- Per-tier results ---
    for tier in ["gold", "silver", "bronze"]:
        tier_rows = []
        for strategy_key, label in [
            ("top1_crowd", "A: Top-1 Crowd Pick"),
            ("top3_crowd", "B: Top-3 Crowd Picks"),
            ("proportional", "C: Proportional ($10)"),
            ("top1_edge5pct", "D: Top-1 + 5% Edge (Kelly)"),
        ]:
            tier_trades = [t for t in results[strategy_key] if t["tier"] == tier]
            tier_rows.append(compute_metrics(tier_trades, label))
        print_table(tier_rows, f"TIER: {tier.upper()}")

    # --- Detailed analysis of proportional strategy ---
    print(f"\n{'='*90}")
    print("  PROPORTIONAL STRATEGY DETAIL")
    print(f"{'='*90}")
    print("Proportional always 'wins' (gets payout from the winning bucket).")
    print("But the payout depends on the overround (sum of all prices).")
    prop_trades = results["proportional"]
    overrounds = []
    for ev in events:
        total_price = sum(ev["entry_prices"].values())
        overrounds.append(total_price)
    print(f"  Avg overround: {np.mean(overrounds):.3f} (1.0 = fair, >1.0 = house edge)")
    print(f"  Min overround: {np.min(overrounds):.3f}")
    print(f"  Max overround: {np.max(overrounds):.3f}")
    print(f"  Proportional ROI = (1/overround - 1) on average")
    print(f"  Expected ROI if overround=1.05: {(1/1.05 - 1)*100:.1f}%")
    print(f"  Expected ROI if overround=1.10: {(1/1.10 - 1)*100:.1f}%")

    # --- Win rate analysis for crowd top pick ---
    print(f"\n{'='*90}")
    print("  CROWD TOP-1 PICK ANALYSIS")
    print(f"{'='*90}")
    top1 = results["top1_crowd"]
    prices_when_won = [t["price"] for t in top1 if t["won"]]
    prices_when_lost = [t["price"] for t in top1 if not t["won"]]
    print(f"  Total events: {len(top1)}")
    print(f"  Wins: {len(prices_when_won)}, Losses: {len(prices_when_lost)}")
    if prices_when_won:
        print(f"  Avg price when WON:  {np.mean(prices_when_won):.3f} (payout: ${10/np.mean(prices_when_won):.1f})")
    if prices_when_lost:
        print(f"  Avg price when LOST: {np.mean(prices_when_lost):.3f}")
    print(f"  Avg top-bucket price: {np.mean([t['price'] for t in top1]):.3f}")

    # Break-even analysis
    avg_price = np.mean([t["price"] for t in top1])
    win_rate_actual = len(prices_when_won) / len(top1) if top1 else 0
    break_even_wr = avg_price  # need to win at least this fraction to break even
    print(f"\n  Break-even win rate at avg price {avg_price:.3f}: {break_even_wr*100:.1f}%")
    print(f"  Actual win rate: {win_rate_actual*100:.1f}%")
    if win_rate_actual > break_even_wr:
        print(f"  --> Crowd top pick BEATS break-even by {(win_rate_actual - break_even_wr)*100:.1f}pp")
    else:
        print(f"  --> Crowd top pick MISSES break-even by {(break_even_wr - win_rate_actual)*100:.1f}pp")

    # --- Per market type ---
    print(f"\n{'='*90}")
    print("  BY MARKET TYPE (Top-1 Crowd)")
    print(f"{'='*90}")
    for mtype in ["weekly", "daily", "short", "monthly"]:
        mtype_trades = [t for t in top1 if any(
            ev["market_type"] == mtype and ev["slug"] == t["slug"] for ev in events
        )]
        if mtype_trades:
            m = compute_metrics(mtype_trades, mtype)
            print(f"  {mtype:<10}: {m['n_bets']:>3} bets, {m['n_wins']:>3} wins, "
                  f"WR={m['win_rate']:.1f}%, ROI={m['roi']:+.1f}%, P&L=${m['total_pnl']:+.0f}")


if __name__ == "__main__":
    main()
