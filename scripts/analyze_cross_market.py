"""
Cross-Market Consistency Arbitrage Analysis: Daily vs Weekly Markets.

Hypothesis: Daily and weekly markets cover overlapping time periods.
If a weekly market for "Feb 10-16" overlaps with daily markets for each
day within that week, pricing inconsistencies between them can be exploited.

Steps:
1. Load market catalog, identify daily and weekly events with date ranges
2. Map daily events that fall within weekly event date ranges
3. Compute implied daily distributions from weekly prices
4. Measure divergence between implied and actual daily prices
5. Test whether the cheaper side wins more often

GO/NO-GO gate: Need >= 10 daily-weekly overlapping pairs.

Usage:
    python scripts/analyze_cross_market.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"
PRICE_HISTORY_PATH = PROJECT_DIR / "data" / "sources" / "polymarket" / "prices" / "price_history.parquet"


# ---------------------------------------------------------------------------
# Step 1: Load and categorize events
# ---------------------------------------------------------------------------
def load_events():
    """Load market catalog and group rows into events."""
    df = pd.read_parquet(CATALOG_PATH)
    print(f"Loaded market catalog: {len(df)} rows, {df['event_id'].nunique()} events")
    print(f"\nMarket type distribution:")
    type_counts = df.groupby("market_type")["event_id"].nunique()
    for mt, cnt in type_counts.items():
        print(f"  {mt}: {cnt} events")

    # Build event-level records
    events = []
    for eid, group in df.groupby("event_id"):
        row = group.iloc[0]
        start = row["start_date"]
        end = row["end_date"]
        if pd.isna(start) or pd.isna(end):
            continue

        # Normalize to date objects for comparison
        if hasattr(start, "date"):
            start_date = start.date() if not isinstance(start, type(None)) else None
            end_date = end.date() if not isinstance(end, type(None)) else None
        else:
            start_date = pd.Timestamp(start).date()
            end_date = pd.Timestamp(end).date()

        events.append({
            "event_id": eid,
            "event_slug": row["event_slug"],
            "event_title": row["event_title"],
            "market_type": row["market_type"],
            "duration_days": int(row["duration_days"]) if pd.notna(row["duration_days"]) else None,
            "start_date": start_date,
            "end_date": end_date,
            "n_buckets": int(row["n_buckets"]),
            "is_resolved": bool(row["is_resolved"]),
            "buckets": group[["bucket_label", "lower_bound", "upper_bound",
                              "price_yes", "is_winner", "token_id_yes"]].to_dict("records"),
        })

    return events, df


# ---------------------------------------------------------------------------
# Step 2: Find daily-weekly overlapping pairs
# ---------------------------------------------------------------------------
def find_overlapping_pairs(events):
    """For each daily event, find weekly events whose date range contains it."""
    daily_events = [e for e in events if e["market_type"] == "daily"]
    weekly_events = [e for e in events if e["market_type"] == "weekly"]

    print(f"\nDaily events with valid dates: {len(daily_events)}")
    print(f"Weekly events with valid dates: {len(weekly_events)}")

    pairs = []
    for daily in daily_events:
        d_start = daily["start_date"]
        d_end = daily["end_date"]

        for weekly in weekly_events:
            w_start = weekly["start_date"]
            w_end = weekly["end_date"]

            # Check if daily event falls within weekly event range
            if d_start >= w_start and d_end <= w_end:
                pairs.append({
                    "daily_event": daily,
                    "weekly_event": weekly,
                    "daily_slug": daily["event_slug"],
                    "weekly_slug": weekly["event_slug"],
                    "daily_date": d_start,
                    "weekly_start": w_start,
                    "weekly_end": w_end,
                    "weekly_duration": weekly["duration_days"],
                })

    print(f"\nOverlapping daily-weekly pairs found: {len(pairs)}")

    if pairs:
        # Show some examples
        print(f"\nSample pairs (first 5):")
        for p in pairs[:5]:
            print(f"  Daily: {p['daily_slug'][:60]}")
            print(f"    -> Weekly: {p['weekly_slug'][:60]}")
            print(f"    Daily date: {p['daily_date']}, Weekly: {p['weekly_start']} to {p['weekly_end']}")
            print()

    return pairs


# ---------------------------------------------------------------------------
# Step 3: Load price histories for overlapping events
# ---------------------------------------------------------------------------
def load_backtest_prices(event_slug):
    """Load price history from backtest dataset for an event."""
    prices_path = BACKTEST_DIR / "events" / event_slug / "prices.parquet"
    if prices_path.exists():
        return pd.read_parquet(prices_path)
    return None


def load_backtest_metadata(event_slug):
    """Load metadata from backtest dataset."""
    meta_path = BACKTEST_DIR / "events" / event_slug / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Step 4: Compute implied daily distribution from weekly prices
# ---------------------------------------------------------------------------
def compute_implied_daily_from_weekly(weekly_buckets, weekly_duration):
    """Convert weekly bucket boundaries to implied daily ranges.

    If a weekly bucket is "140-159" for a 7-day event, it implies
    a daily rate of ~20-23 tweets. We create an implied daily
    distribution by dividing bucket boundaries by duration.

    Returns list of dicts with implied daily bucket boundaries and prices.
    """
    if not weekly_duration or weekly_duration <= 0:
        return None

    implied = []
    for b in weekly_buckets:
        lower = b.get("lower_bound", 0)
        upper = b.get("upper_bound", 99999)
        price = b.get("price_yes", 0.0)

        if pd.isna(lower) or pd.isna(upper):
            continue

        lower = int(lower)
        upper = int(upper)

        implied_lower = lower / weekly_duration
        implied_upper = upper / weekly_duration if upper < 99999 else 99999

        implied.append({
            "weekly_label": b.get("bucket_label", ""),
            "weekly_lower": lower,
            "weekly_upper": upper,
            "weekly_price": float(price) if pd.notna(price) else 0.0,
            "implied_daily_lower": implied_lower,
            "implied_daily_upper": implied_upper,
        })

    return implied


def compute_divergence(daily_buckets, implied_daily, daily_prices_at_time=None):
    """Compute divergence between actual daily bucket prices and weekly-implied daily dist.

    For each daily bucket, find the implied probability from the weekly market
    by summing weekly bucket probabilities that overlap with the daily bucket range.

    Returns divergence metrics.
    """
    if not daily_buckets or not implied_daily:
        return None

    # Build daily bucket price map
    daily_price_map = {}
    for b in daily_buckets:
        label = b.get("bucket_label", "")
        price = float(b.get("price_yes", 0.0)) if pd.notna(b.get("price_yes")) else 0.0
        lower = int(b.get("lower_bound", 0)) if pd.notna(b.get("lower_bound")) else 0
        upper = int(b.get("upper_bound", 99999)) if pd.notna(b.get("upper_bound")) else 99999
        daily_price_map[label] = {
            "price": price,
            "lower": lower,
            "upper": upper,
        }

    # Use provided time-specific prices if available
    if daily_prices_at_time:
        for label, price in daily_prices_at_time.items():
            if label in daily_price_map:
                daily_price_map[label]["price"] = price

    # For each daily bucket, compute implied probability from weekly
    # by finding which weekly implied ranges overlap
    divergences = []
    for label, daily_info in daily_price_map.items():
        d_lower = daily_info["lower"]
        d_upper = daily_info["upper"]
        d_price = daily_info["price"]

        # Sum weekly-implied probability for daily ranges that overlap
        implied_prob = 0.0
        for imp in implied_daily:
            imp_lower = imp["implied_daily_lower"]
            imp_upper = imp["implied_daily_upper"]
            w_price = imp["weekly_price"]

            # Check overlap between daily bucket [d_lower, d_upper] and
            # implied daily range [imp_lower, imp_upper]
            if imp_upper == 99999:
                imp_upper_eff = 10000  # large number
            else:
                imp_upper_eff = imp_upper

            if d_upper == 99999:
                d_upper_eff = 10000
            else:
                d_upper_eff = d_upper

            overlap_lower = max(d_lower, imp_lower)
            overlap_upper = min(d_upper_eff, imp_upper_eff)

            if overlap_upper > overlap_lower:
                # Fraction of the weekly implied range that overlaps with this daily bucket
                imp_range = imp_upper_eff - imp_lower
                if imp_range > 0:
                    overlap_frac = (overlap_upper - overlap_lower) / imp_range
                    implied_prob += w_price * overlap_frac

        divergence = d_price - implied_prob
        divergences.append({
            "bucket_label": label,
            "daily_price": d_price,
            "implied_from_weekly": implied_prob,
            "divergence": divergence,
            "abs_divergence": abs(divergence),
        })

    return divergences


# ---------------------------------------------------------------------------
# Step 5: Analyze divergences and test signal
# ---------------------------------------------------------------------------
def analyze_divergences(pairs):
    """For each pair, compute divergences and test predictive power."""
    all_divergences = []
    pairs_with_data = 0
    pairs_with_prices = 0

    for pair in pairs:
        daily_meta = load_backtest_metadata(pair["daily_slug"])
        weekly_meta = load_backtest_metadata(pair["weekly_slug"])

        if not daily_meta or not weekly_meta:
            continue
        pairs_with_data += 1

        weekly_duration = weekly_meta.get("duration_days", 7)
        weekly_buckets = weekly_meta.get("buckets", [])
        daily_buckets = daily_meta.get("buckets", [])

        # Load price histories to get T-24h prices (not just final)
        daily_prices_df = load_backtest_prices(pair["daily_slug"])
        weekly_prices_df = load_backtest_prices(pair["weekly_slug"])

        # Compute implied daily from weekly
        implied = compute_implied_daily_from_weekly(weekly_buckets, weekly_duration)
        if not implied:
            continue

        # Try to use price snapshots at similar times
        # Use final catalog prices as fallback
        weekly_prices_for_implied = None
        if weekly_prices_df is not None and len(weekly_prices_df) > 0:
            pairs_with_prices += 1
            # Get the latest available weekly prices before the daily event starts
            # This simulates what we'd see in production
            try:
                daily_start = pd.Timestamp(pair["daily_date"])
                if daily_start.tzinfo is None:
                    daily_start = daily_start.tz_localize("UTC")

                # Filter weekly prices to before daily event start
                if "timestamp" in weekly_prices_df.columns:
                    ts_col = "timestamp"
                elif "ts" in weekly_prices_df.columns:
                    ts_col = "ts"
                else:
                    ts_col = weekly_prices_df.columns[0]

                weekly_prices_df[ts_col] = pd.to_datetime(weekly_prices_df[ts_col], utc=True)
                pre_daily = weekly_prices_df[weekly_prices_df[ts_col] < daily_start]

                if len(pre_daily) > 0:
                    # Get last price snapshot for each bucket
                    latest = pre_daily.sort_values(ts_col).groupby("bucket_label").last()
                    weekly_prices_for_implied = latest["price"].to_dict() if "price" in latest.columns else None
            except Exception as e:
                pass  # Fall back to catalog prices

        # If we have time-specific weekly prices, update the implied calc
        if weekly_prices_for_implied:
            for imp in implied:
                w_label = imp["weekly_label"]
                if w_label in weekly_prices_for_implied:
                    imp["weekly_price"] = weekly_prices_for_implied[w_label]

        # Similarly get daily prices at T-24h
        daily_prices_at_time = None
        if daily_prices_df is not None and len(daily_prices_df) > 0:
            try:
                daily_start = pd.Timestamp(pair["daily_date"])
                if daily_start.tzinfo is None:
                    daily_start = daily_start.tz_localize("UTC")

                if "timestamp" in daily_prices_df.columns:
                    ts_col = "timestamp"
                elif "ts" in daily_prices_df.columns:
                    ts_col = "ts"
                else:
                    ts_col = daily_prices_df.columns[0]

                daily_prices_df[ts_col] = pd.to_datetime(daily_prices_df[ts_col], utc=True)
                pre_start = daily_prices_df[daily_prices_df[ts_col] < daily_start]

                if len(pre_start) > 0:
                    latest = pre_start.sort_values(ts_col).groupby("bucket_label").last()
                    daily_prices_at_time = latest["price"].to_dict() if "price" in latest.columns else None
            except Exception:
                pass

        # Compute divergences
        divs = compute_divergence(daily_buckets, implied, daily_prices_at_time)
        if not divs:
            continue

        # Determine which daily bucket actually won
        daily_winner = daily_meta.get("winning_bucket", None)
        weekly_winner = weekly_meta.get("winning_bucket", None)

        for d in divs:
            d["daily_slug"] = pair["daily_slug"]
            d["weekly_slug"] = pair["weekly_slug"]
            d["daily_date"] = str(pair["daily_date"])
            d["daily_winner"] = daily_winner
            d["weekly_winner"] = weekly_winner
            d["is_winner"] = (d["bucket_label"] == daily_winner)
            d["has_price_history"] = (daily_prices_df is not None and weekly_prices_for_implied is not None)

        all_divergences.extend(divs)

    return all_divergences, pairs_with_data, pairs_with_prices


def test_signal(all_divergences):
    """Test whether the cheaper side (daily vs weekly implied) wins more often."""
    if not all_divergences:
        return None

    df = pd.DataFrame(all_divergences)

    print(f"\n{'='*70}")
    print("DIVERGENCE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nTotal bucket-level observations: {len(df)}")
    print(f"Unique daily events: {df['daily_slug'].nunique()}")
    print(f"Observations with price history: {df['has_price_history'].sum()}")

    # Overall divergence stats
    print(f"\nDivergence statistics (daily_price - weekly_implied):")
    print(f"  Mean divergence:    {df['divergence'].mean():.4f}")
    print(f"  Median divergence:  {df['divergence'].median():.4f}")
    print(f"  Std divergence:     {df['divergence'].std():.4f}")
    print(f"  Mean |divergence|:  {df['abs_divergence'].mean():.4f}")
    print(f"  Max |divergence|:   {df['abs_divergence'].max():.4f}")

    # How often are divergences > 5%?
    large_div = df[df["abs_divergence"] > 0.05]
    print(f"\n  Buckets with |divergence| > 5%:  {len(large_div)} / {len(df)} ({100*len(large_div)/len(df):.1f}%)")
    large_div_10 = df[df["abs_divergence"] > 0.10]
    print(f"  Buckets with |divergence| > 10%: {len(large_div_10)} / {len(df)} ({100*len(large_div_10)/len(df):.1f}%)")

    # Per-event divergence
    event_divs = df.groupby("daily_slug").agg(
        mean_abs_div=("abs_divergence", "mean"),
        max_abs_div=("abs_divergence", "max"),
        n_large=("abs_divergence", lambda x: (x > 0.05).sum()),
    ).reset_index()
    print(f"\nPer-event divergence:")
    print(f"  Mean of event-level mean |div|:  {event_divs['mean_abs_div'].mean():.4f}")
    print(f"  Mean of event-level max |div|:   {event_divs['max_abs_div'].mean():.4f}")
    print(f"  Events with any bucket > 5% div: {(event_divs['n_large'] > 0).sum()} / {len(event_divs)}")

    # Signal test: when daily price < weekly implied (positive divergence from weekly),
    # does the daily bucket win more often?
    # Conversely, when daily price > weekly implied, does it lose?
    print(f"\n{'='*70}")
    print("SIGNAL TEST: Does the cheaper market win?")
    print(f"{'='*70}")

    # Focus on buckets with meaningful divergence
    sig_df = df[df["abs_divergence"] > 0.05].copy()
    if len(sig_df) == 0:
        print("\n  NO buckets with > 5% divergence. Cannot test signal.")
        return df

    # When daily_price < weekly_implied (divergence < -0.05): daily is cheaper
    # Hypothesis: cheaper side is underpriced = wins more than price implies
    daily_cheap = sig_df[sig_df["divergence"] < -0.05]
    daily_expensive = sig_df[sig_df["divergence"] > 0.05]

    print(f"\n  Buckets where daily is CHEAPER than weekly-implied ({len(daily_cheap)} obs):")
    if len(daily_cheap) > 0:
        win_rate = daily_cheap["is_winner"].mean()
        avg_price = daily_cheap["daily_price"].mean()
        print(f"    Win rate: {win_rate:.3f} (avg daily price: {avg_price:.3f})")
        print(f"    If win_rate > avg_price, there's edge in buying daily")
        if avg_price > 0:
            print(f"    Edge: {win_rate - avg_price:.3f} ({100*(win_rate-avg_price)/avg_price:.1f}% above price)")
    else:
        print(f"    No observations")

    print(f"\n  Buckets where daily is MORE EXPENSIVE than weekly-implied ({len(daily_expensive)} obs):")
    if len(daily_expensive) > 0:
        win_rate = daily_expensive["is_winner"].mean()
        avg_price = daily_expensive["daily_price"].mean()
        print(f"    Win rate: {win_rate:.3f} (avg daily price: {avg_price:.3f})")
        print(f"    If win_rate < avg_price, there's edge in SELLING daily (buying No)")
        if avg_price > 0:
            print(f"    Edge: {avg_price - win_rate:.3f} ({100*(avg_price-win_rate)/avg_price:.1f}% below price)")
    else:
        print(f"    No observations")

    # Combined signal: trade the divergence
    print(f"\n  Combined: trade divergence (buy cheap, sell expensive):")
    n_trades = len(daily_cheap) + len(daily_expensive)
    if n_trades > 0:
        # For "buy cheap" trades: profit = (1-price)*is_winner - price*(1-is_winner)
        # Simplified: profit = is_winner - price
        profits_cheap = (daily_cheap["is_winner"].astype(float) - daily_cheap["daily_price"]) if len(daily_cheap) > 0 else pd.Series(dtype=float)
        # For "sell expensive" (buy No): profit = (1-is_winner) - (1-price) = price - is_winner
        profits_expensive = (daily_expensive["daily_price"] - daily_expensive["is_winner"].astype(float)) if len(daily_expensive) > 0 else pd.Series(dtype=float)

        all_profits = pd.concat([profits_cheap, profits_expensive])
        print(f"    Total trades: {n_trades}")
        print(f"    Mean profit per trade: {all_profits.mean():.4f}")
        print(f"    Total profit (unit bets): {all_profits.sum():.2f}")
        print(f"    Win rate: {(all_profits > 0).mean():.3f}")

    return df


# ---------------------------------------------------------------------------
# Step 6: Verdict
# ---------------------------------------------------------------------------
def render_verdict(pairs, all_divergences, div_df):
    """Print final PASS/FAIL verdict."""
    print(f"\n{'='*70}")
    print("VERDICT: Cross-Market Consistency Arbitrage")
    print(f"{'='*70}")

    n_pairs = len(pairs)
    reasons = []

    # Check 1: Enough pairs?
    if n_pairs < 10:
        reasons.append(f"FAIL: Only {n_pairs} overlapping pairs (need >= 10)")
    else:
        reasons.append(f"PASS: {n_pairs} overlapping pairs (>= 10)")

    # Check 2: Regular divergences > 5%?
    if div_df is not None and len(div_df) > 0:
        pct_large = (div_df["abs_divergence"] > 0.05).mean()
        if pct_large > 0.10:
            reasons.append(f"PASS: {100*pct_large:.1f}% of buckets have > 5% divergence")
        else:
            reasons.append(f"FAIL: Only {100*pct_large:.1f}% of buckets have > 5% divergence (need > 10%)")
    else:
        reasons.append("FAIL: No divergence data to analyze")

    # Check 3: Signal predictive power?
    if div_df is not None and len(div_df) > 0:
        sig_df = div_df[div_df["abs_divergence"] > 0.05]
        if len(sig_df) > 0:
            daily_cheap = sig_df[sig_df["divergence"] < -0.05]
            if len(daily_cheap) > 0:
                win_rate = daily_cheap["is_winner"].mean()
                avg_price = daily_cheap["daily_price"].mean()
                if avg_price > 0 and win_rate > avg_price * 1.10:  # 10% edge
                    reasons.append(f"PASS: Cheap side wins at {win_rate:.3f} vs price {avg_price:.3f} (>10% edge)")
                else:
                    reasons.append(f"FAIL: Cheap side wins at {win_rate:.3f} vs price {avg_price:.3f} (< 10% edge)")
            else:
                reasons.append("FAIL: No 'daily cheaper' observations to test")
        else:
            reasons.append("FAIL: No significant divergences to test signal")
    else:
        reasons.append("FAIL: No signal data")

    for r in reasons:
        print(f"  {r}")

    all_pass = all(r.startswith("PASS") for r in reasons)
    print(f"\n  OVERALL: {'PASS - Proceed to model creation' if all_pass else 'FAIL - Insufficient evidence for cross-market arb'}")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Cross-Market Consistency Arbitrage Analysis")
    print("Daily vs Weekly Market Overlap Detection")
    print("=" * 70)

    # Step 1: Load events
    events, catalog_df = load_events()
    print(f"\nLoaded {len(events)} events with valid dates")

    # Step 2: Find overlapping pairs
    pairs = find_overlapping_pairs(events)

    # GO/NO-GO gate
    if len(pairs) < 10:
        print(f"\n*** GO/NO-GO: FAIL ***")
        print(f"Only {len(pairs)} overlapping daily-weekly pairs found.")
        print(f"Need at least 10 to proceed. Insufficient data for analysis.")

        # Still show what we found
        if pairs:
            print(f"\nPairs found:")
            for p in pairs:
                print(f"  {p['daily_slug'][:60]} <-> {p['weekly_slug'][:60]}")

        # Diagnose why
        print(f"\n--- Diagnostic ---")
        daily_events = [e for e in events if e["market_type"] == "daily"]
        weekly_events = [e for e in events if e["market_type"] == "weekly"]
        print(f"Daily events: {len(daily_events)}")
        for de in daily_events[:10]:
            print(f"  {de['event_slug'][:60]}  ({de['start_date']} to {de['end_date']})")
        print(f"\nWeekly events: {len(weekly_events)}")
        weekly_dates = [(we["start_date"], we["end_date"], we["event_slug"]) for we in weekly_events]
        weekly_dates.sort()
        for ws, we_d, slug in weekly_dates[:10]:
            print(f"  {slug[:60]}  ({ws} to {we_d})")

        # Check if there's ANY date overlap at all
        daily_dates = set()
        for de in daily_events:
            current = de["start_date"]
            while current <= de["end_date"]:
                daily_dates.add(current)
                current += timedelta(days=1)

        weekly_date_ranges = []
        for we in weekly_events:
            current = we["start_date"]
            while current <= we["end_date"]:
                weekly_date_ranges.append(current)
                current += timedelta(days=1)
        weekly_dates_set = set(weekly_date_ranges)

        overlap_dates = daily_dates & weekly_dates_set
        print(f"\nDaily event dates: {min(daily_dates) if daily_dates else 'none'} to {max(daily_dates) if daily_dates else 'none'}")
        print(f"Weekly event dates: {min(weekly_dates_set) if weekly_dates_set else 'none'} to {max(weekly_dates_set) if weekly_dates_set else 'none'}")
        print(f"Calendar days with both daily AND weekly markets: {len(overlap_dates)}")

        if not overlap_dates:
            print("\n  No temporal overlap at all between daily and weekly markets.")
            print("  Daily markets exist in a different time period than weekly markets.")
        else:
            print(f"  Overlapping calendar dates: {sorted(list(overlap_dates))[:20]}...")

        render_verdict(pairs, [], None)
        return

    # Step 3-4: Analyze divergences
    print(f"\n{'='*70}")
    print("Analyzing price divergences across overlapping pairs...")
    print(f"{'='*70}")

    all_divergences, pairs_with_data, pairs_with_prices = analyze_divergences(pairs)
    print(f"\nPairs with backtest data: {pairs_with_data}")
    print(f"Pairs with price history: {pairs_with_prices}")
    print(f"Total bucket-level divergence observations: {len(all_divergences)}")

    if not all_divergences:
        print("\nNo divergence data could be computed. Missing backtest data for these events.")
        render_verdict(pairs, [], None)
        return

    # Step 5: Test signal
    div_df = test_signal(all_divergences)

    # Step 6: Verdict
    passed = render_verdict(pairs, all_divergences, div_df)

    if passed:
        print(f"\n  Next step: Create src/ml/cross_market_model.py")
        print(f"  The model should adjust daily bucket probabilities based on")
        print(f"  weekly-implied vs daily-actual divergence signals.")


if __name__ == "__main__":
    main()
