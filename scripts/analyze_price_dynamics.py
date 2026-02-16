"""
Analyze CLOB price dynamics to validate intra-market momentum hypothesis.

Hypothesis: Bucket prices evolve over a market's lifetime (3-7 days).
Buckets that have been falling steadily (crowd abandoning) but match
the model's prediction are higher-value entries. Buckets with sharp
recent spikes (crowd piling in late) tend to overshoot.

Steps:
1. Audit price coverage (snapshots per event/bucket)
2. Compute per-bucket momentum features (24h, 48h, acceleration, volatility)
3. Test predictive power: momentum of winning vs losing buckets
4. Check autocorrelation (mean-reverting or trending?)
5. Save momentum features to event features.json files

Usage:
    python scripts/analyze_price_dynamics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

PRICES_PATH = PROJECT_DIR / "data" / "sources" / "polymarket" / "prices" / "price_history.parquet"
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"
EVENTS_DIR = BACKTEST_DIR / "events"

SEP = "=" * 70


def load_prices():
    """Load CLOB price history."""
    if not PRICES_PATH.exists():
        print("ERROR: Price history not found at {}".format(PRICES_PATH))
        sys.exit(1)
    df = pd.read_parquet(PRICES_PATH)
    print("Loaded {} price rows".format(len(df)))
    print("Columns: {}".format(list(df.columns)))
    print("Date range: {} to {}".format(df["timestamp"].min(), df["timestamp"].max()))
    return df


def load_backtest_index():
    """Load backtest index for ground truth."""
    if not INDEX_PATH.exists():
        print("ERROR: Backtest index not found at {}".format(INDEX_PATH))
        sys.exit(1)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: Audit CLOB Price Coverage
# ---------------------------------------------------------------------------

def audit_coverage(df):
    """Check how many distinct timestamps exist per event/bucket."""
    print()
    print(SEP)
    print("STEP 1: CLOB PRICE DATA COVERAGE AUDIT")
    print(SEP)
    print()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Per-event stats
    event_stats = []
    for slug, grp in df.groupby("event_slug"):
        n_buckets = grp["bucket_label"].nunique()
        n_timestamps = grp["timestamp"].nunique()
        earliest = grp["timestamp"].min()
        latest = grp["timestamp"].max()
        span = latest - earliest
        span_hours = span.total_seconds() / 3600

        # Per-bucket snapshot counts
        per_bucket = grp.groupby("bucket_label")["timestamp"].nunique()
        avg_snapshots = per_bucket.mean()
        min_snapshots = per_bucket.min()
        max_snapshots = per_bucket.max()

        event_stats.append({
            "event_slug": slug,
            "n_buckets": n_buckets,
            "n_unique_timestamps": n_timestamps,
            "earliest": earliest,
            "latest": latest,
            "span_hours": round(span_hours, 1),
            "avg_snapshots_per_bucket": round(avg_snapshots, 1),
            "min_snapshots_per_bucket": min_snapshots,
            "max_snapshots_per_bucket": max_snapshots,
        })

    stats_df = pd.DataFrame(event_stats)

    # Summary
    print("Total events with price data: {}".format(len(stats_df)))
    print()
    print("Snapshot coverage distribution:")
    print("  Avg unique timestamps per event: {:.1f}".format(
        stats_df["n_unique_timestamps"].mean()))
    print("  Median: {:.0f}".format(stats_df["n_unique_timestamps"].median()))
    print("  Min: {}".format(stats_df["n_unique_timestamps"].min()))
    print("  Max: {}".format(stats_df["n_unique_timestamps"].max()))
    print()
    print("Time span (hours) distribution:")
    print("  Avg: {:.1f}h".format(stats_df["span_hours"].mean()))
    print("  Median: {:.1f}h".format(stats_df["span_hours"].median()))
    print("  Min: {:.1f}h".format(stats_df["span_hours"].min()))
    print("  Max: {:.1f}h".format(stats_df["span_hours"].max()))
    print()

    # How many events have enough coverage for momentum computation?
    good_coverage = stats_df[
        (stats_df["span_hours"] >= 48) &
        (stats_df["min_snapshots_per_bucket"] >= 3)
    ]
    marginal_coverage = stats_df[
        (stats_df["span_hours"] >= 24) &
        (stats_df["min_snapshots_per_bucket"] >= 2)
    ]
    print("Events with GOOD coverage (>=48h span, >=3 snapshots/bucket): {} / {}".format(
        len(good_coverage), len(stats_df)))
    print("Events with MARGINAL coverage (>=24h span, >=2 snapshots/bucket): {} / {}".format(
        len(marginal_coverage), len(stats_df)))
    print()

    # Show bottom 10 by coverage
    print("--- Worst 10 events by snapshot count ---")
    worst = stats_df.nsmallest(10, "n_unique_timestamps")
    for _, row in worst.iterrows():
        print("  {}: {} timestamps, {:.1f}h span, {} buckets".format(
            row["event_slug"][:60], row["n_unique_timestamps"],
            row["span_hours"], row["n_buckets"]))
    print()

    # Show top 10 by coverage
    print("--- Best 10 events by snapshot count ---")
    best = stats_df.nlargest(10, "n_unique_timestamps")
    for _, row in best.iterrows():
        print("  {}: {} timestamps, {:.1f}h span, {} buckets".format(
            row["event_slug"][:60], row["n_unique_timestamps"],
            row["span_hours"], row["n_buckets"]))

    return stats_df


# ---------------------------------------------------------------------------
# Step 2: Compute Price Momentum Features
# ---------------------------------------------------------------------------

def compute_momentum_features(df, stats_df, index):
    """Compute per-bucket momentum features for events with sufficient coverage."""
    print()
    print(SEP)
    print("STEP 2: COMPUTE PRICE MOMENTUM FEATURES")
    print(SEP)
    print()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Build event metadata lookup
    events_by_slug = {}
    for evt in index.get("events", []):
        events_by_slug[evt["event_slug"]] = evt

    all_momentum = []
    events_processed = 0
    events_skipped = 0

    for slug, grp in df.groupby("event_slug"):
        evt = events_by_slug.get(slug)
        if evt is None:
            continue

        # Get event end date
        end_str = evt.get("end_date")
        if not end_str:
            events_skipped += 1
            continue

        end_dt = pd.Timestamp(end_str, tz="UTC")

        # Reference time: T-24h before close (this is when we'd enter)
        t_entry = end_dt - pd.Timedelta(hours=24)
        t_24h_before = end_dt - pd.Timedelta(hours=48)
        t_48h_before = end_dt - pd.Timedelta(hours=72)

        # Load metadata for winning bucket
        meta_path = EVENTS_DIR / slug / "metadata.json"
        winning_bucket = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                winning_bucket = meta.get("winning_bucket")

        # For each bucket, find closest price at each reference time
        for bucket_label, bgrp in grp.groupby("bucket_label"):
            bgrp = bgrp.sort_values("timestamp")

            price_at_entry = _closest_price(bgrp, t_entry, window_hours=6)
            price_at_24h = _closest_price(bgrp, t_24h_before, window_hours=6)
            price_at_48h = _closest_price(bgrp, t_48h_before, window_hours=6)

            if price_at_entry is None or price_at_24h is None:
                continue

            # Momentum features
            momentum_24h = price_at_entry - price_at_24h
            momentum_48h = (price_at_entry - price_at_48h) if price_at_48h is not None else None
            acceleration = (momentum_24h - (price_at_24h - price_at_48h)) if price_at_48h is not None else None

            # Volatility in 24h window before entry
            window_start = t_entry - pd.Timedelta(hours=24)
            window_data = bgrp[
                (bgrp["timestamp"] >= window_start) &
                (bgrp["timestamp"] <= t_entry)
            ]
            vol_24h = window_data["price"].std() if len(window_data) >= 2 else None

            # Relative momentum (normalize by price level)
            rel_momentum_24h = momentum_24h / max(price_at_24h, 0.01)

            won = (bucket_label == winning_bucket) if winning_bucket else None

            all_momentum.append({
                "event_slug": slug,
                "bucket_label": bucket_label,
                "price_at_entry": price_at_entry,
                "price_at_24h_before": price_at_24h,
                "price_at_48h_before": price_at_48h,
                "momentum_24h": momentum_24h,
                "momentum_48h": momentum_48h,
                "acceleration": acceleration,
                "vol_24h": vol_24h,
                "rel_momentum_24h": rel_momentum_24h,
                "won": won,
                "ground_truth_tier": evt.get("ground_truth_tier"),
            })

        events_processed += 1

    momentum_df = pd.DataFrame(all_momentum)
    print("Processed {} events, skipped {}".format(events_processed, events_skipped))
    print("Total bucket-momentum records: {}".format(len(momentum_df)))

    if len(momentum_df) > 0 and "won" in momentum_df.columns:
        has_outcome = momentum_df[momentum_df["won"].notna()]
        print("Records with outcome data: {}".format(len(has_outcome)))
        if len(has_outcome) > 0:
            print("  Winners: {}".format(int(has_outcome["won"].sum())))
            print("  Losers: {}".format(int((~has_outcome["won"]).sum())))

    return momentum_df


def _closest_price(bgrp, target_time, window_hours=6):
    """Find closest price to target_time within window."""
    window_start = target_time - pd.Timedelta(hours=window_hours)
    window_end = target_time + pd.Timedelta(hours=window_hours)
    windowed = bgrp[
        (bgrp["timestamp"] >= window_start) &
        (bgrp["timestamp"] <= window_end)
    ]
    if windowed.empty:
        return None
    diffs = (windowed["timestamp"] - target_time).abs()
    idx = diffs.idxmin()
    return float(windowed.loc[idx, "price"])


# ---------------------------------------------------------------------------
# Step 3: Statistical Tests
# ---------------------------------------------------------------------------

def test_predictive_power(momentum_df):
    """Test if momentum predicts winning buckets."""
    print()
    print(SEP)
    print("STEP 3: STATISTICAL TESTS - MOMENTUM PREDICTIVE POWER")
    print(SEP)
    print()

    if momentum_df.empty:
        print("FAIL: No momentum data available.")
        return False

    has_outcome = momentum_df[momentum_df["won"].notna()].copy()
    if len(has_outcome) < 20:
        print("FAIL: Insufficient data ({} records with outcomes).".format(len(has_outcome)))
        return False

    winners = has_outcome[has_outcome["won"] == True]
    losers = has_outcome[has_outcome["won"] == False]

    print("Sample sizes: {} winners, {} losers".format(len(winners), len(losers)))
    print()

    signal_found = False

    # Test 1: Raw momentum_24h
    print("--- Test 1: Raw Momentum (24h) ---")
    w_mom = winners["momentum_24h"].dropna()
    l_mom = losers["momentum_24h"].dropna()
    if len(w_mom) >= 5 and len(l_mom) >= 5:
        print("  Winners avg momentum_24h: {:.6f}".format(w_mom.mean()))
        print("  Losers  avg momentum_24h: {:.6f}".format(l_mom.mean()))
        print("  Difference: {:.6f}".format(w_mom.mean() - l_mom.mean()))
        t_stat, p_value = stats.ttest_ind(w_mom, l_mom, equal_var=False)
        print("  Welch's t-test: t={:.3f}, p={:.4f}".format(t_stat, p_value))
        if p_value < 0.05:
            print("  ** SIGNIFICANT at 5% level **")
            signal_found = True
        else:
            print("  Not significant (p >= 0.05)")
    print()

    # Test 2: Relative momentum
    print("--- Test 2: Relative Momentum (24h / price level) ---")
    w_rel = winners["rel_momentum_24h"].dropna()
    l_rel = losers["rel_momentum_24h"].dropna()
    if len(w_rel) >= 5 and len(l_rel) >= 5:
        print("  Winners avg rel_momentum: {:.4f}".format(w_rel.mean()))
        print("  Losers  avg rel_momentum: {:.4f}".format(l_rel.mean()))
        t_stat, p_value = stats.ttest_ind(w_rel, l_rel, equal_var=False)
        print("  Welch's t-test: t={:.3f}, p={:.4f}".format(t_stat, p_value))
        if p_value < 0.05:
            print("  ** SIGNIFICANT at 5% level **")
            signal_found = True
        else:
            print("  Not significant (p >= 0.05)")
    print()

    # Test 3: Momentum 48h
    print("--- Test 3: Raw Momentum (48h) ---")
    w_48 = winners["momentum_48h"].dropna()
    l_48 = losers["momentum_48h"].dropna()
    if len(w_48) >= 5 and len(l_48) >= 5:
        print("  Winners avg momentum_48h: {:.6f}".format(w_48.mean()))
        print("  Losers  avg momentum_48h: {:.6f}".format(l_48.mean()))
        t_stat, p_value = stats.ttest_ind(w_48, l_48, equal_var=False)
        print("  Welch's t-test: t={:.3f}, p={:.4f}".format(t_stat, p_value))
        if p_value < 0.05:
            print("  ** SIGNIFICANT at 5% level **")
            signal_found = True
        else:
            print("  Not significant (p >= 0.05)")
    print()

    # Test 4: Acceleration
    print("--- Test 4: Acceleration (change in momentum) ---")
    w_acc = winners["acceleration"].dropna()
    l_acc = losers["acceleration"].dropna()
    if len(w_acc) >= 5 and len(l_acc) >= 5:
        print("  Winners avg acceleration: {:.6f}".format(w_acc.mean()))
        print("  Losers  avg acceleration: {:.6f}".format(l_acc.mean()))
        t_stat, p_value = stats.ttest_ind(w_acc, l_acc, equal_var=False)
        print("  Welch's t-test: t={:.3f}, p={:.4f}".format(t_stat, p_value))
        if p_value < 0.05:
            print("  ** SIGNIFICANT at 5% level **")
            signal_found = True
        else:
            print("  Not significant (p >= 0.05)")
    print()

    # Test 5: Volatility
    print("--- Test 5: Price Volatility (24h) ---")
    w_vol = winners["vol_24h"].dropna()
    l_vol = losers["vol_24h"].dropna()
    if len(w_vol) >= 5 and len(l_vol) >= 5:
        print("  Winners avg vol_24h: {:.6f}".format(w_vol.mean()))
        print("  Losers  avg vol_24h: {:.6f}".format(l_vol.mean()))
        t_stat, p_value = stats.ttest_ind(w_vol, l_vol, equal_var=False)
        print("  Welch's t-test: t={:.3f}, p={:.4f}".format(t_stat, p_value))
        if p_value < 0.05:
            print("  ** SIGNIFICANT at 5% level **")
            signal_found = True
        else:
            print("  Not significant (p >= 0.05)")
    print()

    # Test 6: Autocorrelation of momentum (is it mean-reverting or trending?)
    print("--- Test 6: Momentum Autocorrelation ---")
    event_momentum = has_outcome.groupby("event_slug")["momentum_24h"].mean().dropna()
    if len(event_momentum) >= 10:
        autocorr = event_momentum.autocorr(lag=1)
        print("  Per-event avg momentum autocorrelation (lag=1): {:.4f}".format(autocorr))
        if autocorr < -0.2:
            print("  -> MEAN-REVERTING (negative autocorrelation)")
        elif autocorr > 0.2:
            print("  -> TRENDING (positive autocorrelation)")
        else:
            print("  -> NO CLEAR PATTERN (weak autocorrelation)")
    else:
        print("  Insufficient events for autocorrelation ({})".format(len(event_momentum)))
    print()

    # Test 7: Contrarian signal -- do falling prices predict winners?
    print("--- Test 7: Contrarian Signal (Falling Prices -> Winner?) ---")
    has_mom = has_outcome[has_outcome["momentum_24h"].notna()].copy()
    if len(has_mom) >= 20:
        falling = has_mom[has_mom["momentum_24h"] < -0.01]
        rising = has_mom[has_mom["momentum_24h"] > 0.01]
        flat = has_mom[(has_mom["momentum_24h"] >= -0.01) & (has_mom["momentum_24h"] <= 0.01)]

        if len(falling) > 0 and len(rising) > 0:
            falling_win_rate = falling["won"].mean()
            rising_win_rate = rising["won"].mean()
            flat_win_rate = flat["won"].mean() if len(flat) > 0 else None

            print("  Falling price buckets: {} records, win rate {:.2%}".format(
                len(falling), falling_win_rate))
            print("  Rising price buckets:  {} records, win rate {:.2%}".format(
                len(rising), rising_win_rate))
            if flat_win_rate is not None:
                print("  Flat price buckets:    {} records, win rate {:.2%}".format(
                    len(flat), flat_win_rate))

            # Chi-squared test for independence
            contingency = pd.crosstab(
                has_mom["momentum_24h"] < -0.01,
                has_mom["won"]
            )
            if contingency.shape == (2, 2):
                chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
                print("  Chi-squared test: chi2={:.3f}, p={:.4f}".format(chi2, p_chi))
                if p_chi < 0.05:
                    print("  ** SIGNIFICANT at 5% level **")
                    signal_found = True
    print()

    # Test 8: By tier
    print("--- Test 8: Momentum by Tier ---")
    for tier in ["gold", "silver", "bronze"]:
        tier_data = has_outcome[has_outcome["ground_truth_tier"] == tier]
        if len(tier_data) < 10:
            continue
        tw = tier_data[tier_data["won"] == True]["momentum_24h"].dropna()
        tl = tier_data[tier_data["won"] == False]["momentum_24h"].dropna()
        if len(tw) >= 3 and len(tl) >= 3:
            print("  {}: winners avg={:.6f}, losers avg={:.6f}, diff={:.6f}".format(
                tier.upper(), tw.mean(), tl.mean(), tw.mean() - tl.mean()))
            t_stat, p_value = stats.ttest_ind(tw, tl, equal_var=False)
            print("         t={:.3f}, p={:.4f} {}".format(
                t_stat, p_value, "**SIG**" if p_value < 0.05 else ""))
    print()

    # Test 9: Effect size (Cohen's d)
    print("--- Test 9: Effect Size (Cohen's d) ---")
    w_mom = winners["momentum_24h"].dropna()
    l_mom = losers["momentum_24h"].dropna()
    if len(w_mom) >= 5 and len(l_mom) >= 5:
        pooled_std = np.sqrt(
            ((len(w_mom) - 1) * w_mom.std() ** 2 + (len(l_mom) - 1) * l_mom.std() ** 2) /
            (len(w_mom) + len(l_mom) - 2)
        )
        if pooled_std > 0:
            cohens_d = (w_mom.mean() - l_mom.mean()) / pooled_std
            print("  Cohen's d (momentum_24h): {:.4f}".format(cohens_d))
            if abs(cohens_d) >= 0.5:
                print("  -> MEDIUM to LARGE effect")
            elif abs(cohens_d) >= 0.2:
                print("  -> SMALL effect")
            else:
                print("  -> NEGLIGIBLE effect")
    print()

    return signal_found


# ---------------------------------------------------------------------------
# Step 4: Save Momentum Features to Event features.json
# ---------------------------------------------------------------------------

def save_momentum_features(momentum_df):
    """Pre-compute and save momentum features into each event's features.json."""
    print()
    print(SEP)
    print("STEP 4: SAVING MOMENTUM FEATURES TO EVENT features.json FILES")
    print(SEP)
    print()

    if momentum_df.empty:
        print("No momentum data to save.")
        return 0

    events_updated = 0
    for slug, grp in momentum_df.groupby("event_slug"):
        feat_path = EVENTS_DIR / slug / "features.json"
        if not feat_path.exists():
            continue

        with open(feat_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        # Build per-bucket momentum dict
        bucket_momentum = {}
        for _, row in grp.iterrows():
            label = row["bucket_label"]
            entry = {}
            for key in ["momentum_24h", "momentum_48h", "acceleration",
                         "vol_24h", "rel_momentum_24h"]:
                val = row[key]
                if pd.notna(val):
                    entry[key] = round(float(val), 6)
                else:
                    entry[key] = None
            bucket_momentum[label] = entry

        # Aggregate event-level stats
        valid_mom = grp["momentum_24h"].dropna()
        event_level = {
            "avg_momentum_24h": round(float(valid_mom.mean()), 6) if len(valid_mom) > 0 else None,
            "std_momentum_24h": round(float(valid_mom.std()), 6) if len(valid_mom) > 1 else None,
            "max_abs_momentum_24h": round(float(valid_mom.abs().max()), 6) if len(valid_mom) > 0 else None,
            "n_buckets_with_momentum": len(bucket_momentum),
        }

        features["price_dynamics"] = {
            "per_bucket": bucket_momentum,
            "event_level": event_level,
        }

        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)

        events_updated += 1

    print("Updated {} event features.json files with price_dynamics data.".format(
        events_updated))
    return events_updated


# ---------------------------------------------------------------------------
# Step 5: Verdict
# ---------------------------------------------------------------------------

def print_verdict(signal_found, momentum_df, stats_df):
    """Print PASS/FAIL verdict."""
    print()
    print(SEP)
    print("VERDICT")
    print(SEP)
    print()

    # Check coverage
    if stats_df is None or len(stats_df) == 0:
        print("FAIL: No price data available.")
        return

    good_coverage = stats_df[
        (stats_df["span_hours"] >= 48) &
        (stats_df["min_snapshots_per_bucket"] >= 3)
    ]
    has_coverage = len(good_coverage) >= 20

    has_momentum = (momentum_df is not None and len(momentum_df) > 100)

    print("Coverage sufficient (>=20 events with >=48h span): {}".format(
        "YES ({} events)".format(len(good_coverage)) if has_coverage else "NO ({} events)".format(len(good_coverage))))
    print("Momentum data available: {}".format(
        "YES ({} records)".format(len(momentum_df)) if has_momentum else "NO"))
    print("Statistical signal found: {}".format("YES" if signal_found else "NO"))
    print()

    if has_coverage and has_momentum and signal_found:
        print("VERDICT: PASS - Proceed to build PriceDynamicsModel")
        print("  Momentum features have been saved to event features.json files.")
        print("  Next: Build src/ml/price_dynamics_model.py and backtest.")
    elif has_coverage and has_momentum:
        print("VERDICT: MARGINAL - Coverage exists but signal is weak.")
        print("  Momentum features saved anyway for exploration.")
        print("  Building model with conservative parameters (low momentum weight).")
        print("  The model will still combine momentum with the proven tail boost.")
    else:
        print("VERDICT: FAIL - Insufficient data or no signal.")
        if not has_coverage:
            print("  Need more price snapshots per event (currently have {} good events).".format(
                len(good_coverage)))


def main():
    print(SEP)
    print("PRICE DYNAMICS ANALYSIS")
    print("Validating intra-market momentum hypothesis")
    print(SEP)

    # Load data
    df = load_prices()
    index = load_backtest_index()

    # Step 1: Audit coverage
    stats_df = audit_coverage(df)

    # Step 2: Compute momentum
    momentum_df = compute_momentum_features(df, stats_df, index)

    # Step 3: Statistical tests
    signal_found = test_predictive_power(momentum_df)

    # Step 4: Save features (regardless of signal -- useful for exploration)
    save_momentum_features(momentum_df)

    # Step 5: Verdict
    print_verdict(signal_found, momentum_df, stats_df)


if __name__ == "__main__":
    main()
