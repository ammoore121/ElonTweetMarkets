"""
Analyze intra-market overround mispricing across all backtest events.

Hypothesis: Bucket prices sum to >1.0 (overround). The overround is NOT distributed
evenly -- some buckets absorb more vig than others. Buckets where
raw_price > normalized_price are systematically overpriced; raw_price < normalized_price
are underpriced.

This script loads the backtest dataset, extracts entry prices at T-24h, and computes:
1. Overround = sum(bucket_prices) - 1.0 for each event
2. Normalized price = price / sum(prices) for each bucket
3. Mispricing = raw_price - normalized_price (positive = overpriced, negative = underpriced)
4. Whether mispricing predicts the winning bucket

Usage:
    python scripts/analyze_overround.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"
EVENTS_DIR = BACKTEST_DIR / "events"


# ---------------------------------------------------------------------------
# Entry price extraction (mirrors BacktestEngine._get_entry_prices)
# ---------------------------------------------------------------------------

def get_entry_prices(metadata: dict, prices_df, entry_hours: int = 24, window_hours: int = 6) -> dict:
    """Get market prices at entry time (T-24h before close).

    Returns dict of bucket_label -> raw market price.
    """
    buckets = metadata.get("buckets", [])
    bucket_prices = {}

    if prices_df is not None and not prices_df.empty:
        try:
            end_str = metadata.get("end_date")
            if end_str:
                end_dt = pd.Timestamp(end_str, tz="UTC")
                target_time = end_dt - pd.Timedelta(hours=entry_hours)
                window_start = target_time - pd.Timedelta(hours=window_hours)
                window_end = target_time + pd.Timedelta(hours=window_hours)

                ts_col = "timestamp"
                if ts_col in prices_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(prices_df[ts_col]):
                        prices_df = prices_df.copy()
                        prices_df[ts_col] = pd.to_datetime(prices_df[ts_col], utc=True)

                    windowed = prices_df[
                        (prices_df[ts_col] >= window_start)
                        & (prices_df[ts_col] <= window_end)
                    ]

                    if not windowed.empty:
                        windowed = windowed.copy()
                        windowed["_diff"] = (windowed[ts_col] - target_time).abs()
                        closest = windowed.sort_values("_diff").drop_duplicates(
                            subset=["bucket_label"], keep="first"
                        )
                        for _, row in closest.iterrows():
                            bucket_prices[str(row["bucket_label"])] = float(row["price"])

                        if bucket_prices:
                            return bucket_prices
        except Exception:
            pass

    # Fallback: metadata final prices
    for bucket in buckets:
        label = bucket["bucket_label"]
        price = bucket.get("price_yes")
        if price is not None:
            bucket_prices[label] = float(price)
        else:
            bucket_prices[label] = 0.0

    return bucket_prices


def classify_bucket_position(bucket: dict, buckets: list[dict]) -> str:
    """Classify bucket as 'lower_tail', 'center', or 'upper_tail'.

    Uses sorted position: bottom/top 25% = tail, middle 50% = center.
    """
    sorted_buckets = sorted(buckets, key=lambda b: int(b["lower_bound"]))
    n = len(sorted_buckets)
    labels = [b["bucket_label"] for b in sorted_buckets]

    idx = labels.index(bucket["bucket_label"]) if bucket["bucket_label"] in labels else -1
    if idx < 0:
        return "unknown"

    frac = idx / max(n - 1, 1)
    if frac <= 0.25:
        return "lower_tail"
    elif frac >= 0.75:
        return "upper_tail"
    else:
        return "center"


def main():
    # Load backtest index
    if not INDEX_PATH.exists():
        print("ERROR: Backtest index not found at {}".format(INDEX_PATH))
        print("       Run: python scripts/build_backtest_dataset.py")
        sys.exit(1)

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    events = index.get("events", [])
    print("Loaded {} events from backtest index".format(len(events)))
    print()

    # -----------------------------------------------------------------------
    # Step 1: Compute overround and mispricing for each event
    # -----------------------------------------------------------------------
    all_overrounds = []
    all_mispricings = []  # list of dicts with per-bucket mispricing info
    events_with_prices = 0
    events_skipped = 0

    for evt in events:
        slug = evt["event_slug"]
        event_dir = EVENTS_DIR / slug

        # Load metadata
        meta_path = event_dir / "metadata.json"
        if not meta_path.exists():
            events_skipped += 1
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load prices
        prices_df = None
        prices_path = event_dir / "prices.parquet"
        if prices_path.exists():
            try:
                prices_df = pd.read_parquet(prices_path)
            except Exception:
                pass

        # Get entry prices at T-24h
        entry_prices = get_entry_prices(metadata, prices_df)

        buckets = metadata.get("buckets", [])
        winning_bucket = metadata.get("winning_bucket")
        tier = metadata.get("ground_truth_tier", "unknown")

        if not entry_prices or not buckets:
            events_skipped += 1
            continue

        # Filter to only buckets with non-zero prices
        bucket_labels_with_prices = [
            b["bucket_label"] for b in buckets
            if entry_prices.get(b["bucket_label"], 0) > 0.001
        ]

        if len(bucket_labels_with_prices) < 2:
            events_skipped += 1
            continue

        # Compute overround using ALL bucket prices (including near-zero)
        raw_prices = {b["bucket_label"]: entry_prices.get(b["bucket_label"], 0.0) for b in buckets}
        price_sum = sum(raw_prices.values())

        if price_sum < 0.5:
            # Prices too low (likely metadata fallback with all zeros)
            events_skipped += 1
            continue

        events_with_prices += 1
        overround = price_sum - 1.0
        all_overrounds.append({
            "slug": slug,
            "tier": tier,
            "n_buckets": len(buckets),
            "price_sum": price_sum,
            "overround": overround,
            "market_type": metadata.get("market_type", "unknown"),
        })

        # Compute per-bucket mispricing
        for b in buckets:
            label = b["bucket_label"]
            raw_price = raw_prices.get(label, 0.0)
            normalized_price = raw_price / price_sum if price_sum > 0 else 0.0
            mispricing = raw_price - normalized_price
            is_winner = (label == winning_bucket)
            position = classify_bucket_position(b, buckets)

            all_mispricings.append({
                "slug": slug,
                "tier": tier,
                "bucket_label": label,
                "raw_price": raw_price,
                "normalized_price": normalized_price,
                "mispricing": mispricing,
                "is_winner": is_winner,
                "position": position,
                "n_buckets": len(buckets),
                "overround": overround,
            })

    print("=" * 70)
    print("OVERROUND ANALYSIS")
    print("=" * 70)
    print()
    print("Events analyzed: {} (skipped {} with insufficient prices)".format(
        events_with_prices, events_skipped))
    print()

    # -----------------------------------------------------------------------
    # Step 2: Overround statistics
    # -----------------------------------------------------------------------
    overrounds = [o["overround"] for o in all_overrounds]
    print("--- Overround Statistics ---")
    print("  Mean:   {:.4f} ({:.2f}%)".format(np.mean(overrounds), np.mean(overrounds) * 100))
    print("  Median: {:.4f} ({:.2f}%)".format(np.median(overrounds), np.median(overrounds) * 100))
    print("  Std:    {:.4f}".format(np.std(overrounds)))
    print("  Min:    {:.4f}".format(np.min(overrounds)))
    print("  Max:    {:.4f}".format(np.max(overrounds)))
    print("  % events with overround > 0: {:.1f}%".format(
        100 * sum(1 for o in overrounds if o > 0) / len(overrounds)))
    print("  % events with overround > 5%: {:.1f}%".format(
        100 * sum(1 for o in overrounds if o > 0.05) / len(overrounds)))
    print()

    # By market type
    print("--- Overround by Market Type ---")
    by_type = defaultdict(list)
    for o in all_overrounds:
        by_type[o["market_type"]].append(o["overround"])
    for mtype, vals in sorted(by_type.items()):
        print("  {:>10s}: mean={:.4f}, n={}".format(mtype, np.mean(vals), len(vals)))
    print()

    # By tier
    print("--- Overround by Tier ---")
    by_tier = defaultdict(list)
    for o in all_overrounds:
        by_tier[o["tier"]].append(o["overround"])
    for tier, vals in sorted(by_tier.items()):
        print("  {:>10s}: mean={:.4f}, n={}".format(tier, np.mean(vals), len(vals)))
    print()

    # -----------------------------------------------------------------------
    # Step 3: Mispricing distribution by bucket position
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_mispricings)

    print("--- Mispricing by Bucket Position ---")
    for pos in ["lower_tail", "center", "upper_tail"]:
        subset = df[df["position"] == pos]
        if subset.empty:
            continue
        print("  {:>12s}: mean_mispricing={:+.6f}, std={:.6f}, n={}".format(
            pos,
            subset["mispricing"].mean(),
            subset["mispricing"].std(),
            len(subset),
        ))
    print()

    # -----------------------------------------------------------------------
    # Step 4: Does mispricing predict winners?
    # -----------------------------------------------------------------------
    print("--- Mispricing vs Winner Prediction ---")

    # Overall: are winners more likely to be underpriced?
    winners = df[df["is_winner"]]
    losers = df[~df["is_winner"]]

    print("  Winner buckets:  mean_mispricing={:+.6f} (n={})".format(
        winners["mispricing"].mean(), len(winners)))
    print("  Loser buckets:   mean_mispricing={:+.6f} (n={})".format(
        losers["mispricing"].mean(), len(losers)))
    print()

    # Check: among underpriced buckets (mispricing < -threshold), what % are winners?
    print("--- Hit Rate Analysis (underpriced = mispricing < -threshold) ---")
    print("  {:>12s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}".format(
        "Threshold", "N_under", "N_winners", "Hit_rate%", "Base_rate%"))

    for threshold in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]:
        underpriced = df[df["mispricing"] < -threshold]
        if len(underpriced) == 0:
            continue
        n_winners_under = underpriced["is_winner"].sum()
        hit_rate = n_winners_under / len(underpriced) * 100

        # Base rate: what fraction of all buckets are winners?
        base_rate = df["is_winner"].sum() / len(df) * 100

        # Lift = hit_rate / base_rate
        lift = hit_rate / base_rate if base_rate > 0 else 0

        print("  {:>12.3f}  {:>8d}  {:>10d}  {:>9.1f}%  {:>9.1f}%  lift={:.2f}x".format(
            threshold, len(underpriced), int(n_winners_under), hit_rate, base_rate, lift))
    print()

    # Also check overpriced
    print("--- Hit Rate Analysis (overpriced = mispricing > +threshold) ---")
    print("  {:>12s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}".format(
        "Threshold", "N_over", "N_winners", "Hit_rate%", "Base_rate%"))

    for threshold in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]:
        overpriced = df[df["mispricing"] > threshold]
        if len(overpriced) == 0:
            continue
        n_winners_over = overpriced["is_winner"].sum()
        hit_rate = n_winners_over / len(overpriced) * 100
        base_rate = df["is_winner"].sum() / len(df) * 100
        lift = hit_rate / base_rate if base_rate > 0 else 0

        print("  {:>12.3f}  {:>8d}  {:>10d}  {:>9.1f}%  {:>9.1f}%  lift={:.2f}x".format(
            threshold, len(overpriced), int(n_winners_over), hit_rate, base_rate, lift))
    print()

    # -----------------------------------------------------------------------
    # Step 5: Mispricing by bucket position AND winner status
    # -----------------------------------------------------------------------
    print("--- Mispricing by Position & Winner Status ---")
    for pos in ["lower_tail", "center", "upper_tail"]:
        pos_df = df[df["position"] == pos]
        if pos_df.empty:
            continue
        w = pos_df[pos_df["is_winner"]]
        l = pos_df[~pos_df["is_winner"]]
        print("  {}:".format(pos))
        if not w.empty:
            print("    Winners: mean_mispricing={:+.6f} (n={})".format(
                w["mispricing"].mean(), len(w)))
        if not l.empty:
            print("    Losers:  mean_mispricing={:+.6f} (n={})".format(
                l["mispricing"].mean(), len(l)))
    print()

    # -----------------------------------------------------------------------
    # Step 6: Correlation between raw_price and mispricing
    # -----------------------------------------------------------------------
    print("--- Mispricing Distribution ---")
    print("  All buckets: mean={:+.6f}, median={:+.6f}, std={:.6f}".format(
        df["mispricing"].mean(), df["mispricing"].median(), df["mispricing"].std()))

    # Mispricing is proportional to raw_price * overround / price_sum
    # So higher-priced buckets absorb more absolute mispricing
    # Check if relative mispricing (mispricing / raw_price) is more uniform
    nonzero = df[df["raw_price"] > 0.01].copy()
    nonzero["relative_mispricing"] = nonzero["mispricing"] / nonzero["raw_price"]

    print("  Non-zero buckets: mean_relative_mispricing={:+.6f} (n={})".format(
        nonzero["relative_mispricing"].mean(), len(nonzero)))
    print("  Relative mispricing std: {:.6f}".format(nonzero["relative_mispricing"].std()))
    print()

    # -----------------------------------------------------------------------
    # Step 7: Theoretical check -- is mispricing just proportional to price?
    # -----------------------------------------------------------------------
    print("--- Key Insight: Is Mispricing Proportional to Raw Price? ---")
    # If mispricing[i] = raw_price[i] - raw_price[i]/sum = raw_price[i] * (1 - 1/sum)
    # = raw_price[i] * overround / (1 + overround)
    # Then relative mispricing = overround / (1 + overround) = CONSTANT for all buckets
    # This means mispricing carries NO additional information beyond overround itself.

    if not nonzero.empty:
        rel_mispricing_vals = nonzero["relative_mispricing"].values
        mean_rel = np.mean(rel_mispricing_vals)
        std_rel = np.std(rel_mispricing_vals)
        cv_rel = std_rel / abs(mean_rel) if abs(mean_rel) > 0 else float("inf")

        print("  Mean relative mispricing: {:.6f}".format(mean_rel))
        print("  Std relative mispricing:  {:.6f}".format(std_rel))
        print("  CV (std/mean):            {:.4f}".format(cv_rel))
        print()

        if cv_rel < 0.1:
            print("  ** RESULT: Relative mispricing is nearly CONSTANT (CV={:.4f}).".format(cv_rel))
            print("     This means mispricing = raw_price * (overround / (1 + overround))")
            print("     Every bucket is equally mispriced in relative terms.")
            print("     There is NO differential signal -- overround is distributed proportionally.")
            print()
            verdict = "FAIL"
        else:
            print("  ** RESULT: Relative mispricing has meaningful variation (CV={:.4f}).".format(cv_rel))
            print("     Some buckets absorb more vig than others.")
            print("     This could be actionable if correlated with winner status.")
            print()
            verdict = "MAYBE"
    else:
        verdict = "FAIL"
        print("  No non-zero buckets to analyze.")
        print()

    # -----------------------------------------------------------------------
    # Step 8: Final verdict
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("VERDICT: {}".format(verdict))
    print("=" * 70)

    if verdict == "FAIL":
        print()
        print("The overround mispricing hypothesis does NOT hold.")
        print()
        print("Mathematical proof:")
        print("  mispricing[i] = price[i] - price[i] / sum(prices)")
        print("                = price[i] * (1 - 1/sum(prices))")
        print("                = price[i] * overround / (1 + overround)")
        print()
        print("Since overround/(1+overround) is constant for all buckets in an event,")
        print("every bucket's mispricing is exactly proportional to its raw price.")
        print("Normalizing prices just divides by a constant -- it cannot create")
        print("differential signal between buckets.")
        print()
        print("RECOMMENDATION: Do NOT build IntraMarketArbModel. This approach has")
        print("zero edge by construction. The overround is a flat tax, not a source")
        print("of exploitable mispricing.")
    elif verdict == "MAYBE":
        print()
        print("There is some variation in relative mispricing across buckets.")
        print("Worth building a model to test if this variation predicts winners.")

    return verdict


if __name__ == "__main__":
    main()
