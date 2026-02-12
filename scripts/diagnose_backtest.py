"""
Diagnostic script to trace through backtest events and identify why the
ensemble model loses 100% of bets.

Findings:
- The RegimeAwareModel assigns enormous probability to extreme tail buckets
  (e.g., 62% to "500+" when actual outcome is 200).
- Even with only 10% weight in the ensemble, this pulls the ensemble
  prediction from ~0.05% (market) to ~6.6% on the tail bucket.
- Since the market prices these tails at 0.05%, the Kelly criterion sees
  a massive "edge" and bets on them every time.
- The tail bucket NEVER wins, so every bet is a loss.
- The model never bets on plausible buckets (where real edge might exist)
  because the model agrees with the crowd on those.

Root cause: The RegimeAwareModel's fixed regime means (20, 40, 65, 85
tweets/day) create a mixture distribution that assigns way too much mass
to the upper tail when duration_days * regime_mean is near or above the
tail bucket threshold.

Fix: Cap the ensemble's per-bucket probability to be no more than
max(2 * market_price, market_price + 0.03) for tail buckets, or reduce
the RegimeAwareModel weight to near-zero when crowd data is available.
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ml.advanced_models import (
    EnsembleModel,
    MarketAdjustedModel,
    RegimeAwareModel,
    _get_crowd_probs,
)
from src.ml.baseline_model import NaiveBucketModel

EVENTS_DIR = PROJECT_DIR / "data" / "backtest" / "events"
INDEX_PATH = PROJECT_DIR / "data" / "backtest" / "backtest_index.json"


def load_event(slug):
    """Load metadata and features for a single event."""
    event_dir = EVENTS_DIR / slug
    metadata = json.load(open(event_dir / "metadata.json"))
    features = json.load(open(event_dir / "features.json"))

    prices = None
    prices_path = event_dir / "prices.parquet"
    if prices_path.exists():
        import pandas as pd

        prices = pd.read_parquet(prices_path)

    return metadata, features, prices


def get_entry_prices_for_event(metadata, prices_df, entry_hours=24):
    """Reproduce get_entry_prices from run_backtest.py."""
    import pandas as pd

    buckets = metadata.get("buckets", [])
    if prices_df is not None and not prices_df.empty:
        end_str = metadata.get("end_date")
        if end_str:
            end_dt = pd.Timestamp(end_str, tz="UTC")
            target_time = end_dt - pd.Timedelta(hours=entry_hours)
            window_start = target_time - pd.Timedelta(hours=6)
            window_end = target_time + pd.Timedelta(hours=6)

            ts_col = "timestamp"
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
                entry_prices = {}
                for _, row in closest.iterrows():
                    entry_prices[str(row["bucket_label"])] = float(row["price"])
                if entry_prices:
                    return entry_prices

    # Fallback
    entry_prices = {}
    for bucket in buckets:
        label = bucket["bucket_label"]
        price = bucket.get("price_yes")
        entry_prices[label] = float(price) if price is not None else 0.0
    return entry_prices


def diagnose_event(slug, entry_hours=24):
    """Full diagnostic trace for one event."""
    metadata, features, prices_df = load_event(slug)

    print("=" * 70)
    print(f"EVENT: {slug}")
    print(f"  Title: {metadata['event_title']}")
    print(f"  Duration: {metadata['duration_days']} days")
    print(f"  Winning bucket: {metadata['winning_bucket']}")
    print(f"  XTracker count: {metadata['xtracker_count']}")
    print()

    # Get entry prices
    entry_prices = get_entry_prices_for_event(metadata, prices_df, entry_hours)

    buckets = metadata["buckets"]
    context = {
        "duration_days": metadata["duration_days"],
        "entry_prices": entry_prices,
    }

    # Run all models
    regime = RegimeAwareModel()
    adjusted = MarketAdjustedModel()
    ensemble = EnsembleModel()

    regime_probs = regime.predict(features, buckets, context=context)
    adjusted_probs = adjusted.predict(features, buckets, context=context)
    ensemble_probs = ensemble.predict(features, buckets, context=context)

    # Find the tail bucket (highest open-ended)
    tail_label = None
    for b in buckets:
        if int(b["upper_bound"]) >= 99999:
            tail_label = b["bucket_label"]
            break

    # Sort buckets by lower_bound for display
    sorted_buckets = sorted(buckets, key=lambda b: int(b["lower_bound"]))

    # Show a comparison table
    print(
        f"{'Bucket':>12s}  {'Market':>8s}  {'Regime':>8s}  {'Adjusted':>8s}  "
        f"{'Ensemble':>8s}  {'Edge':>8s}  {'Bet?':>5s}  {'Winner':>6s}"
    )
    print("-" * 85)

    min_edge = 0.05
    bet_buckets = []

    for b in sorted_buckets:
        label = b["bucket_label"]
        mkt = entry_prices.get(label, 0.0)
        r_prob = regime_probs.get(label, 0.0)
        a_prob = adjusted_probs.get(label, 0.0)
        e_prob = ensemble_probs.get(label, 0.0)
        edge = e_prob - mkt
        is_winner = label == metadata["winning_bucket"]
        bet = edge >= min_edge and 0 < mkt < 1
        marker = "<--" if is_winner else ""
        bet_str = "YES" if bet else ""

        if bet:
            bet_buckets.append(label)

        print(
            f"{label:>12s}  {mkt:>8.4f}  {r_prob:>8.4f}  {a_prob:>8.4f}  "
            f"{e_prob:>8.4f}  {edge:>+8.4f}  {bet_str:>5s}  {marker:>6s}"
        )

    print()

    # Explain the tail bucket problem
    if tail_label:
        print(f"TAIL BUCKET ANALYSIS ({tail_label}):")
        print(f"  Market price:          {entry_prices.get(tail_label, 0):.6f}")
        print(f"  Regime model prob:     {regime_probs.get(tail_label, 0):.6f}")
        print(f"  Market-adjusted prob:  {adjusted_probs.get(tail_label, 0):.6f}")
        print(f"  Ensemble prob:         {ensemble_probs.get(tail_label, 0):.6f}")
        print(
            f"  Regime contribution to ensemble: "
            f"{0.10 * regime_probs.get(tail_label, 0):.6f}"
        )
        print(
            f"  Adjusted contribution to ensemble: "
            f"{0.90 * adjusted_probs.get(tail_label, 0):.6f}"
        )

        if regime_probs.get(tail_label, 0) > 0.01:
            print()
            print(
                f"  ** PROBLEM: Regime model assigns {regime_probs.get(tail_label, 0)*100:.1f}% "
                f"to {tail_label}, market says {entry_prices.get(tail_label, 0)*100:.2f}%"
            )
            print(
                f"  ** Even at 10% weight, regime pulls ensemble to "
                f"{ensemble_probs.get(tail_label, 0)*100:.1f}%, creating false edge of "
                f"{(ensemble_probs.get(tail_label, 0) - entry_prices.get(tail_label, 0))*100:.1f}%"
            )

    print()
    if bet_buckets:
        print(f"BETS PLACED: {bet_buckets}")
        winning = metadata["winning_bucket"]
        if winning in bet_buckets:
            print("  --> WOULD WIN (bet includes winning bucket)")
        else:
            print(f"  --> WOULD LOSE (winning bucket is {winning}, not in bets)")
    else:
        print("NO BETS PLACED")
    print()


def main():
    # Load index to find gold events with bets
    index = json.load(open(INDEX_PATH))
    gold_events = [
        e for e in index["events"] if e.get("ground_truth_tier") == "gold"
    ]

    # Known events that had bets in the backtest results
    events_with_bets = [
        "elon-musk-of-tweets-november-18-november-25",
        "elon-musk-of-tweets-december-9-december-16",
        "elon-musk-of-tweets-january-2-january-9",
        "elon-musk-of-tweets-january-15-january-17",
        "elon-musk-of-tweets-january-26-january-28",
    ]

    print("DIAGNOSTIC: Tracing backtest bug for ensemble model")
    print("=" * 70)
    print()
    print("HYPOTHESIS: RegimeAwareModel wildly overestimates tail probability,")
    print("            contaminating ensemble even at 10% weight, causing bets")
    print("            exclusively on extreme upper tail buckets that never win.")
    print()

    for slug in events_with_bets:
        try:
            diagnose_event(slug)
        except Exception as e:
            print(f"ERROR on {slug}: {e}")
            print()

    # Summary
    print("=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    print()
    print("1. RegimeAwareModel assigns 5-62% probability to extreme tail buckets")
    print("   (e.g., 500+, 580+, 740+, 240+) that the market prices at 0.05-2%.")
    print()
    print("2. This happens because the regime mixture includes 'medium' (40/day)")
    print("   and 'high' (65/day) regimes which, over 7-10 day periods, produce")
    print("   total expected counts near or above the tail bucket threshold.")
    print()
    print("3. Even at 10% ensemble weight, the regime model pulls tail probability")
    print("   from ~0.05% to ~6%, creating a massive false 'edge' of ~5.5%.")
    print()
    print("4. Kelly criterion then bets on these tails every time. Since the tail")
    print("   bucket NEVER wins in the dataset, every bet is a loss.")
    print()
    print("FIX OPTIONS:")
    print("  A. Reduce regime_weight to 0.0 when crowd data is available")
    print("  B. Cap per-bucket probability to max(2*market_price, market+0.03)")
    print("  C. Add tail-capping in EnsembleModel.predict() that prevents the")
    print("     ensemble from exceeding 3x the market price on any single bucket")
    print("  D. All of the above (recommended)")


if __name__ == "__main__":
    main()
