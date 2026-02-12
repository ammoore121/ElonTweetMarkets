"""
Build the unified backtest dataset from market catalog, XTracker mapping,
price history, GDELT news data, and SpaceX launch calendar.

For each resolved event, produces:
  - metadata.json       (event info + ground truth)
  - prices.parquet      (hourly bucket prices for this event)
  - features.json       (pre-event feature vector)

Also writes a master backtest_index.json listing all events.

Input files:
  - data/processed/market_catalog.parquet           (one row per bucket per event)
  - data/processed/xtracker_mapping.parquet          (one row per event, ground truth)
  - data/sources/polymarket/prices/price_history.parquet  (hourly prices, all events)
  - data/sources/xtracker/daily_metrics_full.json    (daily tweet counts)
  - data/sources/gdelt/gdelt_*_timelinevol.json      (news volume per entity)
  - data/sources/gdelt/gdelt_*_timelinetone.json     (news tone per entity)
  - data/sources/calendar/spacex_launches_historical.json  (launch dates)

Output:
  - data/backtest/backtest_index.json
  - data/backtest/events/{event_slug}/metadata.json
  - data/backtest/events/{event_slug}/prices.parquet
  - data/backtest/events/{event_slug}/features.json

Usage:
    python scripts/build_backtest_dataset.py                 # All resolved events
    python scripts/build_backtest_dataset.py --tier gold     # Only gold-tier events
    python scripts/build_backtest_dataset.py --event-slug X  # Single event
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path so we can import src.*
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.features.feature_builder import TweetFeatureBuilder


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Input files
CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
MAPPING_PATH = PROJECT_DIR / "data" / "processed" / "xtracker_mapping.parquet"
PRICE_HISTORY_PATH = (
    PROJECT_DIR / "data" / "sources" / "polymarket" / "prices" / "price_history.parquet"
)

# Output
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
EVENTS_DIR = BACKTEST_DIR / "events"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


# ---------------------------------------------------------------------------
# Data loading (mapping, catalog, prices -- NOT feature sources)
# Feature sources are now loaded lazily inside TweetFeatureBuilder.
# ---------------------------------------------------------------------------
def load_mapping():
    """Load xtracker_mapping.parquet (one row per resolved event)."""
    if not MAPPING_PATH.exists():
        print("  WARNING: {} not found.".format(MAPPING_PATH))
        return pd.DataFrame()
    df = pd.read_parquet(MAPPING_PATH)
    return df


def load_catalog():
    """Load market_catalog.parquet (one row per bucket per event)."""
    if not CATALOG_PATH.exists():
        print("  WARNING: {} not found.".format(CATALOG_PATH))
        return pd.DataFrame()
    df = pd.read_parquet(CATALOG_PATH)
    return df


def load_prices():
    """Load price_history.parquet (hourly prices, all events)."""
    if not PRICE_HISTORY_PATH.exists():
        print("  WARNING: {} not found.".format(PRICE_HISTORY_PATH))
        return pd.DataFrame()
    df = pd.read_parquet(PRICE_HISTORY_PATH)
    return df


# ---------------------------------------------------------------------------
# Per-event output writers
# ---------------------------------------------------------------------------
def write_event_metadata(event_dir, mapping_row, catalog_event):
    """Write metadata.json for a single event."""
    event_dir.mkdir(parents=True, exist_ok=True)

    # Bucket list from catalog
    buckets = []
    for _, row in catalog_event.iterrows():
        buckets.append({
            "bucket_label": str(row["bucket_label"]),
            "lower_bound": int(row["lower_bound"]) if pd.notna(row["lower_bound"]) else None,
            "upper_bound": int(row["upper_bound"]) if pd.notna(row["upper_bound"]) else None,
            "is_winner": bool(row["is_winner"]),
            "price_yes": float(row["price_yes"]) if pd.notna(row["price_yes"]) else None,
        })

    # Handle count_in_winning_bucket which can be bool, None, or string "None"
    ciwb_raw = mapping_row.get("count_in_winning_bucket")
    if ciwb_raw is None or (isinstance(ciwb_raw, str) and ciwb_raw in ("None", "")):
        ciwb = None
    elif isinstance(ciwb_raw, bool):
        ciwb = ciwb_raw
    else:
        try:
            ciwb = bool(ciwb_raw) if pd.notna(ciwb_raw) else None
        except (TypeError, ValueError):
            ciwb = None

    metadata = {
        "event_id": str(mapping_row["event_id"]),
        "event_slug": str(mapping_row["event_slug"]),
        "event_title": str(mapping_row["event_title"]),
        "start_date": (
            str(mapping_row["start_date"])[:10]
            if pd.notna(mapping_row["start_date"]) else None
        ),
        "end_date": (
            str(mapping_row["end_date"])[:10]
            if pd.notna(mapping_row["end_date"]) else None
        ),
        "market_type": str(mapping_row["market_type"]),
        "duration_days": (
            int(mapping_row["duration_days"])
            if pd.notna(mapping_row["duration_days"]) else None
        ),
        "ground_truth_tier": str(mapping_row["ground_truth_tier"]),
        "xtracker_count": (
            float(mapping_row["xtracker_count"])
            if pd.notna(mapping_row["xtracker_count"]) else None
        ),
        "winning_bucket": str(mapping_row["winning_bucket"]),
        "winning_lower": (
            int(mapping_row["winning_lower"])
            if pd.notna(mapping_row["winning_lower"]) else None
        ),
        "winning_upper": (
            int(mapping_row["winning_upper"])
            if pd.notna(mapping_row["winning_upper"]) else None
        ),
        "count_in_winning_bucket": ciwb,
        "n_buckets": len(buckets),
        "buckets": buckets,
    }

    fpath = event_dir / "metadata.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return fpath


def write_event_prices(event_dir, event_id, prices_df):
    """Write prices.parquet for a single event (subset of price_history)."""
    event_dir.mkdir(parents=True, exist_ok=True)

    event_prices = prices_df[prices_df["event_id"] == str(event_id)].copy()

    fpath = event_dir / "prices.parquet"
    if event_prices.empty:
        # Write empty parquet with correct schema
        empty_df = pd.DataFrame(columns=[
            "bucket_label", "token_id", "timestamp", "price",
        ])
        empty_df.to_parquet(fpath, index=False, engine="pyarrow")
    else:
        # Select relevant columns
        cols = ["bucket_label", "token_id", "timestamp", "price"]
        out = event_prices[[c for c in cols if c in event_prices.columns]].copy()
        out = out.sort_values(["bucket_label", "timestamp"]).reset_index(drop=True)
        out.to_parquet(fpath, index=False, engine="pyarrow")

    return fpath, len(event_prices)


def write_event_features(event_dir, features_dict):
    """Write features.json for a single event."""
    event_dir.mkdir(parents=True, exist_ok=True)
    fpath = event_dir / "features.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(features_dict, f, indent=2, default=str)
    return fpath


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_backtest(tier_filter=None, slug_filter=None):
    """Build backtest dataset for resolved events."""
    sep = "=" * 70
    print(sep)
    print("Building Backtest Dataset (using TweetFeatureBuilder)")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Load mapping, catalog, prices
    # ------------------------------------------------------------------
    print("\n--- Loading input data ---")

    mapping = load_mapping()
    if mapping.empty:
        print("ERROR: No mapping data. Run build_xtracker_mapping.py first.")
        return
    print("  XTracker mapping:    {} events".format(len(mapping)))

    catalog = load_catalog()
    if catalog.empty:
        print("ERROR: No catalog data. Run build_market_catalog.py first.")
        return
    print("  Market catalog:      {} rows ({} events)".format(
        len(catalog), catalog["event_id"].nunique()))

    prices = load_prices()
    print("  Price history:       {} rows ({} events)".format(
        len(prices), prices["event_id"].nunique() if not prices.empty else 0))

    # ------------------------------------------------------------------
    # 2. Initialize the TweetFeatureBuilder (lazy-loads data sources)
    # ------------------------------------------------------------------
    print("\n--- Initializing TweetFeatureBuilder ---")
    builder = TweetFeatureBuilder(feature_group="full")
    print("  {}".format(builder.describe().replace("\n", "\n  ")))

    # ------------------------------------------------------------------
    # 3. Filter events
    # ------------------------------------------------------------------
    print("\n--- Filtering events ---")

    events = mapping.copy()

    if tier_filter:
        events = events[events["ground_truth_tier"] == tier_filter]
        print("  Filtered to tier '{}': {} events".format(tier_filter, len(events)))

    if slug_filter:
        events = events[events["event_slug"] == slug_filter]
        print("  Filtered to slug '{}': {} events".format(slug_filter, len(events)))

    if events.empty:
        print("  No events match the filter criteria.")
        return

    tier_counts = events["ground_truth_tier"].value_counts()
    print("  Events to process:   {}".format(len(events)))
    for tier in ["gold", "silver", "bronze"]:
        n = tier_counts.get(tier, 0)
        if n > 0:
            print("    {:8s}: {}".format(tier, n))

    # ------------------------------------------------------------------
    # 4. Process each event
    # ------------------------------------------------------------------
    print("\n--- Processing events ---")
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    index_entries = []
    stats = {
        "total": 0,
        "with_prices": 0,
        "with_temporal": 0,
        "with_media": 0,
        "with_calendar": 0,
        "with_market": 0,
        "with_cross": 0,
    }

    for _, event_row in tqdm(events.iterrows(), total=len(events),
                              desc="Building backtest", unit="event"):
        event_id = str(event_row["event_id"])
        event_slug = str(event_row["event_slug"])

        # Extract date strings
        start_date_str = None
        end_date_str = None
        if pd.notna(event_row["start_date"]):
            start_date_str = str(event_row["start_date"])[:10]
        if pd.notna(event_row["end_date"]):
            end_date_str = str(event_row["end_date"])[:10]

        # Get catalog buckets for this event
        catalog_event = catalog[catalog["event_id"] == event_id]

        # Event output directory
        event_dir = EVENTS_DIR / event_slug

        # --- Metadata ---
        meta_path = write_event_metadata(event_dir, event_row, catalog_event)

        # --- Prices ---
        prices_path, n_price_rows = write_event_prices(
            event_dir, event_id, prices
        )

        # --- Features (computed via TweetFeatureBuilder) ---
        event_metadata = {
            "event_id": event_id,
            "event_slug": event_slug,
            "start_date": start_date_str,
            "end_date": end_date_str,
        }
        all_features = builder.build_features(event_slug, event_metadata)

        # Write features
        feat_path = write_event_features(event_dir, all_features)

        # --- Track stats ---
        temporal = all_features.get("temporal", {})
        media = all_features.get("media", {})
        calendar = all_features.get("calendar", {})
        market = all_features.get("market", {})
        cross = all_features.get("cross", {})

        stats["total"] += 1
        if n_price_rows > 0:
            stats["with_prices"] += 1
        if temporal.get("rolling_avg_7d") is not None:
            stats["with_temporal"] += 1
        if any(v is not None for v in media.values()):
            stats["with_media"] += 1
        if calendar.get("launches_trailing_7d") is not None:
            stats["with_calendar"] += 1
        if market.get("crowd_implied_ev") is not None:
            stats["with_market"] += 1
        if any(v is not None for v in cross.values()):
            stats["with_cross"] += 1

        # --- Index entry ---
        index_entries.append({
            "event_id": event_id,
            "event_slug": event_slug,
            "event_title": str(event_row["event_title"]),
            "start_date": start_date_str,
            "end_date": end_date_str,
            "market_type": str(event_row["market_type"]),
            "ground_truth_tier": str(event_row["ground_truth_tier"]),
            "xtracker_count": (
                float(event_row["xtracker_count"])
                if pd.notna(event_row["xtracker_count"]) else None
            ),
            "winning_bucket": str(event_row["winning_bucket"]),
            "n_price_rows": n_price_rows,
            "has_temporal_features": temporal.get("rolling_avg_7d") is not None,
            "has_market_features": market.get("crowd_implied_ev") is not None,
            "has_cross_features": any(v is not None for v in cross.values()),
            "output_dir": str(event_dir),
        })

    # ------------------------------------------------------------------
    # 5. Write backtest index
    # ------------------------------------------------------------------
    print("\n--- Writing backtest index ---")

    index_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_events": stats["total"],
        "feature_builder": {
            "feature_group": builder.feature_group,
            "n_features": len(builder._required_features),
        },
        "filters": {
            "tier": tier_filter,
            "slug": slug_filter,
        },
        "coverage": {
            "with_prices": stats["with_prices"],
            "with_temporal_features": stats["with_temporal"],
            "with_media_features": stats["with_media"],
            "with_calendar_features": stats["with_calendar"],
            "with_market_features": stats["with_market"],
            "with_cross_features": stats["with_cross"],
        },
        "events": index_entries,
    }

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, default=str)
    print("  Saved: {}".format(INDEX_PATH))

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n{}".format(sep))
    print("Backtest Dataset Summary")
    print(sep)
    print("  Total events:            {}".format(stats["total"]))
    print("  With price data:         {}".format(stats["with_prices"]))
    print("  With temporal features:  {}".format(stats["with_temporal"]))
    print("  With media features:     {}".format(stats["with_media"]))
    print("  With calendar features:  {}".format(stats["with_calendar"]))
    print("  With market features:    {}".format(stats["with_market"]))
    print("  With cross features:     {}".format(stats["with_cross"]))
    print("")
    print("  Output directory:        {}".format(BACKTEST_DIR))
    print("  Index file:              {}".format(INDEX_PATH))

    # Tier breakdown
    tier_summary = {}
    for entry in index_entries:
        tier = entry["ground_truth_tier"]
        tier_summary[tier] = tier_summary.get(tier, 0) + 1
    print("")
    print("  Events by tier:")
    for tier in ["gold", "silver", "bronze"]:
        n = tier_summary.get(tier, 0)
        if n > 0:
            print("    {:8s}: {}".format(tier, n))

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build unified backtest dataset from market catalog, "
                    "XTracker mapping, price history, and feature sources."
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["gold", "silver", "bronze"],
        default=None,
        help="Filter to only events of this ground truth tier.",
    )
    parser.add_argument(
        "--event-slug",
        type=str,
        default=None,
        help="Process only a single event by its slug.",
    )
    args = parser.parse_args()

    build_backtest(tier_filter=args.tier, slug_filter=args.event_slug)

    print("\nDone!")


if __name__ == "__main__":
    main()
