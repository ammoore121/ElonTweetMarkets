"""Fetch current market odds for active Elon tweet count events on Polymarket.

Captures live bucket prices and saves as MarketOdds snapshots.
Deduplicates unchanged prices automatically.

Usage:
    python scripts/fetch_current_odds.py
    python scripts/fetch_current_odds.py --refresh-catalog
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data_sources.polymarket.client import PolymarketClient
from src.paper_trading.schemas import MarketOdds
from src.paper_trading.tracker import PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"


def load_active_events(catalog_path: Path) -> pd.DataFrame:
    """Load active (unresolved, not yet ended) events from catalog."""
    if not catalog_path.exists():
        logger.error("Market catalog not found at %s", catalog_path)
        return pd.DataFrame()

    df = pd.read_parquet(catalog_path)
    now = datetime.now(timezone.utc)

    # Filter to unresolved events with end_date in the future
    if "is_resolved" in df.columns:
        df = df[df["is_resolved"] == False]

    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
        df = df[df["end_date"] > now]

    return df


def compute_implied_ev(bucket_prices: dict, buckets_df: pd.DataFrame) -> float:
    """Compute probability-weighted expected tweet count from bucket prices."""
    ev = 0.0
    for _, row in buckets_df.iterrows():
        label = row["bucket_label"]
        price = bucket_prices.get(label, 0.0)
        lower = int(row["lower_bound"])
        upper = int(row["upper_bound"])
        if upper >= 99999:
            # Open-ended: estimate midpoint from typical bucket width
            widths = []
            for _, r2 in buckets_df.iterrows():
                u2 = int(r2["upper_bound"])
                if u2 < 99999:
                    widths.append(u2 - int(r2["lower_bound"]))
            typical = sum(widths) / len(widths) if widths else 25
            mid = lower + typical / 2
        elif lower <= 0:
            mid = upper / 2
        else:
            mid = (lower + upper) / 2
        ev += price * mid
    return ev


def fetch_odds_for_event(
    client: PolymarketClient,
    event_slug: str,
    event_id: str,
    market_type: str,
    buckets_df: pd.DataFrame,
) -> MarketOdds | None:
    """Fetch live prices for all buckets in one event."""
    bucket_prices = {}
    errors = 0

    for _, row in buckets_df.iterrows():
        token_id = row.get("token_id_yes", "")
        label = row["bucket_label"]
        if not token_id:
            logger.warning("  No token_id for bucket %s, skipping", label)
            continue

        try:
            price = client.get_price(str(token_id), side="buy")
            bucket_prices[label] = float(price)
        except Exception as e:
            # 404 = Polymarket delisted this bucket (near-zero probability)
            bucket_prices[label] = 0.0
            errors += 1

        time.sleep(0.1)  # Rate limit

    # Skip if no live prices at all (all buckets returned 404)
    live_prices = {k: v for k, v in bucket_prices.items() if v > 0}
    if not live_prices:
        logger.warning("  No live prices fetched for %s (all 404)", event_slug)
        return None

    # Compute implied EV
    implied_ev = compute_implied_ev(bucket_prices, buckets_df)

    odds = MarketOdds(
        event_slug=event_slug,
        event_id=str(event_id),
        market_type=market_type,
        bucket_prices=bucket_prices,
        n_buckets=len(bucket_prices),
        implied_ev=implied_ev,
    )
    return odds


def main():
    parser = argparse.ArgumentParser(description="Fetch current market odds")
    parser.add_argument(
        "--refresh-catalog", action="store_true",
        help="Re-run fetch_elon_markets + build_market_catalog first",
    )
    args = parser.parse_args()

    if args.refresh_catalog:
        logger.info("Refreshing market catalog...")
        scripts_dir = PROJECT_DIR / "scripts"
        subprocess.run([sys.executable, str(scripts_dir / "fetch_elon_markets.py")], check=True)
        subprocess.run([sys.executable, str(scripts_dir / "build_market_catalog.py")], check=True)
        logger.info("Catalog refreshed.")

    # Load active events
    active_df = load_active_events(CATALOG_PATH)
    if active_df.empty:
        logger.info("No active events found.")
        return

    # Group by event
    event_groups = active_df.groupby("event_slug")
    logger.info("Found %d active events with %d total buckets",
                len(event_groups), len(active_df))

    # Initialize
    client = PolymarketClient()
    tracker = PerformanceTracker(
        odds_dir=str(PROJECT_DIR / "data" / "odds"),
        signals_dir=str(PROJECT_DIR / "data" / "signals"),
    )

    recorded = 0
    skipped = 0

    for event_slug, buckets_df in event_groups:
        first = buckets_df.iloc[0]
        event_id = str(first.get("event_id", ""))
        market_type = str(first.get("market_type", ""))
        end_date = first.get("end_date", "")

        logger.info("Fetching odds for %s (%s, ends %s, %d buckets)",
                     event_slug, market_type, str(end_date)[:10], len(buckets_df))

        odds = fetch_odds_for_event(client, str(event_slug), event_id, market_type, buckets_df)
        if odds is None:
            continue

        odds_id, was_recorded = tracker.record_odds(odds, check_changed=True)
        if was_recorded:
            recorded += 1
            logger.info("  Recorded odds %s (implied EV=%.1f)", odds_id, odds.implied_ev)
        else:
            skipped += 1
            logger.info("  Prices unchanged, skipped")

    # Summary
    print()
    print("=" * 55)
    print("  ODDS CAPTURE SUMMARY")
    print("=" * 55)
    print("  Active events:     {:>4d}".format(len(event_groups)))
    print("  New snapshots:     {:>4d}".format(recorded))
    print("  Unchanged/skipped: {:>4d}".format(skipped))
    print("=" * 55)


if __name__ == "__main__":
    main()
