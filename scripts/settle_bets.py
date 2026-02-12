"""Settle completed paper trading bets using XTracker as the resolution oracle.

Checks all open betslips, determines if their events have ended,
fetches actual tweet counts from XTracker, and settles bets.

Usage:
    python scripts/settle_bets.py
    python scripts/settle_bets.py --force-event EVENT_SLUG
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data_sources.xtracker.client import XTrackerClient
from src.paper_trading.tracker import PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
ELON_USER_ID = "c4e2a911-36ec-4453-8a39-1edb5e6b2969"


def load_catalog() -> pd.DataFrame:
    """Load market catalog for event metadata."""
    if not CATALOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(CATALOG_PATH)


def get_event_metadata(catalog_df: pd.DataFrame, event_slug: str) -> dict:
    """Extract event metadata from catalog."""
    if catalog_df.empty:
        return {}
    rows = catalog_df[catalog_df["event_slug"] == event_slug]
    if rows.empty:
        return {}
    first = rows.iloc[0]
    return {
        "start_date": str(first.get("start_date", ""))[:10],
        "end_date": str(first.get("end_date", ""))[:10],
        "market_type": str(first.get("market_type", "")),
        "buckets": rows[["bucket_label", "lower_bound", "upper_bound"]].to_dict("records"),
    }


def count_to_winning_bucket(count: int, buckets: list) -> str:
    """Map a tweet count to the winning bucket label."""
    for b in sorted(buckets, key=lambda x: int(x["lower_bound"])):
        lower = int(b["lower_bound"])
        upper = int(b["upper_bound"])
        if lower <= count <= upper:
            return b["bucket_label"]
    # If count exceeds all buckets, return the last (open-ended) bucket
    if buckets:
        last = max(buckets, key=lambda x: int(x["lower_bound"]))
        return last["bucket_label"]
    return ""


def fetch_xtracker_count(client: XTrackerClient, start_date: str, end_date: str) -> int | None:
    """Fetch total tweet count for a date range from XTracker.

    The XTracker API returns records like:
        {"id": ..., "userId": ..., "date": "...", "type": "daily",
         "data": {"count": 55, "cumulative": 0, "trackingId": "..."}}

    The response is either a dict with a "data" key containing a list of records,
    or directly a list of records.
    """
    try:
        result = client.get_daily_metrics(ELON_USER_ID, start_date, end_date)
        if not result:
            return None

        # Normalize: API may return {data: [...]} or just [...]
        if isinstance(result, dict):
            records = result.get("data", result)
            if isinstance(records, dict):
                # Single record wrapped in dict
                records = [records]
        elif isinstance(result, list):
            records = result
        else:
            return None

        if not isinstance(records, list) or len(records) == 0:
            return None

        total = 0
        for record in records:
            # Each record has a nested "data" dict with "count"
            data = record.get("data", record)
            if isinstance(data, dict):
                count = data.get("count", 0)
            else:
                count = record.get("count", 0)
            if count is not None:
                total += int(count)

        return total if total > 0 else None
    except Exception as e:
        logger.warning("XTracker fetch failed for %s to %s: %s", start_date, end_date, e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Settle completed bets")
    parser.add_argument("--force-event", type=str, default=None,
                        help="Force settle a specific event (even if end_date not passed)")
    args = parser.parse_args()

    tracker = PerformanceTracker(
        odds_dir=str(PROJECT_DIR / "data" / "odds"),
        signals_dir=str(PROJECT_DIR / "data" / "signals"),
    )

    open_bets = tracker.get_open_betslips()
    if not open_bets:
        logger.info("No open betslips to settle.")
        tracker.print_performance()
        return

    logger.info("Found %d open betslips", len(open_bets))

    catalog_df = load_catalog()
    client = XTrackerClient()
    now = datetime.now(timezone.utc)

    # Group betslips by event
    events_to_settle = {}
    for bet in open_bets:
        slug = bet.event_slug
        if slug not in events_to_settle:
            events_to_settle[slug] = []
        events_to_settle[slug].append(bet)

    settled_count = 0
    skipped_count = 0

    for event_slug, bets in events_to_settle.items():
        meta = get_event_metadata(catalog_df, event_slug)
        if not meta:
            logger.warning("No catalog metadata for %s, skipping", event_slug)
            skipped_count += len(bets)
            continue

        end_date_str = meta["end_date"]
        if not end_date_str:
            logger.warning("No end_date for %s, skipping", event_slug)
            skipped_count += len(bets)
            continue

        # Check if event has ended
        try:
            end_dt = pd.Timestamp(end_date_str, tz="UTC")
        except Exception:
            logger.warning("Invalid end_date '%s' for %s", end_date_str, event_slug)
            skipped_count += len(bets)
            continue

        if end_dt > now and args.force_event != event_slug:
            logger.info("Event %s still active (ends %s), skipping %d bets",
                         event_slug, end_date_str, len(bets))
            skipped_count += len(bets)
            continue

        # Fetch XTracker count
        start_date_str = meta["start_date"]
        logger.info("Settling %s (%s to %s, %d bets)...",
                     event_slug, start_date_str, end_date_str, len(bets))

        xtracker_count = fetch_xtracker_count(client, start_date_str, end_date_str)
        if xtracker_count is None:
            logger.warning("  XTracker data not available yet for %s, skipping", event_slug)
            skipped_count += len(bets)
            continue

        # Map count to winning bucket
        winning_bucket = count_to_winning_bucket(xtracker_count, meta["buckets"])
        if not winning_bucket:
            logger.warning("  Could not map count %d to a bucket for %s", xtracker_count, event_slug)
            skipped_count += len(bets)
            continue

        logger.info("  XTracker count: %d -> winning bucket: %s", xtracker_count, winning_bucket)

        # Settle all bets for this event
        try:
            settlement_ids = tracker.settle_event(
                event_slug, winning_bucket, xtracker_count=xtracker_count
            )
            settled_count += len(settlement_ids)
            for sid, bet in zip(settlement_ids, bets):
                won = bet.bucket_label == winning_bucket
                result = "WON" if won else "LOST"
                logger.info("  %s: bet on '%s' -> %s ($%.2f wagered)",
                             result, bet.bucket_label, sid, bet.wager)
        except Exception as e:
            logger.error("  Settlement failed for %s: %s", event_slug, e)
            skipped_count += len(bets)

    # Summary
    print()
    print("=" * 55)
    print("  SETTLEMENT SUMMARY")
    print("=" * 55)
    print("  Events checked:    {:>4d}".format(len(events_to_settle)))
    print("  Bets settled:      {:>4d}".format(settled_count))
    print("  Bets skipped:      {:>4d}".format(skipped_count))
    print("=" * 55)

    tracker.print_performance()


if __name__ == "__main__":
    main()
