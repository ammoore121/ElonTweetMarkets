"""
Fetch CLOB price history for all Elon tweet market buckets.

Reads market catalog, fetches hourly prices for each bucket's YES token,
saves per-event parquet files for backtesting.

Usage:
    python scripts/fetch_clob_prices.py                          # Backfill resolved events
    python scripts/fetch_clob_prices.py --event-slug X           # Specific event
    python scripts/fetch_clob_prices.py --resume                 # Resume from checkpoint
    python scripts/fetch_clob_prices.py --force                  # Re-fetch even if file exists
    python scripts/fetch_clob_prices.py --mode daily             # Fetch last 48h for active events
"""
import requests
import json
import time
import argparse
import logging
import signal
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CLOB_URL = "https://clob.polymarket.com"
PRICES_ENDPOINT = f"{CLOB_URL}/prices-history"
MAX_CHUNK_DAYS = 7
RATE_LIMIT_SLEEP = 0.3
MAX_RETRIES = 3

PROJECT_DIR = Path(__file__).parent.parent
CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "sources" / "polymarket" / "prices"
BACKFILL_DIR = OUTPUT_DIR / "events"
DAILY_DIR = OUTPUT_DIR / "daily"
CHECKPOINT_PATH = BACKFILL_DIR / "_checkpoint.json"
LOG_PATH = OUTPUT_DIR / "_fetch_log.txt"

# Graceful shutdown flag
_shutdown_requested = False


def setup_logging():
    """Configure logging to both console and file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("clob_prices")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


def handle_shutdown(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested. Saving checkpoint after current event...")


def load_checkpoint():
    """Load checkpoint file if it exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_events": [], "last_updated": None, "total_events": 0, "total_fetched": 0}


def save_checkpoint(checkpoint):
    """Save checkpoint to disk."""
    BACKFILL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def load_catalog():
    """Load market catalog from parquet."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Market catalog not found at {CATALOG_PATH}. "
            "Run the catalog build step first."
        )
    return pd.read_parquet(CATALOG_PATH)


def to_unix_ts(dt):
    """Convert datetime to Unix timestamp (seconds)."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def fetch_price_history(session, token_id, start_ts, end_ts, logger):
    """
    Fetch price history for a single token in 7-day chunks.

    Returns list of {'t': unix_ts, 'p': price} records.
    """
    all_points = []
    current_start = start_ts

    while current_start < end_ts:
        chunk_end = min(current_start + MAX_CHUNK_DAYS * 86400, end_ts)

        params = {
            "market": token_id,
            "startTs": current_start,
            "endTs": chunk_end,
            "fidelity": 60,
        }

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(PRICES_ENDPOINT, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = (2 ** attempt) * 2
                    logger.warning(
                        f"    Rate limited (429). Backing off {wait}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                history = data.get("history", [])
                all_points.extend(history)
                success = True
                break

            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    # Already handled above via continue
                    pass
                else:
                    logger.warning(f"    HTTP error for token {token_id[:16]}...: {e}")
                    success = True  # Don't retry on non-429 HTTP errors
                    break
            except requests.exceptions.RequestException as e:
                wait = (2 ** attempt) * 1
                logger.warning(
                    f"    Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        if not success:
            logger.error(f"    Failed after {MAX_RETRIES} retries for token {token_id[:16]}...")

        current_start = chunk_end
        time.sleep(RATE_LIMIT_SLEEP)

    return all_points

def process_event(session, event_slug, event_group, logger):
    """
    Fetch price history for all buckets in one event.

    Returns a DataFrame with columns:
        event_id, event_slug, bucket_label, token_id, timestamp, price
    """
    rows = []

    # Compute fetch window with padding
    start_dates = pd.to_datetime(event_group["start_date"])
    end_dates = pd.to_datetime(event_group["end_date"])
    fetch_start = start_dates.min() - timedelta(days=2)
    fetch_end = end_dates.max() + timedelta(days=1)

    # Guard against events with missing date metadata
    if pd.isna(fetch_start) or pd.isna(fetch_end):
        logger.warning(
            f"  Skipping event '{event_slug}': missing start_date or end_date in catalog"
        )
        return pd.DataFrame(columns=["event_id", "event_slug", "bucket_label",
                                      "token_id", "timestamp", "price"])

    start_ts = to_unix_ts(fetch_start)
    end_ts = to_unix_ts(fetch_end)

    event_id = event_group["event_id"].iloc[0]
    buckets = event_group[["bucket_label", "token_id_yes"]].drop_duplicates()

    logger.info(
        f"  Event: {event_slug} ({len(buckets)} buckets, "
        f"{fetch_start.strftime('%Y-%m-%d')} to {fetch_end.strftime('%Y-%m-%d')})"
    )

    for _, bucket_row in buckets.iterrows():
        if _shutdown_requested:
            break

        bucket_label = bucket_row["bucket_label"]
        token_id = bucket_row["token_id_yes"]

        if pd.isna(token_id) or not token_id:
            logger.warning(f"    Skipping bucket '{bucket_label}': no token_id_yes")
            continue

        try:
            points = fetch_price_history(session, str(token_id), start_ts, end_ts, logger)
        except Exception as e:
            logger.error(f"    Error fetching bucket '{bucket_label}': {e}")
            continue

        if not points:
            logger.debug(f"    Bucket '{bucket_label}': no price data")
            continue

        for pt in points:
            rows.append({
                "event_id": event_id,
                "event_slug": event_slug,
                "bucket_label": bucket_label,
                "token_id": token_id,
                "timestamp": datetime.fromtimestamp(pt["t"], tz=timezone.utc),
                "price": float(pt["p"]),
            })

    if not rows:
        return pd.DataFrame(columns=["event_id", "event_slug", "bucket_label",
                                      "token_id", "timestamp", "price"])

    return pd.DataFrame(rows)

def process_event_daily(session, event_slug, event_group, start_ts, end_ts, logger):
    """
    Fetch price history for all buckets in one event over a fixed time window.

    Similar to process_event() but uses explicit start/end timestamps (e.g. 48h window)
    and adds extra columns: market_type, is_resolved, fetch_date.

    Returns a DataFrame with columns:
        event_id, event_slug, bucket_label, token_id, timestamp, price,
        market_type, is_resolved, fetch_date
    """
    rows = []

    event_id = event_group["event_id"].iloc[0]
    # Extract market_type from the event group if available
    market_type = (
        event_group["market_type"].iloc[0]
        if "market_type" in event_group.columns
        else "unknown"
    )
    buckets = event_group[["bucket_label", "token_id_yes"]].drop_duplicates()
    today_str = date.today().isoformat()

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    logger.info(
        f"  Event: {event_slug} ({len(buckets)} buckets, "
        f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')})"
    )

    for _, bucket_row in buckets.iterrows():
        if _shutdown_requested:
            break

        bucket_label = bucket_row["bucket_label"]
        token_id = bucket_row["token_id_yes"]

        if pd.isna(token_id) or not token_id:
            logger.warning(f"    Skipping bucket '{bucket_label}': no token_id_yes")
            continue

        try:
            points = fetch_price_history(session, str(token_id), start_ts, end_ts, logger)
        except Exception as e:
            logger.error(f"    Error fetching bucket '{bucket_label}': {e}")
            continue

        if not points:
            logger.debug(f"    Bucket '{bucket_label}': no price data")
            continue

        for pt in points:
            rows.append({
                "event_id": event_id,
                "event_slug": event_slug,
                "bucket_label": bucket_label,
                "token_id": token_id,
                "timestamp": datetime.fromtimestamp(pt["t"], tz=timezone.utc),
                "price": float(pt["p"]),
                "market_type": market_type,
                "is_resolved": False,
                "fetch_date": today_str,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "event_id", "event_slug", "bucket_label", "token_id",
            "timestamp", "price", "market_type", "is_resolved", "fetch_date",
        ])

    return pd.DataFrame(rows)

def run_backfill(args, logger):
    """Run backfill mode: fetch full price history for resolved events."""
    logger.info("=" * 60)
    logger.info("Polymarket CLOB Price History Fetch (backfill)")
    logger.info("=" * 60)

    # Load catalog
    logger.info("\nLoading market catalog...")
    catalog = load_catalog()
    logger.info(f"  Catalog: {len(catalog)} bucket rows")

    # Filter to resolved events only (unless specific slug requested)
    if args.event_slug:
        catalog = catalog[catalog["event_slug"] == args.event_slug]
        if catalog.empty:
            logger.error(f"  No rows found for event slug '{args.event_slug}'")
            return
        logger.info(f"  Filtered to event: {args.event_slug} ({len(catalog)} buckets)")
    else:
        resolved = catalog[catalog["is_resolved"] == True]
        if resolved.empty:
            logger.warning("  No resolved events found. Fetching all events instead.")
        else:
            catalog = resolved
            logger.info(f"  Filtered to resolved events: {len(catalog)} bucket rows")

    # Group by event slug
    event_groups = dict(list(catalog.groupby("event_slug")))
    event_slugs = sorted(event_groups.keys())
    logger.info(f"  Events to process: {len(event_slugs)}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed_events", []))

    # Determine which events to skip
    BACKFILL_DIR.mkdir(parents=True, exist_ok=True)
    events_to_fetch = []
    for slug in event_slugs:
        parquet_path = BACKFILL_DIR / f"{slug}_prices.parquet"

        if args.resume and slug in completed:
            logger.info(f"  [SKIP/checkpoint] {slug}")
            continue
        elif not args.force and not args.resume and parquet_path.exists():
            logger.info(f"  [SKIP/exists] {slug}")
            if slug not in completed:
                completed.add(slug)
            continue
        else:
            events_to_fetch.append(slug)

    if not events_to_fetch:
        logger.info("\nNo events to fetch. Use --force to re-fetch.")
        return

    logger.info(f"\nFetching price data for {len(events_to_fetch)} events...\n")

    # Setup session
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "ElonTweetMarkets/1.0",
    })

    # Tracking stats
    t_start = time.time()
    total_observations = 0
    events_with_no_data = []
    events_processed = 0

    # Update checkpoint total
    checkpoint["total_events"] = len(event_slugs)

    # Process events
    progress = tqdm(events_to_fetch, desc="Events", unit="event")
    for slug in progress:
        if _shutdown_requested:
            logger.info("\nShutdown requested. Saving checkpoint...")
            break

        progress.set_postfix_str(slug[:40])
        event_group = event_groups[slug]

        df = process_event(session, slug, event_group, logger)

        # Save parquet
        BACKFILL_DIR.mkdir(parents=True, exist_ok=True)
        parquet_path = BACKFILL_DIR / f"{slug}_prices.parquet"

        if df.empty:
            events_with_no_data.append(slug)
            logger.warning(f"  -> No price data for {slug}")
            # Save empty parquet to mark as attempted
            df.to_parquet(parquet_path, index=False)
        else:
            df.to_parquet(parquet_path, index=False)
            n_obs = len(df)
            total_observations += n_obs
            logger.info(f"  -> Saved {parquet_path.name}: {n_obs:,} observations")

        # Update checkpoint
        completed.add(slug)
        events_processed += 1
        checkpoint["completed_events"] = sorted(completed)
        checkpoint["total_fetched"] = len(completed)
        save_checkpoint(checkpoint)

    elapsed = time.time() - t_start

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("CLOB Price Fetch Summary (backfill)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Events processed this run:   {events_processed}")
    logger.info(f"  Total events completed:       {len(completed)} / {len(event_slugs)}")
    logger.info(f"  Total price observations:     {total_observations:,}")
    logger.info(f"  Events with zero price data:  {len(events_with_no_data)}")
    if events_with_no_data:
        for slug in events_with_no_data:
            logger.info(f"    - {slug}")

    # Estimate data size
    total_bytes = sum(
        f.stat().st_size for f in BACKFILL_DIR.glob("*_prices.parquet") if f.is_file()
    )
    if total_bytes > 1_000_000:
        logger.info(f"  Total data size:              {total_bytes / 1_000_000:.1f} MB")
    else:
        logger.info(f"  Total data size:              {total_bytes / 1_000:.1f} KB")

    logger.info(f"  Time elapsed:                 {elapsed:.1f}s")
    logger.info(f"  Output directory:             {BACKFILL_DIR}")
    logger.info(f"  Checkpoint:                   {CHECKPOINT_PATH}")
    logger.info("=" * 60)

def run_daily(args, logger):
    """Run daily mode: fetch last 48h of prices for all active (unresolved) events."""
    logger.info("=" * 60)
    logger.info("Polymarket CLOB Price History Fetch (daily)")
    logger.info("=" * 60)

    # Load catalog
    logger.info("\nLoading market catalog...")
    catalog = load_catalog()
    logger.info(f"  Catalog: {len(catalog)} bucket rows")

    # Filter to unresolved events only
    active = catalog[catalog["is_resolved"] == False]
    if active.empty:
        logger.warning("  No active (unresolved) events found. Nothing to fetch.")
        return
    logger.info(f"  Active (unresolved) events: {len(active)} bucket rows")

    # Group by event slug
    event_groups = dict(list(active.groupby("event_slug")))
    event_slugs = sorted(event_groups.keys())
    logger.info(f"  Events to process: {len(event_slugs)}")

    # Compute 48h window: now - 48h to now
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=48)
    start_ts = to_unix_ts(window_start)
    end_ts = to_unix_ts(now)
    logger.info(
        f"  Window: {window_start.strftime('%Y-%m-%d %H:%M')} to "
        f"{now.strftime('%Y-%m-%d %H:%M')} UTC"
    )

    # Setup session
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "ElonTweetMarkets/1.0",
    })

    # Tracking stats
    t_start = time.time()
    all_dfs = []
    total_observations = 0
    events_with_no_data = []
    events_processed = 0

    # Process events
    progress = tqdm(event_slugs, desc="Events", unit="event")
    for slug in progress:
        if _shutdown_requested:
            logger.info("\nShutdown requested. Stopping...")
            break

        progress.set_postfix_str(slug[:40])
        event_group = event_groups[slug]

        df = process_event_daily(session, slug, event_group, start_ts, end_ts, logger)

        if df.empty:
            events_with_no_data.append(slug)
            logger.warning(f"  -> No price data for {slug}")
        else:
            all_dfs.append(df)
            n_obs = len(df)
            total_observations += n_obs
            logger.info(f"  -> {slug}: {n_obs:,} observations")

        events_processed += 1

    # Combine and save single daily file
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()
    output_path = DAILY_DIR / f"prices_{today_str}.parquet"

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(output_path, index=False)
        logger.info(f"\n-> Saved {output_path.name}: {len(combined):,} total observations")
    else:
        # Save empty parquet with correct schema
        empty = pd.DataFrame(columns=[
            "event_id", "event_slug", "bucket_label", "token_id",
            "timestamp", "price", "market_type", "is_resolved", "fetch_date",
        ])
        empty.to_parquet(output_path, index=False)
        logger.info(f"\n-> Saved empty {output_path.name}: no observations")

    elapsed = time.time() - t_start

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("CLOB Price Fetch Summary (daily)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Events processed:             {events_processed}")
    logger.info(f"  Total price observations:     {total_observations:,}")
    logger.info(f"  Events with zero price data:  {len(events_with_no_data)}")
    if events_with_no_data:
        for slug in events_with_no_data:
            logger.info(f"    - {slug}")
    logger.info(f"  Time elapsed:                 {elapsed:.1f}s")
    logger.info(f"  Output file:                  {output_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fetch CLOB price history for Elon tweet market buckets.")
    parser.add_argument("--mode", choices=["backfill", "daily"], default="backfill", help="Fetch mode")
    parser.add_argument("--event-slug", type=str, default=None, help="Specific event slug (backfill only).")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (backfill only).")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if exists (backfill only).")
    args = parser.parse_args()
    logger = setup_logging()
    signal.signal(signal.SIGINT, handle_shutdown)
    if hasattr(signal, "SIGTERM"): signal.signal(signal.SIGTERM, handle_shutdown)
    if args.mode == "daily": run_daily(args, logger)
    else: run_backfill(args, logger)

if __name__ == "__main__":
    main()
