"""
Fetch trade history for Elon tweet markets on Polymarket.

Uses the public Data API endpoint (no auth required):
    GET https://data-api.polymarket.com/trades?market={condition_id}&limit=N&offset=M

Produces per-event parquet files and a summary JSON with whale detection metrics.

Usage:
    python scripts/fetch_trade_history.py --mode active          # Active (unresolved) events
    python scripts/fetch_trade_history.py --mode backfill        # Resolved events
    python scripts/fetch_trade_history.py --limit 3              # Only first N events (testing)
    python scripts/fetch_trade_history.py --event-slug X         # Specific event
    python scripts/fetch_trade_history.py --dry-run              # Show what would be fetched
    python scripts/fetch_trade_history.py --force                # Re-fetch existing events
    python scripts/fetch_trade_history.py --resume               # Resume from checkpoint
    python scripts/fetch_trade_history.py --whale-threshold 500  # Custom whale threshold (USD)
"""
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_API_URL = "https://data-api.polymarket.com"
TRADES_ENDPOINT = f"{DATA_API_URL}/trades"

# API limits
MAX_PAGE_SIZE = 10_000  # API max per request
DEFAULT_PAGE_SIZE = 5_000  # Use large pages to minimise requests
MAX_OFFSET = 10_000  # API hard cap on offset
RATE_LIMIT_SLEEP = 0.5  # seconds between requests
MAX_RETRIES = 3

# Whale detection
DEFAULT_WHALE_THRESHOLD_USD = 500.0

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "sources" / "polymarket" / "trades"
CHECKPOINT_PATH = OUTPUT_DIR / "_checkpoint.json"
SUMMARY_PATH = OUTPUT_DIR / "trade_summary.json"
LOG_PATH = OUTPUT_DIR / "_fetch_log.txt"

# Graceful shutdown
_shutdown_requested = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Configure logging to both console and file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("trade_history")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

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


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_events": [], "last_updated": None}


def save_checkpoint(checkpoint):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------
def load_catalog():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Market catalog not found at {CATALOG_PATH}. "
            "Run: python scripts/build_market_catalog.py"
        )
    return pd.read_parquet(CATALOG_PATH)


# ---------------------------------------------------------------------------
# Trade fetching
# ---------------------------------------------------------------------------
def fetch_trades_for_condition(
    session, condition_id: str, logger, side: str = None
) -> list[dict]:
    """
    Fetch all trades for a single condition_id (one bucket market).

    The Data API supports:
        market: comma-separated condition IDs
        limit: max 10000
        offset: max 10000
        side: BUY or SELL

    Returns list of raw trade dicts from the API.
    """
    all_trades = []
    offset = 0

    while offset <= MAX_OFFSET:
        params = {
            "market": condition_id,
            "limit": DEFAULT_PAGE_SIZE,
            "offset": offset,
        }
        if side:
            params["side"] = side

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(TRADES_ENDPOINT, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = (2 ** attempt) * 2
                    logger.warning(
                        f"    Rate limited (429). Backing off {wait}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    logger.debug(
                        f"    404 for condition_id {condition_id[:16]}... (no trades)"
                    )
                    return all_trades

                resp.raise_for_status()
                data = resp.json()

                if not isinstance(data, list):
                    logger.warning(
                        f"    Unexpected response type for {condition_id[:16]}...: "
                        f"{type(data).__name__}"
                    )
                    return all_trades

                all_trades.extend(data)
                success = True

                # If we got fewer than page size, no more pages
                if len(data) < DEFAULT_PAGE_SIZE:
                    return all_trades

                break

            except Exception as e:
                wait = (2 ** attempt) * 1
                logger.warning(
                    f"    Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        if not success:
            logger.error(
                f"    Failed after {MAX_RETRIES} retries for condition {condition_id[:16]}..."
            )
            return all_trades

        offset += DEFAULT_PAGE_SIZE
        time.sleep(RATE_LIMIT_SLEEP)

    return all_trades


def normalize_trade(raw: dict, event_slug: str, bucket_label: str, token_id_yes: str) -> dict:
    """
    Normalize a raw trade record from the Data API into our schema.

    API returns fields like:
        proxyWallet, side, asset, conditionId, size, price,
        timestamp, title, slug, eventSlug, outcome, outcomeIndex,
        transactionHash, ...
    """
    price = _safe_float(raw.get("price"))
    size = _safe_float(raw.get("size"))
    trade_value_usd = price * size if price and size else 0.0

    # Parse timestamp - API returns Unix seconds (integer)
    ts_raw = raw.get("timestamp")
    timestamp = None
    if ts_raw is not None:
        try:
            ts_int = int(ts_raw)
            # Data API returns timestamps in seconds
            timestamp = datetime.fromtimestamp(ts_int, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            pass

    return {
        "timestamp": timestamp,
        "token_id": raw.get("asset", ""),
        "condition_id": raw.get("conditionId", ""),
        "event_slug": event_slug,
        "bucket_label": bucket_label,
        "side": (raw.get("side") or "").upper(),
        "price": price,
        "size": size,
        "trade_value_usd": trade_value_usd,
        "outcome": raw.get("outcome", ""),
        "outcome_index": raw.get("outcomeIndex"),
        "transaction_hash": raw.get("transactionHash", ""),
    }


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-event processing
# ---------------------------------------------------------------------------
def process_event(
    session, event_slug: str, event_group: pd.DataFrame, logger
) -> pd.DataFrame:
    """
    Fetch trades for all buckets in one event.

    Uses condition_id (not token_id) since the Data API filters by condition_id.
    """
    all_rows = []

    # Get unique buckets with their condition_ids
    buckets = event_group[
        ["bucket_label", "condition_id", "token_id_yes"]
    ].drop_duplicates(subset=["condition_id"])

    logger.info(
        f"  Event: {event_slug} ({len(buckets)} buckets)"
    )

    for _, bucket_row in buckets.iterrows():
        if _shutdown_requested:
            break

        bucket_label = bucket_row["bucket_label"]
        condition_id = bucket_row["condition_id"]
        token_id_yes = bucket_row.get("token_id_yes", "")

        if pd.isna(condition_id) or not condition_id:
            logger.warning(f"    Skipping bucket '{bucket_label}': no condition_id")
            continue

        raw_trades = fetch_trades_for_condition(session, str(condition_id), logger)

        if not raw_trades:
            logger.debug(f"    Bucket '{bucket_label}': no trades")
            continue

        for raw in raw_trades:
            row = normalize_trade(raw, event_slug, bucket_label, str(token_id_yes))
            all_rows.append(row)

        logger.debug(f"    Bucket '{bucket_label}': {len(raw_trades)} trades")
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        return pd.DataFrame(columns=[
            "timestamp", "token_id", "condition_id", "event_slug",
            "bucket_label", "side", "price", "size", "trade_value_usd",
            "outcome", "outcome_index", "transaction_hash",
        ])

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Summary / whale metrics
# ---------------------------------------------------------------------------
def compute_event_summary(
    df: pd.DataFrame, event_slug: str, whale_threshold: float
) -> dict:
    """Compute per-event summary with whale detection metrics."""
    if df.empty:
        return {
            "event_slug": event_slug,
            "total_trades": 0,
            "total_volume_usd": 0.0,
            "whale_trades": 0,
            "whale_volume_usd": 0.0,
            "per_bucket": {},
        }

    total_trades = len(df)
    total_volume = df["trade_value_usd"].sum()

    whale_mask = df["trade_value_usd"] >= whale_threshold
    whale_trades = int(whale_mask.sum())
    whale_volume = float(df.loc[whale_mask, "trade_value_usd"].sum())

    per_bucket = {}
    for bucket_label, bdf in df.groupby("bucket_label"):
        buy_mask = bdf["side"] == "BUY"
        sell_mask = bdf["side"] == "SELL"
        buy_vol = float(bdf.loc[buy_mask, "trade_value_usd"].sum())
        sell_vol = float(bdf.loc[sell_mask, "trade_value_usd"].sum())

        b_whale = bdf["trade_value_usd"] >= whale_threshold
        whale_buy = float(bdf.loc[b_whale & buy_mask, "trade_value_usd"].sum())
        whale_sell = float(bdf.loc[b_whale & sell_mask, "trade_value_usd"].sum())

        per_bucket[bucket_label] = {
            "trade_count": len(bdf),
            "volume_usd": round(buy_vol + sell_vol, 2),
            "buy_volume": round(buy_vol, 2),
            "sell_volume": round(sell_vol, 2),
            "buy_sell_ratio": round(buy_vol / sell_vol, 3) if sell_vol > 0 else None,
            "whale_count": int(b_whale.sum()),
            "whale_net_direction": round(whale_buy - whale_sell, 2),
        }

    return {
        "event_slug": event_slug,
        "total_trades": total_trades,
        "total_volume_usd": round(total_volume, 2),
        "whale_trades": whale_trades,
        "whale_volume_usd": round(whale_volume, 2),
        "per_bucket": per_bucket,
    }


# ---------------------------------------------------------------------------
# Main runners
# ---------------------------------------------------------------------------
def run(args, logger):
    logger.info("=" * 60)
    logger.info(f"Polymarket Trade History Fetch (mode={args.mode})")
    logger.info("=" * 60)

    # Load catalog
    logger.info("\nLoading market catalog...")
    catalog = load_catalog()
    logger.info(f"  Catalog: {len(catalog)} bucket rows")

    # Filter by mode
    if args.event_slug:
        catalog = catalog[catalog["event_slug"] == args.event_slug]
        if catalog.empty:
            logger.error(f"  No rows found for event slug '{args.event_slug}'")
            return
        logger.info(f"  Filtered to event: {args.event_slug} ({len(catalog)} buckets)")
    elif args.mode == "active":
        catalog = catalog[catalog["is_resolved"] == False]
        logger.info(f"  Active (unresolved) events: {len(catalog)} bucket rows")
    else:  # backfill
        catalog = catalog[catalog["is_resolved"] == True]
        logger.info(f"  Resolved events: {len(catalog)} bucket rows")

    if catalog.empty:
        logger.warning("  No events to process.")
        return

    # Group by event slug
    event_groups = dict(list(catalog.groupby("event_slug")))
    event_slugs = sorted(event_groups.keys())

    # Apply --limit
    if args.limit and args.limit > 0:
        event_slugs = event_slugs[:args.limit]
        logger.info(f"  Limited to first {args.limit} events")

    logger.info(f"  Events to process: {len(event_slugs)}")

    # Checkpoint / skip logic
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get("completed_events", []))

    events_to_fetch = []
    for slug in event_slugs:
        parquet_path = OUTPUT_DIR / f"{slug}.parquet"

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

    # Dry run
    if args.dry_run:
        logger.info(f"\n[DRY RUN] Would fetch trades for {len(events_to_fetch)} events:")
        for slug in events_to_fetch:
            n_buckets = len(event_groups[slug])
            logger.info(f"  - {slug} ({n_buckets} buckets)")
        logger.info("\nRe-run without --dry-run to execute.")
        return

    logger.info(f"\nFetching trades for {len(events_to_fetch)} events...\n")

    # Setup session
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "ElonTweetMarkets/1.0",
    })

    # Process events
    t_start = time.time()
    total_trades = 0
    events_with_no_data = []
    events_processed = 0
    all_summaries = {}

    # Load existing summary if present
    if SUMMARY_PATH.exists():
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
            if isinstance(existing_summary, dict) and "events" in existing_summary:
                all_summaries = existing_summary["events"]
        except (json.JSONDecodeError, KeyError):
            pass

    progress = tqdm(events_to_fetch, desc="Events", unit="event")
    for slug in progress:
        if _shutdown_requested:
            logger.info("\nShutdown requested. Saving checkpoint...")
            break

        progress.set_postfix_str(slug[:40])
        event_group = event_groups[slug]

        df = process_event(session, slug, event_group, logger)

        # Save per-event parquet
        parquet_path = OUTPUT_DIR / f"{slug}.parquet"
        if df.empty:
            events_with_no_data.append(slug)
            logger.warning(f"  -> No trades for {slug}")
            df.to_parquet(parquet_path, index=False)
        else:
            df.to_parquet(parquet_path, index=False)
            n_trades = len(df)
            total_trades += n_trades
            logger.info(
                f"  -> Saved {parquet_path.name}: {n_trades:,} trades, "
                f"${df['trade_value_usd'].sum():,.0f} volume"
            )

        # Compute and store summary
        summary = compute_event_summary(df, slug, args.whale_threshold)
        all_summaries[slug] = summary

        # Update checkpoint
        completed.add(slug)
        events_processed += 1
        checkpoint["completed_events"] = sorted(completed)
        save_checkpoint(checkpoint)

    # Save combined summary
    summary_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "whale_threshold_usd": args.whale_threshold,
        "total_events": len(all_summaries),
        "events": all_summaries,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # Final report
    logger.info(f"\n{'=' * 60}")
    logger.info("Trade History Fetch Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Events processed this run:   {events_processed}")
    logger.info(f"  Total events completed:       {len(completed)}")
    logger.info(f"  Total trades fetched:         {total_trades:,}")
    logger.info(f"  Events with zero trades:      {len(events_with_no_data)}")
    if events_with_no_data:
        for slug in events_with_no_data[:10]:
            logger.info(f"    - {slug}")
        if len(events_with_no_data) > 10:
            logger.info(f"    ... and {len(events_with_no_data) - 10} more")

    total_bytes = sum(
        f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet") if f.is_file()
    )
    if total_bytes > 1_000_000:
        logger.info(f"  Total data size:              {total_bytes / 1_000_000:.1f} MB")
    else:
        logger.info(f"  Total data size:              {total_bytes / 1_000:.1f} KB")

    logger.info(f"  Whale threshold:              ${args.whale_threshold:.0f}")
    logger.info(f"  Time elapsed:                 {elapsed:.1f}s")
    logger.info(f"  Output directory:             {OUTPUT_DIR}")
    logger.info(f"  Summary:                      {SUMMARY_PATH}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch trade history for Elon tweet markets on Polymarket."
    )
    parser.add_argument(
        "--mode", choices=["active", "backfill"], default="active",
        help="Fetch for active (unresolved) or resolved events (default: active)."
    )
    parser.add_argument(
        "--event-slug", type=str, default=None,
        help="Specific event slug to fetch."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N events (for testing)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without making API calls."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if parquet already exists."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skip completed events)."
    )
    parser.add_argument(
        "--whale-threshold", type=float, default=DEFAULT_WHALE_THRESHOLD_USD,
        help=f"USD threshold for whale trades (default: ${DEFAULT_WHALE_THRESHOLD_USD:.0f})."
    )

    args = parser.parse_args()
    logger = setup_logging()

    signal.signal(signal.SIGINT, handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_shutdown)

    run(args, logger)


if __name__ == "__main__":
    main()
