"""Fetch order book snapshots for active Elon tweet markets on Polymarket.

Captures bid/ask depth, spread, mid price, and book imbalance for each bucket
in active (unresolved, not yet ended) events.

Usage:
    python scripts/fetch_orderbook.py              # Fetch order books for active markets
    python scripts/fetch_orderbook.py --dry-run    # List what would be fetched
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "sources" / "polymarket" / "orderbook"

CLOB_URL = "https://clob.polymarket.com"
RATE_LIMIT_SLEEP = 0.5
MAX_RETRIES = 3


def load_active_events() -> pd.DataFrame:
    """Load active (unresolved, not yet ended) events from the market catalog."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Market catalog not found at {CATALOG_PATH}. "
            "Run: python scripts/build_market_catalog.py"
        )

    df = pd.read_parquet(CATALOG_PATH)
    now = datetime.now(timezone.utc)

    # Filter to unresolved events
    if "is_resolved" in df.columns:
        df = df[df["is_resolved"] == False]

    # Filter to events that haven't ended yet
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
        df = df[df["end_date"] > now]

    return df


def fetch_order_book(session: requests.Session, token_id: str) -> dict | None:
    """Fetch the order book for a single token_id.

    Returns raw dict with 'bids' and 'asks' lists, or None on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(
                f"{CLOB_URL}/book",
                params={"token_id": token_id},
                timeout=30,
            )

            if resp.status_code == 404:
                # Delisted token -- return empty book
                return {"bids": [], "asks": []}

            if resp.status_code == 429:
                wait = (2 ** attempt) * 2
                print(f"    Rate limited (429). Backing off {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 1
                print(f"    Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Failed after {MAX_RETRIES} retries for token {token_id[:16]}...: {e}")
                return None

    return None


def compute_book_metrics(book: dict) -> dict:
    """Compute derived metrics from raw order book data.

    Returns dict with: bid_depth_total, ask_depth_total, spread, mid_price, book_imbalance.
    """
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    bid_depth_total = sum(float(b.get("size", 0)) for b in bids)
    ask_depth_total = sum(float(a.get("size", 0)) for a in asks)

    # Best bid = highest bid price, best ask = lowest ask price
    best_bid = max((float(b["price"]) for b in bids), default=0.0)
    best_ask = min((float(a["price"]) for a in asks), default=0.0)

    if best_bid > 0 and best_ask > 0:
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
    else:
        spread = 0.0
        mid_price = best_bid if best_bid > 0 else best_ask

    total_depth = bid_depth_total + ask_depth_total
    if total_depth > 0:
        book_imbalance = (bid_depth_total - ask_depth_total) / total_depth
    else:
        book_imbalance = 0.0

    return {
        "bid_depth_total": round(bid_depth_total, 2),
        "ask_depth_total": round(ask_depth_total, 2),
        "spread": round(spread, 4),
        "mid_price": round(mid_price, 4),
        "book_imbalance": round(book_imbalance, 4),
    }


def run_dry_run(active_df: pd.DataFrame) -> None:
    """Print what would be fetched without making any API calls."""
    event_groups = active_df.groupby("event_slug")
    total_buckets = 0

    print(f"\n{'='*60}")
    print("  DRY RUN -- Order Book Fetch Preview")
    print(f"{'='*60}")
    print(f"  Active events: {len(event_groups)}")
    print(f"  Total buckets: {len(active_df)}")
    print()

    for event_slug, buckets_df in sorted(event_groups):
        first = buckets_df.iloc[0]
        market_type = first.get("market_type", "unknown")
        end_date = first.get("end_date", "")
        n_buckets = len(buckets_df)
        total_buckets += n_buckets

        # Count how many have valid token_ids
        valid = buckets_df["token_id_yes"].notna() & (buckets_df["token_id_yes"] != "")
        n_valid = valid.sum()

        print(f"  {event_slug}")
        print(f"    Type: {market_type}  |  Ends: {str(end_date)[:10]}  |  Buckets: {n_valid}/{n_buckets}")

    est_time = total_buckets * (RATE_LIMIT_SLEEP + 0.3)
    print(f"\n  Estimated fetch time: ~{est_time:.0f}s ({est_time/60:.1f} min)")
    print(f"{'='*60}")


def run_fetch(active_df: pd.DataFrame) -> None:
    """Fetch order books for all active events and save snapshot."""
    event_groups = dict(list(active_df.groupby("event_slug")))
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")

    print(f"\nFetching order books for {len(event_groups)} active events...")

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "ElonTweetMarkets/1.0",
    })

    all_event_snapshots = []
    total_fetched = 0
    total_errors = 0

    for event_slug in sorted(event_groups.keys()):
        buckets_df = event_groups[event_slug]
        n_buckets = len(buckets_df)
        event_buckets = []

        for i, (_, row) in enumerate(buckets_df.iterrows(), 1):
            token_id = row.get("token_id_yes", "")
            bucket_label = row["bucket_label"]

            if pd.isna(token_id) or not token_id:
                print(f"  Skipping {event_slug} bucket '{bucket_label}': no token_id")
                continue

            print(f"Fetching book for {event_slug} bucket {i}/{n_buckets}...")

            book = fetch_order_book(session, str(token_id))
            if book is None:
                total_errors += 1
                continue

            metrics = compute_book_metrics(book)

            bucket_snapshot = {
                "token_id": str(token_id),
                "bucket_label": bucket_label,
                "bids": book.get("bids", []),
                "asks": book.get("asks", []),
                **metrics,
            }
            event_buckets.append(bucket_snapshot)
            total_fetched += 1

            time.sleep(RATE_LIMIT_SLEEP)

        if event_buckets:
            all_event_snapshots.append({
                "event_slug": event_slug,
                "buckets": event_buckets,
            })

    # Build snapshot
    snapshot = {
        "timestamp": now.isoformat(),
        "events": all_event_snapshots,
        "meta": {
            "n_events": len(all_event_snapshots),
            "n_buckets_fetched": total_fetched,
            "n_errors": total_errors,
        },
    }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"snapshot_{timestamp_str}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("  ORDER BOOK SNAPSHOT SUMMARY")
    print(f"{'='*60}")
    print(f"  Timestamp:      {now.isoformat()}")
    print(f"  Events:         {len(all_event_snapshots)}")
    print(f"  Buckets fetched:{total_fetched:>5d}")
    print(f"  Errors:         {total_errors:>5d}")
    print(f"  Output:         {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch order book snapshots for active Elon tweet markets."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be fetched without making API calls",
    )
    args = parser.parse_args()

    # Load active events
    active_df = load_active_events()
    if active_df.empty:
        print("No active markets found (all resolved or ended). Nothing to fetch.")
        return

    if args.dry_run:
        run_dry_run(active_df)
    else:
        run_fetch(active_df)


if __name__ == "__main__":
    main()
