#!/usr/bin/env python3
"""Fetch daily Reddit activity from Elon-related subreddits.

Two modes:
  --mode backfill  : Arctic Shift API for historical data (free, no auth)
  --mode daily     : Reddit JSON API for recent 24h data (free, no auth)

Stores: data/sources/reddit/daily_activity.parquet
        data/sources/reddit/fetch_log.json

Usage:
  python scripts/fetch_reddit_activity.py --mode backfill --start-date 2024-01-01
  python scripts/fetch_reddit_activity.py --mode daily
  python scripts/fetch_reddit_activity.py --mode backfill --start-date 2026-01-01 --end-date 2026-01-07 --dry-run
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

REDDIT_DIR = PROJECT_DIR / "data" / "sources" / "reddit"
REDDIT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REDDIT_DIR / "daily_activity.parquet"
FETCH_LOG_PATH = REDDIT_DIR / "fetch_log.json"

# Arctic Shift API (historical)
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
ARCTIC_SHIFT_SLEEP = 1.0  # seconds between requests

# Reddit JSON API (recent)
REDDIT_JSON_SLEEP = 1.5  # seconds between requests

USER_AGENT = "ElonTweetMarkets/1.0 (research project; Python urllib)"

# Subreddits to track.
# For WSB we add a query filter to only get Elon-related posts.
SUBREDDITS = {
    "elonmusk": None,
    "teslamotors": None,
    "SpaceX": None,
    "dogecoin": None,
    "wallstreetbets": "elon OR musk OR tesla",
}


def _get_json(url: str, retries: int = 3) -> dict | None:
    """Fetch JSON from a URL with retries."""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            elif e.code >= 500:
                wait = 5 * (attempt + 1)
                print(f"    Server error ({e.code}). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"    HTTP {e.code}: {e.reason}")
                return None
        except (URLError, TimeoutError) as e:
            wait = 5 * (attempt + 1)
            print(f"    Connection error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"    Unexpected error: {e}")
            return None
    print(f"    Failed after {retries} retries")
    return None


def _aggregate_posts(posts: list[dict]) -> dict:
    """Compute daily aggregate metrics from a list of post dicts."""
    if not posts:
        return {
            "post_count": 0,
            "comment_count": 0,
            "avg_score": 0.0,
            "top_post_score": 0,
        }
    scores = [p.get("score", 0) or 0 for p in posts]
    comment_counts = [p.get("num_comments", 0) or 0 for p in posts]
    return {
        "post_count": len(posts),
        "comment_count": sum(comment_counts),
        "avg_score": round(sum(scores) / len(scores), 2),
        "top_post_score": max(scores),
    }


# ── Arctic Shift (backfill) ──────────────────────────────────────────────


def _fetch_arctic_shift_day(
    subreddit: str, date_str: str, query: str | None = None
) -> list[dict]:
    """Fetch all posts for a subreddit on a single day via Arctic Shift.

    Uses limit=auto and paginates by adjusting 'after' if we hit the cap.
    """
    next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    all_posts = []
    after_cursor = date_str

    while True:
        url = (
            f"{ARCTIC_SHIFT_BASE}"
            f"?subreddit={subreddit}"
            f"&after={after_cursor}"
            f"&before={next_day}"
            f"&limit=100"
            f"&sort=asc"
            f"&fields=created_utc,score,num_comments,title"
        )
        if query:
            # URL-encode spaces as +
            url += f"&query={query.replace(' ', '+')}"

        data = _get_json(url)
        if data is None:
            break

        posts = data.get("data", [])
        if not posts:
            break

        all_posts.extend(posts)

        # If we got exactly 100, there may be more -- paginate
        if len(posts) >= 100:
            last_ts = posts[-1].get("created_utc", 0)
            # Move cursor past the last post
            after_cursor_dt = datetime.utcfromtimestamp(last_ts)
            after_cursor = after_cursor_dt.strftime("%Y-%m-%dT%H:%M:%S")
            time.sleep(ARCTIC_SHIFT_SLEEP)
        else:
            break

    return all_posts


def backfill(start_date: str, end_date: str, dry_run: bool = False) -> pd.DataFrame:
    """Backfill daily Reddit activity using Arctic Shift API."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    total_requests = total_days * len(SUBREDDITS)

    print(f"Backfill: {start_date} to {end_date} ({total_days} days)")
    print(f"Subreddits: {list(SUBREDDITS.keys())}")
    print(f"Estimated requests: ~{total_requests}")

    if dry_run:
        print("\n[DRY RUN] Would fetch the above. Exiting.")
        return pd.DataFrame()

    rows = []
    request_count = 0

    current = start
    while current < end:
        date_str = current.strftime("%Y-%m-%d")
        day_num = (current - start).days + 1

        for subreddit, query in SUBREDDITS.items():
            request_count += 1
            label = f"[{day_num}/{total_days}] {date_str} r/{subreddit}"
            if query:
                label += f" (query: {query})"

            posts = _fetch_arctic_shift_day(subreddit, date_str, query)
            agg = _aggregate_posts(posts)

            if request_count % 20 == 0 or agg["post_count"] > 0:
                print(
                    f"  {label}: {agg['post_count']} posts, "
                    f"{agg['comment_count']} comments"
                )

            rows.append(
                {
                    "date": date_str,
                    "subreddit": subreddit,
                    **agg,
                }
            )

            time.sleep(ARCTIC_SHIFT_SLEEP)

        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    print(f"\nCollected {len(df)} rows ({request_count} API requests)")
    return df


# ── Reddit JSON API (daily) ──────────────────────────────────────────────


def _fetch_reddit_json_recent(subreddit: str, query: str | None = None) -> list[dict]:
    """Fetch recent posts from Reddit JSON API, filter to last 24h."""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    cutoff_ts = cutoff.timestamp()

    if query:
        # Use search endpoint for keyword filtering
        q = query.replace(" ", "+")
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={q}&restrict_sr=on&sort=new&t=day&limit=100"
        )
    else:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100"

    data = _get_json(url)
    if data is None:
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        created = post.get("created_utc", 0)
        if created >= cutoff_ts:
            posts.append(
                {
                    "created_utc": created,
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "title": post.get("title", ""),
                }
            )

    return posts


def daily_fetch() -> pd.DataFrame:
    """Fetch last-24h Reddit activity via Reddit JSON API."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"Daily fetch for {today}")

    rows = []
    for subreddit, query in SUBREDDITS.items():
        label = f"r/{subreddit}"
        if query:
            label += f" (query: {query})"

        print(f"  Fetching {label}...")
        posts = _fetch_reddit_json_recent(subreddit, query)
        agg = _aggregate_posts(posts)

        print(
            f"    {agg['post_count']} posts, "
            f"{agg['comment_count']} comments, "
            f"avg_score={agg['avg_score']}"
        )

        rows.append(
            {
                "date": today,
                "subreddit": subreddit,
                **agg,
            }
        )

        time.sleep(REDDIT_JSON_SLEEP)

    df = pd.DataFrame(rows)
    return df


# ── Persistence ──────────────────────────────────────────────────────────


def save_results(df: pd.DataFrame):
    """Save or append results to parquet, deduplicating by (date, subreddit)."""
    if df.empty:
        print("No data to save.")
        return

    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        print(f"Existing data: {len(existing)} rows")
        combined = pd.concat([existing, df], ignore_index=True)
        # Keep latest data for each (date, subreddit) pair
        combined = combined.drop_duplicates(
            subset=["date", "subreddit"], keep="last"
        )
        combined = combined.sort_values(["date", "subreddit"]).reset_index(drop=True)
    else:
        combined = df.sort_values(["date", "subreddit"]).reset_index(drop=True)

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(combined)} rows to {OUTPUT_PATH}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Subreddits: {sorted(combined['subreddit'].unique())}")


def update_fetch_log(mode: str, start_date: str, end_date: str, row_count: int):
    """Track what has been fetched."""
    log = {}
    if FETCH_LOG_PATH.exists():
        with open(FETCH_LOG_PATH, "r") as f:
            log = json.load(f)

    if "fetches" not in log:
        log["fetches"] = []

    log["fetches"].append(
        {
            "mode": mode,
            "start_date": start_date,
            "end_date": end_date,
            "rows": row_count,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )
    log["last_updated"] = datetime.utcnow().isoformat() + "Z"

    with open(FETCH_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Updated fetch log: {FETCH_LOG_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fetch daily Reddit activity for Elon-related subreddits"
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="backfill",
        help="backfill (Arctic Shift, historical) or daily (Reddit JSON, recent 24h)",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Start date for backfill (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="End date for backfill (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without making requests",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("REDDIT ACTIVITY FETCH")
    print("=" * 60)

    if args.mode == "backfill":
        df = backfill(args.start_date, args.end_date, args.dry_run)
        if not args.dry_run and not df.empty:
            save_results(df)
            update_fetch_log(args.mode, args.start_date, args.end_date, len(df))
    elif args.mode == "daily":
        df = daily_fetch()
        if not df.empty:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            save_results(df)
            update_fetch_log(args.mode, today, today, len(df))

    print("\nDone!")


if __name__ == "__main__":
    main()
