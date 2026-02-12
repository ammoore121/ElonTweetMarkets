"""
Process Kaggle tweet CSVs into daily counts matching XTracker methodology.
Merges with existing XTracker daily_metrics_full.json into a unified daily_counts.json.

XTracker counts: original posts + reposts + quote tweets (NOT replies).

Input:
    data/tweets/all_musk_posts.csv     (55K tweets, May 2023 - Dec 2024)
    data/tweets/elonmusk.csv           (24K tweets, 2010 - Jun 2023)
    data/sources/xtracker/daily_metrics_full.json  (Oct 2025+)

Output:
    data/processed/daily_counts.json   (unified daily counts, all sources)

Usage: python scripts/process_kaggle_tweets.py
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
TWEETS_DIR = PROJECT / "data" / "tweets"
XTRACKER_PATH = PROJECT / "data" / "sources" / "xtracker" / "daily_metrics_full.json"
OUTPUT_PATH = PROJECT / "data" / "processed" / "daily_counts.json"


def process_all_musk_posts():
    """Process all_musk_posts.csv (May 2023 - Dec 2024).

    Has explicit isReply/isRetweet/isQuote flags.
    XTracker counts: posts + retweets + quotes, NOT replies.
    """
    path = TWEETS_DIR / "all_musk_posts.csv"
    if not path.exists():
        print("  SKIP: all_musk_posts.csv not found")
        return {}

    df = pd.read_csv(path, low_memory=False)
    print(f"  all_musk_posts.csv: {len(df):,} rows")

    # Parse dates
    df["date"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Filter: exclude replies (match XTracker methodology)
    # isReply=True means it's a reply to another tweet
    if "isReply" in df.columns:
        before = len(df)
        df = df[df["isReply"] != True]
        print(f"  Filtered replies: {before:,} -> {len(df):,} ({before - len(df):,} replies removed)")

    # Count by date
    daily = df.groupby("date_str").size().to_dict()
    print(f"  Date range: {min(daily.keys())} to {max(daily.keys())}")
    print(f"  Days with data: {len(daily)}")

    return daily


def process_elonmusk_csv():
    """Process elonmusk.csv (2010 - Jun 2023).

    No reply/retweet flags. Use heuristics:
    - Exclude tweets starting with '@' (likely replies)
    - Include everything else
    """
    path = TWEETS_DIR / "elonmusk.csv"
    if not path.exists():
        print("  SKIP: elonmusk.csv not found")
        return {}

    df = pd.read_csv(path, low_memory=False)
    print(f"  elonmusk.csv: {len(df):,} rows")

    # Parse dates
    df["date"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Heuristic: filter replies (tweets starting with @)
    if "Text" in df.columns:
        before = len(df)
        df = df[~df["Text"].fillna("").str.strip().str.startswith("@")]
        print(f"  Filtered @-replies: {before:,} -> {len(df):,}")

    daily = df.groupby("date_str").size().to_dict()
    print(f"  Date range: {min(daily.keys())} to {max(daily.keys())}")
    print(f"  Days with data: {len(daily)}")

    return daily


def load_xtracker_daily():
    """Load XTracker daily_metrics_full.json (Oct 2025+)."""
    if not XTRACKER_PATH.exists():
        print("  SKIP: XTracker daily metrics not found")
        return {}

    with open(XTRACKER_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    daily = {}
    for record in raw.get("data", []):
        date_str = record["date"][:10]
        count = record["data"]["count"]
        # Only include non-zero days (zeros are tracking window artifacts)
        if count > 0:
            daily[date_str] = count

    print(f"  XTracker: {len(daily)} non-zero days")
    if daily:
        print(f"  Date range: {min(daily.keys())} to {max(daily.keys())}")

    return daily


def merge_sources(kaggle_main, kaggle_old, xtracker):
    """Merge all sources, preferring XTracker > kaggle_main > kaggle_old."""
    all_dates = set()
    all_dates.update(kaggle_main.keys())
    all_dates.update(kaggle_old.keys())
    all_dates.update(xtracker.keys())

    unified = {}
    for d in sorted(all_dates):
        # Priority: XTracker (ground truth) > all_musk_posts (has flags) > elonmusk.csv (heuristic)
        if d in xtracker:
            unified[d] = {"count": xtracker[d], "source": "xtracker"}
        elif d in kaggle_main:
            unified[d] = {"count": kaggle_main[d], "source": "kaggle_main"}
        elif d in kaggle_old:
            unified[d] = {"count": kaggle_old[d], "source": "kaggle_old"}

    return unified


def validate_overlap(kaggle_main, xtracker):
    """Check if Kaggle and XTracker agree where they overlap."""
    overlap_dates = set(kaggle_main.keys()) & set(xtracker.keys())
    if not overlap_dates:
        print("  No overlapping dates between Kaggle and XTracker")
        return

    print(f"\n  Overlap validation ({len(overlap_dates)} dates):")
    diffs = []
    for d in sorted(overlap_dates):
        k = kaggle_main[d]
        x = xtracker[d]
        diff = k - x
        diffs.append(diff)
        if abs(diff) > 5:
            print(f"    {d}: Kaggle={k} XTracker={x} diff={diff:+d}")

    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        print(f"  Mean difference: {avg_diff:+.1f} tweets/day")
        print(f"  Kaggle {'over' if avg_diff > 0 else 'under'}counts by ~{abs(avg_diff):.1f}/day on average")


def main():
    print("=" * 60)
    print("Process Kaggle Tweets -> Daily Counts")
    print("=" * 60)

    # Process each source
    print("\n1. Processing all_musk_posts.csv...")
    kaggle_main = process_all_musk_posts()

    print("\n2. Processing elonmusk.csv...")
    kaggle_old = process_elonmusk_csv()

    print("\n3. Loading XTracker daily metrics...")
    xtracker = load_xtracker_daily()

    # Validate overlap
    print("\n4. Validating overlap...")
    validate_overlap(kaggle_main, xtracker)

    # Merge
    print("\n5. Merging sources...")
    unified = merge_sources(kaggle_main, kaggle_old, xtracker)

    # Stats
    sources = Counter(v["source"] for v in unified.values())
    print(f"  Total days: {len(unified)}")
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt} days")

    dates = sorted(unified.keys())
    print(f"  Full range: {dates[0]} to {dates[-1]}")

    # Check coverage for bronze event period (Apr 2024 - Sept 2025)
    bronze_dates = [d for d in dates if "2024-04" <= d <= "2025-09"]
    print(f"  Bronze period coverage (Apr 2024 - Sep 2025): {len(bronze_dates)} days")

    # Check for gaps
    from datetime import timedelta
    gap_ranges = []
    for i in range(1, len(dates)):
        d1 = datetime.strptime(dates[i-1], "%Y-%m-%d")
        d2 = datetime.strptime(dates[i], "%Y-%m-%d")
        gap = (d2 - d1).days
        if gap > 3:
            gap_ranges.append((dates[i-1], dates[i], gap))

    if gap_ranges:
        print(f"\n  Gaps > 3 days:")
        for start, end, gap in gap_ranges:
            print(f"    {start} to {end} ({gap} days)")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_days": len(unified),
            "date_range": {"start": dates[0], "end": dates[-1]},
            "sources": dict(sources),
        },
        "daily_counts": unified,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print("\nDone!")


if __name__ == "__main__":
    main()
