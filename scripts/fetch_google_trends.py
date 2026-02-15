#!/usr/bin/env python3
"""Fetch Google Trends data for Elon Musk-related queries via pytrends.

Uses overlapping 85-day windows (max for daily granularity) stitched together
with overlap normalization. Sleeps 15s between requests for rate limiting.

Queries: ["Elon Musk", "Tesla", "SpaceX", "Dogecoin", "DOGE government"]

Stores:
  data/sources/trends/google_trends.parquet
"""

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from datetime import datetime, timedelta

import pandas as pd

try:
    from pytrends.request import TrendReq
except ImportError:
    print("ERROR: pytrends not installed. Run: pip install pytrends")
    sys.exit(1)


TRENDS_DIR = PROJECT_DIR / "data" / "sources" / "trends"
TRENDS_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = ["Elon Musk", "Tesla", "SpaceX", "Dogecoin", "DOGE government"]
WINDOW_DAYS = 85  # Max for daily granularity from Google Trends
OVERLAP_DAYS = 15  # Overlap between windows for normalization
SLEEP_SECS = 15   # Rate limiting between requests

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime.now()


def fetch_single_window(pytrends, keywords, start, end):
    """Fetch a single time window from Google Trends."""
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="", gprop="")
    df = pytrends.interest_over_time()
    if df.empty:
        return None
    # Drop isPartial column if present
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df


def stitch_windows(frames):
    """Stitch overlapping windows using overlap normalization."""
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]

    result = frames[0].copy()

    for i in range(1, len(frames)):
        next_frame = frames[i].copy()

        # Find overlapping dates
        overlap_dates = result.index.intersection(next_frame.index)

        if len(overlap_dates) == 0:
            # No overlap -- just concatenate
            result = pd.concat([result, next_frame[~next_frame.index.isin(result.index)]])
            continue

        # Compute scaling factors from overlap region
        for col in result.columns:
            overlap_old = result.loc[overlap_dates, col]
            overlap_new = next_frame.loc[overlap_dates, col]

            # Mean ratio scaling (avoid division by zero)
            old_mean = overlap_old.mean()
            new_mean = overlap_new.mean()

            if new_mean > 0 and old_mean > 0:
                scale = old_mean / new_mean
            else:
                scale = 1.0

            next_frame[col] = next_frame[col] * scale

        # Append non-overlapping dates from next_frame
        new_dates = next_frame.index.difference(result.index)
        if len(new_dates) > 0:
            result = pd.concat([result, next_frame.loc[new_dates]])

    return result.sort_index()


def fetch_trends():
    """Download Google Trends data using overlapping windows."""
    print(f"Fetching Google Trends for: {QUERIES}")
    print(f"  Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Window: {WINDOW_DAYS} days, Overlap: {OVERLAP_DAYS} days")
    print(f"  Sleep between requests: {SLEEP_SECS}s")

    pytrends = TrendReq(hl="en-US", tz=300)

    # Generate window boundaries
    windows = []
    current_start = START_DATE
    while current_start < END_DATE:
        current_end = min(current_start + timedelta(days=WINDOW_DAYS - 1), END_DATE)
        windows.append((current_start, current_end))
        current_start = current_start + timedelta(days=WINDOW_DAYS - OVERLAP_DAYS)

    print(f"  Windows to fetch: {len(windows)}")

    frames = []
    for i, (ws, we) in enumerate(windows):
        print(f"  [{i+1}/{len(windows)}] {ws.strftime('%Y-%m-%d')} to {we.strftime('%Y-%m-%d')}...", end=" ")
        try:
            df = fetch_single_window(pytrends, QUERIES, ws, we)
            if df is not None and not df.empty:
                frames.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("empty")
        except Exception as e:
            print(f"FAILED: {e}")

        if i < len(windows) - 1:
            time.sleep(SLEEP_SECS)

    if not frames:
        print("ERROR: No data fetched")
        return

    # Stitch windows
    print("\n  Stitching windows...")
    combined = stitch_windows(frames)

    # Normalize column names
    combined = combined.reset_index()
    combined = combined.rename(columns={
        "date": "date",
        "Elon Musk": "elon_musk",
        "Tesla": "tesla",
        "SpaceX": "spacex",
        "Dogecoin": "dogecoin",
        "DOGE government": "doge_government",
    })

    # Ensure date column is string
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")

    out_path = TRENDS_DIR / "google_trends.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\n  Saved {len(combined)} rows to {out_path}")
    print(f"  Date range: {combined['date'].iloc[0]} to {combined['date'].iloc[-1]}")
    print(f"  Columns: {list(combined.columns)}")
    return combined


def main():
    print("=" * 60)
    print("GOOGLE TRENDS FETCH")
    print("=" * 60)
    fetch_trends()
    print("\nDone!")


if __name__ == "__main__":
    main()
