"""
Fetch XTracker historical daily metrics for Elon Musk.
XTracker is the resolution oracle for Polymarket tweet markets.

Usage: python scripts/fetch_xtracker_history.py
"""
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://xtracker.polymarket.com/api"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sources" / "xtracker"
USER_ID = "c4e2a911-36ec-4453-8a39-1edb5e6b2969"

session = requests.Session()
session.headers.update({"Accept": "application/json"})


def fetch_user():
    """Fetch Elon Musk user profile."""
    resp = session.get(f"{BASE_URL}/users/elonmusk")
    resp.raise_for_status()
    return resp.json()


def fetch_daily_metrics(start_date: str, end_date: str):
    """Fetch daily metrics for a date range."""
    resp = session.get(
        f"{BASE_URL}/metrics/{USER_ID}",
        params={"type": "daily", "startDate": start_date, "endDate": end_date}
    )
    resp.raise_for_status()
    return resp.json()


def fetch_hourly_metrics(start_date: str, end_date: str):
    """Fetch hourly metrics for a date range."""
    resp = session.get(
        f"{BASE_URL}/metrics/{USER_ID}",
        params={"type": "hourly", "startDate": start_date, "endDate": end_date}
    )
    resp.raise_for_status()
    return resp.json()


def fetch_trackings():
    """Fetch all tracking windows."""
    resp = session.get(f"{BASE_URL}/users/elonmusk/trackings")
    resp.raise_for_status()
    return resp.json()


def fetch_all_trackings():
    """Fetch all trackings across all users."""
    resp = session.get(f"{BASE_URL}/trackings")
    resp.raise_for_status()
    return resp.json()


def save_json(data, filename):
    """Save JSON data to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path} ({path.stat().st_size:,} bytes)")


def main():
    print("=" * 60)
    print("XTracker Data Fetch")
    print("=" * 60)

    # 1. User profile
    print("\n[1/5] Fetching user profile...")
    user = fetch_user()
    save_json(user, "user_elonmusk.json")

    # 2. Trackings
    print("\n[2/5] Fetching trackings...")
    trackings = fetch_trackings()
    save_json(trackings, "elonmusk_trackings.json")

    # 3. All trackings
    print("\n[3/5] Fetching all trackings...")
    try:
        all_trackings = fetch_all_trackings()
        save_json(all_trackings, "all_trackings.json")
    except Exception as e:
        print(f"  Warning: Could not fetch all trackings: {e}")

    # 4. Daily metrics - fetch in monthly chunks going back as far as possible
    print("\n[4/5] Fetching daily metrics (chunked by month)...")
    all_daily = []
    end = datetime.now(tz=None) + timedelta(days=1)  # through tomorrow to catch today
    # Try going back to 2024-01-01 - API will return what it has
    start_target = datetime(2024, 1, 1)
    current = start_target

    while current < end:
        chunk_end = min(current + timedelta(days=90), end)
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        print(f"  Fetching {start_str} to {end_str}...")
        try:
            result = fetch_daily_metrics(start_str, end_str)
            data = result.get("data", result) if isinstance(result, dict) else result
            if isinstance(data, list):
                all_daily.extend(data)
                print(f"    Got {len(data)} records")
            else:
                print(f"    Got response: {str(result)[:200]}")
                if result:
                    all_daily.append(result)
        except Exception as e:
            print(f"    Error: {e}")
        current = chunk_end + timedelta(days=1)
        time.sleep(1)  # Rate limit respect

    save_json({"fetched_at": datetime.now().isoformat(), "count": len(all_daily), "data": all_daily},
              "daily_metrics_full.json")
    print(f"  Total daily records: {len(all_daily)}")

    # 5. Hourly metrics - just recent period (last 3 months)
    print("\n[5/5] Fetching hourly metrics (last 3 months)...")
    all_hourly = []
    hourly_start = end - timedelta(days=90)  # last 3 months
    current = hourly_start

    while current < end:
        chunk_end = min(current + timedelta(days=30), end)
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        print(f"  Fetching {start_str} to {end_str}...")
        try:
            result = fetch_hourly_metrics(start_str, end_str)
            data = result.get("data", result) if isinstance(result, dict) else result
            if isinstance(data, list):
                all_hourly.extend(data)
                print(f"    Got {len(data)} records")
            else:
                print(f"    Got response: {str(result)[:200]}")
        except Exception as e:
            print(f"    Error: {e}")
        current = chunk_end + timedelta(days=1)
        time.sleep(1)

    save_json({"fetched_at": datetime.now().isoformat(), "count": len(all_hourly), "data": all_hourly},
              "hourly_metrics_full.json")
    print(f"  Total hourly records: {len(all_hourly)}")

    print("\n" + "=" * 60)
    print("XTracker fetch complete!")
    print(f"Daily records: {len(all_daily)}")
    print(f"Hourly records: {len(all_hourly)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
