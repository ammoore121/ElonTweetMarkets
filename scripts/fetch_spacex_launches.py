"""
Fetch historical SpaceX launch data from Launch Library 2 API.
Rate limit: 15 requests/hour on free tier.

Usage: python scripts/fetch_spacex_launches.py
"""
import requests
import json
import time
from datetime import datetime
from pathlib import Path

BASE_URL = "https://ll.thespacedevs.com/2.0.0"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sources" / "calendar"

session = requests.Session()
session.headers.update({"Accept": "application/json"})


def fetch_launches(url=None, params=None):
    """Fetch a page of launches."""
    if url is None:
        url = f"{BASE_URL}/launch/"
    resp = session.get(url, params=params, timeout=30)
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
    print("SpaceX Launch History Fetch")
    print("=" * 60)

    # 1. Historical launches (2024-01-01 to present)
    print("\n[1/2] Fetching historical SpaceX launches...")
    all_launches = []
    params = {
        "lsp__name": "SpaceX",
        "net__gte": "2024-01-01",
        "net__lte": "2026-02-10",
        "limit": 100,
        "ordering": "net",
    }

    page = 1
    next_url = None
    while True:
        print(f"  Page {page}...")
        try:
            if next_url:
                result = fetch_launches(url=next_url)
            else:
                result = fetch_launches(params=params)

            launches = result.get("results", [])
            all_launches.extend(launches)
            print(f"    Got {len(launches)} launches (total: {len(all_launches)}/{result.get('count', '?')})")

            next_url = result.get("next")
            if not next_url:
                break
            page += 1
            time.sleep(5)  # Respect 15 req/hr rate limit
        except Exception as e:
            print(f"    Error: {e}")
            break

    save_json({
        "fetched_at": datetime.now().isoformat(),
        "count": len(all_launches),
        "launches": all_launches,
    }, "spacex_launches_historical.json")

    # 2. Upcoming launches
    print("\n[2/2] Fetching upcoming SpaceX launches...")
    try:
        result = fetch_launches(params={
            "lsp__name": "SpaceX",
            "limit": 20,
        })
        upcoming = result.get("results", [])
        # Use the upcoming endpoint
        result2 = fetch_launches(url=f"{BASE_URL}/launch/upcoming/", params={
            "lsp__name": "SpaceX",
            "limit": 20,
        })
        upcoming_launches = result2.get("results", [])
        save_json({
            "fetched_at": datetime.now().isoformat(),
            "count": len(upcoming_launches),
            "launches": upcoming_launches,
        }, "spacex_upcoming_fresh.json")
        print(f"  Got {len(upcoming_launches)} upcoming launches")
    except Exception as e:
        print(f"  Error fetching upcoming: {e}")

    print(f"\n{'=' * 60}")
    print(f"SpaceX fetch complete!")
    print(f"Historical launches: {len(all_launches)}")
    if all_launches:
        dates = [l.get("net", "")[:10] for l in all_launches if l.get("net")]
        print(f"Date range: {min(dates)} to {max(dates)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
