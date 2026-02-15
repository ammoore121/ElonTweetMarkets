#!/usr/bin/env python3
"""Fetch Wikipedia daily pageview counts for Elon Musk and related articles.

Uses Wikimedia REST API (free, no auth needed).
Stores: data/sources/wikipedia/pageviews.json

API docs: https://wikimedia.org/api/rest_v1/
Endpoint: /metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json
import time
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError


WIKI_DIR = PROJECT_DIR / "data" / "sources" / "wikipedia"
WIKI_DIR.mkdir(parents=True, exist_ok=True)

# Articles to track (Wikipedia article titles, underscore-separated)
ARTICLES = [
    "Elon_Musk",
    "Tesla,_Inc.",
    "SpaceX",
    "Dogecoin",
    "Department_of_Government_Efficiency",
]

START_DATE = "20240101"  # YYYYMMDD format for Wikimedia API
END_DATE = datetime.now().strftime("%Y%m%d")

BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
USER_AGENT = "ElonTweetMarkets/1.0 (research project; Python urllib)"


def fetch_article_pageviews(article: str) -> dict:
    """Fetch daily pageviews for a single Wikipedia article.

    Returns: {date_str: views_count, ...}
    """
    url = f"{BASE_URL}/en.wikipedia/all-access/all-agents/{article}/daily/{START_DATE}/{END_DATE}"
    print(f"  Fetching {article}...")

    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"    ERROR {e.code}: {e.reason}")
        return {}
    except Exception as e:
        print(f"    ERROR: {e}")
        return {}

    result = {}
    for item in data.get("items", []):
        # timestamp format: YYYYMMDD00
        ts = item.get("timestamp", "")
        if len(ts) >= 8:
            date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
            result[date_str] = item.get("views", 0)

    print(f"    Got {len(result)} days, range: {min(result.keys()) if result else 'N/A'} to {max(result.keys()) if result else 'N/A'}")
    return result


def main():
    print("=" * 60)
    print("WIKIPEDIA PAGEVIEW FETCH")
    print("=" * 60)

    all_data = {}
    for article in ARTICLES:
        key = article.lower().replace(",_", "_").replace(",", "").replace(".", "")
        views = fetch_article_pageviews(article)
        all_data[key] = {
            "article": article,
            "daily_views": views,
            "total_days": len(views),
            "mean_daily": round(sum(views.values()) / max(len(views), 1), 1),
            "max_daily": max(views.values()) if views else 0,
        }
        time.sleep(0.5)  # Be polite to Wikimedia API

    # Save
    output = {
        "fetched_at": datetime.now().isoformat(),
        "start_date": f"{START_DATE[:4]}-{START_DATE[4:6]}-{START_DATE[6:8]}",
        "end_date": END_DATE,
        "articles": all_data,
    }

    out_path = WIKI_DIR / "pageviews.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
    print("\nSummary:")
    for key, info in all_data.items():
        print(f"  {key}: {info['total_days']} days, mean={info['mean_daily']}/day, max={info['max_daily']}/day")

    print("\nDone!")


if __name__ == "__main__":
    main()
