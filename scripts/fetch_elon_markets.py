"""
Fetch all Elon Musk tweet count prediction markets from Polymarket.
Captures both active and resolved markets with full bucket structure.

Uses slug pattern 'elon-musk-of-tweets' to match these categorical markets.
Also fetches via event slug_contains for broader coverage.

Usage: python scripts/fetch_elon_markets.py
"""
import requests
import json
import time
from datetime import datetime
from pathlib import Path

GAMMA_URL = "https://gamma-api.polymarket.com"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sources" / "polymarket"

session = requests.Session()
session.headers.update({"Accept": "application/json", "User-Agent": "ElonTweetMarkets/1.0"})

# Slug patterns that match Elon tweet count markets
SLUG_PATTERNS = ["elon-musk-of-tweets", "elon-musk-tweets", "elonmusk-tweet"]
# Keywords in question text
KEYWORDS = ["elon", "musk"]
TWEET_KEYWORDS = ["tweet", "post", "# tweets", "# post"]


def is_elon_tweet_market(item):
    """Check if an event or market is an Elon tweet count market."""
    title = (item.get("title") or item.get("question") or "").lower()
    slug = (item.get("slug") or "").lower()
    desc = (item.get("description") or "").lower()

    # Slug-based match (most reliable)
    if "elon-musk" in slug and "tweet" in slug:
        return True

    # Title/question-based match
    has_elon = any(k in title for k in KEYWORDS)
    has_tweet = any(k in title for k in TWEET_KEYWORDS)
    if has_elon and has_tweet:
        return True

    # Description fallback
    has_elon_desc = "elon musk" in desc or "elonmusk" in desc
    has_tweet_desc = any(k in desc for k in TWEET_KEYWORDS)
    if has_elon_desc and has_tweet_desc and ("how many" in desc or "count" in desc or "post" in desc):
        return True

    return False


def paginate_events(max_pages=50, **kwargs):
    """Paginate through events from Gamma API."""
    all_items = []
    offset = 0
    for page in range(max_pages):
        params = {"limit": 100, "offset": offset, **kwargs}
        try:
            resp = session.get(f"{GAMMA_URL}/events", params=params, timeout=30)
            resp.raise_for_status()
            items = resp.json()
        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            break

        if not items:
            break
        all_items.extend(items)
        if len(items) < 100:
            break
        offset += 100
        time.sleep(0.3)

    return all_items


def paginate_markets(max_pages=100, **kwargs):
    """Paginate through markets from Gamma API."""
    all_items = []
    offset = 0
    for page in range(max_pages):
        params = {"limit": 100, "offset": offset, **kwargs}
        try:
            resp = session.get(f"{GAMMA_URL}/markets", params=params, timeout=30)
            resp.raise_for_status()
            items = resp.json()
        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            break

        if not items:
            break
        all_items.extend(items)
        if page % 10 == 0 and page > 0:
            print(f"    ... page {page + 1}, {len(all_items)} markets so far")
        if len(items) < 100:
            break
        offset += 100
        time.sleep(0.3)

    return all_items


def save_json(data, filename):
    """Save JSON data to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path} ({path.stat().st_size:,} bytes)")


def main():
    print("=" * 60)
    print("Polymarket Elon Tweet Markets Fetch")
    print("=" * 60)

    all_events = []
    all_markets = []

    # Strategy 1: Get events with slug containing 'elon-musk'
    print("\n[1/4] Fetching events via slug_contains='elon-musk'...")
    for active_val, closed_val, label in [(True, False, "active"), (False, True, "closed")]:
        print(f"  {label} events...")
        events = paginate_events(
            active=str(active_val).lower(),
            closed=str(closed_val).lower(),
        )
        matches = [e for e in events if is_elon_tweet_market(e)]
        print(f"    Scanned {len(events)} events, matched {len(matches)}")
        all_events.extend(matches)

    # Strategy 2: Get individual markets (these are the bucket outcomes)
    print("\n[2/4] Fetching active markets (scanning for Elon tweet buckets)...")
    active_markets = paginate_markets(active="true", closed="false")
    active_matches = [m for m in active_markets if is_elon_tweet_market(m)]
    print(f"  Scanned {len(active_markets)} markets, matched {len(active_matches)}")
    all_markets.extend(active_matches)

    print("\n[3/4] Fetching closed markets (scanning for Elon tweet buckets)...")
    closed_markets = paginate_markets(active="false", closed="true")
    closed_matches = [m for m in closed_markets if is_elon_tweet_market(m)]
    print(f"  Scanned {len(closed_markets)} markets, matched {len(closed_matches)}")
    all_markets.extend(closed_matches)

    # Strategy 3: For events found, also fetch their child markets directly
    print("\n[4/4] Enriching events with child markets...")
    event_market_ids = set()
    for event in all_events:
        for m in event.get("markets", []):
            mid = m.get("id")
            if mid:
                event_market_ids.add(mid)
                all_markets.append(m)

    # Deduplicate
    seen_event_ids = set()
    unique_events = []
    for e in all_events:
        eid = e.get("id")
        if eid and eid not in seen_event_ids:
            seen_event_ids.add(eid)
            unique_events.append(e)

    seen_market_ids = set()
    unique_markets = []
    for m in all_markets:
        mid = m.get("id")
        if mid and mid not in seen_market_ids:
            seen_market_ids.add(mid)
            unique_markets.append(m)

    # Group markets by neg_risk_market_id (event groups)
    groups = {}
    for m in unique_markets:
        nrm = m.get("neg_risk_market_id") or m.get("negRiskMarketId") or "ungrouped"
        if nrm not in groups:
            groups[nrm] = []
        groups[nrm].append(m)

    # Save
    save_json({
        "fetched_at": datetime.now().isoformat(),
        "event_count": len(unique_events),
        "market_count": len(unique_markets),
        "event_groups": len(groups),
        "events": unique_events,
        "markets": unique_markets,
    }, "elon_tweet_markets_full.json")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Polymarket fetch complete!")
    print(f"Unique events: {len(unique_events)}")
    print(f"Unique individual markets (buckets): {len(unique_markets)}")
    print(f"Event groups (by neg_risk_market_id): {len(groups)}")
    print(f"\nEvent groups:")
    for nrm, markets in sorted(groups.items(), key=lambda x: -len(x[1])):
        questions = [m.get("question", "?")[:60] for m in markets[:2]]
        print(f"  [{len(markets):3d} buckets] {questions[0]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
