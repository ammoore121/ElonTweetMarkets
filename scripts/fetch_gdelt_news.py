"""
Fetch GDELT news data for Elon Musk related entities.
Uses GDELT DOC 2.0 API (free, no auth needed).

Usage: python scripts/fetch_gdelt_news.py
"""
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sources" / "gdelt"

session = requests.Session()

# Entities to track
QUERIES = {
    "elon_musk": '"Elon Musk"',
    "tesla": '"Tesla"',
    "spacex": '"SpaceX"',
    "xai": '"xAI"',
    "doge_elon": '"DOGE" "Elon"',
    "neuralink": '"Neuralink"',
}

# Modes to fetch
MODES = ["timelinevol", "timelinetone", "timelinevolraw"]


def fetch_gdelt(query, mode, start_dt, end_dt):
    """Fetch GDELT data for a query and mode."""
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "STARTDATETIME": start_dt.strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME": end_dt.strftime("%Y%m%d%H%M%S"),
    }
    resp = session.get(DOC_API, params=params, timeout=60)
    resp.raise_for_status()
    # GDELT sometimes returns empty or non-JSON
    text = resp.text.strip()
    if not text:
        return None
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
    print("GDELT News Data Fetch")
    print("=" * 60)

    # GDELT DOC API allows max 3 months per query, so we chunk
    end_date = datetime(2026, 2, 10)
    start_date = datetime(2024, 1, 1)  # ~2 years back

    results = {}
    total_fetches = len(QUERIES) * len(MODES)
    fetch_count = 0

    for entity_key, query in QUERIES.items():
        results[entity_key] = {}
        for mode in MODES:
            fetch_count += 1
            print(f"\n[{fetch_count}/{total_fetches}] {entity_key} - {mode}")

            all_data = []
            current = start_date
            while current < end_date:
                chunk_end = min(current + timedelta(days=89), end_date)
                print(f"  {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
                try:
                    data = fetch_gdelt(query, mode, current, chunk_end)
                    if data:
                        all_data.append({
                            "start": current.isoformat(),
                            "end": chunk_end.isoformat(),
                            "response": data,
                        })
                        # Count data points
                        timeline = data.get("timeline", [])
                        n_points = sum(len(series.get("data", [])) for series in timeline)
                        print(f"    Got {n_points} data points across {len(timeline)} series")
                    else:
                        print(f"    Empty response")
                except Exception as e:
                    print(f"    Error: {e}")

                current = chunk_end + timedelta(days=1)
                time.sleep(2)  # Be respectful to GDELT

            results[entity_key][mode] = all_data

            # Save per-entity per-mode file
            filename = f"gdelt_{entity_key}_{mode}.json"
            save_json({
                "fetched_at": datetime.now().isoformat(),
                "query": query,
                "mode": mode,
                "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "chunks": all_data,
            }, filename)

    # Save combined file
    save_json({
        "fetched_at": datetime.now().isoformat(),
        "queries": {k: v for k, v in QUERIES.items()},
        "modes": MODES,
        "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "data": results,
    }, "gdelt_all_combined.json")

    print(f"\n{'=' * 60}")
    print(f"GDELT fetch complete!")
    print(f"Entities: {len(QUERIES)}")
    print(f"Modes: {len(MODES)}")
    print(f"Total API calls: {fetch_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
