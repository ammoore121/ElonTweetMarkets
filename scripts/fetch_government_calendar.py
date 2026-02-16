#!/usr/bin/env python3
"""Fetch DOGE-related government events (executive orders, bills, hearings, notices).

Sources:
  1. Federal Register API (free, no auth) - executive orders, rules, notices
  2. GovTrack API (free, no auth) - bills in Congress
  3. Congress.gov API (free, requires API key) - bills + hearings

Stores:
  data/sources/government/events.parquet
  data/sources/government/fetch_log.json

Usage:
  python scripts/fetch_government_calendar.py
  python scripts/fetch_government_calendar.py --source federal_register
  python scripts/fetch_government_calendar.py --source govtrack
  python scripts/fetch_government_calendar.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

OUTPUT_DIR = PROJECT_DIR / "data" / "sources" / "government"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

session = requests.Session()
session.headers.update({"Accept": "application/json"})

# Search terms for DOGE-related government activity
FEDERAL_REGISTER_QUERIES = [
    '"Department of Government Efficiency"',
    'DOGE "government efficiency"',
    # Note: '"Elon Musk" executive order' returns only unrelated H-1B rules.
    # The DOGE-specific queries above cover all relevant executive orders.
]

GOVTRACK_QUERIES = [
    "DOGE",
    # Note: "Department of Government Efficiency" returns 400+ false positives on GovTrack
    # (matches any bill mentioning "efficiency" or "government"). "DOGE" is more targeted.
]

START_DATE = "2025-01-01"


# ---------------------------------------------------------------------------
# Federal Register API (no auth)
# ---------------------------------------------------------------------------

def fetch_federal_register(dry_run=False):
    """Fetch DOGE-related documents from the Federal Register API."""
    base_url = "https://www.federalregister.gov/api/v1/documents.json"
    all_docs = []
    seen_doc_numbers = set()

    for query in FEDERAL_REGISTER_QUERIES:
        print(f"  Query: {query}")
        page = 1
        while True:
            params = {
                "conditions[term]": query,
                "conditions[publication_date][gte]": START_DATE,
                "per_page": 20,
                "page": page,
                "fields[]": [
                    "title", "type", "document_number", "publication_date",
                    "abstract", "agencies", "html_url",
                ],
            }

            if dry_run:
                print(f"    [DRY RUN] Would fetch page {page} from {base_url}")
                break

            try:
                resp = session.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"    Error on page {page}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for doc in results:
                doc_num = doc.get("document_number", "")
                if doc_num in seen_doc_numbers:
                    continue
                seen_doc_numbers.add(doc_num)

                # Map Federal Register type to our event_type
                fr_type = (doc.get("type") or "").lower()
                if "presidential" in fr_type:
                    event_type = "executive_order"
                elif "rule" in fr_type:
                    event_type = "rule"
                elif "notice" in fr_type:
                    event_type = "notice"
                else:
                    event_type = "notice"

                agencies = doc.get("agencies", [])
                agency_names = ", ".join(
                    a.get("name", "") for a in agencies if a.get("name")
                )

                all_docs.append({
                    "date": doc.get("publication_date", ""),
                    "event_type": event_type,
                    "source": "federal_register",
                    "title": doc.get("title", ""),
                    "description": (doc.get("abstract") or agency_names or ""),
                })

            total = data.get("total_pages", 1)
            print(f"    Page {page}/{total}: {len(results)} docs (unique total: {len(seen_doc_numbers)})")

            if page >= total:
                break
            page += 1
            time.sleep(1)

    print(f"  Federal Register total: {len(all_docs)} unique documents")
    return all_docs


# ---------------------------------------------------------------------------
# GovTrack API (no auth)
# ---------------------------------------------------------------------------

def fetch_govtrack(dry_run=False):
    """Fetch DOGE-related bills from GovTrack API."""
    base_url = "https://www.govtrack.us/api/v2/bill"
    all_bills = []
    seen_ids = set()

    for query in GOVTRACK_QUERIES:
        print(f"  Query: {query}")
        offset = 0
        limit = 20
        while True:
            params = {
                "q": query,
                "congress": 119,
                "limit": limit,
                "offset": offset,
                "order_by": "-introduced_date",
            }

            if dry_run:
                print(f"    [DRY RUN] Would fetch offset {offset} from {base_url}")
                break

            try:
                resp = session.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"    Error at offset {offset}: {e}")
                break

            objects = data.get("objects", [])
            total = data.get("meta", {}).get("total_count", 0)

            if not objects:
                break

            for bill in objects:
                bill_id = bill.get("id")
                if bill_id in seen_ids:
                    continue
                seen_ids.add(bill_id)

                intro_date = bill.get("introduced_date", "")
                if intro_date and intro_date < START_DATE:
                    continue

                title = bill.get("title", "")
                display = bill.get("display_number", "")
                status = bill.get("current_status_description", "")
                sponsor = bill.get("sponsor", {})
                sponsor_name = sponsor.get("name", "") if sponsor else ""

                all_bills.append({
                    "date": intro_date,
                    "event_type": "bill",
                    "source": "govtrack",
                    "title": f"{display}: {title}" if display else title,
                    "description": f"Status: {status}. Sponsor: {sponsor_name}" if status else "",
                })

            print(f"    Offset {offset}: {len(objects)} bills (total available: {total})")

            offset += limit
            if offset >= total:
                break
            time.sleep(1)

    print(f"  GovTrack total: {len(all_bills)} unique bills")
    return all_bills


# ---------------------------------------------------------------------------
# Congress.gov API (requires API key)
# ---------------------------------------------------------------------------

def fetch_congress(dry_run=False):
    """Fetch DOGE-related bills and hearings from Congress.gov API."""
    api_key = os.environ.get("CONGRESS_API_KEY", "")
    if not api_key:
        print("  WARNING: CONGRESS_API_KEY not found in .env -- skipping Congress.gov")
        return []

    base_url = "https://api.congress.gov/v3"
    all_items = []

    # Bills
    print("  Fetching bills...")
    query = 'DOGE OR "Department of Government Efficiency"'
    offset = 0
    limit = 20
    while True:
        params = {
            "query": query,
            "fromDateTime": f"{START_DATE}T00:00:00Z",
            "offset": offset,
            "limit": limit,
            "api_key": api_key,
        }

        if dry_run:
            print(f"    [DRY RUN] Would fetch bills offset {offset}")
            break

        try:
            resp = session.get(f"{base_url}/bill", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    Error fetching bills at offset {offset}: {e}")
            break

        bills = data.get("bills", [])
        if not bills:
            break

        for bill in bills:
            date = (bill.get("latestAction", {}).get("actionDate", "")
                    or bill.get("updateDate", "")[:10])
            all_items.append({
                "date": date,
                "event_type": "bill",
                "source": "congress",
                "title": bill.get("title", ""),
                "description": bill.get("latestAction", {}).get("text", ""),
            })

        print(f"    Got {len(bills)} bills (offset {offset})")
        total = data.get("pagination", {}).get("count", 0)
        offset += limit
        if offset >= total:
            break
        time.sleep(1)

    # Hearings
    print("  Fetching hearings...")
    offset = 0
    while True:
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "api_key": api_key,
        }

        if dry_run:
            print(f"    [DRY RUN] Would fetch hearings offset {offset}")
            break

        try:
            resp = session.get(f"{base_url}/hearing", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    Error fetching hearings at offset {offset}: {e}")
            break

        hearings = data.get("hearings", [])
        if not hearings:
            break

        for h in hearings:
            date = h.get("date", "")[:10] if h.get("date") else ""
            all_items.append({
                "date": date,
                "event_type": "hearing",
                "source": "congress",
                "title": h.get("title", ""),
                "description": h.get("description", ""),
            })

        print(f"    Got {len(hearings)} hearings (offset {offset})")
        total = data.get("pagination", {}).get("count", 0)
        offset += limit
        if offset >= total:
            break
        time.sleep(1)

    print(f"  Congress.gov total: {len(all_items)} items")
    return all_items


# ---------------------------------------------------------------------------
# Merge & deduplicate
# ---------------------------------------------------------------------------

def merge_and_save(all_events):
    """Merge events, deduplicate, and save to parquet."""
    if not all_events:
        print("\nNo events found -- nothing to save.")
        return None

    df = pd.DataFrame(all_events)

    # Clean dates
    df["date"] = df["date"].astype(str).str[:10]
    df = df[df["date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]

    # Sort by date descending
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # Deduplicate: same date + very similar title (first 60 chars lowercase)
    df["_dedup_key"] = df["date"] + "|" + df["title"].str[:60].str.lower().str.strip()
    df = df.drop_duplicates(subset="_dedup_key", keep="first").drop(columns="_dedup_key")
    df = df.reset_index(drop=True)

    # Save parquet
    out_path = OUTPUT_DIR / "events.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} events to {out_path}")

    # Save fetch log
    log = {
        "fetched_at": datetime.now().isoformat(),
        "total_events": len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "by_type": df["event_type"].value_counts().to_dict(),
        "date_range": {
            "min": df["date"].min(),
            "max": df["date"].max(),
        },
    }
    log_path = OUTPUT_DIR / "fetch_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved fetch log to {log_path}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total events: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nBy source:")
    for src, count in df["source"].value_counts().items():
        print(f"  {src}: {count}")
    print(f"\nBy event type:")
    for etype, count in df["event_type"].value_counts().items():
        print(f"  {etype}: {count}")
    print(f"\nRecent events:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['date']} [{row['event_type']:16s}] {row['title'][:80]}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch DOGE-related government events")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    parser.add_argument("--source", choices=["federal_register", "govtrack", "congress", "all"],
                        default="all", help="Which source to fetch (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("GOVERNMENT CALENDAR FETCH")
    print("=" * 60)

    all_events = []

    if args.source in ("federal_register", "all"):
        print(f"\n[1] Federal Register API (no auth)")
        events = fetch_federal_register(dry_run=args.dry_run)
        all_events.extend(events)

    if args.source in ("govtrack", "all"):
        print(f"\n[2] GovTrack API (no auth)")
        events = fetch_govtrack(dry_run=args.dry_run)
        all_events.extend(events)

    if args.source in ("congress", "all"):
        print(f"\n[3] Congress.gov API (requires CONGRESS_API_KEY)")
        events = fetch_congress(dry_run=args.dry_run)
        all_events.extend(events)

    if not args.dry_run:
        merge_and_save(all_events)

    print("\nDone!")


if __name__ == "__main__":
    main()
