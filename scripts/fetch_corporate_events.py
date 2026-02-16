#!/usr/bin/env python3
"""Fetch Tesla/SpaceX/xAI/Neuralink corporate events for tweet prediction features.

Sources:
  1. yfinance - Tesla earnings dates (past + upcoming)
  2. SEC EDGAR submissions API - Tesla 8-K filings (material events)
  3. Manual curation - High-impact product launches, milestones, funding rounds

Stores:
  data/sources/calendar/corporate_events.parquet

Usage:
  python scripts/fetch_corporate_events.py             # Fetch all sources
  python scripts/fetch_corporate_events.py --source yfinance
  python scripts/fetch_corporate_events.py --source edgar
  python scripts/fetch_corporate_events.py --source manual
  python scripts/fetch_corporate_events.py --dry-run    # Preview without saving
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

OUTPUT_DIR = PROJECT_DIR / "data" / "sources" / "calendar"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "corporate_events.parquet"

# Tesla CIK for SEC EDGAR
TESLA_CIK = "0001318605"
EDGAR_BASE = "https://data.sec.gov/submissions"
EDGAR_USER_AGENT = "ElonTweetMarkets research@example.com"

START_DATE = "2024-01-01"


# ---------------------------------------------------------------------------
# Source 1: yfinance - Tesla earnings dates
# ---------------------------------------------------------------------------

def fetch_yfinance_earnings() -> list[dict]:
    """Fetch Tesla earnings dates from yfinance."""
    print("\n[yfinance] Fetching Tesla earnings dates...")
    events = []

    try:
        tsla = yf.Ticker("TSLA")
        earnings = tsla.earnings_dates
        if earnings is None or earnings.empty:
            print("  WARNING: No earnings dates returned from yfinance")
            return events

        for dt_idx, row in earnings.iterrows():
            date_str = dt_idx.strftime("%Y-%m-%d")
            if date_str < START_DATE:
                continue

            # Determine if actual EPS is available (past) or not (upcoming)
            eps_estimate = row.get("EPS Estimate", None)
            eps_actual = row.get("Reported EPS", None)
            surprise = row.get("Surprise(%)", None)

            desc_parts = []
            if pd.notna(eps_estimate):
                desc_parts.append(f"EPS Est: {eps_estimate}")
            if pd.notna(eps_actual):
                desc_parts.append(f"Reported: {eps_actual}")
            if pd.notna(surprise):
                desc_parts.append(f"Surprise: {surprise}%")

            description = "; ".join(desc_parts) if desc_parts else "Tesla quarterly earnings call"

            events.append({
                "date": date_str,
                "company": "Tesla",
                "event_type": "earnings",
                "title": f"Tesla Earnings Call ({date_str[:7]})",
                "description": description,
                "expected_tweet_impact": "high",
            })

        print(f"  Found {len(events)} earnings dates")
    except Exception as e:
        print(f"  ERROR: {e}")

    return events


# ---------------------------------------------------------------------------
# Source 2: SEC EDGAR - Tesla 8-K filings
# ---------------------------------------------------------------------------

def fetch_edgar_8k() -> list[dict]:
    """Fetch Tesla 8-K filings from SEC EDGAR submissions API."""
    print("\n[EDGAR] Fetching Tesla 8-K filings...")
    events = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": EDGAR_USER_AGENT,
        "Accept": "application/json",
    })

    try:
        # Fetch main submission file
        url = f"{EDGAR_BASE}/CIK{TESLA_CIK}.json"
        print(f"  Requesting {url}")
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        company_name = data.get("name", "Tesla, Inc.")
        print(f"  Company: {company_name}")

        # Process recent filings from main response
        recent = data.get("filings", {}).get("recent", {})
        events.extend(_parse_edgar_filings(recent))

        # Check for older filing pages
        filing_files = data.get("filings", {}).get("files", [])
        for file_info in filing_files:
            fname = file_info.get("name", "")
            if not fname:
                continue
            time.sleep(0.5)  # Rate limit
            file_url = f"{EDGAR_BASE}/{fname}"
            print(f"  Fetching additional filings: {fname}")
            try:
                resp2 = session.get(file_url, timeout=30)
                resp2.raise_for_status()
                older = resp2.json()
                events.extend(_parse_edgar_filings(older))
            except Exception as e:
                print(f"    WARNING: Failed to fetch {fname}: {e}")

        print(f"  Found {len(events)} 8-K filings since {START_DATE}")
    except Exception as e:
        print(f"  ERROR: {e}")

    return events


def _parse_edgar_filings(filings_dict: dict) -> list[dict]:
    """Parse EDGAR filings dict into event records, filtering to 8-K since START_DATE."""
    events = []
    forms = filings_dict.get("form", [])
    dates = filings_dict.get("filingDate", [])
    accessions = filings_dict.get("accessionNumber", [])
    descriptions = filings_dict.get("primaryDocDescription", [])

    for i in range(len(forms)):
        form = forms[i] if i < len(forms) else ""
        date = dates[i] if i < len(dates) else ""
        accession = accessions[i] if i < len(accessions) else ""
        desc = descriptions[i] if i < len(descriptions) else ""

        # Only 8-K filings
        if form not in ("8-K", "8-K/A"):
            continue
        # Only since START_DATE
        if date < START_DATE:
            continue

        # Determine impact based on description
        impact = _classify_8k_impact(desc)

        events.append({
            "date": date,
            "company": "Tesla",
            "event_type": "filing",
            "title": f"Tesla {form}: {desc}" if desc else f"Tesla {form} Filing",
            "description": f"Accession: {accession}",
            "expected_tweet_impact": impact,
        })

    return events


def _classify_8k_impact(description: str) -> str:
    """Classify 8-K filing impact on tweet likelihood."""
    desc_lower = (description or "").lower()
    high_keywords = ["earnings", "ceo", "executive", "acquisition", "merger", "restructur"]
    medium_keywords = ["director", "board", "agreement", "amendment", "regulation"]

    for kw in high_keywords:
        if kw in desc_lower:
            return "high"
    for kw in medium_keywords:
        if kw in desc_lower:
            return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Source 3: Manual curation - High-impact events
# ---------------------------------------------------------------------------

def get_manual_events() -> list[dict]:
    """Return a curated list of high-impact corporate events."""
    print("\n[Manual] Loading curated corporate events...")

    events = [
        # -- Tesla --
        {
            "date": "2024-01-25",
            "company": "Tesla",
            "event_type": "earnings",
            "title": "Tesla Q4 2023 Earnings Call",
            "description": "Q4 2023 results; revenue miss, margin concerns",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-03-01",
            "company": "Tesla",
            "event_type": "product_launch",
            "title": "Tesla FSD v12 Wide Release",
            "description": "FSD v12 neural net end-to-end driving released to all FSD subscribers",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-04-23",
            "company": "Tesla",
            "event_type": "earnings",
            "title": "Tesla Q1 2024 Earnings Call",
            "description": "Q1 2024 results; accelerated affordable model timeline announced",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-06-13",
            "company": "Tesla",
            "event_type": "milestone",
            "title": "Tesla Annual Shareholder Meeting 2024",
            "description": "Musk $56B pay package re-approved; reincorporation to Texas approved",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-07-23",
            "company": "Tesla",
            "event_type": "earnings",
            "title": "Tesla Q2 2024 Earnings Call",
            "description": "Q2 2024 results; robotaxi event announced for October",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-10-10",
            "company": "Tesla",
            "event_type": "product_launch",
            "title": "Tesla Robotaxi (We, Robot) Event",
            "description": "Cybercab robotaxi unveiled, Robovan concept shown, Optimus demo",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-10-23",
            "company": "Tesla",
            "event_type": "earnings",
            "title": "Tesla Q3 2024 Earnings Call",
            "description": "Q3 2024 results; margins beat, 2025 growth guidance",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-11-21",
            "company": "Tesla",
            "event_type": "product_launch",
            "title": "Cybertruck First Anniversary / Foundation Series End",
            "description": "Cybertruck ramp-up milestone, regular production begins",
            "expected_tweet_impact": "medium",
        },
        {
            "date": "2025-01-29",
            "company": "Tesla",
            "event_type": "earnings",
            "title": "Tesla Q4 2024 Earnings Call",
            "description": "Q4 2024 results; FY2024 deliveries, 2025 outlook",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2025-06-01",
            "company": "Tesla",
            "event_type": "product_launch",
            "title": "Tesla Robotaxi Launch Target (Austin)",
            "description": "Planned launch of paid unsupervised FSD robotaxi service in Austin, TX",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2025-10-09",
            "company": "Tesla",
            "event_type": "product_launch",
            "title": "Tesla FSD Unsupervised Expansion",
            "description": "FSD unsupervised expansion beyond Austin test market",
            "expected_tweet_impact": "high",
        },
        # -- xAI --
        {
            "date": "2024-03-17",
            "company": "xAI",
            "event_type": "product_launch",
            "title": "Grok-1 Open Source Release",
            "description": "xAI open-sourced Grok-1 (314B parameter MoE model) on GitHub",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-05-02",
            "company": "xAI",
            "event_type": "funding",
            "title": "xAI Series B - $6B Round",
            "description": "xAI raised $6B Series B at $18B valuation",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-08-13",
            "company": "xAI",
            "event_type": "product_launch",
            "title": "Grok-2 Launch",
            "description": "Grok-2 and Grok-2 mini released; image generation via Flux",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-11-05",
            "company": "xAI",
            "event_type": "milestone",
            "title": "xAI Colossus Cluster Online",
            "description": "100K H100 GPU Colossus supercomputer cluster fully operational in Memphis",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-12-21",
            "company": "xAI",
            "event_type": "funding",
            "title": "xAI Series C - $6B Round",
            "description": "xAI raised $6B Series C at $40B+ valuation",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2025-02-17",
            "company": "xAI",
            "event_type": "product_launch",
            "title": "Grok-3 Launch",
            "description": "Grok-3 released; largest training run, reasoning capabilities",
            "expected_tweet_impact": "high",
        },
        # -- Neuralink --
        {
            "date": "2024-01-29",
            "company": "Neuralink",
            "event_type": "milestone",
            "title": "Neuralink First Human Implant",
            "description": "First human patient received Neuralink brain implant (Noland Arbaugh)",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-03-20",
            "company": "Neuralink",
            "event_type": "milestone",
            "title": "Neuralink Patient Plays Chess",
            "description": "First Neuralink patient demonstrated playing chess with brain implant",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-07-16",
            "company": "Neuralink",
            "event_type": "milestone",
            "title": "Neuralink Second Human Implant",
            "description": "Second human patient received Neuralink implant",
            "expected_tweet_impact": "medium",
        },
        {
            "date": "2024-09-17",
            "company": "Neuralink",
            "event_type": "milestone",
            "title": "Neuralink Blindsight Prototype",
            "description": "FDA Breakthrough Device designation for Blindsight (vision restoration)",
            "expected_tweet_impact": "high",
        },
        # -- SpaceX (non-launch milestones, launches already in separate data) --
        {
            "date": "2024-03-14",
            "company": "SpaceX",
            "event_type": "milestone",
            "title": "Starship IFT-3",
            "description": "Starship third integrated flight test; reached space, breakup on reentry",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-06-06",
            "company": "SpaceX",
            "event_type": "milestone",
            "title": "Starship IFT-4",
            "description": "Starship fourth test; both stages achieved controlled splashdown",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-10-13",
            "company": "SpaceX",
            "event_type": "milestone",
            "title": "Starship IFT-5 - Chopstick Catch",
            "description": "Super Heavy booster caught by launch tower chopsticks; historic first",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2024-11-19",
            "company": "SpaceX",
            "event_type": "milestone",
            "title": "Starship IFT-6",
            "description": "Sixth flight test; booster catch attempt, Starship controlled reentry",
            "expected_tweet_impact": "high",
        },
        {
            "date": "2025-01-16",
            "company": "SpaceX",
            "event_type": "milestone",
            "title": "Starship IFT-7",
            "description": "Seventh flight test; successful booster catch, upper stage breakup",
            "expected_tweet_impact": "high",
        },
        # -- The Boring Company --
        {
            "date": "2024-06-14",
            "company": "Tesla",
            "event_type": "milestone",
            "title": "Vegas Loop Expansion Approval",
            "description": "Las Vegas approved major Boring Company Loop expansion to resort corridor",
            "expected_tweet_impact": "medium",
        },
        # -- DOGE (government) --
        {
            "date": "2025-01-20",
            "company": "xAI",
            "event_type": "milestone",
            "title": "DOGE Department Established",
            "description": "Department of Government Efficiency officially established; Musk as advisor",
            "expected_tweet_impact": "high",
        },
    ]

    # Filter to only events since START_DATE
    events = [e for e in events if e["date"] >= START_DATE]
    print(f"  Loaded {len(events)} curated events")
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_corporate_events(sources: list[str], dry_run: bool = False) -> pd.DataFrame:
    """Fetch and merge corporate events from specified sources."""
    all_events = []

    if "yfinance" in sources or "all" in sources:
        all_events.extend(fetch_yfinance_earnings())

    if "edgar" in sources or "all" in sources:
        all_events.extend(fetch_edgar_8k())

    if "manual" in sources or "all" in sources:
        all_events.extend(get_manual_events())

    if not all_events:
        print("\nNo events collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_events)

    # Deduplicate: same date + company + event_type -> keep first (manual > yfinance dupes)
    df = df.drop_duplicates(subset=["date", "company", "event_type", "title"], keep="first")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"CORPORATE EVENTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total events: {len(df)}")
    print(f"\nBy source (event_type):")
    print(df["event_type"].value_counts().to_string())
    print(f"\nBy company:")
    print(df["company"].value_counts().to_string())
    print(f"\nBy impact:")
    print(df["expected_tweet_impact"].value_counts().to_string())
    print(f"\nDate range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    if dry_run:
        print(f"\n[DRY RUN] Would save {len(df)} events to {OUTPUT_PATH}")
        print(f"\nSample events:")
        print(df.head(20).to_string(index=False))
    else:
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"\nSaved {len(df)} events to {OUTPUT_PATH}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch corporate events for Elon tweet prediction")
    parser.add_argument("--source", default="all", choices=["all", "yfinance", "edgar", "manual"],
                        help="Which source to fetch (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    print("=" * 60)
    print("CORPORATE EVENTS FETCH")
    print(f"Source: {args.source}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    sources = [args.source]
    build_corporate_events(sources, dry_run=args.dry_run)
    print("\nDone!")


if __name__ == "__main__":
    main()
