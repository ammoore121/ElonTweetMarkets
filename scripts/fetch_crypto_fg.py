#!/usr/bin/env python3
"""Fetch Crypto Fear & Greed Index from alternative.me API.

Free API, no auth required. Returns full history of daily 0-100 values.
API: https://api.alternative.me/fng/?limit=0&format=json

Stores:
  data/sources/market/crypto_fear_greed.parquet
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from datetime import datetime

import pandas as pd
import requests


MARKET_DIR = PROJECT_DIR / "data" / "sources" / "market"
MARKET_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def fetch_fear_greed():
    """Download full Crypto Fear & Greed Index history."""
    print(f"Fetching Crypto Fear & Greed Index from alternative.me...")
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = data.get("data", [])
    if not records:
        print("ERROR: No data returned")
        return

    rows = []
    for rec in records:
        ts = int(rec["timestamp"])
        dt = datetime.utcfromtimestamp(ts)
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "fg_value": int(rec["value"]),
            "fg_classification": rec.get("value_classification", ""),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    # Filter to 2024+ for consistency with other sources
    df = df[df["date"] >= "2024-01-01"].reset_index(drop=True)

    # Add derived features
    df["fg_7d_avg"] = df["fg_value"].rolling(7, min_periods=1).mean()
    df["fg_3d_avg"] = df["fg_value"].rolling(3, min_periods=1).mean()
    df["fg_delta"] = df["fg_3d_avg"] - df["fg_7d_avg"]

    # Category based on value
    def categorize_fg(val):
        if pd.isna(val):
            return None
        if val <= 25:
            return "extreme_fear"
        elif val <= 45:
            return "fear"
        elif val <= 55:
            return "neutral"
        elif val <= 75:
            return "greed"
        else:
            return "extreme_greed"

    df["fg_category"] = df["fg_value"].apply(categorize_fg)

    out_path = MARKET_DIR / "crypto_fear_greed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Current value: {df['fg_value'].iloc[-1]} ({df['fg_classification'].iloc[-1]})")
    return df


def main():
    print("=" * 60)
    print("CRYPTO FEAR & GREED INDEX FETCH")
    print("=" * 60)
    fetch_fear_greed()
    print("\nDone!")


if __name__ == "__main__":
    main()
