#!/usr/bin/env python3
"""Fetch VIX (CBOE Volatility Index) daily data via yfinance.

Stores:
  data/sources/market/vix_daily.parquet
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from datetime import datetime

import pandas as pd
import yfinance as yf


MARKET_DIR = PROJECT_DIR / "data" / "sources" / "market"
MARKET_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


def fetch_vix():
    """Download ^VIX daily OHLCV data."""
    print(f"Fetching ^VIX from {START_DATE} to {END_DATE}...")
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False)
    if vix.empty:
        print("ERROR: No VIX data returned")
        return

    # Flatten multi-level columns if present
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix = vix.reset_index()
    vix.columns = [c.lower().replace(" ", "_") for c in vix.columns]

    # Add derived features
    vix["pct_change"] = vix["close"].pct_change()
    vix["pct_change_5d"] = vix["close"].pct_change(periods=5)
    vix["ma5"] = vix["close"].rolling(5).mean()
    vix["ma5_ratio"] = vix["close"] / vix["ma5"]

    # Level category: low (<15), medium (15-25), high (25-35), extreme (>35)
    def categorize_vix(val):
        if pd.isna(val):
            return None
        if val < 15:
            return "low"
        elif val < 25:
            return "medium"
        elif val < 35:
            return "high"
        else:
            return "extreme"

    vix["level_category"] = vix["close"].apply(categorize_vix)

    # Ensure date column is string for consistency
    vix["date"] = pd.to_datetime(vix["date"]).dt.strftime("%Y-%m-%d")

    out_path = MARKET_DIR / "vix_daily.parquet"
    vix.to_parquet(out_path, index=False)
    print(f"  Saved {len(vix)} rows to {out_path}")
    print(f"  Date range: {vix['date'].iloc[0]} to {vix['date'].iloc[-1]}")
    print(f"  Columns: {list(vix.columns)}")
    return vix


def main():
    print("=" * 60)
    print("VIX DATA FETCH")
    print("=" * 60)
    fetch_vix()
    print("\nDone!")


if __name__ == "__main__":
    main()
