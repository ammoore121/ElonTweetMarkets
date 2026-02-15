#!/usr/bin/env python3
"""Fetch Tesla stock and crypto (DOGE, BTC) daily price data via yfinance.

Stores:
  data/sources/market/tesla_daily.parquet
  data/sources/market/crypto_daily.parquet
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json
from datetime import datetime

import pandas as pd
import yfinance as yf


MARKET_DIR = PROJECT_DIR / "data" / "sources" / "market"
MARKET_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


def fetch_tesla():
    """Download TSLA daily OHLCV data."""
    print(f"Fetching TSLA from {START_DATE} to {END_DATE}...")
    tsla = yf.download("TSLA", start=START_DATE, end=END_DATE, progress=False)
    if tsla.empty:
        print("ERROR: No TSLA data returned")
        return

    # Flatten multi-level columns if present
    if isinstance(tsla.columns, pd.MultiIndex):
        tsla.columns = tsla.columns.get_level_values(0)

    tsla = tsla.reset_index()
    tsla.columns = [c.lower().replace(" ", "_") for c in tsla.columns]

    # Add derived features
    tsla["pct_change"] = tsla["close"].pct_change()
    tsla["abs_pct_change"] = tsla["pct_change"].abs()
    tsla["volume_ma5"] = tsla["volume"].rolling(5).mean()
    tsla["volatility_5d"] = tsla["pct_change"].rolling(5).std()
    tsla["gap"] = (tsla["open"] - tsla["close"].shift(1)) / tsla["close"].shift(1)

    # Ensure date column is string for consistency
    tsla["date"] = pd.to_datetime(tsla["date"]).dt.strftime("%Y-%m-%d")

    out_path = MARKET_DIR / "tesla_daily.parquet"
    tsla.to_parquet(out_path, index=False)
    print(f"  Saved {len(tsla)} rows to {out_path}")
    print(f"  Date range: {tsla['date'].iloc[0]} to {tsla['date'].iloc[-1]}")
    print(f"  Columns: {list(tsla.columns)}")
    return tsla


def fetch_crypto():
    """Download DOGE-USD and BTC-USD daily data."""
    frames = []
    for ticker in ["DOGE-USD", "BTC-USD"]:
        symbol = ticker.split("-")[0].lower()
        print(f"Fetching {ticker} from {START_DATE} to {END_DATE}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Add derived features
        df["pct_change"] = df["close"].pct_change()
        df["volatility_5d"] = df["pct_change"].rolling(5).std()

        # Prefix columns with symbol
        rename_cols = {}
        for c in df.columns:
            if c != "date":
                rename_cols[c] = f"{symbol}_{c}"
        df = df.rename(columns=rename_cols)
        frames.append(df)

    if not frames:
        print("ERROR: No crypto data returned")
        return

    # Merge on date
    crypto = frames[0]
    for f in frames[1:]:
        crypto = crypto.merge(f, on="date", how="outer")
    crypto = crypto.sort_values("date").reset_index(drop=True)

    out_path = MARKET_DIR / "crypto_daily.parquet"
    crypto.to_parquet(out_path, index=False)
    print(f"  Saved {len(crypto)} rows to {out_path}")
    print(f"  Date range: {crypto['date'].iloc[0]} to {crypto['date'].iloc[-1]}")
    print(f"  Columns: {list(crypto.columns)}")
    return crypto


def main():
    print("=" * 60)
    print("MARKET DATA FETCH")
    print("=" * 60)
    fetch_tesla()
    print()
    fetch_crypto()
    print("\nDone!")


if __name__ == "__main__":
    main()
