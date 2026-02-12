"""
Consolidate per-event and daily incremental price parquet files into a
single master price_history.parquet.

Pure data transformation - NO API calls. Reads parquet files written by
fetch_clob_prices.py and merges them with enrichment from market_catalog.

Usage:
    python scripts/consolidate_prices.py              # Full rebuild from events/ files
    python scripts/consolidate_prices.py --daily       # Incremental merge of new daily/ files
"""

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICES_DIR = PROJECT_ROOT / "data" / "sources" / "polymarket" / "prices"
EVENTS_DIR = PRICES_DIR / "events"
DAILY_DIR = PRICES_DIR / "daily"
MASTER_FILE = PRICES_DIR / "price_history.parquet"
CATALOG_FILE = PROJECT_ROOT / "data" / "processed" / "market_catalog.parquet"
STATE_FILE = PRICES_DIR / "_consolidation_state.json"

# Master schema column order
MASTER_COLUMNS = [
    "event_id",
    "event_slug",
    "market_type",
    "bucket_label",
    "token_id",
    "timestamp",
    "price",
    "is_resolved",
    "fetch_date",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_state() -> dict:
    """Load consolidation state JSON, returning empty defaults if missing."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_full_build": None,
        "events_consolidated": 0,
        "total_rows": 0,
        "daily_files_merged": [],
        "last_daily_merge": None,
    }


def save_state(state: dict) -> None:
    """Write consolidation state JSON to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def load_catalog_enrichment():
    """Load market catalog and extract event-level enrichment mapping.

    Returns a DataFrame with unique (event_id -> market_type, is_resolved),
    or None if the catalog file is missing.
    """
    if not CATALOG_FILE.exists():
        print(f"  WARNING: market_catalog.parquet not found at {CATALOG_FILE}")
        print("           Skipping market_type / is_resolved enrichment.")
        return None

    catalog = pd.read_parquet(CATALOG_FILE)

    # Keep one row per event_id with its market_type and is_resolved
    enrichment = (
        catalog[["event_id", "market_type", "is_resolved"]]
        .drop_duplicates(subset=["event_id"], keep="first")
        .copy()
    )
    enrichment["event_id"] = enrichment["event_id"].astype(str)
    return enrichment


def read_parquet_files(file_list: list, desc: str = "Reading") -> pd.DataFrame:
    """Read a list of parquet files with tqdm progress, returning concatenated DataFrame."""
    frames = []
    for fpath in tqdm(file_list, desc=desc, unit="file"):
        try:
            df = pd.read_parquet(fpath)
            if df.empty:
                continue
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Failed to read {fpath.name}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def enrich_dataframe(df: pd.DataFrame, enrichment) -> pd.DataFrame:
    """Left-join price data with catalog enrichment on event_id."""
    if enrichment is None or df.empty:
        return df

    df["event_id"] = df["event_id"].astype(str)
    df = df.merge(
        enrichment, on="event_id", how="left", suffixes=("", "_catalog")
    )

    # Use catalog values for market_type and is_resolved if present
    if "market_type_catalog" in df.columns:
        if "market_type" in df.columns:
            df["market_type"] = df["market_type_catalog"].combine_first(
                df["market_type"]
            )
        else:
            df["market_type"] = df["market_type_catalog"]
        df.drop(columns=["market_type_catalog"], inplace=True, errors="ignore")

    if "is_resolved_catalog" in df.columns:
        if "is_resolved" in df.columns:
            df["is_resolved"] = df["is_resolved_catalog"].combine_first(
                df["is_resolved"]
            )
        else:
            df["is_resolved"] = df["is_resolved_catalog"]
        df.drop(columns=["is_resolved_catalog"], inplace=True, errors="ignore")

    n_enriched = df["market_type"].notna().sum() if "market_type" in df.columns else 0
    print(f"  Enriched {n_enriched:,} / {len(df):,} rows with catalog data.")
    return df


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the master schema types and ensure column order."""
    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    # String columns
    for col in ["event_id", "event_slug", "market_type", "bucket_label", "token_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Timestamp: ensure UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Price: float32
    if "price" in df.columns:
        df["price"] = df["price"].astype("float32")

    # is_resolved: bool (fill missing as False)
    if "is_resolved" in df.columns:
        df["is_resolved"] = df["is_resolved"].fillna(False).astype(bool)
    else:
        df["is_resolved"] = False

    # fetch_date: date
    if "fetch_date" in df.columns:
        df["fetch_date"] = pd.to_datetime(df["fetch_date"]).dt.date
    else:
        df["fetch_date"] = date.today()

    # Ensure all master columns exist
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[MASTER_COLUMNS]


def dedup_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate on (event_id, token_id, timestamp), keeping latest fetch_date."""
    if df.empty:
        return df
    df = df.sort_values("fetch_date", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["event_id", "token_id", "timestamp"], keep="first")
    return df


def sort_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (event_slug, bucket_label, timestamp)."""
    if df.empty:
        return df
    return df.sort_values(
        ["event_slug", "bucket_label", "timestamp"]
    ).reset_index(drop=True)


def print_summary(df: pd.DataFrame, label: str = "Master") -> None:
    """Print a clear summary of the consolidated DataFrame."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{label} Summary")
    print(sep)

    print(f"  Total rows:          {len(df):,}")
    n_events = df["event_id"].nunique()
    n_tokens = df["token_id"].nunique()
    print(f"  Unique events:       {n_events}")
    print(f"  Unique tokens:       {n_tokens}")

    # Date range
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        print(f"  Timestamp range:     {ts_min} to {ts_max}")

    # market_type breakdown
    if "market_type" in df.columns:
        mt_counts = df.groupby("market_type")["event_id"].nunique()
        print("\n  Market type breakdown (events):")
        for mt, cnt in sorted(mt_counts.items(), key=lambda x: -x[1]):
            print(f"    {mt:12s}: {cnt} events")

    # is_resolved breakdown
    if "is_resolved" in df.columns:
        resolved = df[df["is_resolved"]]["event_id"].nunique()
        unresolved = df[~df["is_resolved"]]["event_id"].nunique()
        print(f"\n  Resolved events:     {resolved}")
        print(f"  Unresolved events:   {unresolved}")

    # File size
    if MASTER_FILE.exists():
        size_bytes = MASTER_FILE.stat().st_size
        if size_bytes > 1_000_000:
            print(f"\n  File size:           {size_bytes / 1_000_000:.1f} MB")
        else:
            print(f"\n  File size:           {size_bytes / 1_000:.1f} KB")

    print(sep)


# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------
def full_rebuild() -> None:
    """Rebuild price_history.parquet from all events/ and daily/ parquet files."""
    sep = "=" * 60
    print(sep)
    print("Full Rebuild: Consolidating price history")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Read events/ parquet files
    # ------------------------------------------------------------------
    print(f"\nScanning {EVENTS_DIR} for parquet files...")
    event_files = sorted(EVENTS_DIR.glob("*.parquet")) if EVENTS_DIR.exists() else []

    if not event_files:
        print("  No parquet files found in events/ directory.")
        print("  Ensure fetch_clob_prices.py output is in data/sources/polymarket/prices/events/")
    else:
        print(f"  Found {len(event_files)} event parquet files.")

    df_events = read_parquet_files(event_files, desc="Reading events")
    print(f"  Event rows loaded: {len(df_events):,}")

    # ------------------------------------------------------------------
    # 2. Enrich with market catalog
    # ------------------------------------------------------------------
    print("\nLoading market catalog for enrichment...")
    enrichment = load_catalog_enrichment()
    df_events = enrich_dataframe(df_events, enrichment)

    # Add fetch_date
    if not df_events.empty and "fetch_date" not in df_events.columns:
        df_events["fetch_date"] = date.today()

    # ------------------------------------------------------------------
    # 3. Read daily/ parquet files (if any)
    # ------------------------------------------------------------------
    print(f"\nScanning {DAILY_DIR} for daily parquet files...")
    daily_files = sorted(DAILY_DIR.glob("*.parquet")) if DAILY_DIR.exists() else []

    df_daily = pd.DataFrame()
    if not daily_files:
        print("  No daily parquet files found (OK for initial build).")
    else:
        print(f"  Found {len(daily_files)} daily parquet files.")
        df_daily = read_parquet_files(daily_files, desc="Reading daily")
        print(f"  Daily rows loaded: {len(df_daily):,}")

    # ------------------------------------------------------------------
    # 4. Combine everything
    # ------------------------------------------------------------------
    frames = [f for f in [df_events, df_daily] if not f.empty]
    if not frames:
        print("\nNo price data found. Nothing to consolidate.")
        print("Run fetch_clob_prices.py first and place output in events/ subdirectory.")
        return

    df_combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined rows before dedup: {len(df_combined):,}")

    # ------------------------------------------------------------------
    # 5. Enforce schema, dedup, sort
    # ------------------------------------------------------------------
    df_combined = enforce_schema(df_combined)
    df_combined = dedup_prices(df_combined)
    df_combined = sort_prices(df_combined)
    print(f"Rows after dedup:          {len(df_combined):,}")

    # ------------------------------------------------------------------
    # 6. Write master parquet
    # ------------------------------------------------------------------
    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(MASTER_FILE, index=False, engine="pyarrow")
    print(f"\nSaved: {MASTER_FILE}")

    # ------------------------------------------------------------------
    # 7. Write consolidation state
    # ------------------------------------------------------------------
    daily_names = [f.name for f in daily_files]
    state = {
        "last_full_build": datetime.now(timezone.utc).isoformat(),
        "events_consolidated": int(df_combined["event_id"].nunique()),
        "total_rows": len(df_combined),
        "daily_files_merged": daily_names,
        "last_daily_merge": datetime.now(timezone.utc).isoformat() if daily_files else None,
    }
    save_state(state)
    print(f"Saved: {STATE_FILE}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print_summary(df_combined, label="Full Rebuild")


# ---------------------------------------------------------------------------
# Incremental daily merge
# ---------------------------------------------------------------------------
def daily_merge() -> None:
    """Incrementally merge new daily/ parquet files into existing master."""
    sep = "=" * 60
    print(sep)
    print("Incremental Daily Merge")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Load existing master
    # ------------------------------------------------------------------
    if not MASTER_FILE.exists():
        print(f"\nERROR: {MASTER_FILE} not found.")
        print("Run a full rebuild first:  python scripts/consolidate_prices.py")
        return

    print(f"\nLoading existing master: {MASTER_FILE}")
    df_master = pd.read_parquet(MASTER_FILE)
    print(f"  Existing rows: {len(df_master):,}")

    # ------------------------------------------------------------------
    # 2. Load state to find already-merged daily files
    # ------------------------------------------------------------------
    state = load_state()
    already_merged = set(state.get("daily_files_merged", []))
    print(f"  Previously merged daily files: {len(already_merged)}")

    # ------------------------------------------------------------------
    # 3. Find new daily files
    # ------------------------------------------------------------------
    print(f"\nScanning {DAILY_DIR} for new daily parquet files...")
    all_daily_files = sorted(DAILY_DIR.glob("*.parquet")) if DAILY_DIR.exists() else []
    new_daily_files = [f for f in all_daily_files if f.name not in already_merged]

    if not new_daily_files:
        print("  Nothing to merge - no new daily files found.")
        return

    print(f"  New daily files to merge: {len(new_daily_files)}")

    # ------------------------------------------------------------------
    # 4. Read and concat new daily files
    # ------------------------------------------------------------------
    df_new = read_parquet_files(new_daily_files, desc="Reading new daily")
    if df_new.empty:
        print("  All new daily files were empty. Nothing to merge.")
        return

    print(f"  New rows loaded: {len(df_new):,}")

    # Enrich new daily data with catalog if needed
    enrichment = load_catalog_enrichment()
    df_new = enrich_dataframe(df_new, enrichment)

    # ------------------------------------------------------------------
    # 5. Append to master, dedup, sort
    # ------------------------------------------------------------------
    df_new = enforce_schema(df_new)
    df_master = enforce_schema(df_master)

    df_combined = pd.concat([df_master, df_new], ignore_index=True)
    print(f"\nCombined rows before dedup: {len(df_combined):,}")

    df_combined = dedup_prices(df_combined)
    df_combined = sort_prices(df_combined)
    print(f"  Rows after dedup:          {len(df_combined):,}")

    rows_added = len(df_combined) - len(df_master)
    print(f"  Net new rows:              {rows_added:,}")

    # ------------------------------------------------------------------
    # 6. Write updated master
    # ------------------------------------------------------------------
    df_combined.to_parquet(MASTER_FILE, index=False, engine="pyarrow")
    print(f"\nSaved: {MASTER_FILE}")

    # ------------------------------------------------------------------
    # 7. Update state
    # ------------------------------------------------------------------
    merged_names = already_merged | {f.name for f in new_daily_files}
    state["daily_files_merged"] = sorted(merged_names)
    state["last_daily_merge"] = datetime.now(timezone.utc).isoformat()
    state["total_rows"] = len(df_combined)
    state["events_consolidated"] = int(df_combined["event_id"].nunique())
    save_state(state)
    print(f"Saved: {STATE_FILE}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print_summary(df_combined, label="Daily Merge")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Consolidate per-event and daily price parquet files into a master price_history.parquet."
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Incremental merge: only process new daily/ files into existing master.",
    )
    args = parser.parse_args()

    if args.daily:
        daily_merge()
    else:
        full_rebuild()

    print("\nDone!")


if __name__ == "__main__":
    main()
