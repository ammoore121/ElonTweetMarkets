"""
Download Elon Musk tweet history from Kaggle using REST API.
Reads KAGGLE_API_TOKEN from .env file.

Usage: python scripts/fetch_tweet_history.py
"""
import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "data" / "tweets"

load_dotenv(PROJECT / ".env")

KAGGLE_TOKEN = os.environ.get("KAGGLE_API_TOKEN")

DATASETS = [
    {
        "id": "dadalyndell/elon-musk-tweets-2010-to-2025-march",
        "name": "Elon Musk Tweets 2010-2025",
    },
    {
        "id": "aryansingh0909/elon-musk-tweets-updated-daily",
        "name": "Elon Musk Tweets (Daily Updated)",
    },
]


def download_dataset(dataset_id: str, output_dir: Path) -> bool:
    """Download a Kaggle dataset via REST API with bearer token."""
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
    headers = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}

    print(f"  GET {url}")
    resp = requests.get(url, headers=headers, stream=True, timeout=120)

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:300]}")
        return False

    # Download into memory then unzip
    content_length = resp.headers.get("Content-Length", "unknown")
    print(f"  Downloading ({content_length} bytes)...")

    data = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        data.write(chunk)
        downloaded += len(chunk)
        print(f"  ... {downloaded / (1024*1024):.1f} MB", end="\r")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to unzip
    data.seek(0)
    try:
        with zipfile.ZipFile(data) as zf:
            names = zf.namelist()
            print(f"  Zip contains: {names}")
            zf.extractall(output_dir)
            print(f"  Extracted {len(names)} files to {output_dir}")
    except zipfile.BadZipFile:
        # Maybe it's a raw CSV
        data.seek(0)
        fname = dataset_id.split("/")[-1] + ".csv"
        out_path = output_dir / fname
        out_path.write_bytes(data.read())
        print(f"  Saved raw file to {out_path}")

    return True


def main():
    print("=" * 60)
    print("Kaggle Tweet History Fetch (REST API)")
    print("=" * 60)

    if not KAGGLE_TOKEN:
        print("\nERROR: KAGGLE_API_TOKEN not found in .env")
        print("Add: KAGGLE_API_TOKEN=KGAT_your_token_here")
        return

    print(f"Token: {KAGGLE_TOKEN[:10]}...{KAGGLE_TOKEN[-4:]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        print(f"\n--- {ds['name']} ---")
        print(f"  Dataset: {ds['id']}")
        success = download_dataset(ds["id"], OUTPUT_DIR)
        if not success:
            print(f"  FAILED to download {ds['id']}")

    # List what we got
    all_files = sorted(OUTPUT_DIR.glob("*.csv"))
    print(f"\n{'=' * 60}")
    print(f"Files downloaded: {len(all_files)}")
    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    # Quick stats
    if all_files:
        try:
            import pandas as pd
            for csv_file in all_files:
                print(f"\nStats for {csv_file.name}:")
                df = pd.read_csv(csv_file, low_memory=False)
                print(f"  Rows: {len(df):,}")
                print(f"  Columns: {list(df.columns)}")
                # Try to find date column
                for col in df.columns:
                    if "date" in col.lower() or "time" in col.lower() or "created" in col.lower():
                        print(f"  Date column '{col}': {df[col].iloc[0]} ... {df[col].iloc[-1]}")
                        break
        except Exception as e:
            print(f"  Error reading stats: {e}")

    # Save metadata
    meta = {
        "fetched_at": datetime.now().isoformat(),
        "datasets": DATASETS,
        "files": [{"name": f.name, "size_bytes": f.stat().st_size} for f in all_files],
    }
    meta_path = OUTPUT_DIR / "download_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Metadata: {meta_path}")


if __name__ == "__main__":
    main()
