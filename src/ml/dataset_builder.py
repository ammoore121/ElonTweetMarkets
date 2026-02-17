"""
Bucket-level dataset builder for ML models.

Converts the 159 backtest events into ~2,725 (event, bucket) rows suitable
for XGBoost training. Each row represents a single bucket within a single
event, with features at both the event level (113 scalar features) and the
bucket level (structural position, crowd price, etc.).

Usage:
    from src.ml.dataset_builder import build_bucket_dataset, build_single_event_rows

    # Full training dataset
    df = build_bucket_dataset(events)

    # Single event for inference
    rows = build_single_event_rows(features, buckets, context, metadata)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.feature_builder import TweetFeatureBuilder


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EVENTS_DIR = PROJECT_DIR / "data" / "backtest" / "events"


def _bucket_midpoint(bucket: dict, all_buckets: list[dict]) -> float:
    """Compute midpoint for a bucket, handling open-ended cases."""
    lower = int(bucket["lower_bound"])
    upper = int(bucket["upper_bound"])
    if upper >= 99999:
        widths = [
            int(b["upper_bound"]) - int(b["lower_bound"])
            for b in all_buckets
            if int(b["upper_bound"]) < 99999
        ]
        typical = sum(widths) / len(widths) if widths else 25
        return lower + typical / 2
    if lower <= 0:
        return upper / 2
    return (lower + upper) / 2


def _compute_crowd_stats(
    buckets: list[dict], entry_prices: dict
) -> tuple[float, float]:
    """Compute crowd-implied expected value and std dev from entry prices."""
    ev = 0.0
    total_p = 0.0
    for b in buckets:
        label = b["bucket_label"]
        price = entry_prices.get(label, 0.0)
        mid = _bucket_midpoint(b, buckets)
        ev += mid * price
        total_p += price

    if total_p > 0:
        ev /= total_p

    var = 0.0
    for b in buckets:
        label = b["bucket_label"]
        price = entry_prices.get(label, 0.0)
        mid = _bucket_midpoint(b, buckets)
        if total_p > 0:
            var += (price / total_p) * (mid - ev) ** 2

    std = math.sqrt(var) if var > 0 else 50.0
    return ev, std


def _get_heuristic_probs(
    features: dict, buckets: list[dict], context: dict
) -> dict[str, float]:
    """Get TailBoostModel predictions as a meta-feature.

    Lazy import to avoid circular dependencies.
    """
    try:
        from src.ml.duration_model import TailBoostModel
        model = TailBoostModel()
        return model.predict(features, buckets, context=context)
    except Exception:
        n = len(buckets)
        return {b["bucket_label"]: 1.0 / n for b in buckets}


def _load_entry_prices(event_dir: Path, metadata: dict) -> dict[str, float]:
    """Load entry prices at T-24h from price history or fallback to metadata."""
    prices_path = event_dir / "prices.parquet"
    entry_prices = {}

    if prices_path.exists():
        try:
            prices_df = pd.read_parquet(prices_path)
            end_str = metadata.get("end_date")
            if end_str and not prices_df.empty:
                end_dt = pd.Timestamp(end_str, tz="UTC")
                target_time = end_dt - pd.Timedelta(hours=24)
                window_start = target_time - pd.Timedelta(hours=6)
                window_end = target_time + pd.Timedelta(hours=6)

                ts_col = "timestamp"
                if ts_col in prices_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(prices_df[ts_col]):
                        prices_df[ts_col] = pd.to_datetime(prices_df[ts_col], utc=True)

                    windowed = prices_df[
                        (prices_df[ts_col] >= window_start)
                        & (prices_df[ts_col] <= window_end)
                    ]

                    if not windowed.empty:
                        windowed = windowed.copy()
                        windowed["_diff"] = (windowed[ts_col] - target_time).abs()
                        closest = windowed.sort_values("_diff").drop_duplicates(
                            subset=["bucket_label"], keep="first"
                        )
                        for _, row in closest.iterrows():
                            entry_prices[str(row["bucket_label"])] = float(row["price"])

                        if entry_prices:
                            return entry_prices
        except Exception:
            pass

    # Fallback: use final prices from metadata
    for bucket in metadata.get("buckets", []):
        label = bucket["bucket_label"]
        price = bucket.get("price_yes")
        if price is not None:
            entry_prices[label] = float(price)
        else:
            entry_prices[label] = 0.0

    return entry_prices


def build_single_event_rows(
    features: dict,
    buckets: list[dict],
    context: dict,
    metadata: dict,
    entry_prices: Optional[dict] = None,
) -> pd.DataFrame:
    """Build (bucket) rows for a single event.

    Args:
        features: Feature dict from features.json (nested by category).
        buckets: List of bucket dicts from metadata.
        context: Dict with 'duration_days', 'entry_prices'.
        metadata: Full event metadata dict.
        entry_prices: Optional pre-loaded entry prices.

    Returns:
        DataFrame with one row per bucket.
    """
    if entry_prices is None:
        entry_prices = context.get("entry_prices", {})

    # Flatten event-level features
    flat_features = TweetFeatureBuilder.flatten_features(features)

    # Compute crowd stats
    crowd_ev, crowd_std = _compute_crowd_stats(buckets, entry_prices)

    # Get heuristic predictions as meta-feature
    heuristic_probs = _get_heuristic_probs(features, buckets, context)

    n_buckets = len(buckets)
    sorted_buckets = sorted(buckets, key=lambda b: int(b["lower_bound"]))

    rows = []
    for idx, b in enumerate(sorted_buckets):
        label = b["bucket_label"]
        lower = int(b["lower_bound"])
        upper = int(b["upper_bound"])
        mid = _bucket_midpoint(b, buckets)
        width = upper - lower if upper < 99999 else (mid - lower) * 2

        crowd_price = entry_prices.get(label, 0.0)
        heuristic_prob = heuristic_probs.get(label, 1.0 / max(n_buckets, 1))

        # Distance from crowd EV in SD units
        distance_from_ev = (
            (mid - crowd_ev) / crowd_std if crowd_std > 0 else 0.0
        )

        # Is tail bucket (> 0.8 SD from EV)
        is_tail = 1.0 if abs(distance_from_ev) > 0.8 else 0.0

        row = {
            # Event-level identifiers (not features)
            "_event_slug": metadata.get("event_slug", ""),
            "_bucket_label": label,

            # Per-bucket structural features
            "bucket_position_normalized": idx / max(n_buckets - 1, 1),
            "bucket_midpoint": mid,
            "bucket_width": width,
            "bucket_lower": lower,
            "n_buckets": n_buckets,
            "crowd_price": crowd_price,
            "is_tail_bucket": is_tail,
            "distance_from_ev": distance_from_ev,
            "heuristic_prob": heuristic_prob,
            "heuristic_edge": heuristic_prob - crowd_price,

            # Event-level context
            "duration_days": context.get("duration_days", 7),
            "crowd_ev": crowd_ev,
            "crowd_std": crowd_std,
        }

        # Add all flattened event-level features
        row.update(flat_features)

        # Target columns (for training; will be NaN for inference)
        is_winner = b.get("is_winner", False)
        row["_won"] = 1.0 if is_winner else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_bucket_dataset(
    events: list[dict],
    events_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the full bucket-level dataset from all backtest events.

    Args:
        events: List of event dicts from the backtest index.
        events_dir: Path to events directory. Defaults to data/backtest/events/.

    Returns:
        DataFrame with one row per (event, bucket) pair.
        Columns include event-level features, per-bucket features,
        and target (_won).
    """
    if events_dir is None:
        events_dir = EVENTS_DIR

    all_rows = []

    for evt in events:
        slug = evt["event_slug"]
        event_dir = events_dir / slug

        # Load metadata
        meta_path = event_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load features
        feat_path = event_dir / "features.json"
        if not feat_path.exists():
            features = {}
        else:
            with open(feat_path, "r", encoding="utf-8") as f:
                features = json.load(f)

        buckets = metadata.get("buckets", [])
        if not buckets:
            continue

        duration_days = metadata.get("duration_days", 7)

        # Load entry prices
        entry_prices = _load_entry_prices(event_dir, metadata)

        context = {
            "duration_days": duration_days,
            "entry_prices": entry_prices,
        }

        # Build rows for this event
        event_df = build_single_event_rows(
            features=features,
            buckets=buckets,
            context=context,
            metadata=metadata,
            entry_prices=entry_prices,
        )

        # Add event-level metadata columns
        event_df["_event_id"] = str(evt.get("event_id", ""))
        event_df["_start_date"] = evt.get("start_date", "")
        event_df["_end_date"] = evt.get("end_date", "")
        event_df["_market_type"] = evt.get("market_type", "")
        event_df["_ground_truth_tier"] = evt.get("ground_truth_tier", "")

        all_rows.append(event_df)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns (excluding metadata and targets)."""
    return [
        c for c in df.columns
        if not c.startswith("_")
    ]


def get_metadata_columns() -> list[str]:
    """Return metadata column names (prefixed with _)."""
    return [
        "_event_slug", "_bucket_label", "_event_id",
        "_start_date", "_end_date", "_market_type",
        "_ground_truth_tier", "_won",
    ]
