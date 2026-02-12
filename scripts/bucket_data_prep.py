"""Prepare per-bucket dataset for zone-specific model search.

Creates data/analysis/bucket_dataset.json with all events flattened into
bucket-level rows with features, zone assignments, z-scores, and market prices.

Each event's buckets are sorted by lower_bound and assigned to zones 1-5
based on their relative position within the event's bucket list.
"""

import json
import math
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
BACKTEST = PROJECT / "data" / "backtest"
OUTPUT = PROJECT / "data" / "analysis"


def _load_entry_prices(slug: str, meta: dict) -> dict[str, float]:
    """Load T-24h entry prices from prices.parquet for an event."""
    prices_path = BACKTEST / "events" / slug / "prices.parquet"
    entry_prices: dict[str, float] = {}

    if not prices_path.exists():
        return entry_prices

    try:
        df = pd.read_parquet(prices_path)
        end_dt = pd.Timestamp(meta["end_date"], tz="UTC")
        target = end_dt - pd.Timedelta(hours=24)
        window_start = target - pd.Timedelta(hours=6)
        window_end = target + pd.Timedelta(hours=6)

        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            w = df[
                (df["timestamp"] >= window_start)
                & (df["timestamp"] <= window_end)
            ]
            if not w.empty:
                w = w.copy()
                w["_d"] = (w["timestamp"] - target).abs()
                closest = w.sort_values("_d").drop_duplicates(
                    "bucket_label", keep="first"
                )
                for _, row in closest.iterrows():
                    entry_prices[str(row["bucket_label"])] = float(row["price"])
    except Exception as exc:
        print(f"  Warning: failed to load prices for {slug}: {exc}")

    return entry_prices


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with open(BACKTEST / "backtest_index.json", encoding="utf-8") as f:
        index = json.load(f)

    events_out = []
    skipped = {"no_meta": 0, "no_winner": 0, "no_buckets": 0, "no_prices": 0}

    for evt in index["events"]:
        slug = evt["event_slug"]
        meta_path = BACKTEST / "events" / slug / "metadata.json"
        feat_path = BACKTEST / "events" / slug / "features.json"

        if not meta_path.exists():
            skipped["no_meta"] += 1
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        features: dict = {}
        if feat_path.exists():
            with open(feat_path, encoding="utf-8") as f:
                features = json.load(f)

        winning = meta.get("winning_bucket")
        if not winning:
            skipped["no_winner"] += 1
            continue

        buckets = meta.get("buckets", [])
        if not buckets:
            skipped["no_buckets"] += 1
            continue

        # Load entry prices from CLOB
        entry_prices = _load_entry_prices(slug, meta)
        if not entry_prices or all(v < 0.001 for v in entry_prices.values()):
            skipped["no_prices"] += 1
            continue

        # Market features
        market = features.get("market", {})
        crowd_ev = market.get("crowd_implied_ev")
        crowd_std = market.get("crowd_std_dev")

        # Sort buckets by lower_bound
        sorted_buckets = sorted(buckets, key=lambda b: b["lower_bound"])
        n_buckets = len(sorted_buckets)

        # Compute typical bucket width from non-open-ended buckets
        widths = [
            b["upper_bound"] - b["lower_bound"]
            for b in sorted_buckets
            if b["upper_bound"] < 99999
        ]
        typical_width = sum(widths) / len(widths) if widths else 25

        # Normalize entry prices to sum to 1
        price_total = sum(
            entry_prices.get(b["bucket_label"], 0.0) for b in sorted_buckets
        )

        bucket_rows = []
        for i, b in enumerate(sorted_buckets):
            lower = b["lower_bound"]
            upper = b["upper_bound"]

            # Midpoint
            if upper >= 99999:
                midpoint = lower + typical_width / 2
            elif lower <= 0:
                midpoint = upper / 2
            else:
                midpoint = (lower + upper) / 2

            # Width
            width = upper - lower if upper < 99999 else typical_width

            # Relative position in the bucket list (0.0 = bottom, 1.0 = top)
            rel_pos = i / max(n_buckets - 1, 1)

            # Zone 1-5
            zone = min(int(rel_pos * 5) + 1, 5)

            # Z-score relative to crowd implied EV
            z_score = None
            if (
                crowd_ev is not None
                and crowd_std is not None
                and crowd_std > 0
            ):
                z_score = round((midpoint - crowd_ev) / crowd_std, 4)

            # Market price (normalized)
            raw_price = entry_prices.get(b["bucket_label"], 0.0)
            norm_price = raw_price / price_total if price_total > 0 else 0.0

            bucket_rows.append(
                {
                    "bucket_label": b["bucket_label"],
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "midpoint": round(midpoint, 1),
                    "width": round(width, 1),
                    "bucket_index": i,
                    "n_buckets": n_buckets,
                    "relative_position": round(rel_pos, 4),
                    "zone": zone,
                    "z_score": z_score,
                    "is_winner": 1 if b["bucket_label"] == winning else 0,
                    "market_price": round(norm_price, 6),
                    "raw_price": round(raw_price, 6),
                }
            )

        events_out.append(
            {
                "event_slug": slug,
                "event_id": meta.get("event_id"),
                "market_type": meta.get("market_type"),
                "duration_days": meta.get("duration_days"),
                "ground_truth_tier": evt.get("ground_truth_tier"),
                "xtracker_count": meta.get("xtracker_count"),
                "winning_bucket": winning,
                "n_buckets": n_buckets,
                "features": features,
                "buckets": bucket_rows,
            }
        )

    # Summary statistics
    zone_counts: dict[int, int] = {}
    zone_wins: dict[int, int] = {}
    zone_avg_price: dict[int, list] = {}
    total_buckets = 0

    for e in events_out:
        for b in e["buckets"]:
            z = b["zone"]
            zone_counts[z] = zone_counts.get(z, 0) + 1
            zone_avg_price.setdefault(z, []).append(b["market_price"])
            total_buckets += 1
            if b["is_winner"]:
                zone_wins[z] = zone_wins.get(z, 0) + 1

    # Save dataset
    output_path = OUTPUT / "bucket_dataset.json"
    summary = {
        "n_events": len(events_out),
        "n_buckets_total": total_buckets,
        "zone_summary": {},
    }
    for z in sorted(zone_counts):
        n = zone_counts[z]
        w = zone_wins.get(z, 0)
        avg_p = sum(zone_avg_price[z]) / len(zone_avg_price[z])
        summary["zone_summary"][str(z)] = {
            "n_buckets": n,
            "n_winners": w,
            "win_rate": round(w / n, 4) if n > 0 else 0,
            "avg_market_price": round(avg_p, 4),
            "calibration_gap": round(w / n - avg_p, 4) if n > 0 else 0,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "events": events_out},
            f,
            indent=2,
            default=str,
        )

    # Print report
    print(f"Dataset saved to {output_path}")
    print(f"Events: {len(events_out)} (skipped: {skipped})")
    print(f"Total bucket-event pairs: {total_buckets}")
    print()
    print("Zone distribution:")
    print(f"{'Zone':>6s} {'Buckets':>8s} {'Winners':>8s} {'WinRate':>8s} {'AvgMktP':>8s} {'CalGap':>8s}")
    for z in sorted(zone_counts):
        n = zone_counts[z]
        w = zone_wins.get(z, 0)
        avg_p = sum(zone_avg_price[z]) / len(zone_avg_price[z])
        gap = w / n - avg_p if n > 0 else 0
        print(
            f"{z:>6d} {n:>8d} {w:>8d} {w/n:>8.3f} {avg_p:>8.3f} {gap:>+8.3f}"
        )

    # Winning zone distribution
    print()
    print("Where winners land by zone:")
    for z in sorted(zone_wins):
        pct = 100 * zone_wins[z] / sum(zone_wins.values())
        print(f"  Zone {z}: {zone_wins[z]} winners ({pct:.1f}%)")


if __name__ == "__main__":
    main()
