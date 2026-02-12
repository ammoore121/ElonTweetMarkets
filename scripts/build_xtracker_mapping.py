"""
Build the XTracker-to-market mapping for ground truth backtesting.

For each resolved event in the market catalog, determines the actual tweet count
from XTracker data and cross-validates against the winning bucket.

Input files:
  - data/processed/market_catalog.parquet          (one row per bucket per event)
  - data/sources/xtracker/daily_metrics_full.json   (103 daily records)
  - data/sources/xtracker/elonmusk_all_tracking_details.json  (48 tracking periods)
  - data/sources/xtracker/elonmusk_trackings.json   (50 tracking period definitions)

Output files:
  - data/processed/xtracker_mapping.parquet         (one row per event)
  - data/processed/xtracker_mapping_summary.json    (aggregate statistics)

Usage: python scripts/build_xtracker_mapping.py
"""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
XTRACKER_DIR = PROJECT_DIR / "data" / "sources" / "xtracker"

CATALOG_PATH = PROCESSED_DIR / "market_catalog.parquet"
DAILY_METRICS_PATH = XTRACKER_DIR / "daily_metrics_full.json"
TRACKING_DETAILS_PATH = XTRACKER_DIR / "elonmusk_all_tracking_details.json"
TRACKINGS_PATH = XTRACKER_DIR / "elonmusk_trackings.json"

OUTPUT_PARQUET = PROCESSED_DIR / "xtracker_mapping.parquet"
OUTPUT_SUMMARY = PROCESSED_DIR / "xtracker_mapping_summary.json"

# ---------------------------------------------------------------------------
# XTracker data coverage boundary
# ---------------------------------------------------------------------------
XTRACKER_EARLIEST_DATE = pd.Timestamp("2025-10-30", tz="UTC")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_catalog():
    df = pd.read_parquet(CATALOG_PATH)
    resolved = df[df["is_resolved"]]
    winners = resolved[resolved["is_winner"]].copy()
    winners = winners.sort_values("start_date").reset_index(drop=True)
    return winners


def load_tracking_details():
    with open(TRACKING_DETAILS_PATH) as f:
        raw = json.load(f)

    results = []
    for entry in raw:
        if not entry.get("success"):
            continue
        d = entry["data"]
        stats = d.get("stats", {})
        if not stats.get("isComplete"):
            continue

        results.append({
            "tracking_id": d["id"],
            "title": d.get("title", ""),
            "start_date": d["startDate"][:10],
            "end_date": d["endDate"][:10],
            "total": stats["total"],
            "is_complete": True,
            "days_total": stats.get("daysTotal", 0),
            "daily": stats.get("daily", []),
        })

    return results


def load_daily_metrics():
    with open(DAILY_METRICS_PATH) as f:
        raw = json.load(f)

    results = []
    for r in raw["data"]:
        results.append({
            "date": r["date"][:10],
            "count": r["data"]["count"],
            "cumulative": r["data"]["cumulative"],
            "tracking_id": r["data"].get("trackingId", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Build lookups
# ---------------------------------------------------------------------------
def build_tracking_lookup(tracking_details):
    by_end = defaultdict(list)
    seen = set()
    for t in tracking_details:
        key = (t["start_date"], t["end_date"])
        if key in seen:
            continue
        seen.add(key)
        by_end[t["end_date"]].append(t)
    return dict(by_end)


def build_daily_metrics_by_date(daily_metrics):
    by_date = {}
    for dm in daily_metrics:
        by_date[dm["date"]] = dm
    return by_date


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------
def match_tracking_to_event(event_end_date, event_market_type,
                            event_duration_days, tracking_lookup):
    candidates = tracking_lookup.get(event_end_date, [])

    if not candidates:
        end_dt = datetime.strptime(event_end_date, "%Y-%m-%d")
        for delta in [-1, 1]:
            alt_date = (end_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
            candidates = tracking_lookup.get(alt_date, [])
            if candidates:
                break

    if not candidates:
        return None

    target_days = _expected_tracking_days(event_market_type)
    max_tolerance = _duration_tolerance(event_market_type)

    best = None
    best_diff = float("inf")
    for c in candidates:
        sd = datetime.strptime(c["start_date"], "%Y-%m-%d")
        ed = datetime.strptime(c["end_date"], "%Y-%m-%d")
        tracking_days = (ed - sd).days
        diff = abs(tracking_days - target_days)
        if diff <= max_tolerance and diff < best_diff:
            best_diff = diff
            best = c

    return best


def _expected_tracking_days(market_type):
    return {
        "weekly": 7,
        "daily": 2,
        "short": 3,
        "monthly": 30,
    }.get(market_type, 7)


def _duration_tolerance(market_type):
    """Max allowed difference in days between tracking duration and expected."""
    return {
        "weekly": 2,
        "daily": 1,
        "short": 2,
        "monthly": 5,
    }.get(market_type, 3)


# ---------------------------------------------------------------------------
# Daily metrics summation for a date range
# ---------------------------------------------------------------------------
def sum_daily_metrics(start_date_str, end_date_str, daily_by_date):
    sd = datetime.strptime(start_date_str, "%Y-%m-%d")
    ed = datetime.strptime(end_date_str, "%Y-%m-%d")

    total = 0.0
    n_zero = 0
    n_with_data = 0

    current = sd
    while current <= ed:
        date_str = current.strftime("%Y-%m-%d")
        dm = daily_by_date.get(date_str)
        if dm is not None:
            n_with_data += 1
            total += dm["count"]
            if dm["count"] == 0:
                n_zero += 1
        current += timedelta(days=1)

    return total, n_zero, n_with_data


# ---------------------------------------------------------------------------
# Main mapping logic
# ---------------------------------------------------------------------------
def build_mapping(winners, tracking_details, daily_metrics):
    tracking_lookup = build_tracking_lookup(tracking_details)
    daily_by_date = build_daily_metrics_by_date(daily_metrics)

    rows = []
    for _, event in winners.iterrows():
        event_id = str(event["event_id"])
        event_slug = str(event["event_slug"])
        event_title = str(event["event_title"])
        start_date = event["start_date"]
        end_date = event["end_date"]
        market_type = str(event["market_type"])
        duration_days = (
            int(event["duration_days"])
            if pd.notna(event["duration_days"])
            else 0
        )
        winning_bucket = str(event["bucket_label"])
        winning_lower = (
            int(event["lower_bound"])
            if pd.notna(event["lower_bound"])
            else 0
        )
        winning_upper = (
            int(event["upper_bound"])
            if pd.notna(event["upper_bound"])
            else 99999
        )

        # Skip events with missing end_date
        if pd.isna(end_date):
            continue

        end_date_str = end_date.strftime("%Y-%m-%d")

        matched = match_tracking_to_event(
            event_end_date=end_date_str,
            event_market_type=market_type,
            event_duration_days=duration_days,
            tracking_lookup=tracking_lookup,
        )

        xtracker_tracking_total = float("nan")
        tracking_start_str = None
        tracking_end_str = None

        if matched:
            xtracker_tracking_total = float(matched["total"])
            tracking_start_str = matched["start_date"]
            tracking_end_str = matched["end_date"]

        xtracker_daily_sum = float("nan")
        n_zero_days = 0
        n_days_with_data = 0

        if tracking_start_str and tracking_end_str:
            daily_sum, n_zero, n_data = sum_daily_metrics(
                tracking_start_str, tracking_end_str, daily_by_date
            )
            if n_data > 0:
                xtracker_daily_sum = daily_sum
                n_zero_days = n_zero
                n_days_with_data = n_data

        xtracker_count = float("nan")
        if not math.isnan(xtracker_tracking_total):
            xtracker_count = xtracker_tracking_total
        elif not math.isnan(xtracker_daily_sum) and xtracker_daily_sum > 0:
            xtracker_count = xtracker_daily_sum

        has_event_in_xtracker_era = end_date >= XTRACKER_EARLIEST_DATE
        if (not math.isnan(xtracker_tracking_total)
                and n_days_with_data > 0):
            ground_truth_tier = "gold"
        elif not math.isnan(xtracker_tracking_total):
            ground_truth_tier = "silver"
        elif has_event_in_xtracker_era:
            ground_truth_tier = "silver"
        else:
            ground_truth_tier = "bronze"

        if ground_truth_tier == "bronze" and math.isnan(xtracker_count):
            if winning_upper >= 99999:
                xtracker_count = float(
                    winning_lower + max(20, winning_lower // 10))
            else:
                xtracker_count = float(
                    (winning_lower + winning_upper) / 2.0)

        count_in_winning_bucket = False
        discrepancy = ""

        if not math.isnan(xtracker_count) and ground_truth_tier != "bronze":
            if winning_lower <= xtracker_count <= winning_upper:
                count_in_winning_bucket = True
            else:
                count_in_winning_bucket = False
                if xtracker_count < winning_lower:
                    direction = "below"
                    gap = winning_lower - xtracker_count
                else:
                    direction = "above"
                    gap = xtracker_count - winning_upper
                discrepancy = "XTracker count {} is {} winning bucket [{}-{}] by {}".format(
                    int(xtracker_count), direction, winning_lower, winning_upper, int(gap))
        elif ground_truth_tier == "bronze":
            count_in_winning_bucket = True
        elif math.isnan(xtracker_count):
            # Silver/gold with no matched data - mark as unvalidated, not discrepant
            count_in_winning_bucket = None
            discrepancy = "no_xtracker_data"

        rows.append({
            "event_id": event_id,
            "event_slug": event_slug,
            "event_title": event_title,
            "start_date": start_date,
            "end_date": end_date,
            "market_type": market_type,
            "duration_days": duration_days,
            "xtracker_daily_sum": xtracker_daily_sum,
            "xtracker_tracking_total": xtracker_tracking_total,
            "xtracker_count": xtracker_count,
            "ground_truth_tier": ground_truth_tier,
            "winning_bucket": winning_bucket,
            "winning_lower": winning_lower,
            "winning_upper": winning_upper,
            "count_in_winning_bucket": count_in_winning_bucket,
            "discrepancy": discrepancy,
            "n_zero_days": n_zero_days,
            "n_days_with_data": n_days_with_data,
            "tracking_start": tracking_start_str,
            "tracking_end": tracking_end_str,
        })

    result = pd.DataFrame(rows)
    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def build_summary(mapping_df):
    total = len(mapping_df)
    tier_counts = mapping_df["ground_truth_tier"].value_counts().to_dict()
    type_counts = mapping_df["market_type"].value_counts().to_dict()
    gold_silver = mapping_df[mapping_df["ground_truth_tier"].isin(["gold", "silver"])]
    # Only count events that have xtracker data for validation
    with_data = gold_silver[gold_silver["count_in_winning_bucket"].notna()]
    unvalidated = gold_silver[gold_silver["count_in_winning_bucket"].isna()]
    validated = int(with_data["count_in_winning_bucket"].sum())
    discrepant = int(len(with_data) - validated)
    n_unvalidated = int(len(unvalidated))
    discrepancy_list = gold_silver[
        (gold_silver["discrepancy"] != "") & (gold_silver["discrepancy"] != "no_xtracker_data")
    ][
        ["event_id", "event_slug", "market_type", "xtracker_count",
         "winning_bucket", "winning_lower", "winning_upper", "discrepancy"]
    ].to_dict(orient="records")
    with_tracking = int(mapping_df["xtracker_tracking_total"].notna().sum())
    with_daily = int(mapping_df["xtracker_daily_sum"].notna().sum())
    tier_stats = {}
    for tier in ["gold", "silver", "bronze"]:
        subset = mapping_df[mapping_df["ground_truth_tier"] == tier]
        if len(subset) > 0:
            counts = subset["xtracker_count"].dropna()
            tier_stats[tier] = {
                "n_events": int(len(subset)),
                "count_mean": round(float(counts.mean()), 1) if len(counts) > 0 else None,
                "count_median": round(float(counts.median()), 1) if len(counts) > 0 else None,
                "count_min": round(float(counts.min()), 1) if len(counts) > 0 else None,
                "count_max": round(float(counts.max()), 1) if len(counts) > 0 else None,
            }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_events": total,
        "events_by_tier": tier_counts,
        "events_by_type": type_counts,
        "validation": {
            "gold_silver_events": int(len(gold_silver)),
            "with_xtracker_data": int(len(with_data)),
            "validated_in_bucket": validated,
            "discrepancies": discrepant,
            "unvalidated_no_data": n_unvalidated,
            "discrepancy_rate": round(discrepant / max(len(with_data), 1), 4),
            "discrepancy_details": discrepancy_list,
        },
        "xtracker_coverage": {
            "events_with_tracking_total": with_tracking,
            "events_with_daily_metrics": with_daily,
        },
        "tier_statistics": tier_stats,
    }
    return summary



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 70
    print(sep)
    print("Building XTracker-to-Market Mapping")
    print(sep)

    # Load data
    print("")
    print("--- Loading data ---")
    winners = load_catalog()
    print("  Market catalog: {} resolved events with winners".format(len(winners)))

    tracking_details = load_tracking_details()
    print("  Tracking details: {} completed tracking periods".format(
        len(tracking_details)))

    daily_metrics = load_daily_metrics()
    print("  Daily metrics: {} daily records".format(len(daily_metrics)))

    # Build mapping
    print("")
    print("--- Building mapping ---")
    mapping = build_mapping(winners, tracking_details, daily_metrics)

    # Print tier breakdown
    tier_counts = mapping["ground_truth_tier"].value_counts()
    print("")
    print("  Ground truth tiers:")
    for tier in ["gold", "silver", "bronze"]:
        n = tier_counts.get(tier, 0)
        print("    {:8s}: {:3d} events".format(tier, n))
    print("    {:8s}: {:3d} events".format("total", len(mapping)))

    # Print validation results
    gold_silver = mapping[
        mapping["ground_truth_tier"].isin(["gold", "silver"])
    ]
    with_data = gold_silver[gold_silver["count_in_winning_bucket"].notna()]
    unvalidated = gold_silver[gold_silver["count_in_winning_bucket"].isna()]
    validated = int(with_data["count_in_winning_bucket"].sum())
    discrepant = int(len(with_data) - validated)
    print("")
    print("  Validation (gold + silver with XTracker data):")
    print("    Count in winning bucket: {}/{}".format(validated, len(with_data)))
    print("    Discrepancies: {}".format(discrepant))
    if len(unvalidated) > 0:
        print("    Unvalidated (no data):   {}".format(len(unvalidated)))

    if discrepant > 0:
        print("")
        print("  Discrepancy details:")
        disc_rows = with_data[with_data["count_in_winning_bucket"] == False]
        for _, row in disc_rows.iterrows():
            print("    {:>28s} | {:7s} | {}".format(
                row["event_id"], row["market_type"], row["discrepancy"]))

    # Print sample gold-tier events
    gold = mapping[mapping["ground_truth_tier"] == "gold"].head(10)
    if len(gold) > 0:
        print("")
        print("  Sample gold-tier events:")
        for _, row in gold.iterrows():
            match_str = (
                "OK" if row["count_in_winning_bucket"] else "MISMATCH"
            )
            print("    {:>28s} | {} to {} | count={:4d} | winner=[{}-{}] | {}".format(
                row["event_id"],
                row["tracking_start"], row["tracking_end"],
                int(row["xtracker_count"]),
                row["winning_lower"], row["winning_upper"],
                match_str))

    # Build and print summary
    summary = build_summary(mapping)

    # Save outputs
    print("")
    print("--- Saving outputs ---")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mapping.to_parquet(OUTPUT_PARQUET, index=False)
    print("  Parquet: {}".format(OUTPUT_PARQUET))
    print("    {} rows x {} columns".format(len(mapping), len(mapping.columns)))

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Summary: {}".format(OUTPUT_SUMMARY))

    # Final stats
    val = summary["validation"]
    rate = 1 - val["discrepancy_rate"]
    print("")
    print("--- Summary ---")
    print("  Total resolved events:       {}".format(summary["total_events"]))
    print("  Gold tier (daily + tracking): {}".format(
        summary["events_by_tier"].get("gold", 0)))
    print("  Silver tier (tracking only):  {}".format(
        summary["events_by_tier"].get("silver", 0)))
    print("  Bronze tier (no XTracker):    {}".format(
        summary["events_by_tier"].get("bronze", 0)))
    print("  With XTracker data:           {}/{} gold+silver events".format(
        val["with_xtracker_data"], val["gold_silver_events"]))
    print("  Validation rate:              {}/{} ({:.1%} match)".format(
        val["validated_in_bucket"], val["with_xtracker_data"], rate))
    print("  Discrepancy rate:             {:.1%} ({} events)".format(
        val["discrepancy_rate"], val["discrepancies"]))

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
