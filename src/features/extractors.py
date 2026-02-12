"""
Feature extraction functions for Elon Musk tweet count prediction.
=================================================================

Each function computes one category of features from raw data sources.
All functions enforce point-in-time (PIT) correctness: only data from
BEFORE the event start date is used (except calendar features that look
at the event window itself, e.g. launches_during_event).

Categories:
    temporal  - Rolling averages, volatility, trends, regime indicators
    media     - GDELT news volume and tone signals
    calendar  - SpaceX launches, holidays, event duration
    market    - Crowd-derived signals from Polymarket prices
    cross     - Interaction features combining multiple raw signals

Ported from scripts/build_backtest_dataset.py (22 existing factors)
plus 14 new factors defined in src/features/factor_registry.py.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# US Federal holidays (static list covering 2024-2026)
# ---------------------------------------------------------------------------
US_HOLIDAYS = [
    # 2024
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents' Day
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-10-14",  # Columbus Day
    "2024-11-11",  # Veterans Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
    "2024-12-31",  # New Year's Eve
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-10-13",  # Columbus Day
    "2025-11-11",  # Veterans Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    "2025-12-31",  # New Year's Eve
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-04",  # Independence Day (observed Jul 3)
    "2026-09-07",  # Labor Day
    "2026-10-12",  # Columbus Day
    "2026-11-11",  # Veterans Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    "2026-12-31",  # New Year's Eve
]


# ---------------------------------------------------------------------------
# XTracker coverage boundary (now relaxed -- unified daily_counts.json covers
# back to 2010 via Kaggle data; this constant is kept only for reference)
# ---------------------------------------------------------------------------
XTRACKER_EARLIEST_DATE_STR = "2010-06-04"  # Kaggle data start


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _trailing_dates(ref_date_str: str, n_days: int) -> List[str]:
    """Return list of date strings for the N days ending the day before ref_date."""
    ref = datetime.strptime(ref_date_str, "%Y-%m-%d")
    dates = []
    for i in range(1, n_days + 1):
        d = ref - timedelta(days=i)
        dates.append(d.strftime("%Y-%m-%d"))
    return dates


def _date_range(start_str: str, end_str: str) -> List[str]:
    """Generate list of date strings from start to end (inclusive)."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def _safe_mean(values: list) -> Optional[float]:
    """Mean of list, ignoring None/NaN."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _safe_std(values: list) -> Optional[float]:
    """Standard deviation of list (sample), ignoring None/NaN."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    variance = sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)
    return math.sqrt(variance)


def _linear_slope(values: list) -> Optional[float]:
    """Compute linear regression slope for a sequence of values.

    Uses numpy for robustness. Returns None if insufficient data.
    """
    clean = [(i, v) for i, v in enumerate(values)
             if v is not None and not math.isnan(v)]
    if len(clean) < 3:
        return None
    x = np.array([c[0] for c in clean], dtype=float)
    y = np.array([c[1] for c in clean], dtype=float)
    if np.std(x) == 0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# TEMPORAL FEATURES (11 existing + 1 new = 12 total)
# ---------------------------------------------------------------------------
def compute_temporal_features(
    xtracker_daily: dict,
    event_start_date: str,
    event_end_date: Optional[str] = None,
) -> dict:
    """Compute temporal features using only data BEFORE event_start_date.

    Existing (11):
        yesterday_count, rolling_avg_7d, rolling_avg_14d, rolling_avg_28d,
        rolling_std_7d, rolling_std_14d, trend_7d, trend_14d,
        day_of_week, cv_14d, regime_ratio

    New (1):
        weekend_ratio_7d - ratio of weekend to weekday tweet counts in
                           trailing 14 days
    """
    features = {
        "yesterday_count": None,
        "rolling_avg_7d": None,
        "rolling_avg_14d": None,
        "rolling_avg_28d": None,
        "rolling_std_7d": None,
        "rolling_std_14d": None,
        "trend_7d": None,
        "trend_14d": None,
        "day_of_week": None,
        "cv_14d": None,
        "regime_ratio": None,
        # NEW
        "weekend_ratio_7d": None,
    }

    if not xtracker_daily or not event_start_date:
        return features

    # Day of week
    try:
        start_dt = datetime.strptime(event_start_date, "%Y-%m-%d")
        features["day_of_week"] = start_dt.weekday()
    except ValueError:
        pass

    # Check if event is in XTracker era
    if event_start_date < XTRACKER_EARLIEST_DATE_STR:
        return features

    # Helper: get trailing counts excluding zero-artifact days
    def get_trailing_counts(n_days: int) -> List[float]:
        dates = _trailing_dates(event_start_date, n_days)
        counts = []
        for d in dates:
            entry = xtracker_daily.get(d)
            if entry is not None:
                c = entry["count"]
                if c > 0:
                    counts.append(float(c))
        return counts

    # Helper: get trailing date->count pairs including zeros for weekday analysis
    def get_trailing_date_counts(n_days: int) -> List[tuple]:
        """Return [(date_str, count), ...] for N trailing days, keep zeros."""
        dates = _trailing_dates(event_start_date, n_days)
        pairs = []
        for d in dates:
            entry = xtracker_daily.get(d)
            if entry is not None:
                pairs.append((d, float(entry["count"])))
        return pairs

    # Yesterday
    yesterday = _trailing_dates(event_start_date, 1)
    if yesterday:
        entry = xtracker_daily.get(yesterday[0])
        if entry is not None:
            features["yesterday_count"] = float(entry["count"])

    # Rolling averages and std devs
    counts_7 = get_trailing_counts(7)
    counts_14 = get_trailing_counts(14)
    counts_28 = get_trailing_counts(28)

    features["rolling_avg_7d"] = _safe_mean(counts_7) if counts_7 else None
    features["rolling_avg_14d"] = _safe_mean(counts_14) if counts_14 else None
    features["rolling_avg_28d"] = _safe_mean(counts_28) if counts_28 else None
    features["rolling_std_7d"] = _safe_std(counts_7) if len(counts_7) >= 2 else None
    features["rolling_std_14d"] = _safe_std(counts_14) if len(counts_14) >= 2 else None

    # Trends (use chronological order for regression)
    counts_7_chrono = list(reversed(counts_7))
    counts_14_chrono = list(reversed(counts_14))
    features["trend_7d"] = _linear_slope(counts_7_chrono)
    features["trend_14d"] = _linear_slope(counts_14_chrono)

    # Coefficient of variation
    avg_14 = features["rolling_avg_14d"]
    std_14 = features["rolling_std_14d"]
    if avg_14 is not None and std_14 is not None and avg_14 > 0:
        features["cv_14d"] = round(std_14 / avg_14, 4)

    # Regime ratio
    avg_7 = features["rolling_avg_7d"]
    avg_28 = features["rolling_avg_28d"]
    if avg_7 is not None and avg_28 is not None and avg_28 > 0:
        features["regime_ratio"] = round(avg_7 / avg_28, 4)

    # Round numeric values
    for k in ["rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d",
              "rolling_std_7d", "rolling_std_14d", "trend_7d", "trend_14d"]:
        if features[k] is not None:
            features[k] = round(features[k], 2)

    # --- NEW: weekend_ratio_7d ---
    # Use trailing 14 days for enough weekend samples
    date_counts_14 = get_trailing_date_counts(14)
    weekend_counts = []
    weekday_counts = []
    for d_str, c in date_counts_14:
        try:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            if c > 0:  # skip zero-artifact days
                if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                    weekend_counts.append(c)
                else:
                    weekday_counts.append(c)
        except ValueError:
            pass

    if weekend_counts and weekday_counts:
        weekend_mean = sum(weekend_counts) / len(weekend_counts)
        weekday_mean = sum(weekday_counts) / len(weekday_counts)
        if weekday_mean > 0:
            features["weekend_ratio_7d"] = round(weekend_mean / weekday_mean, 4)

    return features


# ---------------------------------------------------------------------------
# MEDIA/GDELT FEATURES (18 existing + 3 new = 21 total)
# ---------------------------------------------------------------------------
def compute_media_features(
    gdelt_data: dict,
    event_start_date: str,
) -> dict:
    """Compute GDELT media features using only data BEFORE event_start_date.

    Existing (18):
        Per entity (elon_musk, tesla, spacex, neuralink):
            {entity}_vol_1d, {entity}_vol_3d, {entity}_vol_7d, {entity}_vol_delta
        Tone (elon_musk only):
            elon_musk_tone_1d, elon_musk_tone_7d

    New (3):
        elon_musk_tone_delta  - 3d tone minus 7d tone (tone shift)
        total_media_vol_7d    - sum of all entity 7d volumes
        media_vol_concentration - Herfindahl index of attention across entities
    """
    ENTITIES = ["elon_musk", "tesla", "spacex", "neuralink"]

    features = {}

    if not event_start_date:
        # Return all-None features
        for entity in ENTITIES:
            for suffix in ["vol_1d", "vol_3d", "vol_7d", "vol_delta"]:
                features["{}_{}".format(entity, suffix)] = None
        features["elon_musk_tone_1d"] = None
        features["elon_musk_tone_7d"] = None
        # New
        features["elon_musk_tone_delta"] = None
        features["total_media_vol_7d"] = None
        features["media_vol_concentration"] = None
        return features

    for entity in ENTITIES:
        vol_key = "{}_vol".format(entity)
        vol_data = gdelt_data.get(vol_key, {})

        # 1-day
        dates_1 = _trailing_dates(event_start_date, 1)
        vals_1 = [vol_data.get(d) for d in dates_1 if d in vol_data]
        features["{}_vol_1d".format(entity)] = (
            round(vals_1[0], 4) if vals_1 else None
        )

        # 3-day
        dates_3 = _trailing_dates(event_start_date, 3)
        vals_3 = [vol_data.get(d) for d in dates_3 if d in vol_data]
        avg_3 = _safe_mean(vals_3) if vals_3 else None
        features["{}_vol_3d".format(entity)] = (
            round(avg_3, 4) if avg_3 is not None else None
        )

        # 7-day
        dates_7 = _trailing_dates(event_start_date, 7)
        vals_7 = [vol_data.get(d) for d in dates_7 if d in vol_data]
        avg_7 = _safe_mean(vals_7) if vals_7 else None
        features["{}_vol_7d".format(entity)] = (
            round(avg_7, 4) if avg_7 is not None else None
        )

        # Delta (spike indicator)
        if avg_3 is not None and avg_7 is not None:
            features["{}_vol_delta".format(entity)] = round(avg_3 - avg_7, 4)
        else:
            features["{}_vol_delta".format(entity)] = None

    # Tone features (elon_musk only)
    tone_data = gdelt_data.get("elon_musk_tone", {})

    dates_1 = _trailing_dates(event_start_date, 1)
    tone_1 = [tone_data.get(d) for d in dates_1 if d in tone_data]
    features["elon_musk_tone_1d"] = (
        round(tone_1[0], 4) if tone_1 else None
    )

    dates_3 = _trailing_dates(event_start_date, 3)
    tone_3_vals = [tone_data.get(d) for d in dates_3 if d in tone_data]
    avg_tone_3 = _safe_mean(tone_3_vals) if tone_3_vals else None

    dates_7 = _trailing_dates(event_start_date, 7)
    tone_7_vals = [tone_data.get(d) for d in dates_7 if d in tone_data]
    avg_tone_7 = _safe_mean(tone_7_vals) if tone_7_vals else None

    features["elon_musk_tone_7d"] = (
        round(avg_tone_7, 4) if avg_tone_7 is not None else None
    )

    # --- NEW: elon_musk_tone_delta ---
    if avg_tone_3 is not None and avg_tone_7 is not None:
        features["elon_musk_tone_delta"] = round(avg_tone_3 - avg_tone_7, 4)
    else:
        features["elon_musk_tone_delta"] = None

    # --- NEW: total_media_vol_7d ---
    entity_vols = []
    for entity in ENTITIES:
        v = features.get("{}_vol_7d".format(entity))
        if v is not None:
            entity_vols.append(v)

    if entity_vols:
        total_vol = sum(entity_vols)
        features["total_media_vol_7d"] = round(total_vol, 4)

        # --- NEW: media_vol_concentration (Herfindahl index) ---
        if total_vol > 0:
            hhi = sum((v / total_vol) ** 2 for v in entity_vols)
            features["media_vol_concentration"] = round(hhi, 4)
        else:
            features["media_vol_concentration"] = None
    else:
        features["total_media_vol_7d"] = None
        features["media_vol_concentration"] = None

    return features


# ---------------------------------------------------------------------------
# CALENDAR/SPACEX FEATURES (4 existing + 2 new = 6 total)
# ---------------------------------------------------------------------------
def compute_calendar_features(
    spacex_launches: List[str],
    event_start_date: str,
    event_end_date: Optional[str] = None,
) -> dict:
    """Compute SpaceX calendar features.

    Existing (4):
        days_since_last_launch, days_to_next_launch,
        launches_trailing_7d, launches_during_event

    New (2):
        is_holiday_week    - 1 if event window overlaps a US holiday
        event_duration_days - number of days in the event window
    """
    features = {
        "days_since_last_launch": None,
        "days_to_next_launch": None,
        "launches_trailing_7d": None,
        "launches_during_event": None,
        # NEW
        "is_holiday_week": None,
        "event_duration_days": None,
    }

    if not event_start_date:
        return features

    try:
        start_dt = datetime.strptime(event_start_date, "%Y-%m-%d")
    except ValueError:
        return features

    # --- NEW: event_duration_days ---
    if event_end_date:
        try:
            end_dt = datetime.strptime(event_end_date, "%Y-%m-%d")
            features["event_duration_days"] = (end_dt - start_dt).days
        except ValueError:
            pass

    # --- NEW: is_holiday_week ---
    if event_end_date:
        event_dates = set(_date_range(event_start_date, event_end_date))
        holiday_overlap = event_dates.intersection(set(US_HOLIDAYS))
        features["is_holiday_week"] = 1 if holiday_overlap else 0
    else:
        features["is_holiday_week"] = 0

    # SpaceX features require launch dates
    if not spacex_launches:
        return features

    # Days since last launch
    past_launches = [d for d in spacex_launches if d < event_start_date]
    if past_launches:
        last_launch = datetime.strptime(past_launches[-1], "%Y-%m-%d")
        features["days_since_last_launch"] = (start_dt - last_launch).days

    # Days to next launch
    future_launches = [d for d in spacex_launches if d >= event_start_date]
    if future_launches:
        next_launch = datetime.strptime(future_launches[0], "%Y-%m-%d")
        features["days_to_next_launch"] = (next_launch - start_dt).days

    # Launches in trailing 7 days (before event start)
    trail_start = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    features["launches_trailing_7d"] = len(
        [d for d in spacex_launches if trail_start <= d < event_start_date]
    )

    # Launches during event window
    if event_end_date:
        features["launches_during_event"] = len(
            [d for d in spacex_launches
             if event_start_date <= d <= event_end_date]
        )

    return features


# ---------------------------------------------------------------------------
# MARKET-DERIVED FEATURES (5 existing + 3 new = 8 total)
# ---------------------------------------------------------------------------
def compute_market_features(
    prices_df,
    catalog_df,
    event_id: str,
    event_end_date: str,
    temporal_features: Optional[dict] = None,
    calendar_features: Optional[dict] = None,
) -> dict:
    """Compute market-derived features at T-24h before event end.

    Existing (5):
        crowd_implied_ev, crowd_std_dev, distribution_entropy,
        n_buckets_with_price, price_shift_24h

    New (3):
        crowd_vs_rolling_avg - (crowd_ev - rolling_avg*duration) / crowd_ev
        crowd_skewness       - skewness of crowd probability distribution
        crowd_kurtosis       - excess kurtosis of crowd probability distribution

    Args:
        prices_df: Price history DataFrame (all events).
        catalog_df: Market catalog DataFrame (all events).
        event_id: Event ID string.
        event_end_date: Event end date string (YYYY-MM-DD).
        temporal_features: Temporal features dict (for cross-referencing rolling_avg).
        calendar_features: Calendar features dict (for event_duration_days).
    """
    import pandas as pd

    features = {
        "crowd_implied_ev": None,
        "crowd_std_dev": None,
        "distribution_entropy": None,
        "n_buckets_with_price": None,
        "price_shift_24h": None,
        # NEW
        "crowd_vs_rolling_avg": None,
        "crowd_skewness": None,
        "crowd_kurtosis": None,
    }

    if prices_df is None or prices_df.empty:
        return features
    if catalog_df is None or catalog_df.empty:
        return features
    if not event_end_date:
        return features

    # Get buckets for this event from catalog
    event_buckets = catalog_df[
        catalog_df["event_id"] == str(event_id)
    ].copy()

    if event_buckets.empty:
        return features

    # Get price data for this event
    event_prices = prices_df[
        prices_df["event_id"] == str(event_id)
    ].copy()

    if event_prices.empty:
        return features

    # Target time: T-24h before end
    try:
        end_dt = pd.Timestamp(event_end_date, tz="UTC")
    except Exception:
        return features

    target_time_24h = end_dt - pd.Timedelta(hours=24)
    target_time_48h = end_dt - pd.Timedelta(hours=48)

    def get_snapshot_at(target_time):
        """Get price snapshot closest to target_time (within 6h window)."""
        window_start = target_time - pd.Timedelta(hours=6)
        window_end = target_time + pd.Timedelta(hours=6)

        windowed = event_prices[
            (event_prices["timestamp"] >= window_start)
            & (event_prices["timestamp"] <= window_end)
        ]

        if windowed.empty:
            return None

        windowed = windowed.copy()
        windowed["time_diff"] = (
            windowed["timestamp"] - target_time
        ).abs()

        closest = (
            windowed.sort_values("time_diff")
            .drop_duplicates(subset=["bucket_label"], keep="first")
        )

        return closest

    snapshot_24h = get_snapshot_at(target_time_24h)
    if snapshot_24h is None or snapshot_24h.empty:
        return features

    # Merge with bucket bounds from catalog
    bucket_info = event_buckets[
        ["bucket_label", "lower_bound", "upper_bound"]
    ].drop_duplicates(subset=["bucket_label"])

    merged = snapshot_24h.merge(bucket_info, on="bucket_label", how="left")
    merged = merged[merged["lower_bound"].notna() & merged["upper_bound"].notna()]

    if merged.empty:
        return features

    # Compute midpoints
    def bucket_midpoint(lower, upper):
        lower = int(lower)
        upper = int(upper)
        if upper >= 99999:
            return lower + 20  # Open-ended bucket
        return (lower + upper) / 2.0

    merged["midpoint"] = merged.apply(
        lambda r: bucket_midpoint(r["lower_bound"], r["upper_bound"]),
        axis=1,
    )

    # Normalize prices to probabilities
    prices_arr = merged["price"].values.astype(float)
    price_sum = prices_arr.sum()
    if price_sum <= 0:
        return features

    probs = prices_arr / price_sum
    midpoints = merged["midpoint"].values.astype(float)

    features["n_buckets_with_price"] = int(len(merged))

    # Crowd implied EV
    implied_ev = float(np.sum(midpoints * probs))
    features["crowd_implied_ev"] = round(implied_ev, 2)

    # Crowd std dev
    variance = float(np.sum(probs * (midpoints - implied_ev) ** 2))
    std_dev = math.sqrt(max(0, variance))
    features["crowd_std_dev"] = round(std_dev, 2)

    # Distribution entropy
    nonzero_probs = probs[probs > 0]
    entropy = float(-np.sum(nonzero_probs * np.log(nonzero_probs)))
    features["distribution_entropy"] = round(entropy, 4)

    # Price shift: implied EV at T-48h vs T-24h
    snapshot_48h = get_snapshot_at(target_time_48h)
    if snapshot_48h is not None and not snapshot_48h.empty:
        merged_48 = snapshot_48h.merge(bucket_info, on="bucket_label", how="left")
        merged_48 = merged_48[
            merged_48["lower_bound"].notna() & merged_48["upper_bound"].notna()
        ]
        if not merged_48.empty:
            merged_48["midpoint"] = merged_48.apply(
                lambda r: bucket_midpoint(r["lower_bound"], r["upper_bound"]),
                axis=1,
            )
            prices_48 = merged_48["price"].values.astype(float)
            price_sum_48 = prices_48.sum()
            if price_sum_48 > 0:
                probs_48 = prices_48 / price_sum_48
                ev_48 = float(np.sum(merged_48["midpoint"].values * probs_48))
                features["price_shift_24h"] = round(implied_ev - ev_48, 2)

    # --- NEW: crowd_skewness ---
    if std_dev > 0:
        skewness = float(np.sum(probs * ((midpoints - implied_ev) / std_dev) ** 3))
        features["crowd_skewness"] = round(skewness, 4)

    # --- NEW: crowd_kurtosis (excess kurtosis) ---
    if std_dev > 0:
        kurtosis = float(
            np.sum(probs * ((midpoints - implied_ev) / std_dev) ** 4) - 3
        )
        features["crowd_kurtosis"] = round(kurtosis, 4)

    # --- NEW: crowd_vs_rolling_avg ---
    # Requires temporal rolling_avg_7d and calendar event_duration_days
    rolling_avg = None
    duration = None
    if temporal_features:
        rolling_avg = temporal_features.get("rolling_avg_7d")
    if calendar_features:
        duration = calendar_features.get("event_duration_days")

    if rolling_avg is not None and duration is not None and implied_ev != 0:
        expected_from_rolling = rolling_avg * duration
        features["crowd_vs_rolling_avg"] = round(
            (implied_ev - expected_from_rolling) / implied_ev, 4
        )

    return features


# ---------------------------------------------------------------------------
# CROSS-INTERACTION FEATURES (5 new)
# ---------------------------------------------------------------------------
def compute_cross_features(
    temporal: dict,
    media: dict,
    calendar: dict,
    market: dict,
) -> dict:
    """Compute cross-interaction features that combine signals from multiple
    raw data categories.

    New (5):
        bad_press_x_low_activity   - negative tone * activity deficit
        launch_busy_x_trend_down   - launches_trailing_7d * max(0, -trend_7d)
        regime_transition_flag     - 1 if regime_ratio deviates >0.3 from 1.0
        high_vol_x_high_entropy    - vol_delta * above-median entropy
        momentum_reversal_signal   - composite mean-reversion indicator
    """
    features = {
        "bad_press_x_low_activity": None,
        "launch_busy_x_trend_down": None,
        "regime_transition_flag": None,
        "high_vol_x_high_entropy": None,
        "momentum_reversal_signal": None,
    }

    # --- bad_press_x_low_activity ---
    # max(0, -elon_musk_tone_7d / tone_std_proxy) *
    # max(0, (rolling_avg_28d - rolling_avg_7d) / rolling_std_14d)
    tone_7d = media.get("elon_musk_tone_7d")
    avg_7d = temporal.get("rolling_avg_7d")
    avg_28d = temporal.get("rolling_avg_28d")
    std_14d = temporal.get("rolling_std_14d")

    if tone_7d is not None and avg_7d is not None and avg_28d is not None and std_14d is not None:
        # Tone component: higher when tone is more negative
        # Use a reasonable tone_std proxy of 1.0 (typical GDELT tone range)
        tone_component = max(0.0, -tone_7d / 1.0)
        # Activity deficit component: higher when recent activity is below long-term
        if std_14d > 0:
            activity_deficit = max(0.0, (avg_28d - avg_7d) / std_14d)
        else:
            activity_deficit = 0.0
        features["bad_press_x_low_activity"] = round(
            tone_component * activity_deficit, 4
        )

    # --- launch_busy_x_trend_down ---
    launches = calendar.get("launches_trailing_7d")
    trend_7d = temporal.get("trend_7d")

    if launches is not None and trend_7d is not None:
        features["launch_busy_x_trend_down"] = round(
            launches * max(0.0, -trend_7d), 4
        )

    # --- regime_transition_flag ---
    regime_ratio = temporal.get("regime_ratio")
    if regime_ratio is not None:
        features["regime_transition_flag"] = (
            1 if abs(regime_ratio - 1.0) > 0.3 else 0
        )

    # --- high_vol_x_high_entropy ---
    # Product of above-zero GDELT volume delta and above-median entropy
    # Median entropy is approximately 1.5 based on EDA
    vol_delta = media.get("elon_musk_vol_delta")
    entropy = market.get("distribution_entropy")
    MEDIAN_ENTROPY = 1.5

    if vol_delta is not None and entropy is not None:
        features["high_vol_x_high_entropy"] = round(
            max(0.0, vol_delta) * max(0.0, entropy - MEDIAN_ENTROPY), 4
        )

    # --- momentum_reversal_signal ---
    # sign(rolling_avg_28d - rolling_avg_7d) * cv_14d * abs(regime_ratio - 1.0)
    cv_14d = temporal.get("cv_14d")

    if avg_28d is not None and avg_7d is not None and cv_14d is not None and regime_ratio is not None:
        sign_val = 1.0 if (avg_28d - avg_7d) > 0 else (-1.0 if (avg_28d - avg_7d) < 0 else 0.0)
        features["momentum_reversal_signal"] = round(
            sign_val * cv_14d * abs(regime_ratio - 1.0), 4
        )

    return features
