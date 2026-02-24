"""
Feature extraction functions for Elon Musk tweet count prediction.
=================================================================

Each function computes one category of features from raw data sources.
All functions enforce point-in-time (PIT) correctness: only data from
BEFORE the event start date is used (except calendar features that look
at the event window itself, e.g. launches_during_event).

Categories:
    temporal   - Rolling averages, volatility, trends, regime indicators
    media      - GDELT news volume and tone signals
    calendar   - SpaceX launches, holidays, event duration
    government - Government events (exec orders, rules, notices)
    corporate  - Corporate events (earnings, launches, filings)
    market     - Crowd-derived signals from Polymarket prices
    cross      - Interaction features combining multiple raw signals
    reddit     - Reddit post/comment activity across Elon-related subreddits

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
        "day_of_week_sin": None,
        "day_of_week_cos": None,
        # Sparse-data-aware features (Issue #5)
        "data_coverage_7d": None,
        "data_coverage_14d": None,
        "rolling_avg_7d_calendar": None,
    }

    if not xtracker_daily or not event_start_date:
        return features

    # Day of week
    try:
        start_dt = datetime.strptime(event_start_date, "%Y-%m-%d")
        dow = start_dt.weekday()
        features["day_of_week"] = dow
        features["day_of_week_sin"] = round(math.sin(2 * math.pi * dow / 7), 4)
        features["day_of_week_cos"] = round(math.cos(2 * math.pi * dow / 7), 4)
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

    # Helper: calendar-day average (sum of ALL counts including zeros / n_days)
    def get_trailing_total_and_coverage(n_days: int) -> tuple:
        """Return (total_count, n_entries_with_data) for N trailing calendar days.

        Unlike get_trailing_counts() which skips zero-count days (correct for
        filtering noon-to-noon artifacts), this sums ALL entries and divides
        by calendar days to give a conservative daily average.
        """
        dates = _trailing_dates(event_start_date, n_days)
        total = 0.0
        n_with_data = 0
        for d in dates:
            entry = xtracker_daily.get(d)
            if entry is not None:
                total += float(entry["count"])
                n_with_data += 1
        return total, n_with_data

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

    # --- Sparse-data-aware features (Issue #5) ---
    total_7, n_data_7 = get_trailing_total_and_coverage(7)
    total_14, n_data_14 = get_trailing_total_and_coverage(14)
    features["data_coverage_7d"] = round(n_data_7 / 7, 4)
    features["data_coverage_14d"] = round(n_data_14 / 14, 4)
    features["rolling_avg_7d_calendar"] = round(total_7 / 7, 2) if n_data_7 > 0 else None

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
        features["gdelt_entity_divergence"] = None
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

    # --- NEW: gdelt_entity_divergence ---
    # abs(elon_vol_1d - tesla_vol_1d) / max(elon_vol_1d, tesla_vol_1d)
    elon_vol = features.get("elon_musk_vol_1d")
    tesla_vol = features.get("tesla_vol_1d")
    if elon_vol is not None and tesla_vol is not None:
        max_vol = max(elon_vol, tesla_vol)
        if max_vol > 0:
            features["gdelt_entity_divergence"] = round(
                abs(elon_vol - tesla_vol) / max_vol, 4
            )
        else:
            features["gdelt_entity_divergence"] = None
    else:
        features["gdelt_entity_divergence"] = None

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
# GOVERNMENT EVENT FEATURES (3 new)
# ---------------------------------------------------------------------------
def compute_government_features(
    govt_events_df: "pd.DataFrame",
    event_start_date: str,
) -> dict:
    """Compute government event features using only data BEFORE event_start_date.

    Government (3 factors):
        govt_event_flag_7d        - Binary: any govt event in 7 days before event start
        govt_event_count_trailing_7d - Count of govt events in 7 days before event start
        govt_exec_order_flag_7d   - Binary: any executive order in 7 days before event start
    """
    features = {
        "govt_event_flag_7d": 0,
        "govt_event_count_trailing_7d": 0,
        "govt_exec_order_flag_7d": 0,
    }

    if not event_start_date or govt_events_df is None or govt_events_df.empty:
        return features

    try:
        start_dt = datetime.strptime(event_start_date, "%Y-%m-%d")
    except ValueError:
        return features

    window_start = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    # Filter events in the 7-day window before event start
    mask = (govt_events_df["date"] >= window_start) & (govt_events_df["date"] < event_start_date)
    window_events = govt_events_df[mask]

    count = len(window_events)
    features["govt_event_count_trailing_7d"] = count
    features["govt_event_flag_7d"] = 1 if count > 0 else 0

    # Check for executive orders specifically
    if count > 0 and "event_type" in window_events.columns:
        exec_orders = window_events[window_events["event_type"] == "executive_order"]
        features["govt_exec_order_flag_7d"] = 1 if len(exec_orders) > 0 else 0

    return features


# ---------------------------------------------------------------------------
# CORPORATE EVENT FEATURES (3 new)
# ---------------------------------------------------------------------------
def compute_corporate_features(
    corporate_events_df: "pd.DataFrame",
    event_start_date: str,
    event_end_date: Optional[str] = None,
) -> dict:
    """Compute corporate event features around the event window.

    Corporate (3 factors):
        corporate_event_flag_7d   - Binary: any corporate event in 7d around event window
        corporate_event_count_7d  - Count of corporate events in 7d around event window
        tesla_earnings_flag_14d   - Binary: Tesla earnings within 14d of event start
    """
    features = {
        "corporate_event_flag_7d": 0,
        "corporate_event_count_7d": 0,
        "tesla_earnings_flag_14d": 0,
    }

    if not event_start_date or corporate_events_df is None or corporate_events_df.empty:
        return features

    try:
        start_dt = datetime.strptime(event_start_date, "%Y-%m-%d")
    except ValueError:
        return features

    # 7-day window around the event (before start to after end, or +7d if no end)
    window_start = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    if event_end_date:
        try:
            end_dt = datetime.strptime(event_end_date, "%Y-%m-%d")
            window_end = (end_dt + timedelta(days=7)).strftime("%Y-%m-%d")
        except ValueError:
            window_end = (start_dt + timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        window_end = (start_dt + timedelta(days=7)).strftime("%Y-%m-%d")

    # Filter corporate events in 7d window around event
    mask = (
        (corporate_events_df["date"] >= window_start)
        & (corporate_events_df["date"] <= window_end)
    )
    window_events = corporate_events_df[mask]

    count = len(window_events)
    features["corporate_event_count_7d"] = count
    features["corporate_event_flag_7d"] = 1 if count > 0 else 0

    # Tesla earnings within 14 days (before or after) of event start
    earnings_window_start = (start_dt - timedelta(days=14)).strftime("%Y-%m-%d")
    earnings_window_end = (start_dt + timedelta(days=14)).strftime("%Y-%m-%d")

    earnings_mask = (
        (corporate_events_df["date"] >= earnings_window_start)
        & (corporate_events_df["date"] <= earnings_window_end)
        & (corporate_events_df["company"] == "Tesla")
        & (corporate_events_df["event_type"] == "earnings")
    )
    tesla_earnings = corporate_events_df[earnings_mask]
    features["tesla_earnings_flag_14d"] = 1 if len(tesla_earnings) > 0 else 0

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
    event_start_date: Optional[str] = None,
) -> dict:
    """Compute market-derived features at T-24h before event end.

    Existing (5):
        crowd_implied_ev, crowd_std_dev, distribution_entropy,
        n_buckets_with_price, price_shift_24h

    New (7):
        crowd_vs_rolling_avg       - (crowd_ev - rolling_avg*duration) / crowd_ev
        crowd_skewness             - skewness of crowd probability distribution
        crowd_kurtosis             - excess kurtosis of crowd probability distribution
        market_overround           - sum(bucket_prices) - 1.0 (vig in market)
        hours_until_resolution     - hours from event start to market close
        bucket_relative_mispricings - per-bucket: price[i] - price[i]/sum(prices)
        bucket_position_normalized - per-bucket: bucket_index / total_buckets

    Args:
        prices_df: Price history DataFrame (all events).
        catalog_df: Market catalog DataFrame (all events).
        event_id: Event ID string.
        event_end_date: Event end date string (YYYY-MM-DD).
        temporal_features: Temporal features dict (for cross-referencing rolling_avg).
        calendar_features: Calendar features dict (for event_duration_days).
        event_start_date: Event start date string (YYYY-MM-DD) for hours_until_resolution.
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
        "market_overround": None,
        "hours_until_resolution": None,
        "bucket_relative_mispricings": None,
        "bucket_position_normalized": None,
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

    # --- NEW: market_overround ---
    # Sum of raw (unnormalized) bucket prices minus 1.0
    features["market_overround"] = round(float(price_sum) - 1.0, 4)

    # --- NEW: hours_until_resolution ---
    # Hours from event start to event end (market close)
    if event_start_date and event_end_date:
        try:
            start_dt_utc = pd.Timestamp(event_start_date, tz="UTC")
            end_dt_utc = pd.Timestamp(event_end_date, tz="UTC")
            hours_diff = (end_dt_utc - start_dt_utc).total_seconds() / 3600.0
            if hours_diff > 0:
                features["hours_until_resolution"] = round(hours_diff, 1)
        except Exception:
            pass

    # --- NEW: bucket_relative_mispricings ---
    # Per-bucket: raw_price[i] - (raw_price[i] / sum(raw_prices))
    # This shows how much each bucket deviates from its fair-share of 1.0
    mispricings = {}
    sorted_merged = merged.sort_values("lower_bound")
    for _, row in sorted_merged.iterrows():
        label = row["bucket_label"]
        raw_price = float(row["price"])
        fair_share = raw_price / price_sum  # normalized probability
        mispricings[label] = round(raw_price - fair_share, 4)
    features["bucket_relative_mispricings"] = mispricings

    # --- NEW: bucket_position_normalized ---
    # Per-bucket: index / total_buckets (0=lowest, ~1=highest)
    n_total = len(sorted_merged)
    positions = {}
    for idx, (_, row) in enumerate(sorted_merged.iterrows()):
        label = row["bucket_label"]
        positions[label] = round(idx / max(n_total - 1, 1), 4)
    features["bucket_position_normalized"] = positions

    return features


# ---------------------------------------------------------------------------
# REDDIT ACTIVITY FEATURES (7 new)
# ---------------------------------------------------------------------------
def compute_reddit_features(
    reddit_data: "pd.DataFrame",
    event_start_date: str,
) -> dict:
    """Compute Reddit activity features using only data BEFORE event_start_date.

    Reddit (7 factors):
        reddit_total_posts_7d      - Total posts across all subreddits (trailing 7d)
        reddit_total_comments_7d   - Total comments across all subreddits (trailing 7d)
        reddit_post_delta          - (3d avg - 7d avg) / 7d avg post count
        reddit_elonmusk_posts_7d   - Posts in r/elonmusk (trailing 7d avg)
        reddit_teslamotors_posts_7d - Posts in r/teslamotors (trailing 7d avg)
        reddit_attention_concentration - HHI of post volume across subreddits
        reddit_top_score_7d        - Avg top_post_score across subreddits (trailing 7d)
    """
    features = {
        "reddit_total_posts_7d": None,
        "reddit_total_comments_7d": None,
        "reddit_post_delta": None,
        "reddit_elonmusk_posts_7d": None,
        "reddit_teslamotors_posts_7d": None,
        "reddit_attention_concentration": None,
        "reddit_top_score_7d": None,
    }

    if not event_start_date or reddit_data is None or reddit_data.empty:
        return features

    # Filter to data before event start
    prior = reddit_data[reddit_data["date"] < event_start_date].copy()
    if prior.empty:
        return features

    # 7-day trailing window
    dates_7 = _trailing_dates(event_start_date, 7)
    dates_3 = _trailing_dates(event_start_date, 3)

    window_7d = prior[prior["date"].isin(dates_7)]
    window_3d = prior[prior["date"].isin(dates_3)]

    if window_7d.empty:
        return features

    # Aggregate across all subreddits per day, then average
    daily_7d = window_7d.groupby("date").agg(
        posts=("post_count", "sum"),
        comments=("comment_count", "sum"),
        top_score=("top_post_score", "max"),
    )

    avg_posts_7d = daily_7d["posts"].mean()
    avg_comments_7d = daily_7d["comments"].mean()
    avg_top_score_7d = daily_7d["top_score"].mean()

    features["reddit_total_posts_7d"] = round(float(avg_posts_7d), 2)
    features["reddit_total_comments_7d"] = round(float(avg_comments_7d), 2)
    features["reddit_top_score_7d"] = round(float(avg_top_score_7d), 2)

    # Post delta (spike indicator): 3d vs 7d
    if not window_3d.empty:
        daily_3d = window_3d.groupby("date")["post_count"].sum()
        avg_posts_3d = daily_3d.mean()
        if avg_posts_7d > 0:
            features["reddit_post_delta"] = round(
                float((avg_posts_3d - avg_posts_7d) / avg_posts_7d), 4
            )

    # Per-subreddit 7d averages
    sub_avgs = window_7d.groupby("subreddit")["post_count"].mean()

    elonmusk_avg = sub_avgs.get("elonmusk")
    if elonmusk_avg is not None:
        features["reddit_elonmusk_posts_7d"] = round(float(elonmusk_avg), 2)

    teslamotors_avg = sub_avgs.get("teslamotors")
    if teslamotors_avg is not None:
        features["reddit_teslamotors_posts_7d"] = round(float(teslamotors_avg), 2)

    # Attention concentration (HHI across subreddits)
    total_sub_posts = float(sub_avgs.sum())
    if total_sub_posts > 0:
        hhi = sum((float(v) / total_sub_posts) ** 2 for v in sub_avgs.values)
        features["reddit_attention_concentration"] = round(hhi, 4)

    return features


# ---------------------------------------------------------------------------
# CROSS-INTERACTION FEATURES (5 new)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FINANCIAL FEATURES (Tesla stock + crypto prices)
# ---------------------------------------------------------------------------
def compute_financial_features(
    tesla_data: "pd.DataFrame",
    crypto_data: "pd.DataFrame",
    event_start_date: str,
    vix_data: "pd.DataFrame" = None,
    crypto_fg_data: "pd.DataFrame" = None,
) -> dict:
    """Compute financial market features using only data BEFORE event_start_date.

    Tesla (6 factors):
        tsla_pct_change_1d, tsla_pct_change_5d, tsla_volatility_5d,
        tsla_volume_ratio, tsla_drawdown_5d, tsla_gap_1d

    Crypto (6 factors):
        doge_pct_change_1d, doge_pct_change_5d, doge_volatility_5d,
        btc_pct_change_1d, btc_pct_change_5d, btc_volatility_5d

    VIX (5 factors):
        vix_close, vix_pct_change_1d, vix_pct_change_5d,
        vix_level_category, vix_ma5_ratio

    Crypto Fear & Greed (4 factors):
        crypto_fg_value, crypto_fg_7d_avg, crypto_fg_delta, crypto_fg_category
    """
    import pandas as pd

    features = {
        "tsla_pct_change_1d": None,
        "tsla_pct_change_5d": None,
        "tsla_volatility_5d": None,
        "tsla_volume_ratio": None,
        "tsla_drawdown_5d": None,
        "tsla_gap_1d": None,
        "doge_pct_change_1d": None,
        "doge_pct_change_5d": None,
        "doge_volatility_5d": None,
        "btc_pct_change_1d": None,
        "btc_pct_change_5d": None,
        "btc_volatility_5d": None,
        # VIX
        "vix_close": None,
        "vix_pct_change_1d": None,
        "vix_pct_change_5d": None,
        "vix_level_category": None,
        "vix_ma5_ratio": None,
        # Crypto Fear & Greed
        "crypto_fg_value": None,
        "crypto_fg_7d_avg": None,
        "crypto_fg_delta": None,
        "crypto_fg_category": None,
    }

    if not event_start_date:
        return features

    # --- Tesla features ---
    if tesla_data is not None and not tesla_data.empty:
        prior = tesla_data[tesla_data["date"] < event_start_date].copy()
        if len(prior) >= 1:
            last = prior.iloc[-1]
            pct = last.get("pct_change")
            if pct is not None and not (isinstance(pct, float) and math.isnan(pct)):
                features["tsla_pct_change_1d"] = round(float(pct), 6)
            gap = last.get("gap")
            if gap is not None and not (isinstance(gap, float) and math.isnan(gap)):
                features["tsla_gap_1d"] = round(float(gap), 6)

        if len(prior) >= 5:
            last5 = prior.tail(5)
            # 5-day return
            close_5ago = last5.iloc[0].get("close")
            close_now = last5.iloc[-1].get("close")
            if close_5ago and close_now and close_5ago > 0:
                features["tsla_pct_change_5d"] = round(
                    (float(close_now) - float(close_5ago)) / float(close_5ago), 6
                )
            # 5-day volatility
            vol = last5.get("pct_change")
            if vol is not None:
                clean = vol.dropna()
                if len(clean) >= 2:
                    features["tsla_volatility_5d"] = round(float(clean.std()), 6)
            # Volume ratio (last day vs 5-day MA)
            vol_ma = last5["volume"].mean()
            if vol_ma > 0:
                features["tsla_volume_ratio"] = round(
                    float(last5.iloc[-1]["volume"]) / float(vol_ma), 4
                )
            # Drawdown from 5-day high
            high_5d = last5["close"].max()
            if high_5d > 0:
                features["tsla_drawdown_5d"] = round(
                    (float(last5.iloc[-1]["close"]) - float(high_5d)) / float(high_5d), 6
                )

    # --- Crypto features ---
    if crypto_data is not None and not crypto_data.empty:
        prior = crypto_data[crypto_data["date"] < event_start_date].copy()

        for symbol in ["doge", "btc"]:
            pct_col = f"{symbol}_pct_change"
            close_col = f"{symbol}_close"
            vol_col = f"{symbol}_volatility_5d"

            if len(prior) >= 1:
                last = prior.iloc[-1]
                pct = last.get(pct_col)
                if pct is not None and not (isinstance(pct, float) and math.isnan(pct)):
                    features[f"{symbol}_pct_change_1d"] = round(float(pct), 6)

            if len(prior) >= 5:
                last5 = prior.tail(5)
                # 5-day return
                c0 = last5.iloc[0].get(close_col)
                c1 = last5.iloc[-1].get(close_col)
                if c0 and c1 and float(c0) > 0:
                    features[f"{symbol}_pct_change_5d"] = round(
                        (float(c1) - float(c0)) / float(c0), 6
                    )
                # 5-day volatility
                pcts = last5.get(pct_col)
                if pcts is not None:
                    clean = pcts.dropna()
                    if len(clean) >= 2:
                        features[f"{symbol}_volatility_5d"] = round(float(clean.std()), 6)

    # --- VIX features ---
    if vix_data is not None and not vix_data.empty:
        prior = vix_data[vix_data["date"] < event_start_date].copy()
        if len(prior) >= 1:
            last = prior.iloc[-1]
            close = last.get("close")
            if close is not None and not (isinstance(close, float) and math.isnan(close)):
                features["vix_close"] = round(float(close), 2)
            pct = last.get("pct_change")
            if pct is not None and not (isinstance(pct, float) and math.isnan(pct)):
                features["vix_pct_change_1d"] = round(float(pct), 6)
            pct5 = last.get("pct_change_5d")
            if pct5 is not None and not (isinstance(pct5, float) and math.isnan(pct5)):
                features["vix_pct_change_5d"] = round(float(pct5), 6)
            cat = last.get("level_category")
            if cat is not None and cat != "":
                features["vix_level_category"] = str(cat)
            ma5r = last.get("ma5_ratio")
            if ma5r is not None and not (isinstance(ma5r, float) and math.isnan(ma5r)):
                features["vix_ma5_ratio"] = round(float(ma5r), 4)

    # --- Crypto Fear & Greed features ---
    if crypto_fg_data is not None and not crypto_fg_data.empty:
        prior = crypto_fg_data[crypto_fg_data["date"] < event_start_date].copy()
        if len(prior) >= 1:
            last = prior.iloc[-1]
            fg_val = last.get("fg_value")
            if fg_val is not None and not (isinstance(fg_val, float) and math.isnan(fg_val)):
                features["crypto_fg_value"] = int(fg_val)
            fg_7d = last.get("fg_7d_avg")
            if fg_7d is not None and not (isinstance(fg_7d, float) and math.isnan(fg_7d)):
                features["crypto_fg_7d_avg"] = round(float(fg_7d), 2)
            fg_delta = last.get("fg_delta")
            if fg_delta is not None and not (isinstance(fg_delta, float) and math.isnan(fg_delta)):
                features["crypto_fg_delta"] = round(float(fg_delta), 2)
            fg_cat = last.get("fg_category")
            if fg_cat is not None and fg_cat != "":
                features["crypto_fg_category"] = str(fg_cat)

    return features


# ---------------------------------------------------------------------------
# ATTENTION FEATURES (Wikipedia pageviews)
# ---------------------------------------------------------------------------
def compute_attention_features(
    wiki_data: dict,
    event_start_date: str,
) -> dict:
    """Compute Wikipedia pageview attention features (PIT-correct).

    Attention (8 factors):
        wiki_elon_musk_7d, wiki_elon_musk_delta,
        wiki_tesla_7d, wiki_tesla_delta,
        wiki_doge_7d, wiki_doge_delta,
        wiki_total_7d, wiki_attention_concentration
    """
    ENTITIES = ["elon_musk", "tesla_inc", "dogecoin"]

    features = {
        "wiki_elon_musk_7d": None,
        "wiki_elon_musk_delta": None,
        "wiki_tesla_7d": None,
        "wiki_tesla_delta": None,
        "wiki_doge_7d": None,
        "wiki_doge_delta": None,
        "wiki_total_7d": None,
        "wiki_attention_concentration": None,
        # NEW
        "wikipedia_entity_divergence": None,
    }

    if not event_start_date or not wiki_data:
        return features

    short_names = {"elon_musk": "elon_musk", "tesla_inc": "tesla", "dogecoin": "doge"}

    entity_7d_avgs = []
    for entity in ENTITIES:
        daily = wiki_data.get(entity, {}).get("daily_views", {})
        if not daily:
            continue

        short = short_names[entity]

        # 3-day trailing
        dates_3 = _trailing_dates(event_start_date, 3)
        vals_3 = [daily.get(d) for d in dates_3 if d in daily]
        avg_3 = _safe_mean(vals_3) if vals_3 else None

        # 7-day trailing
        dates_7 = _trailing_dates(event_start_date, 7)
        vals_7 = [daily.get(d) for d in dates_7 if d in daily]
        avg_7 = _safe_mean(vals_7) if vals_7 else None

        if avg_7 is not None:
            features[f"wiki_{short}_7d"] = round(avg_7, 1)
            entity_7d_avgs.append(avg_7)

        if avg_3 is not None and avg_7 is not None and avg_7 > 0:
            features[f"wiki_{short}_delta"] = round((avg_3 - avg_7) / avg_7, 4)

    # Total and concentration
    if entity_7d_avgs:
        total = sum(entity_7d_avgs)
        features["wiki_total_7d"] = round(total, 1)
        if total > 0:
            hhi = sum((v / total) ** 2 for v in entity_7d_avgs)
            features["wiki_attention_concentration"] = round(hhi, 4)

    # --- NEW: wikipedia_entity_divergence ---
    # elon_pageviews / tesla_pageviews ratio (7d) vs 30d average of that ratio
    # Spike in this ratio without Tesla = personal event
    elon_daily = wiki_data.get("elon_musk", {}).get("daily_views", {})
    tesla_daily = wiki_data.get("tesla_inc", {}).get("daily_views", {})

    if elon_daily and tesla_daily:
        # 7-day ratio
        dates_7 = _trailing_dates(event_start_date, 7)
        elon_7 = [elon_daily.get(d) for d in dates_7 if d in elon_daily]
        tesla_7 = [tesla_daily.get(d) for d in dates_7 if d in tesla_daily]
        avg_elon_7 = _safe_mean(elon_7) if elon_7 else None
        avg_tesla_7 = _safe_mean(tesla_7) if tesla_7 else None

        # 30-day ratio (baseline)
        dates_30 = _trailing_dates(event_start_date, 30)
        ratios_30 = []
        for d in dates_30:
            e_val = elon_daily.get(d)
            t_val = tesla_daily.get(d)
            if e_val is not None and t_val is not None and t_val > 0:
                ratios_30.append(e_val / t_val)
        avg_ratio_30 = _safe_mean(ratios_30) if ratios_30 else None

        if (avg_elon_7 is not None and avg_tesla_7 is not None
                and avg_tesla_7 > 0 and avg_ratio_30 is not None and avg_ratio_30 > 0):
            current_ratio = avg_elon_7 / avg_tesla_7
            features["wikipedia_entity_divergence"] = round(
                current_ratio / avg_ratio_30, 4
            )

    return features


# ---------------------------------------------------------------------------
# TRENDS FEATURES (Google Trends)
# ---------------------------------------------------------------------------
def compute_trends_features(
    trends_data: "pd.DataFrame",
    event_start_date: str,
) -> dict:
    """Compute Google Trends features using only data BEFORE event_start_date.

    Trends (10 factors):
        gt_elon_musk_7d, gt_elon_musk_delta,
        gt_tesla_7d, gt_tesla_delta,
        gt_spacex_7d, gt_spacex_delta,
        gt_dogecoin_7d, gt_dogecoin_delta,
        gt_total_7d, gt_concentration
    """
    import pandas as pd

    ENTITIES = ["elon_musk", "tesla", "spacex", "dogecoin"]

    features = {
        "gt_elon_musk_7d": None,
        "gt_elon_musk_delta": None,
        "gt_tesla_7d": None,
        "gt_tesla_delta": None,
        "gt_spacex_7d": None,
        "gt_spacex_delta": None,
        "gt_dogecoin_7d": None,
        "gt_dogecoin_delta": None,
        "gt_total_7d": None,
        "gt_concentration": None,
    }

    if not event_start_date or trends_data is None or trends_data.empty:
        return features

    # Build date->row lookup
    prior = trends_data[trends_data["date"] < event_start_date].copy()
    if prior.empty:
        return features

    date_lookup = dict(zip(prior["date"], range(len(prior))))

    entity_7d_avgs = []
    for entity in ENTITIES:
        col = entity  # Column names match entity keys

        if col not in prior.columns:
            continue

        # 3-day trailing
        dates_3 = _trailing_dates(event_start_date, 3)
        vals_3 = []
        for d in dates_3:
            if d in date_lookup:
                idx = date_lookup[d]
                val = prior.iloc[idx].get(col)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    vals_3.append(float(val))
        avg_3 = _safe_mean(vals_3) if vals_3 else None

        # 7-day trailing
        dates_7 = _trailing_dates(event_start_date, 7)
        vals_7 = []
        for d in dates_7:
            if d in date_lookup:
                idx = date_lookup[d]
                val = prior.iloc[idx].get(col)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    vals_7.append(float(val))
        avg_7 = _safe_mean(vals_7) if vals_7 else None

        if avg_7 is not None:
            features[f"gt_{entity}_7d"] = round(avg_7, 2)
            entity_7d_avgs.append(avg_7)

        if avg_3 is not None and avg_7 is not None and avg_7 > 0:
            features[f"gt_{entity}_delta"] = round((avg_3 - avg_7) / avg_7, 4)

    # Total and concentration (HHI)
    if entity_7d_avgs:
        total = sum(entity_7d_avgs)
        features["gt_total_7d"] = round(total, 2)
        if total > 0:
            hhi = sum((v / total) ** 2 for v in entity_7d_avgs)
            features["gt_concentration"] = round(hhi, 4)

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
        # NEW
        "cross_market_daily_weekly_div": 0,
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
