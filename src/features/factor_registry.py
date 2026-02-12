"""
Comprehensive factor registry for Elon Musk tweet count prediction.
====================================================================

Defines ~30 predictive factors organized by category, with metadata,
computation formulas, rationale, and data-source references. This module
serves as the single source of truth for which features the model consumes.

Categories:
    temporal   - Rolling averages, volatility, trends, regime indicators
    media      - GDELT news volume and tone signals
    calendar   - SpaceX launches, day-of-week, holiday effects
    market     - Crowd-derived signals from Polymarket prices
    cross      - Interaction features combining multiple raw signals

Usage:
    from src.features.factor_registry import FACTOR_REGISTRY, print_factor_summary

    for factor in FACTOR_REGISTRY:
        print(factor["name"], factor["category"])

    print_factor_summary()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Factor dataclass
# ---------------------------------------------------------------------------
@dataclass
class Factor:
    """A single predictive factor (feature) definition."""

    name: str
    category: str  # temporal | media | calendar | market | cross
    description: str
    formula: str
    rationale: str
    data_source: str
    coverage_pct: float  # Estimated % of backtest events that have this factor
    expected_range: str  # Human-readable expected value range
    eda_signal: str = ""  # Key EDA finding, if any
    existing: bool = False  # True if already computed in build_backtest_dataset.py
    priority: str = "high"  # high | medium | low

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# TEMPORAL factors (~10)
# Derived from XTracker daily tweet counts (103 days: 2025-10-30 to 2026-02-09)
# ---------------------------------------------------------------------------
TEMPORAL_FACTORS: List[Factor] = [
    Factor(
        name="rolling_avg_7d",
        category="temporal",
        description="Mean daily tweet count over the trailing 7 days before event start",
        formula="mean(daily_count[t-7 : t-1]), excluding zero-artifact days",
        rationale=(
            "Most direct predictor. 7-day window balances recency and noise. "
            "EDA: corr=0.72 with actual count when scaled by duration."
        ),
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,  # 42/159 events
        expected_range="20-90 tweets/day",
        eda_signal="corr=0.72, MAPE=23.8% as standalone predictor",
        existing=True,
        priority="high",
    ),
    Factor(
        name="rolling_avg_14d",
        category="temporal",
        description="Mean daily tweet count over the trailing 14 days",
        formula="mean(daily_count[t-14 : t-1]), excluding zero-artifact days",
        rationale=(
            "Smoother baseline than 7d, captures medium-term activity level. "
            "Less sensitive to single-day spikes."
        ),
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,
        expected_range="20-80 tweets/day",
        existing=True,
        priority="high",
    ),
    Factor(
        name="rolling_avg_28d",
        category="temporal",
        description="Mean daily tweet count over the trailing 28 days",
        formula="mean(daily_count[t-28 : t-1]), excluding zero-artifact days",
        rationale=(
            "Long-term baseline. Useful as denominator in regime_ratio. "
            "Slow-moving anchor for mean-reversion signals."
        ),
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,
        expected_range="25-70 tweets/day",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="yesterday_count",
        category="temporal",
        description="Tweet count on the single day before the event starts",
        formula="daily_count[t-1]",
        rationale=(
            "Captures very-recent momentum. High yesterday often means "
            "high tomorrow (short-term autocorrelation)."
        ),
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,
        expected_range="0-150 tweets",
        existing=True,
        priority="high",
    ),
    Factor(
        name="rolling_std_7d",
        category="temporal",
        description="Standard deviation of daily counts over trailing 7 days",
        formula="std(daily_count[t-7 : t-1])",
        rationale=(
            "Volatility measure. High std = regime uncertainty. "
            "EDA: volatility autocorrelation is -0.45, suggesting mean reversion."
        ),
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,
        expected_range="5-40 tweets/day",
        eda_signal="Volatility mean-reverts (autocorrelation -0.45)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="rolling_std_14d",
        category="temporal",
        description="Standard deviation of daily counts over trailing 14 days",
        formula="std(daily_count[t-14 : t-1])",
        rationale="Smoother volatility measure. Robust to outlier days.",
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,
        expected_range="5-35 tweets/day",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="trend_7d",
        category="temporal",
        description="Linear regression slope of daily counts over trailing 7 days",
        formula="polyfit(range(7), daily_count[t-7:t-1], 1)[0]",
        rationale=(
            "Direction and speed of recent trend. Positive slope = "
            "activity accelerating. Used with regime_ratio for trend detection."
        ),
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,
        expected_range="-20 to +20 tweets/day per day",
        existing=True,
        priority="high",
    ),
    Factor(
        name="cv_14d",
        category="temporal",
        description="Coefficient of variation (std/mean) over trailing 14 days",
        formula="rolling_std_14d / rolling_avg_14d",
        rationale=(
            "Normalized volatility. High CV = uncertain regime, wider prediction "
            "intervals needed. EDA: high-CV events have 2x larger prediction errors."
        ),
        data_source="XTracker daily_metrics_full.json (derived)",
        coverage_pct=26.4,
        expected_range="0.1-0.8",
        eda_signal="High CV events have 2x larger rolling-avg prediction errors",
        existing=True,
        priority="high",
    ),
    Factor(
        name="regime_ratio",
        category="temporal",
        description="Ratio of 7-day to 28-day rolling average",
        formula="rolling_avg_7d / rolling_avg_28d",
        rationale=(
            "Regime change detector. >1 = recent acceleration, <1 = deceleration. "
            "EDA: crowd error is 2x larger during regime transitions."
        ),
        data_source="XTracker daily_metrics_full.json (derived)",
        coverage_pct=26.4,
        expected_range="0.5-2.0",
        eda_signal="Crowd error 2x larger during regime transitions",
        existing=True,
        priority="high",
    ),
    Factor(
        name="day_of_week",
        category="temporal",
        description="Day of week the event starts (0=Monday, 6=Sunday)",
        formula="event_start_date.weekday()",
        rationale=(
            "Captures weekly periodicity. Most markets start on Friday, "
            "but varying start days may affect first-day tweeting behavior."
        ),
        data_source="Event metadata",
        coverage_pct=100.0,
        expected_range="0-6 (categorical)",
        existing=True,
        priority="low",
    ),
    Factor(
        name="weekend_ratio_7d",
        category="temporal",
        description=(
            "Ratio of mean weekend (Sat+Sun) to weekday tweet counts "
            "over trailing 14 days"
        ),
        formula=(
            "mean(count[sat,sun] in last 14d) / mean(count[mon-fri] in last 14d)"
        ),
        rationale=(
            "Detects weekend vs weekday pattern shifts. If Musk tweets more "
            "on weekends recently, a week starting Saturday has upside."
        ),
        data_source="XTracker daily_metrics_full.json",
        coverage_pct=26.4,
        expected_range="0.5-1.5",
        existing=False,
        priority="medium",
    ),
]


# ---------------------------------------------------------------------------
# MEDIA/GDELT factors (~8)
# Derived from GDELT DOC API timelines: 4 entities x 3 modes, Jan 2024 - Feb 2026
# ---------------------------------------------------------------------------
MEDIA_FACTORS: List[Factor] = [
    Factor(
        name="elon_musk_vol_7d",
        category="media",
        description="Mean GDELT normalized volume for 'Elon Musk' over trailing 7 days",
        formula="mean(gdelt_elon_musk_timelinevol[t-7 : t-1])",
        rationale=(
            "Baseline Musk media attention. Higher volume = more events to "
            "react to = potentially more tweets."
        ),
        data_source="GDELT gdelt_elon_musk_timelinevol.json",
        coverage_pct=98.1,  # 156/159 events
        expected_range="0.1-1.0 (normalized)",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="elon_musk_vol_delta",
        category="media",
        description=(
            "GDELT volume spike indicator: 3-day mean minus 7-day mean "
            "for 'Elon Musk'"
        ),
        formula="mean(vol[t-3:t-1]) - mean(vol[t-7:t-1])",
        rationale=(
            "Detects sudden surges in Musk media coverage. A positive delta "
            "signals a breaking story that may trigger tweet storms."
        ),
        data_source="GDELT gdelt_elon_musk_timelinevol.json",
        coverage_pct=98.1,
        expected_range="-0.3 to +0.3",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="elon_musk_tone_7d",
        category="media",
        description="Mean GDELT tone (sentiment) for 'Elon Musk' over trailing 7 days",
        formula="mean(gdelt_elon_musk_timelinetone[t-7 : t-1])",
        rationale=(
            "Negative tone = bad press about Musk. EDA: r=-0.51 with daily "
            "rate. Bad press correlates with MORE tweeting (defensive/reactive)."
        ),
        data_source="GDELT gdelt_elon_musk_timelinetone.json",
        coverage_pct=98.1,
        expected_range="-3.0 to +1.0",
        eda_signal="r=-0.51 with daily tweet rate (bad press = more tweets)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="elon_musk_tone_delta",
        category="media",
        description="Change in Musk media tone: 3-day mean minus 7-day mean",
        formula="mean(tone[t-3:t-1]) - mean(tone[t-7:t-1])",
        rationale=(
            "Captures tone shifts. A sudden drop in tone (more negative) "
            "may trigger defensive tweeting ahead of the prediction window."
        ),
        data_source="GDELT gdelt_elon_musk_timelinetone.json",
        coverage_pct=98.1,
        expected_range="-1.5 to +1.5",
        existing=False,
        priority="medium",
    ),
    Factor(
        name="tesla_vol_7d",
        category="media",
        description="Mean GDELT normalized volume for 'Tesla' over trailing 7 days",
        formula="mean(gdelt_tesla_timelinevol[t-7 : t-1])",
        rationale=(
            "Tesla news often triggers Musk tweets (earnings, recalls, "
            "stock moves). Separate signal from personal Musk coverage."
        ),
        data_source="GDELT gdelt_tesla_timelinevol.json",
        coverage_pct=98.1,
        expected_range="0.1-0.8 (normalized)",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="spacex_vol_7d",
        category="media",
        description="Mean GDELT normalized volume for 'SpaceX' over trailing 7 days",
        formula="mean(gdelt_spacex_timelinevol[t-7 : t-1])",
        rationale=(
            "SpaceX news (launches, Starship tests) can trigger tweeting. "
            "But EDA shows launches_trailing_7d has r=-0.52, so high SpaceX "
            "activity may actually REDUCE personal tweeting."
        ),
        data_source="GDELT gdelt_spacex_timelinevol.json",
        coverage_pct=98.1,
        expected_range="0.05-0.6 (normalized)",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="total_media_vol_7d",
        category="media",
        description=(
            "Sum of GDELT 7-day volumes across all four entities "
            "(Elon, Tesla, SpaceX, Neuralink)"
        ),
        formula=(
            "elon_musk_vol_7d + tesla_vol_7d + spacex_vol_7d + neuralink_vol_7d"
        ),
        rationale=(
            "Aggregate media attention proxy. High total volume = Musk is in "
            "the news cycle from multiple angles. Captures cross-entity storms."
        ),
        data_source="GDELT (all entity volumes combined)",
        coverage_pct=98.1,
        expected_range="0.3-2.5",
        existing=False,
        priority="medium",
    ),
    Factor(
        name="media_vol_concentration",
        category="media",
        description=(
            "Herfindahl index of media volume across entities: "
            "sum(share_i^2) where share_i = entity_vol / total_vol"
        ),
        formula=(
            "sum((entity_vol_7d / total_media_vol_7d)^2) for each entity"
        ),
        rationale=(
            "Measures whether media attention is concentrated on one entity "
            "(e.g., Tesla-only firestorm) vs. spread evenly. Concentrated "
            "attention may trigger more focused tweet responses."
        ),
        data_source="GDELT (derived from entity volumes)",
        coverage_pct=98.1,
        expected_range="0.25-1.0 (0.25=even, 1.0=single entity)",
        existing=False,
        priority="low",
    ),
]


# ---------------------------------------------------------------------------
# CALENDAR/SPACEX factors (~5)
# Derived from Launch Library 2 API (323 launches, 2024-2026)
# ---------------------------------------------------------------------------
CALENDAR_FACTORS: List[Factor] = [
    Factor(
        name="launches_trailing_7d",
        category="calendar",
        description="Number of SpaceX launches in the 7 days before event start",
        formula="count(launches where date in [t-7, t-1])",
        rationale=(
            "Strongest calendar signal. EDA: r=-0.52 with daily tweet rate. "
            "Busy launch weeks likely occupy Musk's attention away from X."
        ),
        data_source="Launch Library 2 spacex_launches_historical.json",
        coverage_pct=98.7,  # 157/159 events
        expected_range="0-5 launches",
        eda_signal="r=-0.52 with daily tweet rate (strongest calendar signal)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="launches_during_event",
        category="calendar",
        description="Number of SpaceX launches during the prediction window",
        formula="count(launches where date in [event_start, event_end])",
        rationale=(
            "Forward-looking launch count during the market's window. "
            "Can be computed from the published schedule."
        ),
        data_source="Launch Library 2 spacex_launches_historical.json",
        coverage_pct=98.7,
        expected_range="0-5 launches",
        existing=True,
        priority="high",
    ),
    Factor(
        name="days_to_next_launch",
        category="calendar",
        description="Days from event start to the next scheduled SpaceX launch",
        formula="min(launch_date - event_start) for launch_date >= event_start",
        rationale=(
            "Imminent launch = potential distraction. Zero means launch day "
            "coincides with event start."
        ),
        data_source="Launch Library 2 spacex_launches_historical.json",
        coverage_pct=98.7,
        expected_range="0-14 days",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="is_holiday_week",
        category="calendar",
        description=(
            "Binary flag: 1 if the event window overlaps a major US holiday "
            "(Thanksgiving, Christmas, New Year, July 4th, etc.)"
        ),
        formula=(
            "1 if any(holiday_date in [event_start, event_end]) else 0, "
            "holidays defined from a static list"
        ),
        rationale=(
            "Holiday weeks often show reduced tweeting due to travel/family time. "
            "Christmas 2025 week had notably low counts."
        ),
        data_source="Static holiday calendar (US federal holidays)",
        coverage_pct=100.0,
        expected_range="0 or 1",
        existing=False,
        priority="medium",
    ),
    Factor(
        name="event_duration_days",
        category="calendar",
        description="Number of days in the prediction window (event_end - event_start)",
        formula="(event_end_date - event_start_date).days",
        rationale=(
            "Critical scaling factor. Daily markets (2-3 days) vs weekly (7 days) "
            "vs monthly have fundamentally different count ranges. Must be used "
            "to convert daily rate predictions to total count."
        ),
        data_source="Event metadata (start_date, end_date)",
        coverage_pct=100.0,
        expected_range="2-31 days",
        existing=False,
        priority="high",
    ),
]


# ---------------------------------------------------------------------------
# MARKET-DERIVED factors (~7)
# Derived from Polymarket CLOB price history (hourly bucket prices)
# ---------------------------------------------------------------------------
MARKET_FACTORS: List[Factor] = [
    Factor(
        name="crowd_implied_ev",
        category="market",
        description=(
            "Crowd-implied expected value: sum(bucket_midpoint * bucket_prob) "
            "at T-24h before event close"
        ),
        formula="sum(midpoint_i * (price_i / sum(prices))) for all buckets",
        rationale=(
            "The market's best guess of the final count. Strong baseline "
            "predictor. EDA: MAPE ~25% on weekly markets."
        ),
        data_source="Polymarket price_history.parquet + market_catalog.parquet",
        coverage_pct=87.4,  # 139/159 events
        expected_range="30-600 tweets",
        existing=True,
        priority="high",
    ),
    Factor(
        name="crowd_std_dev",
        category="market",
        description=(
            "Crowd-implied standard deviation of the distribution at T-24h"
        ),
        formula="sqrt(sum(prob_i * (midpoint_i - implied_ev)^2))",
        rationale=(
            "Measures crowd uncertainty. Wide std = market disagrees on outcome. "
            "Events with high crowd uncertainty may have more exploitable "
            "mispricings."
        ),
        data_source="Polymarket price_history.parquet",
        coverage_pct=87.4,
        expected_range="10-120 tweets",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="distribution_entropy",
        category="market",
        description="Shannon entropy of the crowd distribution at T-24h",
        formula="-sum(p_i * log(p_i)) for p_i > 0",
        rationale=(
            "High entropy = crowd sees many outcomes as plausible (flat distribution). "
            "Low entropy = crowd is concentrated on a few buckets. "
            "High-entropy markets may be more beatable."
        ),
        data_source="Polymarket price_history.parquet",
        coverage_pct=87.4,
        expected_range="0.5-3.0 nats",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="price_shift_24h",
        category="market",
        description="Change in crowd implied EV from T-48h to T-24h",
        formula="crowd_implied_ev(T-24h) - crowd_implied_ev(T-48h)",
        rationale=(
            "Market momentum: positive = crowd is revising upward. "
            "Captures late-breaking information flowing into prices."
        ),
        data_source="Polymarket price_history.parquet",
        coverage_pct=87.4,
        expected_range="-100 to +100 tweets",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="crowd_vs_rolling_avg",
        category="market",
        description=(
            "Difference between crowd implied EV and rolling_avg_7d * duration"
        ),
        formula="crowd_implied_ev - (rolling_avg_7d * event_duration_days)",
        rationale=(
            "Measures divergence between crowd and simple model. If crowd is "
            "much higher than the rolling average extrapolation, the crowd may "
            "be overreacting to recent events. EDA: crowd overestimates by "
            "+27% on daily markets."
        ),
        data_source="Derived (market + temporal)",
        coverage_pct=26.4,  # Need both market and temporal
        expected_range="-200 to +200 tweets",
        eda_signal="Crowd overestimates by +27% on daily markets",
        existing=False,
        priority="high",
    ),
    Factor(
        name="crowd_skewness",
        category="market",
        description=(
            "Skewness of the crowd-implied distribution at T-24h: "
            "sum(prob * ((mid - ev) / std)^3)"
        ),
        formula="sum(p_i * ((midpoint_i - implied_ev) / crowd_std_dev)^3)",
        rationale=(
            "Positive skew = crowd puts more weight on high-count tails. "
            "Negative skew = crowd expects downside risk. Asymmetric crowd "
            "expectations reveal directional biases."
        ),
        data_source="Polymarket price_history.parquet",
        coverage_pct=87.4,
        expected_range="-2.0 to +2.0",
        existing=False,
        priority="medium",
    ),
    Factor(
        name="crowd_kurtosis",
        category="market",
        description=(
            "Excess kurtosis of the crowd-implied distribution at T-24h"
        ),
        formula="sum(p_i * ((midpoint_i - implied_ev) / crowd_std_dev)^4) - 3",
        rationale=(
            "High kurtosis = crowd distribution has fat tails (extreme outcomes "
            "seen as more likely than normal). Low kurtosis = crowd is tightly "
            "concentrated. Complements entropy."
        ),
        data_source="Polymarket price_history.parquet",
        coverage_pct=87.4,
        expected_range="-2.0 to +5.0",
        existing=False,
        priority="low",
    ),
]


# ---------------------------------------------------------------------------
# CROSS-INTERACTION factors (~5)
# Derived by combining signals from multiple raw data sources
# ---------------------------------------------------------------------------
CROSS_FACTORS: List[Factor] = [
    Factor(
        name="bad_press_x_low_activity",
        category="cross",
        description=(
            "Interaction: negative media tone AND below-average recent tweeting. "
            "Product of z-scored tone (inverted) and z-scored activity deficit."
        ),
        formula=(
            "max(0, -elon_musk_tone_7d / tone_std) * "
            "max(0, (rolling_avg_28d - rolling_avg_7d) / rolling_std_14d)"
        ),
        rationale=(
            "Combines two strong EDA signals: negative tone (r=-0.51) and "
            "activity downturn. When Musk is tweeting less than usual AND "
            "press is negative, a tweet storm (mean reversion + defensive "
            "response) becomes more likely."
        ),
        data_source="GDELT tone + XTracker daily counts",
        coverage_pct=26.4,  # Needs both GDELT and temporal
        expected_range="0-5 (unbounded positive)",
        existing=False,
        priority="high",
    ),
    Factor(
        name="launch_busy_x_trend_down",
        category="cross",
        description=(
            "Interaction: busy SpaceX launch week AND declining tweet trend"
        ),
        formula=(
            "launches_trailing_7d * max(0, -trend_7d)"
        ),
        rationale=(
            "Amplifies the launch distraction signal when the trend is already "
            "declining. Launches alone have r=-0.52, and a negative trend "
            "reinforces the downward pressure on tweet counts."
        ),
        data_source="Launch Library 2 + XTracker daily counts",
        coverage_pct=26.4,
        expected_range="0-50",
        existing=False,
        priority="medium",
    ),
    Factor(
        name="regime_transition_flag",
        category="cross",
        description=(
            "Binary flag: 1 if regime_ratio deviates more than 0.3 from 1.0, "
            "indicating the recent week is materially different from the "
            "trailing 4-week average"
        ),
        formula="1 if abs(regime_ratio - 1.0) > 0.3 else 0",
        rationale=(
            "EDA: crowd error is 2x larger during regime transitions. This "
            "flag signals when a regime shift is underway, which is precisely "
            "when the model can add value over the crowd."
        ),
        data_source="XTracker daily counts (derived from regime_ratio)",
        coverage_pct=26.4,
        expected_range="0 or 1",
        eda_signal="Crowd error 2x larger during regime transitions",
        existing=False,
        priority="high",
    ),
    Factor(
        name="high_vol_x_high_entropy",
        category="cross",
        description=(
            "Interaction: high media volatility AND high crowd uncertainty. "
            "Product of above-median GDELT volume delta and above-median entropy."
        ),
        formula=(
            "max(0, elon_musk_vol_delta) * "
            "max(0, distribution_entropy - median_entropy)"
        ),
        rationale=(
            "When there is a media surge AND the crowd is uncertain, the "
            "information has not yet been incorporated into prices. This "
            "identifies events where fresh model predictions are most valuable."
        ),
        data_source="GDELT volume + Polymarket prices",
        coverage_pct=87.4,
        expected_range="0-1.0",
        existing=False,
        priority="medium",
    ),
    Factor(
        name="momentum_reversal_signal",
        category="cross",
        description=(
            "Composite mean-reversion indicator combining volatility mean-reversion "
            "and trend direction"
        ),
        formula=(
            "sign(rolling_avg_28d - rolling_avg_7d) * cv_14d * "
            "abs(regime_ratio - 1.0)"
        ),
        rationale=(
            "EDA shows volatility mean-reverts (autocorrelation -0.45) and "
            "crowd captures only 48% of downswings. This factor is positive "
            "when recent activity is below long-term average with high "
            "variability (upward reversion likely), or when recent activity "
            "is above average with high variability (downward correction likely)."
        ),
        data_source="XTracker daily counts (derived)",
        coverage_pct=26.4,
        expected_range="-0.5 to +0.5",
        eda_signal=(
            "Volatility autocorrelation -0.45; crowd only captures 48% "
            "of downswings but 105% of upswings"
        ),
        existing=False,
        priority="high",
    ),
]


# ---------------------------------------------------------------------------
# COMPLETE REGISTRY
# ---------------------------------------------------------------------------
FACTOR_REGISTRY: List[Factor] = (
    TEMPORAL_FACTORS + MEDIA_FACTORS + CALENDAR_FACTORS
    + MARKET_FACTORS + CROSS_FACTORS
)

# Quick-lookup by name
FACTOR_BY_NAME: dict[str, Factor] = {f.name: f for f in FACTOR_REGISTRY}

# Category grouping
FACTORS_BY_CATEGORY: dict[str, List[Factor]] = {}
for _f in FACTOR_REGISTRY:
    FACTORS_BY_CATEGORY.setdefault(_f.category, []).append(_f)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_factor_summary() -> None:
    """Print a formatted summary table of all factors to stdout."""

    sep = "=" * 110
    thin_sep = "-" * 110

    print(sep)
    print("  ELON TWEET MARKETS -- COMPREHENSIVE FACTOR REGISTRY")
    print("  {} total factors across {} categories".format(
        len(FACTOR_REGISTRY),
        len(FACTORS_BY_CATEGORY),
    ))
    print(sep)

    header = (
        "{:<4s} {:<30s} {:<10s} {:<8s} {:<10s} {:<8s} {:<38s}"
    ).format("#", "Factor Name", "Category", "Priority", "Coverage", "Status", "Key Signal / Rationale")
    print(header)
    print(thin_sep)

    idx = 1
    for category in ["temporal", "media", "calendar", "market", "cross"]:
        factors = FACTORS_BY_CATEGORY.get(category, [])
        if not factors:
            continue

        # Category header
        print("\n  [{}] ({} factors)".format(category.upper(), len(factors)))
        print(thin_sep)

        for f in factors:
            status = "EXISTS" if f.existing else "NEW"
            signal = f.eda_signal if f.eda_signal else f.rationale[:36] + "..."
            # Truncate signal to fit
            if len(signal) > 38:
                signal = signal[:35] + "..."

            line = "{:<4d} {:<30s} {:<10s} {:<8s} {:<10s} {:<8s} {:<38s}".format(
                idx,
                f.name,
                f.category,
                f.priority,
                "{:.0f}%".format(f.coverage_pct),
                status,
                signal,
            )
            print(line)
            idx += 1

    print(thin_sep)

    # Summary stats
    existing_count = sum(1 for f in FACTOR_REGISTRY if f.existing)
    new_count = len(FACTOR_REGISTRY) - existing_count
    high_count = sum(1 for f in FACTOR_REGISTRY if f.priority == "high")
    medium_count = sum(1 for f in FACTOR_REGISTRY if f.priority == "medium")
    low_count = sum(1 for f in FACTOR_REGISTRY if f.priority == "low")

    print("\n  SUMMARY")
    print("  Total factors:    {}".format(len(FACTOR_REGISTRY)))
    print("  Already computed: {} (existing in build_backtest_dataset.py)".format(
        existing_count))
    print("  New to implement: {}".format(new_count))
    print("")
    print("  By priority:  high={}, medium={}, low={}".format(
        high_count, medium_count, low_count))
    print("")
    print("  By category:")
    for cat in ["temporal", "media", "calendar", "market", "cross"]:
        factors = FACTORS_BY_CATEGORY.get(cat, [])
        print("    {:<12s}: {} factors".format(cat, len(factors)))

    # Coverage tiers
    print("")
    print("  Coverage tiers:")
    full = sum(1 for f in FACTOR_REGISTRY if f.coverage_pct >= 95)
    partial = sum(1 for f in FACTOR_REGISTRY if 25 <= f.coverage_pct < 95)
    sparse = sum(1 for f in FACTOR_REGISTRY if f.coverage_pct < 25)
    print("    Full (>=95%):    {} factors (GDELT, calendar, market-derived)".format(full))
    print("    Partial (25-95%): {} factors (market-derived)".format(partial))
    print("    Sparse (<25%):   {} factors (XTracker-era temporal only)".format(sparse))

    # Key EDA findings driving factor design
    print("")
    print("  KEY EDA FINDINGS INFORMING FACTOR DESIGN:")
    print("    1. SpaceX launches_trailing_7d: r=-0.52 (busy launch week = fewer tweets)")
    print("    2. Negative media tone (r=-0.51): bad press = more tweets")
    print("    3. Volatility mean-reverts (autocorrelation -0.45)")
    print("    4. Crowd captures only 48% of downswings but 105% of upswings")
    print("    5. Crowd overestimates by +27% on daily markets")
    print("    6. Crowd error is 2x larger during regime transitions")
    print("    7. High-CV events have 2x larger prediction errors")

    print(sep)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_registry() -> bool:
    """Run validation checks on the factor registry."""
    print("\n--- Factor Registry Validation ---\n")
    errors = 0

    # Check for unique names
    names = [f.name for f in FACTOR_REGISTRY]
    if len(names) != len(set(names)):
        dupes = [n for n in names if names.count(n) > 1]
        print("  FAIL: Duplicate factor names: {}".format(set(dupes)))
        errors += 1
    else:
        print("  PASS: All {} factor names are unique".format(len(names)))

    # Check valid categories
    valid_cats = {"temporal", "media", "calendar", "market", "cross"}
    bad_cats = [f.name for f in FACTOR_REGISTRY if f.category not in valid_cats]
    if bad_cats:
        print("  FAIL: Invalid categories for: {}".format(bad_cats))
        errors += 1
    else:
        print("  PASS: All categories are valid ({})".format(
            ", ".join(sorted(valid_cats))))

    # Check valid priorities
    valid_prios = {"high", "medium", "low"}
    bad_prios = [f.name for f in FACTOR_REGISTRY if f.priority not in valid_prios]
    if bad_prios:
        print("  FAIL: Invalid priorities for: {}".format(bad_prios))
        errors += 1
    else:
        print("  PASS: All priorities are valid")

    # Check coverage percentages
    bad_cov = [f.name for f in FACTOR_REGISTRY
               if not (0 <= f.coverage_pct <= 100)]
    if bad_cov:
        print("  FAIL: Invalid coverage_pct for: {}".format(bad_cov))
        errors += 1
    else:
        print("  PASS: All coverage percentages in [0, 100]")

    # Check required fields are non-empty
    for f in FACTOR_REGISTRY:
        for field_name in ["description", "formula", "rationale", "data_source"]:
            val = getattr(f, field_name)
            if not val or not val.strip():
                print("  FAIL: Factor '{}' has empty '{}'".format(
                    f.name, field_name))
                errors += 1

    if errors == 0:
        print("\n  ALL CHECKS PASSED ({} factors validated)".format(
            len(FACTOR_REGISTRY)))
    else:
        print("\n  {} ERRORS found".format(errors))

    return errors == 0


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_factor_summary()
    print("")
    ok = validate_registry()
    if not ok:
        exit(1)
