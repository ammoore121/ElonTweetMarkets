"""
Comprehensive factor registry for Elon Musk tweet count prediction.
====================================================================

Defines ~69 predictive factors organized by category, with metadata,
computation formulas, rationale, and data-source references. This module
serves as the single source of truth for which features the model consumes.

Categories:
    temporal   - Rolling averages, volatility, trends, regime indicators
    media      - GDELT news volume and tone signals
    calendar   - SpaceX launches, day-of-week, holiday effects
    market     - Crowd-derived signals from Polymarket prices
    cross      - Interaction features combining multiple raw signals
    financial  - Tesla stock and crypto (DOGE, BTC) price/volatility signals
    attention  - Wikipedia pageview signals as public attention proxies
    trends     - Google Trends search interest signals
    government - Federal Register + GovTrack event signals
    corporate  - Corporate events (earnings, launches) from yfinance + SEC EDGAR
    reddit     - Reddit community activity signals

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
    category: str  # temporal | media | calendar | market | cross | financial | attention | trends | government | corporate | reddit
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
# FINANCIAL factors (~12)
# Derived from Tesla stock (yfinance TSLA) and crypto (yfinance DOGE-USD, BTC-USD)
# ---------------------------------------------------------------------------
FINANCIAL_FACTORS: List[Factor] = [
    Factor(
        name="tsla_pct_change_1d",
        category="financial",
        description="Tesla 1-day percentage price change",
        formula="(close[t-1] - close[t-2]) / close[t-2]",
        rationale=(
            "Immediate Tesla stock momentum. Large daily moves may trigger "
            "Elon tweets about stock/market."
        ),
        data_source="yfinance TSLA daily OHLCV (data/sources/market/tesla_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.10 to +0.10",
        existing=True,
        priority="high",
    ),
    Factor(
        name="tsla_pct_change_5d",
        category="financial",
        description="Tesla 5-day cumulative return",
        formula="(close[t-1] - close[t-5]) / close[t-5]",
        rationale=(
            "Medium-term Tesla momentum. Sustained rallies or sell-offs "
            "over a week may shift Musk's tweeting focus toward markets."
        ),
        data_source="yfinance TSLA daily OHLCV (data/sources/market/tesla_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.20 to +0.20",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="tsla_volatility_5d",
        category="financial",
        description="Tesla 5-day realized volatility (std of daily returns)",
        formula="std(pct_change[t-5:t-1])",
        rationale=(
            "High TSLA vol = Elon likely tweeting about markets. "
            "Signal threshold: >3% = elevated."
        ),
        data_source="yfinance TSLA daily OHLCV (data/sources/market/tesla_daily.parquet)",
        coverage_pct=98.7,
        expected_range="0.01-0.08",
        eda_signal="Used by SignalEnhancedTailModel (threshold=0.03)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="tsla_volume_ratio",
        category="financial",
        description="Tesla volume ratio (last day vs 5-day moving average)",
        formula="volume[t-1] / mean(volume[t-5:t-1])",
        rationale=(
            "Unusual volume signals institutional activity or news events "
            "around Tesla that may provoke Musk to tweet."
        ),
        data_source="yfinance TSLA daily OHLCV (data/sources/market/tesla_daily.parquet)",
        coverage_pct=98.7,
        expected_range="0.3-3.0",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="tsla_drawdown_5d",
        category="financial",
        description="Tesla drawdown from 5-day closing high",
        formula="(close[t-1] - max(close[t-5:t-1])) / max(close[t-5:t-1])",
        rationale=(
            "Stock dropping = defensive tweeting. "
            "Signal threshold: <-5%."
        ),
        data_source="yfinance TSLA daily OHLCV (data/sources/market/tesla_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.15 to 0.00",
        eda_signal="Used by SignalEnhancedTailModel (threshold=-0.05)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="tsla_gap_1d",
        category="financial",
        description="Tesla overnight gap (open vs previous close)",
        formula="(open[t-1] - close[t-2]) / close[t-2]",
        rationale=(
            "Overnight gaps reflect after-hours news. Large gaps may "
            "indicate events Musk reacts to early in the day."
        ),
        data_source="yfinance TSLA daily OHLCV (data/sources/market/tesla_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.05 to +0.05",
        existing=True,
        priority="low",
    ),
    Factor(
        name="doge_pct_change_1d",
        category="financial",
        description="Dogecoin 1-day percentage price change",
        formula="(close[t-1] - close[t-2]) / close[t-2]",
        rationale=(
            "DOGE pump/dump = Elon crypto tweet storm. "
            "Signal threshold: >5% daily move."
        ),
        data_source="yfinance DOGE-USD (data/sources/market/crypto_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.15 to +0.15",
        eda_signal="Used by SignalEnhancedTailModel (threshold=0.05)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="doge_pct_change_5d",
        category="financial",
        description="Dogecoin 5-day cumulative return",
        formula="(close[t-1] - close[t-5]) / close[t-5]",
        rationale=(
            "Medium-term DOGE momentum. Sustained DOGE moves often "
            "accompany sustained Musk engagement on crypto topics."
        ),
        data_source="yfinance DOGE-USD (data/sources/market/crypto_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.30 to +0.30",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="doge_volatility_5d",
        category="financial",
        description="Dogecoin 5-day realized volatility (std of daily returns)",
        formula="std(pct_change[t-5:t-1])",
        rationale=(
            "High DOGE volatility indicates crypto market turbulence "
            "that may trigger Musk tweets about crypto/memes."
        ),
        data_source="yfinance DOGE-USD (data/sources/market/crypto_daily.parquet)",
        coverage_pct=98.7,
        expected_range="0.02-0.12",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="btc_pct_change_1d",
        category="financial",
        description="Bitcoin 1-day percentage price change",
        formula="(close[t-1] - close[t-2]) / close[t-2]",
        rationale=(
            "Broader crypto market context. BTC moves often accompany "
            "DOGE moves."
        ),
        data_source="yfinance BTC-USD (data/sources/market/crypto_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.10 to +0.10",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="btc_pct_change_5d",
        category="financial",
        description="Bitcoin 5-day cumulative return",
        formula="(close[t-1] - close[t-5]) / close[t-5]",
        rationale=(
            "Medium-term BTC trend. Provides context for whether "
            "DOGE moves are crypto-wide or DOGE-specific."
        ),
        data_source="yfinance BTC-USD (data/sources/market/crypto_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.15 to +0.15",
        existing=True,
        priority="low",
    ),
    Factor(
        name="btc_volatility_5d",
        category="financial",
        description="Bitcoin 5-day realized volatility (std of daily returns)",
        formula="std(pct_change[t-5:t-1])",
        rationale=(
            "Broad crypto volatility regime indicator. High BTC vol "
            "signals turbulent crypto markets overall."
        ),
        data_source="yfinance BTC-USD (data/sources/market/crypto_daily.parquet)",
        coverage_pct=98.7,
        expected_range="0.01-0.06",
        existing=True,
        priority="low",
    ),
]


# ---------------------------------------------------------------------------
# ATTENTION factors (~8)
# Derived from Wikipedia pageviews via Wikimedia REST API
# ---------------------------------------------------------------------------
ATTENTION_FACTORS: List[Factor] = [
    Factor(
        name="wiki_elon_musk_7d",
        category="attention",
        description="7-day average daily pageviews for the Elon_Musk Wikipedia article",
        formula="mean(pageviews['Elon_Musk'][t-7:t-1])",
        rationale=(
            "Public attention proxy. Average ~61.5K/day. "
            "Spikes during controversies."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="20000-120000 views/day",
        existing=True,
        priority="high",
    ),
    Factor(
        name="wiki_elon_musk_delta",
        category="attention",
        description=(
            "Elon Musk pageview spike: (3-day avg - 7-day avg) / 7-day avg"
        ),
        formula="(mean(pv[t-3:t-1]) - mean(pv[t-7:t-1])) / mean(pv[t-7:t-1])",
        rationale=(
            "Attention spike detector. Signal threshold: >15% above "
            "7-day avg = spike."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="-0.30 to +1.00",
        eda_signal="Used by SignalEnhancedTailModel (threshold=0.15)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="wiki_tesla_7d",
        category="attention",
        description="7-day average daily pageviews for the Tesla_Inc Wikipedia article",
        formula="mean(pageviews['Tesla_Inc'][t-7:t-1])",
        rationale=(
            "Tesla-specific public attention. Spikes around earnings, "
            "recalls, and major product announcements."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="15000-80000 views/day",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="wiki_tesla_delta",
        category="attention",
        description="Tesla pageview spike: (3-day avg - 7-day avg) / 7-day avg",
        formula="(mean(pv[t-3:t-1]) - mean(pv[t-7:t-1])) / mean(pv[t-7:t-1])",
        rationale=(
            "Tesla attention spike detector. Sudden interest in Tesla "
            "may correlate with Musk tweeting about company news."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="-0.30 to +1.00",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="wiki_doge_7d",
        category="attention",
        description="7-day average daily pageviews for the Dogecoin Wikipedia article",
        formula="mean(pageviews['Dogecoin'][t-7:t-1])",
        rationale=(
            "Dogecoin public interest proxy. Spikes during meme cycles "
            "and Musk crypto tweets."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="5000-50000 views/day",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="wiki_doge_delta",
        category="attention",
        description="Dogecoin pageview spike: (3-day avg - 7-day avg) / 7-day avg",
        formula="(mean(pv[t-3:t-1]) - mean(pv[t-7:t-1])) / mean(pv[t-7:t-1])",
        rationale=(
            "Dogecoin attention spike detector. DOGE interest spikes "
            "often follow or precede Musk crypto engagement."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="-0.30 to +1.00",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="wiki_total_7d",
        category="attention",
        description=(
            "Sum of all entity 7-day pageview averages "
            "(Elon Musk + Tesla + Dogecoin)"
        ),
        formula="wiki_elon_musk_7d + wiki_tesla_7d + wiki_doge_7d",
        rationale=(
            "Aggregate attention proxy across all Musk-related entities."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="40000-250000 views/day",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="wiki_attention_concentration",
        category="attention",
        description=(
            "Herfindahl index of pageview attention across entities"
        ),
        formula="sum((entity_7d / wiki_total_7d)^2)",
        rationale=(
            "Concentrated attention (one entity dominates) vs spread "
            "attention. High concentration may signal a focused event."
        ),
        data_source="Wikimedia REST API (data/sources/wikipedia/pageviews.json)",
        coverage_pct=98.7,
        expected_range="0.33-1.0 (0.33=even across 3, 1.0=single entity)",
        existing=True,
        priority="low",
    ),
]


# ---------------------------------------------------------------------------
# VIX factors (5) — added to FINANCIAL category
# Derived from CBOE VIX index via yfinance (^VIX daily OHLCV)
# ---------------------------------------------------------------------------
VIX_FACTORS: List[Factor] = [
    Factor(
        name="vix_close",
        category="financial",
        description="VIX closing value on the day before event start",
        formula="vix_close[t-1]",
        rationale=(
            "Market fear gauge. High VIX (>25) indicates market stress "
            "that may trigger more Elon tweets about markets/economy."
        ),
        data_source="yfinance ^VIX (data/sources/market/vix_daily.parquet)",
        coverage_pct=98.7,
        expected_range="10-50",
        existing=True,
        priority="high",
    ),
    Factor(
        name="vix_pct_change_1d",
        category="financial",
        description="VIX 1-day percentage change",
        formula="(vix_close[t-1] - vix_close[t-2]) / vix_close[t-2]",
        rationale="Sudden VIX spikes signal market shocks that may trigger tweets.",
        data_source="yfinance ^VIX (data/sources/market/vix_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.20 to +0.30",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="vix_pct_change_5d",
        category="financial",
        description="VIX 5-day percentage change",
        formula="(vix_close[t-1] - vix_close[t-5]) / vix_close[t-5]",
        rationale="Medium-term fear trend. Sustained VIX rise = ongoing uncertainty.",
        data_source="yfinance ^VIX (data/sources/market/vix_daily.parquet)",
        coverage_pct=98.7,
        expected_range="-0.40 to +0.60",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="vix_level_category",
        category="financial",
        description="VIX level category: low (<15), medium (15-25), high (25-35), extreme (>35)",
        formula="categorize(vix_close[t-1])",
        rationale="Categorical regime indicator for market stress level.",
        data_source="yfinance ^VIX (data/sources/market/vix_daily.parquet)",
        coverage_pct=98.7,
        expected_range="low/medium/high/extreme (categorical)",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="vix_ma5_ratio",
        category="financial",
        description="Ratio of VIX close to its 5-day moving average",
        formula="vix_close[t-1] / ma5(vix_close)[t-1]",
        rationale=(
            "VIX above its MA5 (ratio >1.10) signals acute stress spike. "
            "Used as signal threshold by SignalEnhancedTailModelV4."
        ),
        data_source="yfinance ^VIX (data/sources/market/vix_daily.parquet)",
        coverage_pct=98.7,
        expected_range="0.80-1.40",
        existing=True,
        priority="high",
    ),
]


# ---------------------------------------------------------------------------
# CRYPTO FEAR & GREED factors (4) — added to FINANCIAL category
# Derived from alternative.me Crypto Fear & Greed Index API
# ---------------------------------------------------------------------------
CRYPTO_FG_FACTORS: List[Factor] = [
    Factor(
        name="crypto_fg_value",
        category="financial",
        description="Crypto Fear & Greed Index value (0-100) on day before event",
        formula="fg_value[t-1]",
        rationale=(
            "Crypto market sentiment. Extreme fear (<25) or greed (>75) "
            "often triggers Musk crypto tweets."
        ),
        data_source="alternative.me API (data/sources/market/crypto_fear_greed.parquet)",
        coverage_pct=98.7,
        expected_range="0-100",
        existing=True,
        priority="high",
    ),
    Factor(
        name="crypto_fg_7d_avg",
        category="financial",
        description="7-day rolling average of Crypto Fear & Greed Index",
        formula="mean(fg_value[t-7:t-1])",
        rationale="Smoothed crypto sentiment baseline.",
        data_source="alternative.me API (data/sources/market/crypto_fear_greed.parquet)",
        coverage_pct=98.7,
        expected_range="10-90",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="crypto_fg_delta",
        category="financial",
        description="Crypto F&G momentum: 3-day avg minus 7-day avg",
        formula="fg_3d_avg - fg_7d_avg",
        rationale="Rapid sentiment shift detector. Large positive = greed spike.",
        data_source="alternative.me API (data/sources/market/crypto_fear_greed.parquet)",
        coverage_pct=98.7,
        expected_range="-30 to +30",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="crypto_fg_category",
        category="financial",
        description="Crypto F&G category: extreme_fear/fear/neutral/greed/extreme_greed",
        formula="categorize(fg_value[t-1])",
        rationale="Categorical crypto sentiment regime.",
        data_source="alternative.me API (data/sources/market/crypto_fear_greed.parquet)",
        coverage_pct=98.7,
        expected_range="extreme_fear/fear/neutral/greed/extreme_greed (categorical)",
        existing=True,
        priority="low",
    ),
]


# ---------------------------------------------------------------------------
# TRENDS factors (10)
# Derived from Google Trends via pytrends
# ---------------------------------------------------------------------------
TRENDS_FACTORS: List[Factor] = [
    Factor(
        name="gt_elon_musk_7d",
        category="trends",
        description="7-day average Google Trends interest for 'Elon Musk'",
        formula="mean(trends['elon_musk'][t-7:t-1])",
        rationale=(
            "Google search interest as a public attention proxy. "
            "Complementary to Wikipedia pageviews — captures different audience."
        ),
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="0-100 (relative interest)",
        existing=True,
        priority="high",
    ),
    Factor(
        name="gt_elon_musk_delta",
        category="trends",
        description="Elon Musk Google Trends spike: (3d avg - 7d avg) / 7d avg",
        formula="(mean(gt[t-3:t-1]) - mean(gt[t-7:t-1])) / mean(gt[t-7:t-1])",
        rationale=(
            "Search interest spike detector. >15% = breaking story. "
            "Used as signal by SignalEnhancedTailModelV4."
        ),
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="-0.50 to +1.00",
        existing=True,
        priority="high",
    ),
    Factor(
        name="gt_tesla_7d",
        category="trends",
        description="7-day average Google Trends interest for 'Tesla'",
        formula="mean(trends['tesla'][t-7:t-1])",
        rationale="Tesla-specific search interest. Spikes around earnings, product launches.",
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="0-100 (relative interest)",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="gt_tesla_delta",
        category="trends",
        description="Tesla Google Trends spike: (3d avg - 7d avg) / 7d avg",
        formula="(mean(gt[t-3:t-1]) - mean(gt[t-7:t-1])) / mean(gt[t-7:t-1])",
        rationale="Tesla search interest spike detector.",
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="-0.50 to +1.00",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="gt_spacex_7d",
        category="trends",
        description="7-day average Google Trends interest for 'SpaceX'",
        formula="mean(trends['spacex'][t-7:t-1])",
        rationale="SpaceX-specific search interest. Complements launch calendar data.",
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="0-100 (relative interest)",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="gt_spacex_delta",
        category="trends",
        description="SpaceX Google Trends spike: (3d avg - 7d avg) / 7d avg",
        formula="(mean(gt[t-3:t-1]) - mean(gt[t-7:t-1])) / mean(gt[t-7:t-1])",
        rationale="SpaceX search interest spike detector.",
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="-0.50 to +1.00",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="gt_dogecoin_7d",
        category="trends",
        description="7-day average Google Trends interest for 'Dogecoin'",
        formula="mean(trends['dogecoin'][t-7:t-1])",
        rationale="Dogecoin search interest. Meme coin interest spikes = Musk tweet triggers.",
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="0-100 (relative interest)",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="gt_dogecoin_delta",
        category="trends",
        description="Dogecoin Google Trends spike: (3d avg - 7d avg) / 7d avg",
        formula="(mean(gt[t-3:t-1]) - mean(gt[t-7:t-1])) / mean(gt[t-7:t-1])",
        rationale="Dogecoin search interest spike detector.",
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="-0.50 to +1.00",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="gt_total_7d",
        category="trends",
        description="Sum of all entity 7-day Google Trends averages",
        formula="gt_elon_musk_7d + gt_tesla_7d + gt_spacex_7d + gt_dogecoin_7d",
        rationale="Aggregate search attention proxy across all Musk-related entities.",
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="0-400",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="gt_concentration",
        category="trends",
        description="Herfindahl index of Google Trends attention across entities",
        formula="sum((entity_7d / gt_total_7d)^2)",
        rationale=(
            "Concentrated search interest vs spread. High = one topic dominates."
        ),
        data_source="Google Trends via pytrends (data/sources/trends/google_trends.parquet)",
        coverage_pct=95.0,
        expected_range="0.25-1.0",
        existing=True,
        priority="low",
    ),
]


# ---------------------------------------------------------------------------
# GOVERNMENT factors (3)
# Derived from Federal Register + GovTrack (data/sources/government/events.parquet)
# ---------------------------------------------------------------------------
GOVERNMENT_FACTORS: List[Factor] = [
    Factor(
        name="govt_event_flag_7d",
        category="government",
        description="Binary: any government event in 7 days before event start",
        formula="1 if any govt event in [t-7, t), else 0",
        rationale="Government actions (exec orders, rules) may trigger Elon tweets.",
        data_source="Federal Register + GovTrack (data/sources/government/events.parquet)",
        coverage_pct=95.0,
        expected_range="0 or 1",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="govt_event_count_trailing_7d",
        category="government",
        description="Count of government events in 7 days before event start",
        formula="count(govt_events in [t-7, t))",
        rationale="Higher government activity density may increase tweet volume.",
        data_source="Federal Register + GovTrack (data/sources/government/events.parquet)",
        coverage_pct=95.0,
        expected_range="0-10",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="govt_exec_order_flag_7d",
        category="government",
        description="Binary: any executive order in 7 days before event start",
        formula="1 if any exec order in [t-7, t), else 0",
        rationale="Executive orders are high-salience events that may provoke Elon response.",
        data_source="Federal Register + GovTrack (data/sources/government/events.parquet)",
        coverage_pct=95.0,
        expected_range="0 or 1",
        existing=True,
        priority="medium",
    ),
]


# ---------------------------------------------------------------------------
# CORPORATE factors (3)
# Derived from yfinance + SEC EDGAR (data/sources/calendar/corporate_events.parquet)
# ---------------------------------------------------------------------------
CORPORATE_FACTORS: List[Factor] = [
    Factor(
        name="corporate_event_flag_7d",
        category="corporate",
        description="Binary: any corporate event in 7d around event window",
        formula="1 if any corporate event in [t-7, t_end+7], else 0",
        rationale="Corporate events (earnings, launches) may modulate tweet behavior.",
        data_source="yfinance + SEC EDGAR (data/sources/calendar/corporate_events.parquet)",
        coverage_pct=95.0,
        expected_range="0 or 1",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="corporate_event_count_7d",
        category="corporate",
        description="Count of corporate events in 7d around event window",
        formula="count(corporate_events in [t-7, t_end+7])",
        rationale="Multiple corporate events may amplify tweet activity.",
        data_source="yfinance + SEC EDGAR (data/sources/calendar/corporate_events.parquet)",
        coverage_pct=95.0,
        expected_range="0-5",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="tesla_earnings_flag_14d",
        category="corporate",
        description="Binary: Tesla earnings within 14d of event start",
        formula="1 if Tesla earnings in [t-14, t+14], else 0",
        rationale="Tesla earnings are high-attention events driving Elon activity.",
        data_source="yfinance + SEC EDGAR (data/sources/calendar/corporate_events.parquet)",
        coverage_pct=95.0,
        expected_range="0 or 1",
        existing=True,
        priority="high",
    ),
]


# ---------------------------------------------------------------------------
# REDDIT factors (7)
# Derived from Reddit API (data/sources/reddit/daily_activity.parquet)
# ---------------------------------------------------------------------------
REDDIT_FACTORS: List[Factor] = [
    Factor(
        name="reddit_total_posts_7d",
        category="reddit",
        description="Total posts across all subreddits (trailing 7d)",
        formula="sum(posts across r/elonmusk, r/teslamotors, r/SpaceX, r/dogecoin, r/technology) for [t-7, t)",
        rationale="Aggregate Reddit post volume as community attention proxy.",
        data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
        coverage_pct=95.0,
        expected_range="50-500",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="reddit_total_comments_7d",
        category="reddit",
        description="Total comments across all subreddits (trailing 7d)",
        formula="sum(comments across subreddits) for [t-7, t)",
        rationale="Comment volume as engagement intensity proxy.",
        data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
        coverage_pct=95.0,
        expected_range="500-10000",
        existing=True,
        priority="low",
    ),
    Factor(
        name="reddit_post_delta",
        category="reddit",
        description="Reddit post momentum: (3d avg - 7d avg) / 7d avg",
        formula="(mean(posts[t-3:t-1]) - mean(posts[t-7:t-1])) / mean(posts[t-7:t-1])",
        rationale="Spike in Reddit posting activity may foreshadow tweet activity.",
        data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
        coverage_pct=95.0,
        expected_range="-0.50 to +1.00",
        existing=True,
        priority="medium",
    ),
    Factor(
        name="reddit_elonmusk_posts_7d",
        category="reddit",
        description="Posts in r/elonmusk (trailing 7d avg)",
        formula="mean(r/elonmusk daily posts) for [t-7, t)",
        rationale="Direct Elon community activity level.",
        data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
        coverage_pct=95.0,
        expected_range="0-50",
        existing=True,
        priority="high",
    ),
    Factor(
        name="reddit_teslamotors_posts_7d",
        category="reddit",
        description="Posts in r/teslamotors (trailing 7d avg)",
        formula="mean(r/teslamotors daily posts) for [t-7, t)",
        rationale="Tesla community activity as attention proxy.",
        data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
        coverage_pct=95.0,
        expected_range="0-100",
        existing=True,
        priority="low",
    ),
    Factor(
        name="reddit_attention_concentration",
        category="reddit",
        description="Herfindahl index of post volume across subreddits",
        formula="sum((sub_posts / total_posts)^2)",
        rationale="Concentrated attention vs spread. High = one subreddit dominates.",
        data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
        coverage_pct=95.0,
        expected_range="0.20-1.0",
        existing=True,
        priority="low",
    ),
    Factor(
        name="reddit_top_score_7d",
        category="reddit",
        description="Avg top_post_score across subreddits (trailing 7d)",
        formula="mean(top_post_score across subreddits) for [t-7, t)",
        rationale="Viral post engagement as attention intensity proxy.",
        data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
        coverage_pct=95.0,
        expected_range="0-5000",
        existing=True,
        priority="low",
    ),
]


# ---------------------------------------------------------------------------
# COMPLETE REGISTRY
# ---------------------------------------------------------------------------
FACTOR_REGISTRY: List[Factor] = (
    TEMPORAL_FACTORS + MEDIA_FACTORS + CALENDAR_FACTORS
    + MARKET_FACTORS + CROSS_FACTORS + FINANCIAL_FACTORS
    + VIX_FACTORS + CRYPTO_FG_FACTORS + ATTENTION_FACTORS + TRENDS_FACTORS
    + GOVERNMENT_FACTORS + CORPORATE_FACTORS + REDDIT_FACTORS
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
    for category in ["temporal", "media", "calendar", "market", "cross",
                      "financial", "attention", "trends", "government",
                      "corporate", "reddit"]:
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
    for cat in ["temporal", "media", "calendar", "market", "cross",
             "financial", "attention", "trends"]:
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
    valid_cats = {"temporal", "media", "calendar", "market", "cross",
                   "financial", "attention", "trends", "government",
                   "corporate", "reddit"}
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
