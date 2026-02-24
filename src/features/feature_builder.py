"""
TweetFeatureBuilder -- declarative, lazy-loaded feature computation.
====================================================================

Pattern from EsportsBetting: declarative feature groups, lazy extractors,
point-in-time (PIT) correctness. Models specify which feature group they
need; the builder only computes those features.

Usage:
    from src.features.feature_builder import TweetFeatureBuilder

    builder = TweetFeatureBuilder(feature_group="full")
    features = builder.build_features(event_slug, event_metadata)
    all_features = builder.build_features_batch(events_list)

Feature groups:
    base            - Core temporal features only (11 factors)
    full            - All 36 factors across all categories
    market_adjusted - Features used by MarketAdjustedModel
    temporal_only   - Temporal + calendar (no market/media dependency)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.extractors import (
    compute_temporal_features,
    compute_media_features,
    compute_calendar_features,
    compute_government_features,
    compute_corporate_features,
    compute_market_features,
    compute_cross_features,
    compute_financial_features,
    compute_attention_features,
    compute_trends_features,
    compute_reddit_features,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DAILY_METRICS_PATH = (
    PROJECT_DIR / "data" / "sources" / "xtracker" / "daily_metrics_full.json"
)
UNIFIED_COUNTS_PATH = (
    PROJECT_DIR / "data" / "processed" / "daily_counts.json"
)
GDELT_DIR = PROJECT_DIR / "data" / "sources" / "gdelt"
SPACEX_PATH = (
    PROJECT_DIR / "data" / "sources" / "calendar" / "spacex_launches_historical.json"
)
CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
PRICE_HISTORY_PATH = (
    PROJECT_DIR / "data" / "sources" / "polymarket" / "prices" / "price_history.parquet"
)
TESLA_PATH = PROJECT_DIR / "data" / "sources" / "market" / "tesla_daily.parquet"
CRYPTO_PATH = PROJECT_DIR / "data" / "sources" / "market" / "crypto_daily.parquet"
WIKI_PATH = PROJECT_DIR / "data" / "sources" / "wikipedia" / "pageviews.json"
VIX_PATH = PROJECT_DIR / "data" / "sources" / "market" / "vix_daily.parquet"
TRENDS_PATH = PROJECT_DIR / "data" / "sources" / "trends" / "google_trends.parquet"
CRYPTO_FG_PATH = PROJECT_DIR / "data" / "sources" / "market" / "crypto_fear_greed.parquet"
GOVT_EVENTS_PATH = PROJECT_DIR / "data" / "sources" / "government" / "events.parquet"
CORPORATE_EVENTS_PATH = (
    PROJECT_DIR / "data" / "sources" / "calendar" / "corporate_events.parquet"
)
REDDIT_PATH = PROJECT_DIR / "data" / "sources" / "reddit" / "daily_activity.parquet"

# GDELT entity keys (filename stems)
GDELT_ENTITIES = ["elon_musk", "tesla", "spacex", "neuralink"]


# ---------------------------------------------------------------------------
# Feature group registry
# ---------------------------------------------------------------------------

# All factor names, organized by category
TEMPORAL_FEATURES = [
    "yesterday_count", "rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d",
    "rolling_std_7d", "rolling_std_14d", "trend_7d", "trend_14d",
    "day_of_week", "cv_14d", "regime_ratio",
    # NEW
    "weekend_ratio_7d",
    "day_of_week_sin", "day_of_week_cos",
]

MEDIA_FEATURES = [
    "elon_musk_vol_1d", "elon_musk_vol_3d", "elon_musk_vol_7d", "elon_musk_vol_delta",
    "tesla_vol_1d", "tesla_vol_3d", "tesla_vol_7d", "tesla_vol_delta",
    "spacex_vol_1d", "spacex_vol_3d", "spacex_vol_7d", "spacex_vol_delta",
    "neuralink_vol_1d", "neuralink_vol_3d", "neuralink_vol_7d", "neuralink_vol_delta",
    "elon_musk_tone_1d", "elon_musk_tone_7d",
    # NEW
    "elon_musk_tone_delta", "total_media_vol_7d", "media_vol_concentration",
    "gdelt_entity_divergence",
]

CALENDAR_FEATURES = [
    "days_since_last_launch", "days_to_next_launch",
    "launches_trailing_7d", "launches_during_event",
    # NEW
    "is_holiday_week", "event_duration_days",
]

GOVERNMENT_FEATURES = [
    "govt_event_flag_7d", "govt_event_count_trailing_7d",
    "govt_exec_order_flag_7d",
]

CORPORATE_FEATURES = [
    "corporate_event_flag_7d", "corporate_event_count_7d",
    "tesla_earnings_flag_14d",
]

MARKET_FEATURES = [
    "crowd_implied_ev", "crowd_std_dev", "distribution_entropy",
    "n_buckets_with_price", "price_shift_24h",
    # NEW
    "crowd_vs_rolling_avg", "crowd_skewness", "crowd_kurtosis",
    "market_overround", "hours_until_resolution",
    "bucket_relative_mispricings", "bucket_position_normalized",
]

CROSS_FEATURES = [
    "bad_press_x_low_activity", "launch_busy_x_trend_down",
    "regime_transition_flag", "high_vol_x_high_entropy",
    "momentum_reversal_signal",
    # NEW
    "cross_market_daily_weekly_div",
]

FINANCIAL_FEATURES = [
    "tsla_pct_change_1d", "tsla_pct_change_5d", "tsla_volatility_5d",
    "tsla_volume_ratio", "tsla_drawdown_5d", "tsla_gap_1d",
    "doge_pct_change_1d", "doge_pct_change_5d", "doge_volatility_5d",
    "btc_pct_change_1d", "btc_pct_change_5d", "btc_volatility_5d",
    # VIX
    "vix_close", "vix_pct_change_1d", "vix_pct_change_5d",
    "vix_level_category", "vix_ma5_ratio",
    # Crypto Fear & Greed
    "crypto_fg_value", "crypto_fg_7d_avg", "crypto_fg_delta", "crypto_fg_category",
]

ATTENTION_FEATURES = [
    "wiki_elon_musk_7d", "wiki_elon_musk_delta",
    "wiki_tesla_7d", "wiki_tesla_delta",
    "wiki_doge_7d", "wiki_doge_delta",
    "wiki_total_7d", "wiki_attention_concentration",
    # NEW
    "wikipedia_entity_divergence",
]

TRENDS_FEATURES = [
    "gt_elon_musk_7d", "gt_elon_musk_delta",
    "gt_tesla_7d", "gt_tesla_delta",
    "gt_spacex_7d", "gt_spacex_delta",
    "gt_dogecoin_7d", "gt_dogecoin_delta",
    "gt_total_7d", "gt_concentration",
]

REDDIT_FEATURES = [
    "reddit_total_posts_7d", "reddit_total_comments_7d",
    "reddit_post_delta", "reddit_elonmusk_posts_7d",
    "reddit_teslamotors_posts_7d", "reddit_attention_concentration",
    "reddit_top_score_7d",
]

ALL_FEATURES = (
    TEMPORAL_FEATURES + MEDIA_FEATURES + CALENDAR_FEATURES
    + GOVERNMENT_FEATURES + CORPORATE_FEATURES
    + MARKET_FEATURES + CROSS_FEATURES
    + FINANCIAL_FEATURES + ATTENTION_FEATURES + TRENDS_FEATURES
    + REDDIT_FEATURES
)


FEATURE_GROUPS: Dict[str, Dict[str, Any]] = {
    "base": {
        "features": [
            "rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d",
            "rolling_std_7d", "rolling_std_14d", "trend_7d",
            "day_of_week", "cv_14d", "regime_ratio",
            "yesterday_count", "weekend_ratio_7d",
            "day_of_week_sin", "day_of_week_cos",
        ],
        "categories": {"temporal"},
        "description": "Core temporal features only (14 factors)",
    },
    "full": {
        "features": ALL_FEATURES,
        "categories": {"temporal", "media", "calendar", "government", "corporate", "market", "cross", "financial", "attention", "trends", "reddit"},
        "description": "All available features across all categories",
    },
    "market_adjusted": {
        "features": [
            # Temporal
            "rolling_avg_7d", "rolling_std_7d", "trend_7d",
            "cv_14d", "regime_ratio", "weekend_ratio_7d",
            # Media
            "elon_musk_vol_7d", "elon_musk_vol_delta",
            "elon_musk_tone_7d", "elon_musk_tone_delta",
            "total_media_vol_7d", "gdelt_entity_divergence",
            # Calendar
            "launches_trailing_7d", "launches_during_event",
            "is_holiday_week", "event_duration_days",
            # Market
            "crowd_implied_ev", "crowd_std_dev", "distribution_entropy",
            "crowd_vs_rolling_avg", "crowd_skewness",
            "market_overround", "hours_until_resolution",
            # Cross
            "bad_press_x_low_activity", "regime_transition_flag",
            "momentum_reversal_signal", "cross_market_daily_weekly_div",
            # Financial (new)
            "tsla_pct_change_1d", "tsla_volatility_5d", "tsla_drawdown_5d",
            "doge_pct_change_1d", "doge_volatility_5d",
            # Temporal (cyclical)
            "day_of_week_sin", "day_of_week_cos",
            # Attention (new)
            "wiki_elon_musk_7d", "wiki_elon_musk_delta",
            "wikipedia_entity_divergence",
            # VIX
            "vix_close", "vix_ma5_ratio",
            # Crypto Fear & Greed
            "crypto_fg_value", "crypto_fg_delta",
            # Trends
            "gt_elon_musk_7d", "gt_elon_musk_delta",
            # Government
            "govt_event_flag_7d", "govt_exec_order_flag_7d",
            # Corporate
            "corporate_event_flag_7d", "tesla_earnings_flag_14d",
            # Reddit
            "reddit_total_posts_7d", "reddit_post_delta",
            "reddit_elonmusk_posts_7d",
        ],
        "categories": {"temporal", "media", "calendar", "government", "corporate", "market", "cross", "financial", "attention", "trends", "reddit"},
        "description": "Features for market-adjusted predictions",
    },
    "temporal_only": {
        "features": TEMPORAL_FEATURES + CALENDAR_FEATURES,
        "categories": {"temporal", "calendar"},
        "description": "Temporal + calendar features, no market/media dependency (18 factors)",
    },
}


# ---------------------------------------------------------------------------
# Data loaders (lazy, called once per builder lifetime)
# ---------------------------------------------------------------------------
def _load_xtracker_daily() -> dict:
    """Load daily tweet counts by merging unified daily_counts.json + XTracker daily_metrics.

    Loads both sources and merges them so the most recent data is always available,
    even if daily_counts.json (rebuilt manually) is stale. XTracker entries override
    unified entries for the same date when XTracker has a non-zero count.

    Returns dict: date_str -> {count, cumulative, tracking_id}
    """
    by_date = {}

    # Layer 1: Unified file (Kaggle + XTracker merged, may be stale)
    if UNIFIED_COUNTS_PATH.exists():
        with open(UNIFIED_COUNTS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for date_str, entry in raw.get("daily_counts", {}).items():
            count = entry["count"] if isinstance(entry, dict) else entry
            by_date[date_str] = {
                "count": count,
                "cumulative": 0,
                "tracking_id": "",
            }

    # Layer 2: XTracker daily metrics (refreshed by cron, overlay newer data)
    if DAILY_METRICS_PATH.exists():
        with open(DAILY_METRICS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for record in raw.get("data", []):
            date_str = record["date"][:10]
            count = record["data"]["count"]
            # XTracker overrides unified when it has a non-zero count,
            # or when the date doesn't exist in unified at all
            if count > 0 or date_str not in by_date:
                by_date[date_str] = {
                    "count": count,
                    "cumulative": record["data"].get("cumulative", 0),
                    "tracking_id": record["data"].get("trackingId", ""),
                }

    return by_date


def _load_gdelt_data() -> dict:
    """Load GDELT volume and tone timelines for all entities.

    Returns dict: {
        "elon_musk_vol": {date_str: value, ...},
        "elon_musk_tone": {date_str: value, ...},
        ...
    }
    """
    gdelt = {}

    for entity in GDELT_ENTITIES:
        for mode, suffix in [("timelinevol", "vol"), ("timelinetone", "tone")]:
            key = "{}_{}".format(entity, suffix)
            fpath = GDELT_DIR / "gdelt_{}_{}.json".format(entity, mode)
            if not fpath.exists():
                gdelt[key] = {}
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, IOError):
                gdelt[key] = {}
                continue

            date_values = {}
            for chunk in raw.get("chunks", []):
                response = chunk.get("response", {})
                for timeline in response.get("timeline", []):
                    for point in timeline.get("data", []):
                        raw_date = point.get("date", "")
                        if len(raw_date) >= 8:
                            date_str = "{}-{}-{}".format(
                                raw_date[:4], raw_date[4:6], raw_date[6:8]
                            )
                            date_values[date_str] = float(point.get("value", 0))
            gdelt[key] = date_values

    return gdelt


def _load_spacex_launches() -> List[str]:
    """Load SpaceX launch dates. Returns sorted list of date strings YYYY-MM-DD."""
    if not SPACEX_PATH.exists():
        return []

    with open(SPACEX_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dates = []
    for launch in raw.get("launches", []):
        net = launch.get("net", "")
        if net and len(net) >= 10:
            dates.append(net[:10])

    return sorted(set(dates))


def _load_tesla_data() -> pd.DataFrame:
    """Load Tesla daily OHLCV data from parquet."""
    if not TESLA_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(TESLA_PATH)


def _load_crypto_data() -> pd.DataFrame:
    """Load crypto (DOGE, BTC) daily data from parquet."""
    if not CRYPTO_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(CRYPTO_PATH)


def _load_wiki_data() -> dict:
    """Load Wikipedia pageview data from JSON.

    Returns dict: {entity_key: {article, daily_views: {date: count}, ...}}
    """
    if not WIKI_PATH.exists():
        return {}
    try:
        with open(WIKI_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("articles", {})
    except (json.JSONDecodeError, IOError):
        return {}


def _load_vix_data() -> pd.DataFrame:
    """Load VIX daily OHLCV data from parquet."""
    if not VIX_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(VIX_PATH)


def _load_trends_data() -> pd.DataFrame:
    """Load Google Trends data from parquet."""
    if not TRENDS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(TRENDS_PATH)


def _load_crypto_fg_data() -> pd.DataFrame:
    """Load Crypto Fear & Greed data from parquet."""
    if not CRYPTO_FG_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(CRYPTO_FG_PATH)


def _load_govt_events() -> pd.DataFrame:
    """Load government events from parquet.

    Schema: date, event_type, source, title, description
    Ensures date column is string (YYYY-MM-DD) for consistent comparisons.
    """
    if not GOVT_EVENTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(GOVT_EVENTS_PATH)
    if not df.empty and "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]
    return df


def _load_corporate_events() -> pd.DataFrame:
    """Load corporate events from parquet.

    Schema: date, company, event_type, title, description, expected_tweet_impact
    Ensures date column is string (YYYY-MM-DD) for consistent comparisons.
    """
    if not CORPORATE_EVENTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(CORPORATE_EVENTS_PATH)
    if not df.empty and "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]
    return df


def _load_reddit_data() -> pd.DataFrame:
    """Load Reddit daily activity data from parquet."""
    if not REDDIT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(REDDIT_PATH)


def _load_catalog() -> pd.DataFrame:
    """Load market_catalog.parquet."""
    if not CATALOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(CATALOG_PATH)


def _load_prices() -> pd.DataFrame:
    """Load price_history.parquet."""
    if not PRICE_HISTORY_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(PRICE_HISTORY_PATH)


# ---------------------------------------------------------------------------
# TweetFeatureBuilder
# ---------------------------------------------------------------------------
class TweetFeatureBuilder:
    """Computes features for tweet count prediction markets.

    Pattern from EsportsBetting: declarative feature groups, lazy extractors.

    Args:
        feature_group: Name of the feature group to compute. One of:
            "base", "full", "market_adjusted", "temporal_only".
            Defaults to "full".

    Usage:
        builder = TweetFeatureBuilder(feature_group="full")

        # Single event
        features = builder.build_features("event-slug", {
            "event_id": "12345",
            "start_date": "2025-11-15",
            "end_date": "2025-11-22",
        })

        # Batch
        all_features = builder.build_features_batch([
            {"event_slug": "...", "event_id": "...", "start_date": "...", ...},
            ...
        ])
    """

    # Class-level reference to all known feature groups
    FEATURE_GROUPS = FEATURE_GROUPS

    def __init__(self, feature_group: str = "full"):
        if feature_group not in FEATURE_GROUPS:
            raise ValueError(
                "Unknown feature_group '{}'. Choose from: {}".format(
                    feature_group, list(FEATURE_GROUPS.keys())
                )
            )

        self.feature_group = feature_group
        self._group_config = FEATURE_GROUPS[feature_group]
        self._required_features = set(self._group_config["features"])
        self._required_categories = self._group_config["categories"]

        # Lazy-loaded data sources (loaded on first access)
        self._xtracker_data: Optional[dict] = None
        self._gdelt_data: Optional[dict] = None
        self._spacex_launches: Optional[List[str]] = None
        self._catalog_df: Optional[pd.DataFrame] = None
        self._prices_df: Optional[pd.DataFrame] = None
        self._tesla_data: Optional[pd.DataFrame] = None
        self._crypto_data: Optional[pd.DataFrame] = None
        self._wiki_data: Optional[dict] = None
        self._vix_data: Optional[pd.DataFrame] = None
        self._trends_data: Optional[pd.DataFrame] = None
        self._crypto_fg_data: Optional[pd.DataFrame] = None
        self._govt_events: Optional[pd.DataFrame] = None
        self._corporate_events: Optional[pd.DataFrame] = None
        self._reddit_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Lazy data source accessors
    # ------------------------------------------------------------------
    @property
    def xtracker_data(self) -> dict:
        if self._xtracker_data is None:
            self._xtracker_data = _load_xtracker_daily()
        return self._xtracker_data

    @property
    def gdelt_data(self) -> dict:
        if self._gdelt_data is None:
            self._gdelt_data = _load_gdelt_data()
        return self._gdelt_data

    @property
    def spacex_launches(self) -> List[str]:
        if self._spacex_launches is None:
            self._spacex_launches = _load_spacex_launches()
        return self._spacex_launches

    @property
    def catalog_df(self) -> pd.DataFrame:
        if self._catalog_df is None:
            self._catalog_df = _load_catalog()
        return self._catalog_df

    @property
    def prices_df(self) -> pd.DataFrame:
        if self._prices_df is None:
            self._prices_df = _load_prices()
        return self._prices_df

    @property
    def tesla_data(self) -> pd.DataFrame:
        if self._tesla_data is None:
            self._tesla_data = _load_tesla_data()
        return self._tesla_data

    @property
    def crypto_data(self) -> pd.DataFrame:
        if self._crypto_data is None:
            self._crypto_data = _load_crypto_data()
        return self._crypto_data

    @property
    def wiki_data(self) -> dict:
        if self._wiki_data is None:
            self._wiki_data = _load_wiki_data()
        return self._wiki_data

    @property
    def vix_data(self) -> pd.DataFrame:
        if self._vix_data is None:
            self._vix_data = _load_vix_data()
        return self._vix_data

    @property
    def trends_data(self) -> pd.DataFrame:
        if self._trends_data is None:
            self._trends_data = _load_trends_data()
        return self._trends_data

    @property
    def crypto_fg_data(self) -> pd.DataFrame:
        if self._crypto_fg_data is None:
            self._crypto_fg_data = _load_crypto_fg_data()
        return self._crypto_fg_data

    @property
    def govt_events(self) -> pd.DataFrame:
        if self._govt_events is None:
            self._govt_events = _load_govt_events()
        return self._govt_events

    @property
    def corporate_events(self) -> pd.DataFrame:
        if self._corporate_events is None:
            self._corporate_events = _load_corporate_events()
        return self._corporate_events

    @property
    def reddit_data(self) -> pd.DataFrame:
        if self._reddit_data is None:
            self._reddit_data = _load_reddit_data()
        return self._reddit_data

    # ------------------------------------------------------------------
    # Category need checks
    # ------------------------------------------------------------------
    def _needs(self, category: str) -> bool:
        """Check if this feature group needs features from the given category."""
        return category in self._required_categories

    # ------------------------------------------------------------------
    # Core build method
    # ------------------------------------------------------------------
    def build_features(
        self,
        event_slug: str,
        event_metadata: dict,
    ) -> dict:
        """Compute all required features for a single event.

        Args:
            event_slug: Event slug string (for logging/identification).
            event_metadata: Dict with at least:
                - event_id: str
                - start_date: str (YYYY-MM-DD)
                - end_date: str (YYYY-MM-DD)

        Returns:
            dict with top-level keys matching the categories computed:
                {temporal: {}, media: {}, calendar: {}, market: {}, cross: {}}
            Keys for unused categories will be empty dicts.
        """
        event_id = str(event_metadata.get("event_id", ""))
        start_date = event_metadata.get("start_date")
        end_date = event_metadata.get("end_date")

        # Ensure date strings are clean
        if start_date and len(str(start_date)) >= 10:
            start_date = str(start_date)[:10]
        else:
            start_date = None

        if end_date and len(str(end_date)) >= 10:
            end_date = str(end_date)[:10]
        else:
            end_date = None

        result = {}

        # --- 1. Temporal features ---
        if self._needs("temporal"):
            temporal = compute_temporal_features(
                self.xtracker_data, start_date, end_date
            )
        else:
            temporal = {}
        result["temporal"] = temporal

        # --- 2. Media features ---
        if self._needs("media"):
            media = compute_media_features(self.gdelt_data, start_date)
        else:
            media = {}
        result["media"] = media

        # --- 3. Calendar features ---
        if self._needs("calendar"):
            calendar = compute_calendar_features(
                self.spacex_launches, start_date, end_date
            )
        else:
            calendar = {}
        result["calendar"] = calendar

        # --- 3b. Government features ---
        if self._needs("government"):
            government = compute_government_features(
                self.govt_events, start_date
            )
        else:
            government = {}
        result["government"] = government

        # --- 3c. Corporate features ---
        if self._needs("corporate"):
            corporate = compute_corporate_features(
                self.corporate_events, start_date, end_date
            )
        else:
            corporate = {}
        result["corporate"] = corporate

        # --- 4. Market features (depends on temporal + calendar for cross-refs) ---
        if self._needs("market"):
            market = compute_market_features(
                self.prices_df,
                self.catalog_df,
                event_id,
                end_date,
                temporal_features=temporal,
                calendar_features=calendar,
                event_start_date=start_date,
            )
        else:
            market = {}
        result["market"] = market

        # --- 5. Cross features (computed AFTER base features) ---
        if self._needs("cross"):
            cross = compute_cross_features(temporal, media, calendar, market)
        else:
            cross = {}
        result["cross"] = cross

        # --- 6. Financial features (Tesla stock + crypto + VIX + crypto F&G) ---
        if self._needs("financial"):
            financial = compute_financial_features(
                self.tesla_data, self.crypto_data, start_date,
                vix_data=self.vix_data,
                crypto_fg_data=self.crypto_fg_data,
            )
        else:
            financial = {}
        result["financial"] = financial

        # --- 7. Attention features (Wikipedia pageviews) ---
        if self._needs("attention"):
            attention = compute_attention_features(
                self.wiki_data, start_date
            )
        else:
            attention = {}
        result["attention"] = attention

        # --- 8. Trends features (Google Trends) ---
        if self._needs("trends"):
            trends = compute_trends_features(
                self.trends_data, start_date
            )
        else:
            trends = {}
        result["trends"] = trends

        # --- 9. Reddit features ---
        if self._needs("reddit"):
            reddit = compute_reddit_features(
                self.reddit_data, start_date
            )
        else:
            reddit = {}
        result["reddit"] = reddit

        return result

    # ------------------------------------------------------------------
    # Batch build
    # ------------------------------------------------------------------
    def build_features_batch(
        self,
        events: List[dict],
    ) -> Dict[str, dict]:
        """Compute features for multiple events.

        Data sources are loaded once (lazily on first event) and reused
        across all events.

        Args:
            events: List of dicts, each with:
                - event_slug: str
                - event_id: str
                - start_date: str
                - end_date: str

        Returns:
            dict mapping event_slug -> features_dict
        """
        results = {}
        for event in events:
            slug = str(event.get("event_slug", "unknown"))
            features = self.build_features(slug, event)
            results[slug] = features
        return results

    # ------------------------------------------------------------------
    # Utility: flatten features to a single dict
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_features(features: dict) -> dict:
        """Flatten nested {category: {feature: value}} into a single dict.

        Useful for DataFrame construction.
        """
        flat = {}
        for category, feats in features.items():
            if isinstance(feats, dict):
                for k, v in feats.items():
                    flat[k] = v
        return flat

    # ------------------------------------------------------------------
    # Info / debug
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Return a human-readable description of this builder."""
        lines = [
            "TweetFeatureBuilder(group='{}')".format(self.feature_group),
            "  Description: {}".format(self._group_config["description"]),
            "  Categories:  {}".format(sorted(self._required_categories)),
            "  # Features:  {}".format(len(self._required_features)),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "TweetFeatureBuilder(feature_group='{}', n_features={})".format(
            self.feature_group, len(self._required_features)
        )
