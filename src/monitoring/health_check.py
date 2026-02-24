"""Pre-signal health checks for the paper trading pipeline.

Validates feature completeness, data freshness, and model readiness
BEFORE signals are generated. Designed to catch the class of failures
discovered in Feb 2026: stale temporal features, OOD XGBoost inputs,
and missing critical data.

Usage:
    from src.monitoring.health_check import SignalHealthCheck

    hc = SignalHealthCheck()

    # Once at startup
    freshness = hc.check_data_freshness()

    # Per event, after features are built
    completeness = hc.check_feature_completeness(features)
    distribution = hc.check_feature_distribution(features, model_id="xgb_residual_v1")
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Critical vs optional features
# ---------------------------------------------------------------------------
# Critical: signal generation is unreliable without these.
# At least one of rolling_avg_7d or rolling_avg_14d must be present.
CRITICAL_FEATURES = {
    ("temporal", "cv_14d"),
    ("calendar", "event_duration_days"),
}
CRITICAL_ROLLING_AVG = [
    ("temporal", "rolling_avg_7d"),
    ("temporal", "rolling_avg_14d"),
]

OPTIONAL_FEATURES = [
    ("temporal", "trend_7d"),
    ("temporal", "regime_ratio"),
    ("temporal", "yesterday_count"),
    ("financial", "tsla_pct_change_1d"),
    ("financial", "doge_pct_change_1d"),
    ("financial", "vix_close"),
    ("media", "elon_musk_vol_7d"),
    ("media", "elon_musk_tone_7d"),
    ("attention", "wiki_elon_musk_7d"),
]

# ---------------------------------------------------------------------------
# Data source freshness config
# ---------------------------------------------------------------------------
# (label, path relative to PROJECT_DIR, max_stale_days)
DATA_FRESHNESS_SOURCES = [
    ("xtracker", "data/sources/xtracker/daily_metrics_full.json", 2),
    ("daily_counts", "data/processed/daily_counts.json", 7),
    ("market_catalog", "data/processed/market_catalog.parquet", 2),
    ("tesla_daily", "data/sources/market/tesla_daily.parquet", 3),
    ("crypto_daily", "data/sources/market/crypto_daily.parquet", 3),
    ("vix_daily", "data/sources/market/vix_daily.parquet", 3),
    ("crypto_fear_greed", "data/sources/market/crypto_fear_greed.parquet", 3),
    ("wikipedia", "data/sources/wikipedia/pageviews.json", 7),
    ("google_trends", "data/sources/trends/google_trends.parquet", 14),
]


# ---------------------------------------------------------------------------
# SignalHealthCheck
# ---------------------------------------------------------------------------
class SignalHealthCheck:
    """Validates feature completeness and model readiness before signals are generated."""

    def check_feature_completeness(self, features: dict) -> dict:
        """Check that critical features are not null/NaN.

        Args:
            features: Nested dict {category: {feature_name: value, ...}, ...}
                      as returned by TweetFeatureBuilder.build_features().

        Returns:
            {
                "passed": bool,
                "missing_critical": ["temporal.rolling_avg_7d", ...],
                "missing_optional": ["financial.vix_close", ...],
                "completeness_pct": 0.85,
            }
        """
        missing_critical = []
        missing_optional = []

        # Check critical features
        for category, name in CRITICAL_FEATURES:
            val = features.get(category, {}).get(name)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                missing_critical.append("{}.{}".format(category, name))

        # Check rolling avg (at least one must be present)
        has_rolling = False
        for category, name in CRITICAL_ROLLING_AVG:
            val = features.get(category, {}).get(name)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                has_rolling = True
                break
        if not has_rolling:
            missing_critical.append("temporal.rolling_avg_7d|rolling_avg_14d")

        # Check optional features
        for category, name in OPTIONAL_FEATURES:
            val = features.get(category, {}).get(name)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                missing_optional.append("{}.{}".format(category, name))

        # Compute overall completeness across all features
        total = 0
        non_null = 0
        for category, feats in features.items():
            if not isinstance(feats, dict):
                continue
            for name, val in feats.items():
                total += 1
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    non_null += 1

        completeness_pct = non_null / total if total > 0 else 0.0
        passed = len(missing_critical) == 0

        return {
            "passed": passed,
            "missing_critical": missing_critical,
            "missing_optional": missing_optional,
            "completeness_pct": round(completeness_pct, 4),
        }

    def check_feature_distribution(self, features: dict, model_id: str) -> dict:
        """Check if live features are within training distribution bounds.

        For XGBoost models, loads feature_config.json from models/{model_id}/
        and checks for features >3 SD from training mean. Currently a stub
        that checks config existence; full stats require training_stats.json
        to be generated during training.

        Args:
            features: Nested feature dict from TweetFeatureBuilder.
            model_id: Model identifier (e.g. "xgb_residual_v1").

        Returns:
            {
                "passed": bool,
                "ood_features": [{"name": ..., "live_value": ..., ...}],
                "note": str,
            }
        """
        config_path = PROJECT_DIR / "models" / model_id / "feature_config.json"
        stats_path = PROJECT_DIR / "models" / model_id / "training_stats.json"

        if not config_path.exists():
            return {
                "passed": True,
                "ood_features": [],
                "note": "No feature_config.json found for {}; skipping distribution check.".format(model_id),
            }

        # Load feature config to know which features the model uses
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        feature_columns = config.get("feature_columns", [])

        # If training stats exist, do real OOD check
        if not stats_path.exists():
            return {
                "passed": True,
                "ood_features": [],
                "note": "No training_stats.json for {}; distribution check unavailable. "
                        "Generate with: train_xgb_model.py --save-stats".format(model_id),
            }

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # Flatten live features for lookup
        flat = {}
        for category, feats in features.items():
            if isinstance(feats, dict):
                flat.update(feats)

        ood_features = []
        for col in feature_columns:
            if col not in stats:
                continue
            live_val = flat.get(col)
            if live_val is None or (isinstance(live_val, float) and math.isnan(live_val)):
                continue

            train_mean = stats[col].get("mean", 0.0)
            train_std = stats[col].get("std", 1.0)
            if train_std <= 0:
                continue

            z = abs(float(live_val) - train_mean) / train_std
            if z > 3.0:
                ood_features.append({
                    "name": col,
                    "live_value": round(float(live_val), 4),
                    "train_mean": round(train_mean, 4),
                    "train_std": round(train_std, 4),
                    "z_score": round(z, 2),
                })

        passed = len(ood_features) == 0
        return {
            "passed": passed,
            "ood_features": ood_features,
            "note": "",
        }

    def check_regime_alignment(self, features: dict) -> dict:
        """Check if temporal features diverge from crowd expectations.

        Detects regime shifts by comparing rolling-average-based expected
        tweet counts against crowd-implied EV. Also flags sparse XTracker
        data coverage.

        Args:
            features: Nested feature dict from TweetFeatureBuilder.

        Returns:
            {
                "passed": bool,
                "warnings": ["warning message", ...],
                "regime_ratio": float or None,
                "data_coverage_7d": float or None,
            }
        """
        warnings = []
        temporal = features.get("temporal", {})
        market = features.get("market", {})
        calendar = features.get("calendar", {})

        # Use calendar-day average if available, else standard rolling avg
        rolling_avg = temporal.get("rolling_avg_7d_calendar")
        if rolling_avg is None:
            rolling_avg = temporal.get("rolling_avg_7d")
        if rolling_avg is None:
            rolling_avg = temporal.get("rolling_avg_14d")

        crowd_ev = market.get("crowd_implied_ev")
        duration = calendar.get("event_duration_days")
        data_coverage = temporal.get("data_coverage_7d")

        regime_ratio = None

        # Check temporal-vs-crowd divergence
        if rolling_avg is not None and duration is not None and crowd_ev is not None:
            if crowd_ev > 0 and duration > 0:
                expected_total = rolling_avg * duration
                regime_ratio = expected_total / crowd_ev
                if regime_ratio > 1.8:
                    warnings.append(
                        "REGIME DIVERGENCE: temporal features predict {:.0f} tweets "
                        "but crowd expects {:.0f} (ratio {:.2f}x — temporal may be "
                        "inflated by sparse data)".format(
                            expected_total, crowd_ev, regime_ratio
                        )
                    )
                elif regime_ratio < 0.55:
                    warnings.append(
                        "REGIME DIVERGENCE: temporal features predict {:.0f} tweets "
                        "but crowd expects {:.0f} (ratio {:.2f}x — temporal may be "
                        "stale or crowd is overestimating)".format(
                            expected_total, crowd_ev, regime_ratio
                        )
                    )

        # Check data coverage
        if data_coverage is not None and data_coverage < 0.5:
            warnings.append(
                "SPARSE DATA: only {:.0f}% XTracker coverage in 7-day window — "
                "rolling averages may be unreliable".format(data_coverage * 100)
            )

        passed = len(warnings) == 0
        return {
            "passed": passed,
            "warnings": warnings,
            "regime_ratio": round(regime_ratio, 4) if regime_ratio is not None else None,
            "data_coverage_7d": data_coverage,
        }

    def check_data_freshness(self) -> dict:
        """Check that data files are fresh enough for live trading.

        Inspects file modification times for key data sources.
        Flags anything older than its configured threshold.

        Returns:
            {
                "passed": bool,
                "stale_sources": [
                    {"source": "xtracker", "last_modified": "2026-02-10", "days_stale": 13},
                ],
                "missing_sources": ["source_name", ...],
            }
        """
        now = datetime.now(timezone.utc)
        stale = []
        missing = []

        for label, rel_path, max_days in DATA_FRESHNESS_SOURCES:
            full_path = PROJECT_DIR / rel_path
            if not full_path.exists():
                missing.append(label)
                continue

            mtime = datetime.fromtimestamp(full_path.stat().st_mtime, tz=timezone.utc)
            age_days = (now - mtime).total_seconds() / 86400.0

            if age_days > max_days:
                stale.append({
                    "source": label,
                    "last_modified": mtime.strftime("%Y-%m-%d %H:%M"),
                    "days_stale": round(age_days, 1),
                    "threshold_days": max_days,
                })

        passed = len(stale) == 0 and len(missing) == 0
        return {
            "passed": passed,
            "stale_sources": stale,
            "missing_sources": missing,
        }
