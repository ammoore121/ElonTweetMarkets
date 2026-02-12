"""Feature engineering package for Elon Musk tweet count prediction.

Exports:
    TweetFeatureBuilder  - Main builder class (declarative feature groups)
    FEATURE_GROUPS       - Registry of available feature groups
    compute_temporal_features   - Temporal feature extractor
    compute_media_features      - GDELT media feature extractor
    compute_calendar_features   - SpaceX/calendar feature extractor
    compute_market_features     - Market-derived feature extractor
    compute_cross_features      - Cross-interaction feature extractor
"""

from src.features.feature_builder import TweetFeatureBuilder, FEATURE_GROUPS
from src.features.extractors import (
    compute_temporal_features,
    compute_media_features,
    compute_calendar_features,
    compute_market_features,
    compute_cross_features,
)

__all__ = [
    "TweetFeatureBuilder",
    "FEATURE_GROUPS",
    "compute_temporal_features",
    "compute_media_features",
    "compute_calendar_features",
    "compute_market_features",
    "compute_cross_features",
]
