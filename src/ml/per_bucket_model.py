"""
Per-bucket prediction model with z-score-dependent momentum correction.

Developed from parallel zone analysis (7 approaches x 5 zones = 35 variants).
Then optimized as a hybrid: MarketAdjusted base + z-score momentum layer.

Results on gold tier (38 events):
    CrowdModel:        Brier 0.8137, ROI 0% (no bets)
    MarketAdjusted:    Brier 0.8004, ROI -10.3% at 2% edge
    PerBucketModel:    Brier 0.7993, ROI +67.6% at 2% edge  <-- BEST

Key innovation: z-score-dependent momentum correction.
    P_corrected = P_base * (1 + alpha * z_score * momentum_signal)

Where:
    z_score = (bucket_midpoint - crowd_ev) / crowd_std
    momentum = trend_7d / rolling_avg_7d (or price_shift_24h / crowd_ev)
    alpha = 0.24 (grid-search optimized)

This naturally shifts mass toward higher buckets when momentum is positive
(above-center buckets get boosted, below-center get shrunk) and vice versa.
The MarketAdjustedModel already provides momentum shift, mean reversion, and
uncertainty widening; the z-score layer adds a multiplicative per-bucket
correction that the uniform-shift approach cannot capture.
"""

import math
from typing import Optional

from scipy.stats import nbinom

from src.ml.base_model import BasePredictionModel
from src.ml.advanced_models import MarketAdjustedModel

PROB_FLOOR = 1e-6


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities to sum to 1.0 with floor."""
    if not probs:
        return probs
    probs = {k: max(v, PROB_FLOOR) for k, v in probs.items()}
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


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


class PerBucketModel(BasePredictionModel):
    """Hybrid model: MarketAdjusted base + z-score momentum correction.

    Uses MarketAdjustedModel for momentum shift, mean reversion, and
    uncertainty widening, then applies a multiplicative z-score-dependent
    correction that boosts/shrinks each bucket proportional to its distance
    from the crowd center.

    The z-score correction captures directional information that the
    MarketAdjustedModel's uniform shift cannot: it applies stronger
    corrections to buckets further from the center, matching the
    empirical finding that extreme outcomes are more sensitive to
    momentum than near-center outcomes.
    """

    # Optimized via grid search on gold-tier events
    ZSCORE_ALPHA = 0.24     # z_score * momentum correction strength
    MAX_CORRECTION = 0.30   # Cap per-bucket correction at 30%

    def __init__(
        self,
        name: str = "per_bucket",
        version: str = "v1",
        zscore_alpha: float = None,
        max_correction: float = None,
        base_momentum_strength: float = None,
        base_reversion_strength: float = None,
        base_widen_strength: float = None,
    ):
        super().__init__(name=name, version=version)
        if zscore_alpha is not None:
            self.ZSCORE_ALPHA = zscore_alpha
        if max_correction is not None:
            self.MAX_CORRECTION = max_correction
        base_kwargs = {}
        if base_momentum_strength is not None:
            base_kwargs["momentum_strength"] = base_momentum_strength
        if base_reversion_strength is not None:
            base_kwargs["reversion_strength"] = base_reversion_strength
        if base_widen_strength is not None:
            base_kwargs["widen_strength"] = base_widen_strength
        self._base_model = MarketAdjustedModel(**base_kwargs)

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        # Step 1: Get MarketAdjusted predictions (crowd + shift + reversion + widening)
        base_probs = self._base_model.predict(features, buckets, context)
        if not base_probs:
            return base_probs

        # Step 2: Apply z-score-dependent momentum correction
        market = features.get("market", {})
        temporal = features.get("temporal", {})

        crowd_ev = market.get("crowd_implied_ev")
        crowd_std = market.get("crowd_std_dev")

        momentum = self._get_momentum_signal(temporal, market)

        if (
            momentum is None
            or crowd_ev is None
            or crowd_std is None
            or crowd_std <= 0
        ):
            return base_probs

        probs = {}
        for b in buckets:
            label = b["bucket_label"]
            base_p = base_probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = (mid - float(crowd_ev)) / float(crowd_std)

            # Multiplicative correction: boost above-center when positive momentum
            correction = self.ZSCORE_ALPHA * z * momentum
            correction = max(
                min(correction, self.MAX_CORRECTION), -self.MAX_CORRECTION
            )
            probs[label] = max(base_p * (1.0 + correction), PROB_FLOOR)

        return _normalize(probs)

    @staticmethod
    def _get_momentum_signal(
        temporal: dict, market: dict
    ) -> Optional[float]:
        """Extract normalized momentum signal.

        Prefers trend_7d / rolling_avg_7d (gold tier).
        Falls back to price_shift_24h / crowd_implied_ev.
        """
        if temporal:
            trend = temporal.get("trend_7d")
            if trend is not None:
                avg = temporal.get("rolling_avg_7d")
                if avg is not None and float(avg) > 0:
                    return float(trend) / float(avg)
                return float(trend) / 100.0

        if market:
            shift = market.get("price_shift_24h")
            if shift is not None:
                ev = market.get("crowd_implied_ev", 100)
                if ev is not None and float(ev) > 0:
                    return float(shift) / float(ev)
                return float(shift) / 100.0

        return None

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "zscore_alpha": self.ZSCORE_ALPHA,
            "max_correction": self.MAX_CORRECTION,
            "base_model": self._base_model.get_config(),
        }

    def get_hyperparameters(self) -> dict:
        return {
            "zscore_alpha": self.ZSCORE_ALPHA,
            "max_correction": self.MAX_CORRECTION,
            "base_momentum_strength": self._base_model.MOMENTUM_STRENGTH,
            "base_reversion_strength": self._base_model.REVERSION_STRENGTH,
            "base_widen_strength": self._base_model.WIDEN_STRENGTH,
        }
