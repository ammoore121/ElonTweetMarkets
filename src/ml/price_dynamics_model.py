"""
Price dynamics model for Elon Musk tweet count prediction markets.

Combines the proven structural tail boost with intra-market price momentum
signals. The hypothesis is that bucket prices that have been falling
(crowd abandoning) tend to be underpriced if they match model predictions,
while buckets with sharp recent spikes (crowd piling in late) overshoot.

The model reads pre-computed momentum features from features["price_dynamics"]
which are saved per-event by scripts/analyze_price_dynamics.py.

Approach:
1. Start from crowd prices (normalized)
2. Apply tail boost (structural edge from TailBoost pattern)
3. For each bucket, apply momentum adjustment:
   - Negative momentum (falling price) -> boost probability (contrarian)
   - Strong positive momentum (rising sharply) -> reduce probability (overshoot)
4. Combine adjustments with configurable weights

Usage:
    model = PriceDynamicsModel()
    probs = model.predict(features, buckets, context={"entry_prices": {...}})
"""

import math
from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.advanced_models import (
    _get_crowd_probs,
    _normalize,
    _compute_implied_ev,
    PROB_FLOOR,
)
from src.ml.duration_model import _bucket_midpoint, _classify_duration


class PriceDynamicsModel(BasePredictionModel):
    """Tail boost + intra-market price momentum model.

    Combines two proven signals:
    1. Structural tail boost (crowd underprices tails)
    2. Price momentum contrarian signal (falling buckets are underpriced)

    The momentum adjustment is applied per-bucket using pre-computed features
    from features["price_dynamics"]["per_bucket"][bucket_label].

    Hyperparameters:
        base_tail_boost: Multiplier for tail bucket probabilities (>1 = boost)
        tail_threshold_sd: How many SDs from mean to qualify as "tail"
        momentum_weight: How strongly to adjust for momentum signal
        mean_reversion_strength: Cap on contrarian adjustment per bucket
        short_shrink: Downward shift for short-duration events
    """

    BASE_TAIL_BOOST = 1.35
    TAIL_THRESHOLD_SD = 0.8
    MOMENTUM_WEIGHT = 0.15
    MEAN_REVERSION_STRENGTH = 0.20
    SHORT_SHRINK = 0.18

    def __init__(
        self,
        name: str = "price_dynamics",
        version: str = "v1",
        base_tail_boost: float = None,
        tail_threshold_sd: float = None,
        momentum_weight: float = None,
        mean_reversion_strength: float = None,
        short_shrink: float = None,
    ):
        super().__init__(name=name, version=version)
        if base_tail_boost is not None:
            self.BASE_TAIL_BOOST = base_tail_boost
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd
        if momentum_weight is not None:
            self.MOMENTUM_WEIGHT = momentum_weight
        if mean_reversion_strength is not None:
            self.MEAN_REVERSION_STRENGTH = mean_reversion_strength
        if short_shrink is not None:
            self.SHORT_SHRINK = short_shrink

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        # Step 0: Get crowd probabilities
        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is None:
            n = len(buckets)
            return {b["bucket_label"]: 1.0 / n for b in buckets}

        probs = dict(crowd_probs)

        # Step 1: Duration-based shrink for short events
        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        if _classify_duration(duration_days) == "short" and self.SHORT_SHRINK > 0:
            from src.ml.advanced_models import _shift_distribution
            probs = _shift_distribution(probs, buckets, -self.SHORT_SHRINK)

        # Step 2: Compute crowd EV and std for tail classification
        market = features.get("market", {})
        crowd_ev = market.get("crowd_implied_ev")
        crowd_std = market.get("crowd_std_dev")

        if crowd_ev is None or crowd_std is None or float(crowd_std) <= 0:
            crowd_ev = _compute_implied_ev(buckets, crowd_probs)
            var = 0.0
            for b in buckets:
                mid = _bucket_midpoint(b, buckets)
                p = crowd_probs.get(b["bucket_label"], 0.0)
                var += p * (mid - crowd_ev) ** 2
            crowd_std = math.sqrt(var) if var > 0 else 50.0

        crowd_ev = float(crowd_ev)
        crowd_std = float(crowd_std)

        # Step 3: Get per-bucket momentum data
        price_dynamics = features.get("price_dynamics", {})
        per_bucket_momentum = price_dynamics.get("per_bucket", {})
        event_level = price_dynamics.get("event_level", {})

        # Normalize momentum using event-level std (if available)
        mom_std = event_level.get("std_momentum_24h")
        if mom_std is not None and mom_std > 0:
            mom_std = float(mom_std)
        else:
            mom_std = None

        # Step 4: Apply tail boost + momentum adjustment per bucket
        adjusted = {}
        for b in buckets:
            label = b["bucket_label"]
            p = probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = abs(mid - crowd_ev) / crowd_std if crowd_std > 0 else 0

            # Tail boost (structural edge)
            if z >= self.TAIL_THRESHOLD_SD:
                p = p * self.BASE_TAIL_BOOST

            # Momentum adjustment (contrarian signal)
            bucket_mom = per_bucket_momentum.get(label, {})
            mom_24h = bucket_mom.get("momentum_24h")

            if mom_24h is not None:
                mom_24h = float(mom_24h)

                # Normalize momentum to z-score if possible
                if mom_std is not None and mom_std > 0:
                    mom_z = mom_24h / mom_std
                else:
                    # Fallback: normalize by price level
                    price_level = probs.get(label, 0.05)
                    mom_z = mom_24h / max(price_level, 0.01)

                # Momentum-following adjustment:
                # Positive momentum (rising price) -> positive adjustment (boost)
                # Negative momentum (falling price) -> negative adjustment (reduce)
                # Data shows: winners have +0.07 avg momentum, losers -0.009
                # Rising buckets win 26.4% vs falling 7.4%
                adjustment = mom_z * self.MOMENTUM_WEIGHT

                # Cap the adjustment
                adjustment = max(
                    -self.MEAN_REVERSION_STRENGTH,
                    min(self.MEAN_REVERSION_STRENGTH, adjustment)
                )

                # Apply multiplicatively (1 + adjustment)
                multiplier = 1.0 + adjustment
                multiplier = max(0.5, min(1.5, multiplier))  # Safety bounds
                p = p * multiplier

            adjusted[label] = max(p, PROB_FLOOR)

        return _normalize(adjusted)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "momentum_weight": self.MOMENTUM_WEIGHT,
            "mean_reversion_strength": self.MEAN_REVERSION_STRENGTH,
            "short_shrink": self.SHORT_SHRINK,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "momentum_weight": self.MOMENTUM_WEIGHT,
            "mean_reversion_strength": self.MEAN_REVERSION_STRENGTH,
            "short_shrink": self.SHORT_SHRINK,
        }
