"""
Cross-market consistency arbitrage model for Elon Musk tweet count prediction markets.

Exploits pricing divergences between overlapping daily and weekly markets.
When daily bucket prices diverge from weekly-implied prices, the cheaper
side tends to win (87.9% edge observed in cross-market analysis).

The model reads pre-computed cross-market features from features["cross_market"]
which are saved per-event by scripts/analyze_cross_market.py.

Approach:
1. Start from crowd prices (normalized)
2. Apply tail boost (structural edge from TailBoost pattern)
3. For each bucket, check cross-market divergence:
   - Daily cheaper than weekly implies -> boost probability (underpriced)
   - Daily more expensive than weekly implies -> reduce probability (overpriced)
4. Normalize and return

When cross_market features are not available (most historical events), the
model falls back to pure tail boost, matching TailBoostModel performance.

Usage:
    model = CrossMarketArbModel()
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


class CrossMarketArbModel(BasePredictionModel):
    """Cross-market consistency arbitrage model.

    Adjusts bucket probabilities based on daily-vs-weekly divergence.
    Falls back to tail boost when cross-market data is unavailable.

    The key insight from cross-market analysis:
    - 39.5% of buckets have >5% divergence between daily and weekly prices
    - The cheap side wins at 10.8% vs avg price 5.7% (87.9% edge)
    - This is an exploitable structural inefficiency

    Hyperparameters:
        base_tail_boost: Multiplier for tail bucket probabilities (>1 = boost)
        tail_threshold_sd: How many SDs from mean to qualify as "tail"
        divergence_weight: How strongly to adjust for cross-market divergence
        divergence_threshold: Minimum absolute divergence to act on
        short_shrink: Downward shift for short-duration events
    """

    BASE_TAIL_BOOST = 1.30
    TAIL_THRESHOLD_SD = 1.0
    DIVERGENCE_WEIGHT = 0.20
    DIVERGENCE_THRESHOLD = 0.05
    SHORT_SHRINK = 0.18

    def __init__(
        self,
        name: str = "cross_market_arb",
        version: str = "v1",
        base_tail_boost: float = None,
        tail_threshold_sd: float = None,
        divergence_weight: float = None,
        divergence_threshold: float = None,
        short_shrink: float = None,
    ):
        super().__init__(name=name, version=version)
        if base_tail_boost is not None:
            self.BASE_TAIL_BOOST = base_tail_boost
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd
        if divergence_weight is not None:
            self.DIVERGENCE_WEIGHT = divergence_weight
        if divergence_threshold is not None:
            self.DIVERGENCE_THRESHOLD = divergence_threshold
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

        # Step 3: Get cross-market divergence data (if available)
        cross_market = features.get("cross_market", {})
        per_bucket_divergence = cross_market.get("per_bucket", {})
        has_cross_market = bool(per_bucket_divergence)

        # Step 4: Apply tail boost + cross-market divergence adjustment
        adjusted = {}
        for b in buckets:
            label = b["bucket_label"]
            p = probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = abs(mid - crowd_ev) / crowd_std if crowd_std > 0 else 0

            # Tail boost (structural edge -- always applied)
            if z >= self.TAIL_THRESHOLD_SD:
                p = p * self.BASE_TAIL_BOOST

            # Cross-market divergence adjustment (only when data available)
            if has_cross_market:
                bucket_data = per_bucket_divergence.get(label, {})

                # divergence = daily_price - weekly_implied_price
                # Negative divergence means daily is cheaper than weekly implies
                # (underpriced -> boost). Positive means overpriced -> reduce.
                divergence = bucket_data.get("divergence")

                if divergence is not None:
                    divergence = float(divergence)

                    if abs(divergence) >= self.DIVERGENCE_THRESHOLD:
                        # Scale adjustment by divergence magnitude
                        # Negative divergence -> positive adjustment (boost)
                        # Positive divergence -> negative adjustment (reduce)
                        adjustment = -divergence * self.DIVERGENCE_WEIGHT

                        # Cap the adjustment to avoid extreme swings
                        adjustment = max(-0.30, min(0.30, adjustment))

                        # Apply multiplicatively
                        multiplier = 1.0 + adjustment
                        multiplier = max(0.5, min(1.5, multiplier))
                        p = p * multiplier

            adjusted[label] = max(p, PROB_FLOOR)

        return _normalize(adjusted)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "divergence_weight": self.DIVERGENCE_WEIGHT,
            "divergence_threshold": self.DIVERGENCE_THRESHOLD,
            "short_shrink": self.SHORT_SHRINK,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "divergence_weight": self.DIVERGENCE_WEIGHT,
            "divergence_threshold": self.DIVERGENCE_THRESHOLD,
            "short_shrink": self.SHORT_SHRINK,
        }
