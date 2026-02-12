"""
Asymmetric Momentum Model for Elon Musk tweet count prediction markets.

Key insight from feature analysis: The crowd captures only 48% of downswings
but 105% of upswings. When Elon's tweet count is dropping, the crowd doesn't
adjust fast enough. This model exploits that asymmetry.

Two variants:
    1. AsymmetricMomentumModel: Direct on crowd prices + asymmetric z-score
       + directional mean reversion + conditional widening.
    2. AsymmetricPerBucketModel: Like PerBucketModel but with asymmetric alpha.
       Uses MarketAdjustedModel as base, then applies asymmetric z-score
       correction on top. This is the more conservative approach since it
       reuses the well-calibrated MarketAdjusted base.
"""

import math
from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.advanced_models import (
    MarketAdjustedModel,
    _normalize,
    _get_crowd_probs,
    _shift_distribution,
    _widen_distribution,
    _compute_implied_ev,
    PROB_FLOOR,
)


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


# ---------------------------------------------------------------------------
# Variant 1: Direct asymmetric model on crowd prices
# ---------------------------------------------------------------------------

class AsymmetricMomentumModel(BasePredictionModel):
    """Model exploiting asymmetric crowd reaction to momentum.

    The crowd under-reacts to downswings (captures 48%) but fully tracks
    upswings (captures 105%). This model applies stronger corrections
    during negative momentum periods.

    Layers (applied directly to crowd prices):
        1. Crowd prices as base (normalized)
        2. Asymmetric z-score momentum correction
        3. Directional mean reversion (stronger when crowd is above rolling avg)
        4. Conditional variance widening (widen more during downswings)

    All parameters are tunable via constructor for grid search.
    """

    def __init__(
        self,
        name: str = "asymmetric_momentum",
        version: str = "v1",
        # Core asymmetry parameters
        alpha_down: float = 0.40,
        alpha_up: float = 0.10,
        max_correction: float = 0.35,
        # Mean reversion (directional)
        reversion_down: float = 0.25,
        reversion_up: float = 0.10,
        # Variance widening
        widen_on_downswing: float = 0.15,
        widen_base: float = 0.10,
        # Tail protection
        tail_cap_multiplier: float = 3.0,
    ):
        super().__init__(name=name, version=version)
        self.alpha_down = alpha_down
        self.alpha_up = alpha_up
        self.max_correction = max_correction
        self.reversion_down = reversion_down
        self.reversion_up = reversion_up
        self.widen_on_downswing = widen_on_downswing
        self.widen_base = widen_base
        self.tail_cap_multiplier = tail_cap_multiplier

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        # Step 1: Start from crowd prices
        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is None:
            n = len(buckets)
            return {b["bucket_label"]: 1.0 / n for b in buckets}

        probs = dict(crowd_probs)

        temporal = features.get("temporal", {})
        market = features.get("market", {})
        media = features.get("media", {}) or features.get("gdelt", {})

        momentum = self._get_momentum(temporal, market)
        crowd_ev = market.get("crowd_implied_ev") if market else None
        crowd_std = market.get("crowd_std_dev") if market else None
        rolling_avg = self._get_rolling_avg(temporal)

        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        # Step 2: Asymmetric z-score momentum correction
        if (
            momentum is not None
            and crowd_ev is not None
            and crowd_std is not None
            and float(crowd_std) > 0
        ):
            probs = self._apply_asymmetric_zscore(
                probs, buckets, momentum, float(crowd_ev), float(crowd_std)
            )

        # Step 3: Directional mean reversion
        if rolling_avg is not None and crowd_ev is not None:
            probs = self._apply_directional_reversion(
                probs, buckets, float(crowd_ev), rolling_avg, duration_days
            )

        # Step 4: Conditional variance widening
        if momentum is not None:
            probs = self._apply_conditional_widening(
                probs, buckets, momentum, temporal, media
            )

        # Step 5: Tail protection cap
        if context and "entry_prices" in context:
            entry_prices = context["entry_prices"]
            if entry_prices:
                for b in buckets:
                    label = b["bucket_label"]
                    mkt = entry_prices.get(label, 0.0)
                    if isinstance(mkt, (int, float)) and mkt > 0:
                        cap = max(
                            self.tail_cap_multiplier * mkt,
                            mkt + 0.03,
                        )
                        if probs.get(label, 0.0) > cap:
                            probs[label] = cap

        return _normalize(probs)

    def _apply_asymmetric_zscore(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        momentum: float,
        crowd_ev: float,
        crowd_std: float,
    ) -> dict[str, float]:
        """Apply z-score correction with different alpha for up/down momentum."""
        if abs(momentum) < 0.001:
            return probs

        alpha = self.alpha_down if momentum < 0 else self.alpha_up

        new_probs = {}
        for b in buckets:
            label = b["bucket_label"]
            base_p = probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = (mid - crowd_ev) / crowd_std

            correction = alpha * z * momentum
            correction = max(
                min(correction, self.max_correction), -self.max_correction
            )
            new_probs[label] = max(base_p * (1.0 + correction), PROB_FLOOR)

        return _normalize(new_probs)

    def _apply_directional_reversion(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        crowd_ev: float,
        rolling_avg: float,
        duration_days: int,
    ) -> dict[str, float]:
        """Mean reversion with asymmetric strength."""
        model_ev = rolling_avg * duration_days

        if crowd_ev <= 0 or model_ev <= 0:
            return probs

        divergence = (crowd_ev - model_ev) / crowd_ev

        if abs(divergence) < 0.03:
            return probs

        strength = self.reversion_down if divergence > 0 else self.reversion_up

        shift = -divergence * strength
        shift = max(min(shift, 0.15), -0.15)

        if abs(shift) > 0.005:
            probs = _shift_distribution(probs, buckets, shift)

        return probs

    def _apply_conditional_widening(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        momentum: float,
        temporal: dict,
        media: dict,
    ) -> dict[str, float]:
        """Widen distribution conditionally -- more during downswings."""
        widen_factor = 1.0

        if temporal:
            cv = temporal.get("cv_14d")
            if cv is not None:
                cv = float(cv)
                if cv > 0.35:
                    widen_factor += (cv - 0.35) * self.widen_base * 5.0

        if momentum < -0.05:
            widen_factor += abs(momentum) * self.widen_on_downswing * 3.0

        if media:
            elon_delta = media.get("elon_musk_vol_delta")
            if elon_delta is not None:
                elon_delta = float(elon_delta)
                if elon_delta > 0.05:
                    widen_factor += elon_delta * self.widen_base * 2.0

        widen_factor = max(min(widen_factor, 1.8), 0.9)

        if abs(widen_factor - 1.0) > 0.01:
            probs = _widen_distribution(probs, buckets, widen_factor)

        return probs

    def _get_momentum(
        self, temporal: dict, market: dict
    ) -> Optional[float]:
        """Extract normalized momentum signal."""
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

    @staticmethod
    def _get_rolling_avg(temporal: dict) -> Optional[float]:
        if not temporal:
            return None
        for key in ["rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d"]:
            val = temporal.get(key)
            if val is not None:
                return float(val)
        return None

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "alpha_down": self.alpha_down,
            "alpha_up": self.alpha_up,
            "max_correction": self.max_correction,
            "reversion_down": self.reversion_down,
            "reversion_up": self.reversion_up,
            "widen_on_downswing": self.widen_on_downswing,
            "widen_base": self.widen_base,
            "tail_cap_multiplier": self.tail_cap_multiplier,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "alpha_down": self.alpha_down,
            "alpha_up": self.alpha_up,
            "max_correction": self.max_correction,
            "reversion_down": self.reversion_down,
            "reversion_up": self.reversion_up,
            "widen_on_downswing": self.widen_on_downswing,
            "widen_base": self.widen_base,
            "tail_cap_multiplier": self.tail_cap_multiplier,
        }

    def __repr__(self) -> str:
        return (
            "AsymmetricMomentumModel(name={!r}, alpha_down={}, alpha_up={})".format(
                self.name, self.alpha_down, self.alpha_up
            )
        )


# ---------------------------------------------------------------------------
# Variant 2: Asymmetric z-score on MarketAdjusted base (like PerBucketModel)
# ---------------------------------------------------------------------------

class AsymmetricPerBucketModel(BasePredictionModel):
    """MarketAdjusted base + asymmetric z-score momentum correction.

    Like PerBucketModel, uses MarketAdjustedModel for the heavy lifting
    (momentum shift, mean reversion, widening), then adds a per-bucket
    z-score correction -- but with ASYMMETRIC alpha:

        - alpha_down (stronger) when momentum is negative (downswing)
        - alpha_up (weaker) when momentum is positive (upswing)

    This directly addresses the finding that the crowd captures only 48%
    of downswings. The MarketAdjusted base already provides some correction;
    the asymmetric z-score layer adds a stronger directional tilt when
    the crowd is likely under-reacting.

    Compared to PerBucketModel (symmetric alpha=0.24):
        - Downswing events get alpha > 0.24 (stronger correction)
        - Upswing events get alpha < 0.24 (weaker, trust crowd more)
    """

    def __init__(
        self,
        name: str = "asymmetric_per_bucket",
        version: str = "v1",
        alpha_down: float = 0.36,
        alpha_up: float = 0.12,
        max_correction: float = 0.30,
        tail_cap_multiplier: float = 3.0,
    ):
        super().__init__(name=name, version=version)
        self.alpha_down = alpha_down
        self.alpha_up = alpha_up
        self.max_correction = max_correction
        self.tail_cap_multiplier = tail_cap_multiplier
        self._base_model = MarketAdjustedModel()

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        # Step 1: Get MarketAdjusted base predictions
        base_probs = self._base_model.predict(features, buckets, context)
        if not base_probs:
            return base_probs

        # Step 2: Apply asymmetric z-score momentum correction
        market = features.get("market", {})
        temporal = features.get("temporal", {})

        crowd_ev = market.get("crowd_implied_ev") if market else None
        crowd_std = market.get("crowd_std_dev") if market else None
        momentum = self._get_momentum_signal(temporal, market)

        if (
            momentum is None
            or crowd_ev is None
            or crowd_std is None
            or float(crowd_std) <= 0
        ):
            return base_probs

        crowd_ev = float(crowd_ev)
        crowd_std = float(crowd_std)

        # Choose alpha based on momentum direction
        alpha = self.alpha_down if momentum < 0 else self.alpha_up

        probs = {}
        for b in buckets:
            label = b["bucket_label"]
            base_p = base_probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = (mid - crowd_ev) / crowd_std

            correction = alpha * z * momentum
            correction = max(
                min(correction, self.max_correction), -self.max_correction
            )
            probs[label] = max(base_p * (1.0 + correction), PROB_FLOOR)

        # Tail protection
        if context and "entry_prices" in context:
            entry_prices = context["entry_prices"]
            if entry_prices:
                for b in buckets:
                    label = b["bucket_label"]
                    mkt = entry_prices.get(label, 0.0)
                    if isinstance(mkt, (int, float)) and mkt > 0:
                        cap = max(
                            self.tail_cap_multiplier * mkt,
                            mkt + 0.03,
                        )
                        if probs.get(label, 0.0) > cap:
                            probs[label] = cap

        return _normalize(probs)

    @staticmethod
    def _get_momentum_signal(
        temporal: dict, market: dict
    ) -> Optional[float]:
        """Extract normalized momentum signal (same as PerBucketModel)."""
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
            "alpha_down": self.alpha_down,
            "alpha_up": self.alpha_up,
            "max_correction": self.max_correction,
            "tail_cap_multiplier": self.tail_cap_multiplier,
            "base_model": self._base_model.get_config(),
        }

    def get_hyperparameters(self) -> dict:
        return {
            "alpha_down": self.alpha_down,
            "alpha_up": self.alpha_up,
            "max_correction": self.max_correction,
            "tail_cap_multiplier": self.tail_cap_multiplier,
            "base_momentum_strength": self._base_model.MOMENTUM_STRENGTH,
            "base_reversion_strength": self._base_model.REVERSION_STRENGTH,
            "base_widen_strength": self._base_model.WIDEN_STRENGTH,
        }

    def __repr__(self) -> str:
        return (
            "AsymmetricPerBucketModel(name={!r}, "
            "alpha_down={}, alpha_up={})".format(
                self.name, self.alpha_down, self.alpha_up
            )
        )
