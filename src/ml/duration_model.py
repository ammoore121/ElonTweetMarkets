"""
Duration-adaptive tail model for Elon Musk tweet count prediction markets.

Key insight: Different event durations have different crowd biases.
- SHORT events (<=4 days, ~10 buckets): crowd overestimates by ~27%
- MEDIUM events (5-14 days, ~26 buckets): crowd is better calibrated
- LONG events (>14 days, ~51 buckets): high uncertainty, need wider tails

Additionally, extreme outcomes happen more often than the crowd expects.
Tail boosting steals probability from the center and redistributes to both
tails, exploiting the crowd's tendency to concentrate mass around the mode.

Five model variants:
1. DurationShrinkModel: Duration-based EV shrink only
2. TailBoostModel: Tail boosting only (power-law on tails, sharpen center)
3. DurationTailModel: Combined duration shrink + tail boost
4. PowerLawTailModel: P_adj = P^alpha where alpha<1 for tails, alpha>1 for center
5. DurationOverlayModel: MarketAdjusted base + duration-adaptive overlay

Results on gold tier (38 events):
    See scripts/test_duration.py for latest grid search results.
"""

import math
from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.advanced_models import (
    MarketAdjustedModel,
    _get_crowd_probs,
    _normalize,
    _compute_implied_ev,
    _shift_distribution,
    _widen_distribution,
    _negbin_bucket_probs,
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


def _classify_duration(duration_days: int) -> str:
    """Classify event into short/medium/long based on duration."""
    if duration_days <= 4:
        return "short"
    elif duration_days <= 14:
        return "medium"
    else:
        return "long"


def _get_sorted_buckets(buckets: list[dict]) -> list[dict]:
    """Return buckets sorted by lower_bound."""
    return sorted(buckets, key=lambda b: int(b["lower_bound"]))


# ---------------------------------------------------------------------------
# Variant 1: Duration-Based Shrink Only
# ---------------------------------------------------------------------------

class DurationShrinkModel(BasePredictionModel):
    """Adjusts crowd EV based on event duration.

    For short events, shrinks crowd's expected value downward (crowd
    overestimates on short events by ~27%). For medium events, applies
    a smaller correction. For long events, widens the distribution
    to account for high uncertainty.
    """

    SHORT_SHRINK = 0.20      # Shrink short-event EV by 20%
    MEDIUM_MOMENTUM = 0.10   # Small momentum correction for medium
    LONG_WIDEN = 1.30        # Widen factor for long events

    def __init__(
        self,
        name: str = "duration_shrink",
        version: str = "v1",
        short_shrink: float = None,
        medium_momentum: float = None,
        long_widen: float = None,
    ):
        super().__init__(name=name, version=version)
        if short_shrink is not None:
            self.SHORT_SHRINK = short_shrink
        if medium_momentum is not None:
            self.MEDIUM_MOMENTUM = medium_momentum
        if long_widen is not None:
            self.LONG_WIDEN = long_widen

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is None:
            # Fallback: uniform
            n = len(buckets)
            return {b["bucket_label"]: 1.0 / n for b in buckets}

        probs = dict(crowd_probs)
        duration_class = _classify_duration(duration_days)

        if duration_class == "short":
            # Shrink EV: shift mass toward lower buckets
            shift = -self.SHORT_SHRINK
            probs = _shift_distribution(probs, buckets, shift)

        elif duration_class == "medium":
            # Apply small momentum-based correction
            temporal = features.get("temporal", {})
            trend = None
            for key in ["trend_7d", "trend_14d"]:
                val = temporal.get(key) if temporal else None
                if val is not None:
                    trend = float(val)
                    break

            if trend is not None:
                rolling_avg = None
                for key in ["rolling_avg_7d", "rolling_avg_14d"]:
                    val = temporal.get(key) if temporal else None
                    if val is not None:
                        rolling_avg = float(val)
                        break

                if rolling_avg and rolling_avg > 0:
                    relative_trend = trend / rolling_avg
                    shift = relative_trend * self.MEDIUM_MOMENTUM
                    shift = max(min(shift, 0.15), -0.15)
                    if abs(shift) > 0.005:
                        probs = _shift_distribution(probs, buckets, shift)

        elif duration_class == "long":
            # Widen distribution for long events
            probs = _widen_distribution(probs, buckets, self.LONG_WIDEN)

        return _normalize(probs)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "short_shrink": self.SHORT_SHRINK,
            "medium_momentum": self.MEDIUM_MOMENTUM,
            "long_widen": self.LONG_WIDEN,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "short_shrink": self.SHORT_SHRINK,
            "medium_momentum": self.MEDIUM_MOMENTUM,
            "long_widen": self.LONG_WIDEN,
        }


# ---------------------------------------------------------------------------
# Variant 2: Tail Boost Only
# ---------------------------------------------------------------------------

class TailBoostModel(BasePredictionModel):
    """Boosts probability on both tails at the expense of the center.

    The crowd tends to over-concentrate mass around the mode. This model
    redistributes probability from center buckets to tail buckets.

    A bucket is "tail" if its midpoint is more than tail_threshold
    standard deviations from the crowd's expected value.

    tail_boost_factor: multiplicative boost for tail buckets (e.g. 1.3 = +30%)
    center_power: raise center bucket probabilities to this power (>1 = sharpen)
    """

    TAIL_BOOST_FACTOR = 1.30     # Boost tail probs by 30%
    TAIL_THRESHOLD_SD = 1.0      # Buckets > 1 SD from EV are "tails"
    CENTER_POWER = 1.0           # No extra center sharpening by default

    def __init__(
        self,
        name: str = "tail_boost",
        version: str = "v1",
        tail_boost_factor: float = None,
        tail_threshold_sd: float = None,
        center_power: float = None,
    ):
        super().__init__(name=name, version=version)
        if tail_boost_factor is not None:
            self.TAIL_BOOST_FACTOR = tail_boost_factor
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd
        if center_power is not None:
            self.CENTER_POWER = center_power

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is None:
            n = len(buckets)
            return {b["bucket_label"]: 1.0 / n for b in buckets}

        market = features.get("market", {})
        crowd_ev = market.get("crowd_implied_ev")
        crowd_std = market.get("crowd_std_dev")

        if crowd_ev is None or crowd_std is None or float(crowd_std) <= 0:
            # Cannot determine tails without EV/std, compute from prices
            crowd_ev = _compute_implied_ev(buckets, crowd_probs)
            # Estimate std from distribution
            var = 0.0
            for b in buckets:
                mid = _bucket_midpoint(b, buckets)
                p = crowd_probs.get(b["bucket_label"], 0.0)
                var += p * (mid - crowd_ev) ** 2
            crowd_std = math.sqrt(var) if var > 0 else 50.0

        crowd_ev = float(crowd_ev)
        crowd_std = float(crowd_std)

        probs = {}
        for b in buckets:
            label = b["bucket_label"]
            p = crowd_probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)

            z = abs(mid - crowd_ev) / crowd_std if crowd_std > 0 else 0

            if z >= self.TAIL_THRESHOLD_SD:
                # Tail bucket: boost
                probs[label] = p * self.TAIL_BOOST_FACTOR
            else:
                # Center bucket: optionally sharpen
                if self.CENTER_POWER != 1.0:
                    probs[label] = max(p ** self.CENTER_POWER, PROB_FLOOR)
                else:
                    probs[label] = p

        return _normalize(probs)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "tail_boost_factor": self.TAIL_BOOST_FACTOR,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "center_power": self.CENTER_POWER,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "tail_boost_factor": self.TAIL_BOOST_FACTOR,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "center_power": self.CENTER_POWER,
        }


# ---------------------------------------------------------------------------
# Variant 2b: Regime-Aware Tail Boost (Issue #5)
# ---------------------------------------------------------------------------

class RegimeAwareTailBoostModel(TailBoostModel):
    """Tail boost that adjusts asymmetrically based on regime signals.

    When rolling_avg * duration < crowd_ev (crowd overestimating):
        -> boost LOWER tail more, UPPER tail less
    When rolling_avg * duration > crowd_ev (crowd underestimating):
        -> boost UPPER tail more, LOWER tail less
    When ambiguous or no temporal data:
        -> fall back to symmetric (standard TailBoost)

    This fixes the core Issue #5 problem: standard TailBoost boosts both
    tails equally, which in an overestimating regime pushes bets upward —
    further from reality.
    """

    REGIME_STRENGTH = 0.6   # How much to skew (0=symmetric, 1=fully asymmetric)

    def __init__(
        self,
        name: str = "regime_tail",
        version: str = "v1",
        tail_boost_factor: float = None,
        tail_threshold_sd: float = None,
        center_power: float = None,
        regime_strength: float = None,
    ):
        super().__init__(
            name=name,
            version=version,
            tail_boost_factor=tail_boost_factor,
            tail_threshold_sd=tail_threshold_sd,
            center_power=center_power,
        )
        if regime_strength is not None:
            self.REGIME_STRENGTH = regime_strength

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is None:
            n = len(buckets)
            return {b["bucket_label"]: 1.0 / n for b in buckets}

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

        # Compute regime signal
        regime_signal = self._compute_regime_signal(features, context, crowd_ev)

        probs = {}
        for b in buckets:
            label = b["bucket_label"]
            p = crowd_probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)

            z = abs(mid - crowd_ev) / crowd_std if crowd_std > 0 else 0

            if z >= self.TAIL_THRESHOLD_SD:
                # Determine tail direction relative to crowd EV
                if mid < crowd_ev:
                    # Lower tail: boost more when crowd overestimates (regime_signal < 0)
                    directional_mult = 1 + self.REGIME_STRENGTH * (-regime_signal)
                else:
                    # Upper tail: boost more when crowd underestimates (regime_signal > 0)
                    directional_mult = 1 + self.REGIME_STRENGTH * regime_signal

                # Clamp multiplier to reasonable range [0.3, 2.0]
                directional_mult = max(0.3, min(2.0, directional_mult))
                probs[label] = p * self.TAIL_BOOST_FACTOR * directional_mult
            else:
                # Center bucket: optionally sharpen
                if self.CENTER_POWER != 1.0:
                    probs[label] = max(p ** self.CENTER_POWER, PROB_FLOOR)
                else:
                    probs[label] = p

        return _normalize(probs)

    def _compute_regime_signal(
        self,
        features: dict,
        context: Optional[dict],
        crowd_ev: float,
    ) -> float:
        """Compute regime signal: negative = crowd overestimating, positive = underestimating.

        Returns value clipped to [-1, +1]. Returns 0 when data is insufficient.
        """
        temporal = features.get("temporal", {})
        calendar = features.get("calendar", {})

        # Prefer calendar-day average (conservative), fall back to standard
        rolling_avg = temporal.get("rolling_avg_7d_calendar")
        if rolling_avg is None:
            rolling_avg = temporal.get("rolling_avg_7d")
        if rolling_avg is None:
            rolling_avg = temporal.get("rolling_avg_14d")

        if rolling_avg is None:
            return 0.0

        # Check data coverage — defer to symmetric if sparse
        data_coverage = temporal.get("data_coverage_7d")
        if data_coverage is not None and data_coverage < 0.5:
            return 0.0

        # Get event duration
        duration = calendar.get("event_duration_days")
        if duration is None and context:
            duration = context.get("duration_days")
        if duration is None or duration <= 0:
            return 0.0

        if crowd_ev <= 0:
            return 0.0

        expected_total = rolling_avg * duration
        regime_signal = (expected_total - crowd_ev) / crowd_ev

        # Clip to [-1, +1]
        return max(-1.0, min(1.0, regime_signal))

    def get_config(self) -> dict:
        config = super().get_config()
        config["regime_strength"] = self.REGIME_STRENGTH
        return config

    def get_hyperparameters(self) -> dict:
        params = super().get_hyperparameters()
        params["regime_strength"] = self.REGIME_STRENGTH
        return params


# ---------------------------------------------------------------------------
# Variant 3: Combined Duration + Tail Boost
# ---------------------------------------------------------------------------

class DurationTailModel(BasePredictionModel):
    """Combined duration-adaptive shrink with tail boosting.

    First applies duration-based EV adjustment (shrink for short events,
    widen for long events), then applies tail boosting to both tails.
    """

    SHORT_SHRINK = 0.20
    MEDIUM_MOMENTUM = 0.10
    LONG_WIDEN = 1.30
    TAIL_BOOST_FACTOR = 1.20
    TAIL_THRESHOLD_SD = 1.0

    def __init__(
        self,
        name: str = "duration_tail",
        version: str = "v1",
        short_shrink: float = None,
        medium_momentum: float = None,
        long_widen: float = None,
        tail_boost_factor: float = None,
        tail_threshold_sd: float = None,
    ):
        super().__init__(name=name, version=version)
        if short_shrink is not None:
            self.SHORT_SHRINK = short_shrink
        if medium_momentum is not None:
            self.MEDIUM_MOMENTUM = medium_momentum
        if long_widen is not None:
            self.LONG_WIDEN = long_widen
        if tail_boost_factor is not None:
            self.TAIL_BOOST_FACTOR = tail_boost_factor
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is None:
            n = len(buckets)
            return {b["bucket_label"]: 1.0 / n for b in buckets}

        probs = dict(crowd_probs)
        duration_class = _classify_duration(duration_days)

        # Step 1: Duration-based adjustment
        if duration_class == "short":
            shift = -self.SHORT_SHRINK
            probs = _shift_distribution(probs, buckets, shift)
        elif duration_class == "medium":
            temporal = features.get("temporal", {})
            trend = None
            for key in ["trend_7d", "trend_14d"]:
                val = temporal.get(key) if temporal else None
                if val is not None:
                    trend = float(val)
                    break
            if trend is not None:
                rolling_avg = None
                for key in ["rolling_avg_7d", "rolling_avg_14d"]:
                    val = temporal.get(key) if temporal else None
                    if val is not None:
                        rolling_avg = float(val)
                        break
                if rolling_avg and rolling_avg > 0:
                    relative_trend = trend / rolling_avg
                    shift = relative_trend * self.MEDIUM_MOMENTUM
                    shift = max(min(shift, 0.15), -0.15)
                    if abs(shift) > 0.005:
                        probs = _shift_distribution(probs, buckets, shift)
        elif duration_class == "long":
            probs = _widen_distribution(probs, buckets, self.LONG_WIDEN)

        # Step 2: Tail boosting
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

        boosted = {}
        for b in buckets:
            label = b["bucket_label"]
            p = probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = abs(mid - crowd_ev) / crowd_std if crowd_std > 0 else 0

            if z >= self.TAIL_THRESHOLD_SD:
                boosted[label] = p * self.TAIL_BOOST_FACTOR
            else:
                boosted[label] = p

        return _normalize(boosted)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "short_shrink": self.SHORT_SHRINK,
            "medium_momentum": self.MEDIUM_MOMENTUM,
            "long_widen": self.LONG_WIDEN,
            "tail_boost_factor": self.TAIL_BOOST_FACTOR,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "short_shrink": self.SHORT_SHRINK,
            "medium_momentum": self.MEDIUM_MOMENTUM,
            "long_widen": self.LONG_WIDEN,
            "tail_boost_factor": self.TAIL_BOOST_FACTOR,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
        }


# ---------------------------------------------------------------------------
# Variant 4: Power-Law Tail Adjustment
# ---------------------------------------------------------------------------

class PowerLawTailModel(BasePredictionModel):
    """Power-law tail adjustment: P_adj = P^alpha.

    For tail buckets (|z| > threshold), alpha < 1 flattens probabilities
    (raises low-prob tails). For center buckets, alpha > 1 sharpens
    (concentrates mass at peak). This naturally transfers mass from
    center to tails.

    tail_alpha < 1: boost tails (e.g. 0.8 raises 0.01 to 0.016)
    center_alpha > 1: sharpen center (e.g. 1.2 lowers 0.10 to 0.063)
    """

    TAIL_ALPHA = 0.80           # Power for tail buckets (<1 = boost)
    CENTER_ALPHA = 1.10         # Power for center buckets (>1 = sharpen)
    TAIL_THRESHOLD_SD = 1.0     # SD threshold for tail classification

    def __init__(
        self,
        name: str = "power_law_tail",
        version: str = "v1",
        tail_alpha: float = None,
        center_alpha: float = None,
        tail_threshold_sd: float = None,
    ):
        super().__init__(name=name, version=version)
        if tail_alpha is not None:
            self.TAIL_ALPHA = tail_alpha
        if center_alpha is not None:
            self.CENTER_ALPHA = center_alpha
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is None:
            n = len(buckets)
            return {b["bucket_label"]: 1.0 / n for b in buckets}

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

        probs = {}
        for b in buckets:
            label = b["bucket_label"]
            p = max(crowd_probs.get(label, PROB_FLOOR), PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = abs(mid - crowd_ev) / crowd_std if crowd_std > 0 else 0

            if z >= self.TAIL_THRESHOLD_SD:
                probs[label] = max(p ** self.TAIL_ALPHA, PROB_FLOOR)
            else:
                probs[label] = max(p ** self.CENTER_ALPHA, PROB_FLOOR)

        return _normalize(probs)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "tail_alpha": self.TAIL_ALPHA,
            "center_alpha": self.CENTER_ALPHA,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "tail_alpha": self.TAIL_ALPHA,
            "center_alpha": self.CENTER_ALPHA,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
        }


# ---------------------------------------------------------------------------
# Variant 5: MarketAdjusted Base + Duration Overlay
# ---------------------------------------------------------------------------

class DurationOverlayModel(BasePredictionModel):
    """MarketAdjusted model with duration-adaptive overlay.

    Uses MarketAdjustedModel as the base (which already applies momentum,
    mean reversion, and uncertainty widening), then adds duration-specific
    corrections:

    - SHORT events: additional downward shrink (crowd overestimates)
    - MEDIUM events: keep MarketAdjusted as-is (it handles these well)
    - LONG events: additional widening

    Plus optional tail boosting on top of everything.

    This is the kitchen-sink model: MarketAdjusted's proven corrections
    + duration awareness + tail boosting.
    """

    SHORT_SHRINK = 0.12         # Additional shrink after MarketAdjusted
    LONG_WIDEN = 1.15           # Additional widening after MarketAdjusted
    TAIL_BOOST_FACTOR = 1.15    # Mild tail boost on top of everything
    TAIL_THRESHOLD_SD = 1.2     # Slightly further out to avoid over-boosting

    def __init__(
        self,
        name: str = "duration_overlay",
        version: str = "v1",
        short_shrink: float = None,
        long_widen: float = None,
        tail_boost_factor: float = None,
        tail_threshold_sd: float = None,
    ):
        super().__init__(name=name, version=version)
        self._base_model = MarketAdjustedModel()
        if short_shrink is not None:
            self.SHORT_SHRINK = short_shrink
        if long_widen is not None:
            self.LONG_WIDEN = long_widen
        if tail_boost_factor is not None:
            self.TAIL_BOOST_FACTOR = tail_boost_factor
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        # Step 1: Get MarketAdjusted predictions
        base_probs = self._base_model.predict(features, buckets, context)
        if not base_probs:
            return base_probs

        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        probs = dict(base_probs)
        duration_class = _classify_duration(duration_days)

        # Step 2: Duration-specific overlay
        if duration_class == "short" and self.SHORT_SHRINK > 0:
            shift = -self.SHORT_SHRINK
            probs = _shift_distribution(probs, buckets, shift)
        elif duration_class == "long" and self.LONG_WIDEN > 1.0:
            probs = _widen_distribution(probs, buckets, self.LONG_WIDEN)
        # medium: no additional adjustment beyond MarketAdjusted

        # Step 3: Tail boosting (if enabled)
        if self.TAIL_BOOST_FACTOR > 1.0:
            market = features.get("market", {})
            crowd_ev = market.get("crowd_implied_ev")
            crowd_std = market.get("crowd_std_dev")

            if crowd_ev is None or crowd_std is None or float(crowd_std) <= 0:
                crowd_probs = _get_crowd_probs(buckets, context)
                if crowd_probs:
                    crowd_ev = _compute_implied_ev(buckets, crowd_probs)
                    var = 0.0
                    for b in buckets:
                        mid = _bucket_midpoint(b, buckets)
                        p = crowd_probs.get(b["bucket_label"], 0.0)
                        var += p * (mid - crowd_ev) ** 2
                    crowd_std = math.sqrt(var) if var > 0 else 50.0

            if crowd_ev is not None and crowd_std is not None:
                crowd_ev = float(crowd_ev)
                crowd_std = float(crowd_std)
                if crowd_std > 0:
                    boosted = {}
                    for b in buckets:
                        label = b["bucket_label"]
                        p = probs.get(label, PROB_FLOOR)
                        mid = _bucket_midpoint(b, buckets)
                        z = abs(mid - crowd_ev) / crowd_std
                        if z >= self.TAIL_THRESHOLD_SD:
                            boosted[label] = p * self.TAIL_BOOST_FACTOR
                        else:
                            boosted[label] = p
                    probs = boosted

        return _normalize(probs)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "short_shrink": self.SHORT_SHRINK,
            "long_widen": self.LONG_WIDEN,
            "tail_boost_factor": self.TAIL_BOOST_FACTOR,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "base_model": self._base_model.get_config(),
        }

    def get_hyperparameters(self) -> dict:
        return {
            "short_shrink": self.SHORT_SHRINK,
            "long_widen": self.LONG_WIDEN,
            "tail_boost_factor": self.TAIL_BOOST_FACTOR,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
        }
