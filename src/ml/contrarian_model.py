"""
Contrarian Price-Reversal Model for Elon Musk tweet count prediction markets.

Key insight: Market prices shift in the 24 hours before resolution. The
price_shift_24h feature measures how much the crowd's expected value changed.
Markets tend to OVERREACT to recent information -- a big shift up is often
followed by the actual result being lower than the shifted expectation, and
vice versa.

This model:
1. Starts from crowd/market prices (like MarketAdjustedModel)
2. Detects large 24h price shifts and bets AGAINST the direction
3. Exploits crowd skewness: when the market is very right-skewed, the center
   may be overestimated
4. Uses distribution entropy: low entropy (overconfident market) could mean-revert

Variants controlled by hyperparameters:
- Pure price-reversal (reversal_strength > 0, skew_correction = 0)
- Pure skewness correction (reversal_strength = 0, skew_correction > 0)
- Combined reversal + skewness
- Threshold-gated (only apply when shift exceeds shift_threshold)
- MarketAdjusted base (use_adjusted_base = True)

Usage:
    model = ContrarianModel(
        reversal_strength=0.20,
        shift_threshold=10.0,
        skew_correction=0.05,
        entropy_correction=0.0,
    )
    probs = model.predict(features, buckets, context={"entry_prices": {...}})
"""

import math
from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.advanced_models import (
    PROB_FLOOR,
    _normalize,
    _uniform,
    _get_crowd_probs,
    _shift_distribution,
    _widen_distribution,
    MarketAdjustedModel,
)


class ContrarianModel(BasePredictionModel):
    """Contrarian price-reversal model that fades recent market moves.

    The model starts from crowd prices (or MarketAdjustedModel output) and
    applies corrections that bet against the direction of recent price movement.

    Hyperparameters:
        reversal_strength: How aggressively to reverse recent price shifts.
            0.0 = no reversal. 0.30 = shift 30% of shift back.
        shift_threshold: Minimum absolute price_shift_24h to trigger reversal.
            Filters out noise. In units of implied EV change.
        skew_correction: How aggressively to correct for crowd skewness.
            When crowd is very right-skewed (skewness > 1), shift mass left.
            When left-skewed (skewness < -1), shift mass right.
        entropy_correction: How aggressively to widen overconfident markets.
            When entropy is low (market concentrates on few buckets), widen.
        use_adjusted_base: If True, use MarketAdjustedModel output as the
            starting point (applies momentum, reversion, etc. first).
            If False, use raw crowd prices.
        max_reversal_shift: Cap on the absolute shift amount applied. Prevents
            extreme corrections.
    """

    def __init__(
        self,
        name: str = "contrarian",
        version: str = "v1",
        reversal_strength: float = 0.20,
        shift_threshold: float = 10.0,
        skew_correction: float = 0.05,
        entropy_correction: float = 0.0,
        use_adjusted_base: bool = False,
        max_reversal_shift: float = 0.20,
    ):
        super().__init__(name=name, version=version)
        self.reversal_strength = reversal_strength
        self.shift_threshold = shift_threshold
        self.skew_correction = skew_correction
        self.entropy_correction = entropy_correction
        self.use_adjusted_base = use_adjusted_base
        self.max_reversal_shift = max_reversal_shift

        # Lazy-initialized MarketAdjustedModel (only if use_adjusted_base)
        self._adjusted_model = None

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return probability distribution across buckets."""
        if not buckets:
            return {}

        # -----------------------------------------------------------
        # Step 1: Get base probabilities
        # -----------------------------------------------------------
        if self.use_adjusted_base:
            # Use MarketAdjustedModel output as the starting point
            if self._adjusted_model is None:
                self._adjusted_model = MarketAdjustedModel()
            probs = self._adjusted_model.predict(features, buckets, context)
        else:
            # Use raw crowd prices
            probs = _get_crowd_probs(buckets, context)
            if probs is None:
                # No crowd data -- fall back to uniform
                return _uniform(buckets)

        # -----------------------------------------------------------
        # Step 2: Extract market features
        # -----------------------------------------------------------
        market = features.get("market", {})
        price_shift_24h = market.get("price_shift_24h")
        crowd_skewness = market.get("crowd_skewness")
        crowd_std_dev = market.get("crowd_std_dev")
        distribution_entropy = market.get("distribution_entropy")
        crowd_implied_ev = market.get("crowd_implied_ev")

        # -----------------------------------------------------------
        # Step 3: Apply price reversal correction
        # -----------------------------------------------------------
        if (
            price_shift_24h is not None
            and self.reversal_strength > 0
        ):
            price_shift_24h = float(price_shift_24h)
            abs_shift = abs(price_shift_24h)

            # Only apply if shift exceeds threshold
            if abs_shift >= self.shift_threshold:
                # The reversal goes AGAINST the shift direction:
                # If market shifted UP (positive shift), we shift DOWN
                # If market shifted DOWN (negative shift), we shift UP

                # Scale the correction by the size of the shift relative to
                # the crowd's standard deviation (if available) to normalize
                # across different market scales
                if crowd_std_dev is not None and float(crowd_std_dev) > 0:
                    normalized_shift = price_shift_24h / float(crowd_std_dev)
                elif crowd_implied_ev is not None and float(crowd_implied_ev) > 0:
                    normalized_shift = price_shift_24h / float(crowd_implied_ev)
                else:
                    # Fallback: use absolute shift scaled to a reasonable range
                    normalized_shift = price_shift_24h / 50.0

                # Reversal direction is opposite to shift
                reversal_amount = -normalized_shift * self.reversal_strength

                # Cap the reversal
                reversal_amount = max(
                    min(reversal_amount, self.max_reversal_shift),
                    -self.max_reversal_shift,
                )

                if abs(reversal_amount) > 0.005:
                    probs = _shift_distribution(probs, buckets, reversal_amount)

        # -----------------------------------------------------------
        # Step 4: Apply skewness correction
        # -----------------------------------------------------------
        if (
            crowd_skewness is not None
            and self.skew_correction > 0
        ):
            crowd_skewness = float(crowd_skewness)

            # When crowd is very right-skewed (skewness > 1), they are
            # placing too much weight on the right tail relative to center.
            # This often means the center of mass is pulled too high.
            # Correction: shift mass LEFT (negative shift).
            #
            # When crowd is left-skewed (skewness < -1), the reverse applies.
            #
            # Only correct when skewness is extreme (abs > 1.0)
            if abs(crowd_skewness) > 1.0:
                # Direction: opposite to skewness
                # Right-skewed -> shift left (negative)
                # Left-skewed -> shift right (positive)
                skew_shift = -crowd_skewness * self.skew_correction

                # Scale down for very extreme skewness to avoid over-correction
                if abs(crowd_skewness) > 3.0:
                    skew_shift *= 3.0 / abs(crowd_skewness)

                # Cap
                skew_shift = max(min(skew_shift, 0.15), -0.15)

                if abs(skew_shift) > 0.005:
                    probs = _shift_distribution(probs, buckets, skew_shift)

        # -----------------------------------------------------------
        # Step 5: Apply entropy correction (widen overconfident markets)
        # -----------------------------------------------------------
        if (
            distribution_entropy is not None
            and self.entropy_correction > 0
        ):
            distribution_entropy = float(distribution_entropy)
            n_buckets = len(buckets)

            if n_buckets > 1:
                # Maximum entropy for n buckets is log(n)
                max_entropy = math.log(n_buckets)

                # Normalized entropy: 0 = perfectly concentrated, 1 = uniform
                if max_entropy > 0:
                    normalized_entropy = distribution_entropy / max_entropy
                else:
                    normalized_entropy = 1.0

                # Low entropy (< 0.5) means the market is overconfident
                # Widen the distribution to account for uncertainty
                if normalized_entropy < 0.6:
                    confidence_excess = 0.6 - normalized_entropy
                    widen_factor = 1.0 + confidence_excess * self.entropy_correction * 5.0
                    widen_factor = min(widen_factor, 1.5)  # Cap widening

                    if widen_factor > 1.01:
                        probs = _widen_distribution(probs, buckets, widen_factor)

        return _normalize(probs)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "reversal_strength": self.reversal_strength,
            "shift_threshold": self.shift_threshold,
            "skew_correction": self.skew_correction,
            "entropy_correction": self.entropy_correction,
            "use_adjusted_base": self.use_adjusted_base,
            "max_reversal_shift": self.max_reversal_shift,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "reversal_strength": self.reversal_strength,
            "shift_threshold": self.shift_threshold,
            "skew_correction": self.skew_correction,
            "entropy_correction": self.entropy_correction,
            "use_adjusted_base": self.use_adjusted_base,
            "max_reversal_shift": self.max_reversal_shift,
        }

    def __repr__(self) -> str:
        return (
            "ContrarianModel(rev={}, thresh={}, skew={}, entropy={}, "
            "adjusted_base={})".format(
                self.reversal_strength,
                self.shift_threshold,
                self.skew_correction,
                self.entropy_correction,
                self.use_adjusted_base,
            )
        )
