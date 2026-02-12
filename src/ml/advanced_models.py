"""
Advanced prediction models for Elon Musk tweet count prediction markets.

Three models designed to beat the crowd baseline (Brier ~0.81):

1. RegimeAwareModel: Detects tweeting regimes (high/medium/low) from trailing
   XTracker data and uses a mixture of Negative Binomial distributions weighted
   by regime probabilities. Key idea: when recent variance is high, spread mass
   wider; when trend is strong, shift the distribution.

2. MarketAdjustedModel: Starts from crowd/market prices and applies small
   feature-based adjustments. This is the most promising approach since it
   leverages existing market information rather than replacing it.

3. EnsembleModel: Weighted average of RegimeAwareModel and MarketAdjustedModel.

Usage:
    model = RegimeAwareModel()
    probs = model.predict(features, buckets, context={"duration_days": 7})

    model = MarketAdjustedModel()
    probs = model.predict(features, buckets, context={"entry_prices": {...}})

    model = EnsembleModel()
    probs = model.predict(features, buckets, context={"duration_days": 7, "entry_prices": {...}})
"""

import math
from typing import Optional

from scipy.stats import nbinom

from src.ml.base_model import BasePredictionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROB_FLOOR = 1e-6


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities to sum to 1.0, with floor."""
    if not probs:
        return probs
    # Floor
    probs = {k: max(v, PROB_FLOOR) for k, v in probs.items()}
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


def _uniform(buckets: list[dict]) -> dict[str, float]:
    """Return uniform distribution across buckets."""
    n = len(buckets)
    if n == 0:
        return {}
    p = 1.0 / n
    return {b["bucket_label"]: p for b in buckets}


def _negbin_bucket_probs(
    total_mean: float,
    total_var: float,
    buckets: list[dict],
) -> dict[str, float]:
    """Compute per-bucket probabilities using Negative Binomial CDF.

    Negative Binomial parameterization (scipy convention):
        mean = n * (1 - p) / p
        var  = n * (1 - p) / p^2

    Solving for (n, p):
        p = mean / var
        n = mean^2 / (var - mean)
    """
    # Guard against degenerate cases
    if total_var <= total_mean:
        total_var = total_mean * 1.1

    p_nb = total_mean / total_var
    n_nb = (total_mean ** 2) / (total_var - total_mean)

    # Clamp parameters to valid ranges
    p_nb = max(min(p_nb, 0.9999), 1e-6)
    n_nb = max(n_nb, 0.01)

    probs = {}
    for bucket in buckets:
        lower = int(bucket["lower_bound"])
        upper = int(bucket["upper_bound"])

        if upper >= 99999:
            # Open-ended upper bucket: P(X >= lower)
            if lower <= 0:
                prob = 1.0
            else:
                prob = 1.0 - nbinom.cdf(lower - 1, n_nb, p_nb)
        elif lower <= 0:
            # Bottom bucket: P(X <= upper)
            prob = nbinom.cdf(upper, n_nb, p_nb)
        else:
            # Middle bucket: P(lower <= X <= upper)
            prob = nbinom.cdf(upper, n_nb, p_nb) - nbinom.cdf(
                lower - 1, n_nb, p_nb
            )

        probs[bucket["bucket_label"]] = max(prob, PROB_FLOOR)

    return _normalize(probs)


def _get_crowd_probs(
    buckets: list[dict],
    context: Optional[dict],
) -> Optional[dict[str, float]]:
    """Extract crowd probabilities from entry_prices in context.

    Returns normalized probabilities or None if not available.
    """
    if not context or "entry_prices" not in context:
        return None

    prices = context["entry_prices"]
    if not prices or not isinstance(prices, dict):
        return None

    bucket_labels = {b["bucket_label"] for b in buckets}
    filtered = {
        k: max(float(v), PROB_FLOOR)
        for k, v in prices.items()
        if k in bucket_labels
    }

    if not filtered:
        return None

    # Check if all prices are zero or near-zero (no real market data)
    if all(v < 0.001 for v in filtered.values()):
        return None

    # Add floor for missing buckets
    for b in buckets:
        if b["bucket_label"] not in filtered:
            filtered[b["bucket_label"]] = PROB_FLOOR

    return _normalize(filtered)


def _compute_implied_ev(buckets: list[dict], probs: dict[str, float]) -> float:
    """Compute expected value (midpoint-weighted) from a probability distribution."""
    ev = 0.0
    for b in buckets:
        label = b["bucket_label"]
        lower = int(b["lower_bound"])
        upper = int(b["upper_bound"])
        p = probs.get(label, 0.0)
        if upper >= 99999:
            # For open-ended bucket, use lower + half of typical bucket width
            # Estimate bucket width from other buckets
            widths = [
                int(bb["upper_bound"]) - int(bb["lower_bound"])
                for bb in buckets
                if int(bb["upper_bound"]) < 99999
            ]
            typical_width = sum(widths) / len(widths) if widths else 25
            midpoint = lower + typical_width / 2
        elif lower <= 0:
            midpoint = upper / 2
        else:
            midpoint = (lower + upper) / 2
        ev += midpoint * p
    return ev


def _shift_distribution(
    probs: dict[str, float],
    buckets: list[dict],
    shift_amount: float,
) -> dict[str, float]:
    """Shift probability mass toward higher or lower buckets.

    shift_amount > 0: shift toward higher buckets (counts increasing)
    shift_amount < 0: shift toward lower buckets (counts decreasing)

    The shift is implemented by redistributing mass proportional to
    the distance of each bucket's midpoint from the current expected value,
    in the direction of the shift.
    """
    if abs(shift_amount) < 0.001:
        return probs

    # Sort buckets by lower_bound
    sorted_buckets = sorted(buckets, key=lambda b: int(b["lower_bound"]))
    n = len(sorted_buckets)

    if n < 2:
        return probs

    # Compute midpoints
    midpoints = {}
    for b in sorted_buckets:
        label = b["bucket_label"]
        lower = int(b["lower_bound"])
        upper = int(b["upper_bound"])
        if upper >= 99999:
            widths = [
                int(bb["upper_bound"]) - int(bb["lower_bound"])
                for bb in buckets
                if int(bb["upper_bound"]) < 99999
            ]
            typical_width = sum(widths) / len(widths) if widths else 25
            midpoints[label] = lower + typical_width / 2
        else:
            midpoints[label] = (lower + upper) / 2

    # Compute current EV
    current_ev = sum(probs.get(l, 0) * midpoints[l] for l in midpoints)

    # Apply shift by moving mass in the direction indicated
    # shift_amount is a fractional shift: 0.1 means shift 10% of mass
    # toward higher buckets
    new_probs = dict(probs)

    # Create an ordered list of labels by midpoint
    ordered_labels = [b["bucket_label"] for b in sorted_buckets]

    # For each pair of adjacent buckets, transfer mass in the shift direction
    # This is essentially a diffusion/advection step
    abs_shift = min(abs(shift_amount), 0.3)  # Cap at 30% transfer

    if shift_amount > 0:
        # Shift mass from lower to higher buckets
        for i in range(n - 1):
            label_from = ordered_labels[i]
            label_to = ordered_labels[i + 1]
            transfer = new_probs.get(label_from, 0) * abs_shift
            new_probs[label_from] = new_probs.get(label_from, 0) - transfer
            new_probs[label_to] = new_probs.get(label_to, 0) + transfer
    else:
        # Shift mass from higher to lower buckets
        for i in range(n - 1, 0, -1):
            label_from = ordered_labels[i]
            label_to = ordered_labels[i - 1]
            transfer = new_probs.get(label_from, 0) * abs_shift
            new_probs[label_from] = new_probs.get(label_from, 0) - transfer
            new_probs[label_to] = new_probs.get(label_to, 0) + transfer

    return _normalize(new_probs)


def _widen_distribution(
    probs: dict[str, float],
    buckets: list[dict],
    widen_factor: float,
) -> dict[str, float]:
    """Widen (or narrow) the probability distribution around its mean.

    widen_factor > 1: spread mass toward tails (increase uncertainty)
    widen_factor < 1: concentrate mass toward center (decrease uncertainty)
    widen_factor = 1: no change

    Implementation: blend current distribution toward uniform by widen_factor.
    """
    if abs(widen_factor - 1.0) < 0.001:
        return probs

    n = len(buckets)
    if n == 0:
        return probs

    uniform_p = 1.0 / n

    # widen_factor > 1 means blend toward uniform
    # Compute blend weight: how much to mix toward uniform
    # widen_factor = 1.0 -> blend = 0 (keep original)
    # widen_factor = 2.0 -> blend = 0.5 (halfway to uniform)
    # Cap the blend at 0.5 to avoid losing too much signal
    blend = min(max((widen_factor - 1.0) / 2.0, -0.3), 0.5)

    new_probs = {}
    for b in buckets:
        label = b["bucket_label"]
        p_orig = probs.get(label, uniform_p)
        if blend >= 0:
            new_probs[label] = p_orig * (1 - blend) + uniform_p * blend
        else:
            # Negative blend: sharpen (concentrate more mass at peaks)
            # Raise probabilities to a power > 1
            power = 1.0 + abs(blend)
            new_probs[label] = p_orig ** power

    return _normalize(new_probs)


# ---------------------------------------------------------------------------
# Model 1: RegimeAwareModel
# ---------------------------------------------------------------------------

class RegimeAwareModel(BasePredictionModel):
    """Regime-aware model using mixture of Negative Binomial distributions.

    Detects the current tweeting regime (high/medium/low) from trailing
    XTracker data and produces a mixture distribution:

    - HIGH regime: daily mean ~ 55-80 tweets (weekly ~385-560)
    - MEDIUM regime: daily mean ~ 35-55 tweets (weekly ~245-385)
    - LOW regime: daily mean ~ 10-35 tweets (weekly ~70-245)

    When temporal features are available, uses them to weight regimes.
    When not available, falls back to a prior based on market implied EV
    or to a wide NegBin distribution.

    Key innovations:
    - cv_14d (coefficient of variation) detects regime instability
    - trend_7d detects momentum
    - regime_ratio detects if we're transitioning between regimes
    """

    # Historical regime parameters (daily tweet rates)
    # Derived from XTracker data analysis
    REGIMES = {
        "low": {"mean": 20.0, "std": 8.0, "prior_weight": 0.20},
        "medium": {"mean": 40.0, "std": 12.0, "prior_weight": 0.35},
        "high": {"mean": 65.0, "std": 18.0, "prior_weight": 0.30},
        "very_high": {"mean": 85.0, "std": 20.0, "prior_weight": 0.15},
    }

    def __init__(self, name: str = "regime_aware", version: str = "v1"):
        super().__init__(name=name, version=version)

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return probability distribution across buckets."""
        if not buckets:
            return {}

        # Extract duration from context
        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        temporal = features.get("temporal", {})
        market = features.get("market", {})

        # Compute regime weights based on available features
        regime_weights = self._compute_regime_weights(temporal, market, duration_days)

        # For each regime, compute NegBin bucket probabilities
        mixture_probs = {b["bucket_label"]: 0.0 for b in buckets}

        for regime_name, regime_params in self.REGIMES.items():
            weight = regime_weights.get(regime_name, 0.0)
            if weight < 0.001:
                continue

            daily_mean = regime_params["mean"]
            daily_std = regime_params["std"]

            # Apply trend adjustment to mean
            trend = self._get_trend(temporal)
            if trend is not None:
                # trend is daily slope: positive = increasing tweets
                # Adjust mean by trend * days into the future (conservative: 3 days)
                daily_mean += trend * 3.0

            # Scale to event duration
            total_mean = daily_mean * duration_days
            total_var = (daily_std ** 2) * duration_days

            # If cv is high, inflate variance further
            cv = self._get_cv(temporal)
            if cv is not None and cv > 0.3:
                # High CV means regime is unstable - widen distribution
                variance_multiplier = 1.0 + (cv - 0.3) * 2.0
                total_var *= variance_multiplier

            # Ensure overdispersion
            if total_var <= total_mean:
                total_var = total_mean * 1.2

            regime_probs = _negbin_bucket_probs(total_mean, total_var, buckets)

            for label, p in regime_probs.items():
                mixture_probs[label] += weight * p

        return _normalize(mixture_probs)

    def _compute_regime_weights(
        self,
        temporal: dict,
        market: dict,
        duration_days: int,
    ) -> dict[str, float]:
        """Compute weights for each regime based on available features."""
        weights = {k: v["prior_weight"] for k, v in self.REGIMES.items()}

        rolling_avg = self._get_rolling_avg(temporal)
        rolling_std = self._get_rolling_std(temporal)

        if rolling_avg is not None and rolling_avg > 0:
            # We have temporal data - use it to strongly weight the right regime
            # Compute likelihood of observed avg under each regime
            for regime_name, regime_params in self.REGIMES.items():
                regime_mean = regime_params["mean"]
                regime_std = regime_params["std"]

                # Gaussian likelihood of observed daily avg
                z = (rolling_avg - regime_mean) / regime_std
                likelihood = math.exp(-0.5 * z * z)
                weights[regime_name] *= likelihood

        elif market:
            # No temporal data but we have market-implied EV
            crowd_ev = market.get("crowd_implied_ev")
            if crowd_ev is not None and crowd_ev > 0:
                # Convert to daily rate
                daily_implied = crowd_ev / max(duration_days, 1)

                for regime_name, regime_params in self.REGIMES.items():
                    regime_mean = regime_params["mean"]
                    regime_std = regime_params["std"]
                    z = (daily_implied - regime_mean) / regime_std
                    likelihood = math.exp(-0.5 * z * z)
                    weights[regime_name] *= likelihood

        # If regime_ratio suggests transition, spread mass wider
        regime_ratio = temporal.get("regime_ratio") if temporal else None
        if regime_ratio is not None:
            if regime_ratio > 1.3:
                # Trending up: boost high regimes
                weights["high"] *= 1.5
                weights["very_high"] *= 1.5
            elif regime_ratio < 0.7:
                # Trending down: boost low regimes
                weights["low"] *= 1.5
                weights["medium"] *= 1.2

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    @staticmethod
    def _get_rolling_avg(temporal: dict) -> Optional[float]:
        if not temporal:
            return None
        for key in ["rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d"]:
            val = temporal.get(key)
            if val is not None:
                return float(val)
        return None

    @staticmethod
    def _get_rolling_std(temporal: dict) -> Optional[float]:
        if not temporal:
            return None
        for key in ["rolling_std_7d", "rolling_std_14d"]:
            val = temporal.get(key)
            if val is not None:
                return float(val)
        return None

    @staticmethod
    def _get_trend(temporal: dict) -> Optional[float]:
        if not temporal:
            return None
        for key in ["trend_7d", "trend_14d"]:
            val = temporal.get(key)
            if val is not None:
                return float(val)
        return None

    @staticmethod
    def _get_cv(temporal: dict) -> Optional[float]:
        if not temporal:
            return None
        val = temporal.get("cv_14d")
        if val is not None:
            return float(val)
        return None

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "regimes": {k: {"mean": v["mean"], "std": v["std"]} for k, v in self.REGIMES.items()},
        }

    def __repr__(self) -> str:
        return "RegimeAwareModel(name={!r}, version={!r})".format(self.name, self.version)


# ---------------------------------------------------------------------------
# Model 2: MarketAdjustedModel
# ---------------------------------------------------------------------------

class MarketAdjustedModel(BasePredictionModel):
    """Market-adjusted model: starts from crowd prices and applies feature-based adjustments.

    This model leverages the insight that crowd prices already encode significant
    information (Brier 0.81) and makes small, principled adjustments:

    1. Momentum adjustment: If rolling trend suggests counts are increasing or
       decreasing, shift mass accordingly.
    2. Variance adjustment: If GDELT news volume is spiking or cv_14d is high,
       widen the distribution (more uncertainty favors tails over peaks).
    3. Mean reversion: If crowd_implied_ev is far from the rolling average,
       regress slightly toward the rolling average.
    4. Regime mismatch: If temporal data suggests a different regime than
       the crowd expects, apply a moderate correction.

    Key principle: make SMALL adjustments. The crowd is usually approximately
    right; we only need to be slightly better on average.
    """

    # Tunable hyperparameters (optimized via grid search on gold-tier events)
    MOMENTUM_STRENGTH = 0.16      # How aggressively to shift for momentum
    WIDEN_STRENGTH = 0.18         # How aggressively to widen for uncertainty
    REVERSION_STRENGTH = 0.28    # How aggressively to regress to rolling avg
    REGIME_CORRECTION = 0.00     # Regime correction disabled (hurts calibration)

    def __init__(self, name: str = "market_adjusted", version: str = "v1"):
        super().__init__(name=name, version=version)

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return probability distribution across buckets."""
        if not buckets:
            return {}

        # Extract duration
        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        # Start from crowd prices
        crowd_probs = _get_crowd_probs(buckets, context)

        if crowd_probs is None:
            # No crowd data: fall back to a regime-aware NegBin
            return self._fallback_predict(features, buckets, duration_days)

        # Extract features
        temporal = features.get("temporal", {})
        gdelt = features.get("media", {}) or features.get("gdelt", {})
        market = features.get("market", {})

        probs = dict(crowd_probs)

        # --- Adjustment 1: Momentum shift ---
        probs = self._apply_momentum_shift(probs, buckets, temporal, duration_days)

        # --- Adjustment 2: Variance/uncertainty widening ---
        probs = self._apply_uncertainty_widening(probs, buckets, temporal, gdelt)

        # --- Adjustment 3: Mean reversion ---
        probs = self._apply_mean_reversion(
            probs, buckets, temporal, market, duration_days
        )

        # --- Adjustment 4: Regime mismatch correction ---
        probs = self._apply_regime_correction(
            probs, buckets, temporal, market, duration_days
        )

        return _normalize(probs)

    def _apply_momentum_shift(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        temporal: dict,
        duration_days: int,
    ) -> dict[str, float]:
        """If trend is significant, shift mass toward direction of trend."""
        if not temporal:
            return probs

        trend = None
        for key in ["trend_7d", "trend_14d"]:
            val = temporal.get(key)
            if val is not None:
                trend = float(val)
                break

        if trend is None:
            return probs

        # Trend is daily slope. A trend of +2 means ~2 more tweets/day than before.
        # Scale by how impactful this is relative to the daily average.
        rolling_avg = None
        for key in ["rolling_avg_7d", "rolling_avg_14d"]:
            val = temporal.get(key)
            if val is not None:
                rolling_avg = float(val)
                break

        if rolling_avg is not None and rolling_avg > 0:
            # Relative trend: if trend is 10% of daily avg, that's significant
            relative_trend = trend / rolling_avg
        else:
            # No rolling avg, use absolute threshold
            relative_trend = trend / 40.0  # Assume ~40 tweets/day average

        # Convert to shift amount: capped at moderate levels
        shift = relative_trend * self.MOMENTUM_STRENGTH
        shift = max(min(shift, 0.15), -0.15)

        if abs(shift) > 0.005:
            probs = _shift_distribution(probs, buckets, shift)

        return probs

    def _apply_uncertainty_widening(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        temporal: dict,
        gdelt: dict,
    ) -> dict[str, float]:
        """Widen distribution when uncertainty signals are high."""
        widen_factor = 1.0

        # Signal 1: High coefficient of variation in recent tweets
        if temporal:
            cv = temporal.get("cv_14d")
            if cv is not None:
                cv = float(cv)
                if cv > 0.35:
                    # High variability: widen
                    widen_factor += (cv - 0.35) * self.WIDEN_STRENGTH * 5.0

        # Signal 2: GDELT news volume spike (more news = more uncertainty)
        if gdelt:
            elon_delta = gdelt.get("elon_musk_vol_delta")
            if elon_delta is not None:
                elon_delta = float(elon_delta)
                if elon_delta > 0.05:
                    # News volume increasing significantly
                    widen_factor += elon_delta * self.WIDEN_STRENGTH * 3.0

            # Also check absolute GDELT volume
            elon_vol_1d = gdelt.get("elon_musk_vol_1d")
            if elon_vol_1d is not None:
                elon_vol_1d = float(elon_vol_1d)
                if elon_vol_1d > 0.5:
                    # Very high news volume
                    widen_factor += (elon_vol_1d - 0.5) * self.WIDEN_STRENGTH * 2.0

        # Signal 3: Regime ratio far from 1.0 (transitioning between regimes)
        if temporal:
            regime_ratio = temporal.get("regime_ratio")
            if regime_ratio is not None:
                regime_ratio = float(regime_ratio)
                deviation = abs(regime_ratio - 1.0)
                if deviation > 0.2:
                    widen_factor += deviation * self.WIDEN_STRENGTH * 2.0

        widen_factor = max(min(widen_factor, 2.0), 0.8)  # Cap widening

        if abs(widen_factor - 1.0) > 0.01:
            probs = _widen_distribution(probs, buckets, widen_factor)

        return probs

    def _apply_mean_reversion(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        temporal: dict,
        market: dict,
        duration_days: int,
    ) -> dict[str, float]:
        """If crowd EV diverges from rolling average, regress toward the average."""
        if not temporal or not market:
            return probs

        crowd_ev = market.get("crowd_implied_ev")
        if crowd_ev is None:
            return probs
        crowd_ev = float(crowd_ev)

        # Get model's estimate of expected total from temporal features
        rolling_avg = None
        for key in ["rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d"]:
            val = temporal.get(key)
            if val is not None:
                rolling_avg = float(val)
                break

        if rolling_avg is None or rolling_avg <= 0:
            return probs

        model_ev = rolling_avg * duration_days

        # How far is the crowd from our rolling average estimate?
        if crowd_ev <= 0:
            return probs

        divergence = (crowd_ev - model_ev) / crowd_ev
        # divergence > 0: crowd thinks higher than rolling avg
        # divergence < 0: crowd thinks lower than rolling avg

        # Only apply reversion if divergence is meaningful
        if abs(divergence) < 0.05:
            return probs

        # Shift distribution toward model_ev (away from crowd_ev)
        # The shift direction is opposite to divergence
        shift = -divergence * self.REVERSION_STRENGTH
        shift = max(min(shift, 0.12), -0.12)

        if abs(shift) > 0.005:
            probs = _shift_distribution(probs, buckets, shift)

        return probs

    def _apply_regime_correction(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        temporal: dict,
        market: dict,
        duration_days: int,
    ) -> dict[str, float]:
        """Detect regime mismatch between crowd and temporal signals."""
        if not temporal:
            return probs

        rolling_avg = None
        for key in ["rolling_avg_7d", "rolling_avg_14d"]:
            val = temporal.get(key)
            if val is not None:
                rolling_avg = float(val)
                break

        if rolling_avg is None or rolling_avg <= 0:
            return probs

        rolling_std = None
        for key in ["rolling_std_7d", "rolling_std_14d"]:
            val = temporal.get(key)
            if val is not None:
                rolling_std = float(val)
                break

        if rolling_std is None:
            return probs

        # Compute what a NegBin from temporal features would predict
        total_mean = rolling_avg * duration_days
        total_var = (rolling_std ** 2) * duration_days

        if total_var <= total_mean:
            total_var = total_mean * 1.2

        model_probs = _negbin_bucket_probs(total_mean, total_var, buckets)

        # Blend a small amount of model probs into crowd probs
        blend = self.REGIME_CORRECTION  # 10% model, 90% adjusted crowd

        new_probs = {}
        for b in buckets:
            label = b["bucket_label"]
            crowd_p = probs.get(label, PROB_FLOOR)
            model_p = model_probs.get(label, PROB_FLOOR)
            new_probs[label] = crowd_p * (1 - blend) + model_p * blend

        return _normalize(new_probs)

    def _fallback_predict(
        self,
        features: dict,
        buckets: list[dict],
        duration_days: int,
    ) -> dict[str, float]:
        """Fallback prediction when no crowd data is available.

        Uses a wider NegBin based on available signals.
        """
        temporal = features.get("temporal", {})
        market = features.get("market", {})

        # Try to get daily mean from temporal
        daily_mean = None
        for key in ["rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d"]:
            if temporal:
                val = temporal.get(key)
                if val is not None:
                    daily_mean = float(val)
                    break

        if daily_mean is None or daily_mean <= 0:
            # Try market implied EV
            if market:
                crowd_ev = market.get("crowd_implied_ev")
                if crowd_ev is not None and float(crowd_ev) > 0:
                    daily_mean = float(crowd_ev) / max(duration_days, 1)

        if daily_mean is None or daily_mean <= 0:
            # No data at all - use a wide prior centered around historical median
            daily_mean = 45.0

        # Get daily std
        daily_std = None
        if temporal:
            for key in ["rolling_std_7d", "rolling_std_14d"]:
                val = temporal.get(key)
                if val is not None:
                    daily_std = float(val)
                    break

        if daily_std is None or daily_std <= 0:
            daily_std = daily_mean * 0.4  # Assume 40% CV if unknown

        total_mean = daily_mean * duration_days
        total_var = (daily_std ** 2) * duration_days

        # Inflate variance since we have no crowd data to calibrate against
        total_var *= 1.5

        if total_var <= total_mean:
            total_var = total_mean * 1.3

        return _negbin_bucket_probs(total_mean, total_var, buckets)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "momentum_strength": self.MOMENTUM_STRENGTH,
            "widen_strength": self.WIDEN_STRENGTH,
            "reversion_strength": self.REVERSION_STRENGTH,
            "regime_correction": self.REGIME_CORRECTION,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "momentum_strength": self.MOMENTUM_STRENGTH,
            "widen_strength": self.WIDEN_STRENGTH,
            "reversion_strength": self.REVERSION_STRENGTH,
            "regime_correction": self.REGIME_CORRECTION,
        }

    def __repr__(self) -> str:
        return "MarketAdjustedModel(name={!r}, version={!r})".format(self.name, self.version)


# ---------------------------------------------------------------------------
# Model 3: EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel(BasePredictionModel):
    """Ensemble of RegimeAwareModel and MarketAdjustedModel.

    Combines predictions from both models using configurable weights.
    The MarketAdjustedModel gets more weight by default since it leverages
    crowd prices which are already strong.
    """

    def __init__(
        self,
        name: str = "ensemble",
        version: str = "v1",
        regime_weight: float = 0.10,
        adjusted_weight: float = 0.90,
    ):
        super().__init__(name=name, version=version)
        self.regime_weight = regime_weight
        self.adjusted_weight = adjusted_weight
        self.regime_model = RegimeAwareModel()
        self.adjusted_model = MarketAdjustedModel()

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return weighted-average probability distribution across buckets.

        When crowd/market prices are available, the MarketAdjustedModel already
        incorporates regime signals via its momentum, mean-reversion, and
        uncertainty adjustments.  Using the raw RegimeAwareModel on top would
        contaminate the ensemble with grossly mis-calibrated tail probabilities
        (the regime mixture assigns 60-100% to extreme tails that the market
        correctly prices near zero).  Therefore, when crowd data exists we set
        the regime weight to 0 and rely entirely on the MarketAdjustedModel.

        Additionally, a market-relative cap prevents any single bucket from
        exceeding 3x the market price (or market + 3%), which guards against
        false edge signals on extreme tails.
        """
        if not buckets:
            return {}

        regime_probs = self.regime_model.predict(features, buckets, context)
        adjusted_probs = self.adjusted_model.predict(features, buckets, context)

        # Determine weights - when crowd data is available the adjusted model
        # is far better calibrated; the regime model's fixed-mean mixture
        # creates massive tail over-estimation that contaminates the ensemble.
        crowd_probs = _get_crowd_probs(buckets, context)
        if crowd_probs is not None:
            # Crowd data available: trust the market-adjusted model fully.
            # The regime model's tail contamination causes 100% loss rate.
            r_weight = 0.0
            a_weight = 1.0
        else:
            # No crowd data: lean more on regime model
            r_weight = 0.60
            a_weight = 0.40

        # Weighted average
        ensemble_probs = {}
        for b in buckets:
            label = b["bucket_label"]
            p_regime = regime_probs.get(label, PROB_FLOOR)
            p_adjusted = adjusted_probs.get(label, PROB_FLOOR)
            ensemble_probs[label] = r_weight * p_regime + a_weight * p_adjusted

        # Market-relative cap: prevent any bucket from exceeding
        # max(3 * market_price, market_price + 0.03).  This guards against
        # extreme tail over-estimation even if weights are changed later.
        if crowd_probs is not None:
            entry_prices = (
                context.get("entry_prices", {}) if context else {}
            )
            if entry_prices:
                for b in buckets:
                    label = b["bucket_label"]
                    mkt = entry_prices.get(label, 0.0)
                    if mkt > 0:
                        cap = max(3.0 * mkt, mkt + 0.03)
                        if ensemble_probs.get(label, 0.0) > cap:
                            ensemble_probs[label] = cap

        return _normalize(ensemble_probs)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "regime_weight": self.regime_weight,
            "adjusted_weight": self.adjusted_weight,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "regime_weight": self.regime_weight,
            "adjusted_weight": self.adjusted_weight,
        }

    def __repr__(self) -> str:
        return "EnsembleModel(name={!r}, version={!r}, regime_w={}, adjusted_w={})".format(
            self.name, self.version, self.regime_weight, self.adjusted_weight
        )
