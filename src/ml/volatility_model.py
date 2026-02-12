"""
Volatility Regime Model for Elon Musk tweet count prediction markets.

Key insight: Volatility mean-reverts (autocorrelation -0.45).
- After a HIGH volatility period, the next period is calmer.
  -> The market still prices in wide uncertainty -> SHARPEN distribution
- After a LOW volatility period, the next period is wilder.
  -> The market prices in narrow confidence -> WIDEN distribution

This is the OPPOSITE of what a naive volatility model would do!

The model starts from crowd prices (like MarketAdjustedModel) and applies
volatility-regime-dependent adjustments. It does NOT apply momentum/reversion
(those are handled by MarketAdjustedModel). This model focuses purely on the
shape of the distribution: sharper vs wider.

Approaches implemented:
1. Power-law sharpening/widening: P_adjusted = P^alpha / sum(P^alpha)
   - alpha > 1 sharpens (concentrates mass at peaks)
   - alpha < 1 widens (spreads mass toward tails)
2. Blend-toward-uniform widening (from advanced_models)
3. Conditional application: only act in extreme regimes (configurable)
"""

from typing import Optional

from src.ml.base_model import BasePredictionModel

PROB_FLOOR = 1e-6


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities to sum to 1.0, with floor."""
    if not probs:
        return probs
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


def _get_crowd_probs(
    buckets: list[dict],
    context: Optional[dict],
) -> Optional[dict[str, float]]:
    """Extract crowd probabilities from entry_prices in context."""
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

    if all(v < 0.001 for v in filtered.values()):
        return None

    for b in buckets:
        if b["bucket_label"] not in filtered:
            filtered[b["bucket_label"]] = PROB_FLOOR

    return _normalize(filtered)


def _power_law_adjust(
    probs: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """Apply power-law transformation to probability distribution.

    P_adjusted[i] = P[i]^alpha / sum(P[j]^alpha)

    alpha > 1: sharpens (concentrates mass at high-probability buckets)
    alpha < 1: widens (spreads mass toward low-probability buckets)
    alpha = 1: no change
    """
    if abs(alpha - 1.0) < 0.001:
        return probs

    new_probs = {}
    for k, p in probs.items():
        p_safe = max(p, PROB_FLOOR)
        new_probs[k] = p_safe ** alpha

    return _normalize(new_probs)


def _blend_toward_uniform(
    probs: dict[str, float],
    buckets: list[dict],
    blend: float,
) -> dict[str, float]:
    """Blend distribution toward uniform.

    blend > 0: move toward uniform (widen)
    blend < 0: move away from uniform (sharpen via power law)
    """
    n = len(buckets)
    if n == 0 or abs(blend) < 0.001:
        return probs

    uniform_p = 1.0 / n

    if blend >= 0:
        new_probs = {}
        for b in buckets:
            label = b["bucket_label"]
            p_orig = probs.get(label, uniform_p)
            new_probs[label] = p_orig * (1 - blend) + uniform_p * blend
        return _normalize(new_probs)
    else:
        # Negative blend: sharpen via power law
        power = 1.0 + abs(blend)
        return _power_law_adjust(probs, power)


class VolatilityRegimeModel(BasePredictionModel):
    """Volatility regime model exploiting vol mean-reversion.

    Uses the coefficient of variation (cv_14d) or rolling_std_7d to classify
    the current volatility regime:
    - HIGH vol regime (cv above threshold): next period will be CALMER
      -> SHARPEN crowd distribution (concentrate mass)
    - LOW vol regime (cv below threshold): next period will be WILDER
      -> WIDEN crowd distribution (spread mass to tails)
    - NEUTRAL regime: no adjustment (pass through crowd)

    Hyperparameters:
        vol_metric: Which metric to use ("cv_14d", "rolling_std_7d", "rolling_std_14d")
        vol_threshold_high: Above this = high vol regime
        vol_threshold_low: Below this = low vol regime
        sharpen_alpha: Power-law exponent for sharpening (> 1.0)
        widen_alpha: Power-law exponent for widening (< 1.0)
        use_blend: If True, use blend-toward-uniform instead of power-law
        sharpen_blend: Blend factor for sharpening (negative = sharpen)
        widen_blend: Blend factor for widening (positive = widen)
        extreme_only: If True, only apply in top/bottom quartile
        reversion_strength: Scale factor for adjustment magnitude (0-1)
        market_cap: Max ratio of model prob to market prob (prevents extreme bets)
    """

    def __init__(
        self,
        name: str = "volatility_regime",
        version: str = "v1",
        vol_metric: str = "cv_14d",
        vol_threshold_high: float = 0.45,
        vol_threshold_low: float = 0.35,
        sharpen_alpha: float = 1.3,
        widen_alpha: float = 0.8,
        use_blend: bool = False,
        sharpen_blend: float = -0.15,
        widen_blend: float = 0.15,
        extreme_only: bool = False,
        reversion_strength: float = 1.0,
        market_cap: float = 3.0,
    ):
        super().__init__(name=name, version=version)
        self.vol_metric = vol_metric
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.sharpen_alpha = sharpen_alpha
        self.widen_alpha = widen_alpha
        self.use_blend = use_blend
        self.sharpen_blend = sharpen_blend
        self.widen_blend = widen_blend
        self.extreme_only = extreme_only
        self.reversion_strength = reversion_strength
        self.market_cap = market_cap

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return probability distribution across buckets."""
        if not buckets:
            return {}

        # Start from crowd prices
        crowd_probs = _get_crowd_probs(buckets, context)

        if crowd_probs is None:
            # No crowd data: return uniform (we have no edge without crowd)
            return _uniform(buckets)

        # Get volatility metric
        vol_value = self._get_vol_metric(features)

        if vol_value is None:
            # No volatility data: return crowd as-is
            return crowd_probs

        # Determine regime and apply adjustment
        probs = dict(crowd_probs)

        # Classify regime
        regime = self._classify_regime(vol_value)

        if regime == "neutral":
            return crowd_probs

        # Apply mean-reversion logic:
        # HIGH vol -> next period calmer -> SHARPEN
        # LOW vol -> next period wilder -> WIDEN
        if regime == "high":
            probs = self._sharpen(probs, buckets, vol_value)
        elif regime == "low":
            probs = self._widen(probs, buckets, vol_value)

        # Market-relative cap
        if self.market_cap > 0:
            entry_prices = (
                context.get("entry_prices", {}) if context else {}
            )
            if entry_prices:
                for b in buckets:
                    label = b["bucket_label"]
                    mkt = float(entry_prices.get(label, 0.0))
                    if mkt > 0:
                        cap = max(self.market_cap * mkt, mkt + 0.03)
                        if probs.get(label, 0.0) > cap:
                            probs[label] = cap

        return _normalize(probs)

    def _get_vol_metric(self, features: dict) -> Optional[float]:
        """Extract the volatility metric from features."""
        temporal = features.get("temporal", {})
        if not temporal:
            return None

        if self.vol_metric == "cv_14d":
            val = temporal.get("cv_14d")
            if val is not None:
                return float(val)
            return None

        elif self.vol_metric == "rolling_std_7d":
            val = temporal.get("rolling_std_7d")
            if val is not None:
                # Normalize std by rolling avg to get relative volatility
                avg = temporal.get("rolling_avg_7d")
                if avg is not None and float(avg) > 0:
                    return float(val) / float(avg)
                return float(val) / 40.0  # fallback normalization
            return None

        elif self.vol_metric == "rolling_std_14d":
            val = temporal.get("rolling_std_14d")
            if val is not None:
                avg = temporal.get("rolling_avg_14d")
                if avg is not None and float(avg) > 0:
                    return float(val) / float(avg)
                return float(val) / 40.0
            return None

        else:
            # Try the metric name directly
            val = temporal.get(self.vol_metric)
            if val is not None:
                return float(val)
            return None

    def _classify_regime(self, vol_value: float) -> str:
        """Classify volatility regime based on thresholds.

        Note: extreme_only is handled by the grid search using wider
        threshold gaps (e.g., high=0.55, low=0.25) so that fewer events
        are classified as non-neutral.
        """
        if vol_value >= self.vol_threshold_high:
            return "high"
        elif vol_value <= self.vol_threshold_low:
            return "low"
        else:
            return "neutral"

    def _sharpen(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        vol_value: float,
    ) -> dict[str, float]:
        """Sharpen distribution (high vol -> next period calmer).

        Scale the sharpening intensity by how far vol is above threshold
        and by reversion_strength.
        """
        # How far above the high threshold
        excess = vol_value - self.vol_threshold_high
        intensity = min(excess / 0.2, 1.0) * self.reversion_strength

        if self.use_blend:
            blend = self.sharpen_blend * intensity
            return _blend_toward_uniform(probs, buckets, blend)
        else:
            # Power law: alpha interpolates from 1.0 to sharpen_alpha
            alpha = 1.0 + (self.sharpen_alpha - 1.0) * intensity
            return _power_law_adjust(probs, alpha)

    def _widen(
        self,
        probs: dict[str, float],
        buckets: list[dict],
        vol_value: float,
    ) -> dict[str, float]:
        """Widen distribution (low vol -> next period wilder).

        Scale the widening intensity by how far vol is below threshold
        and by reversion_strength.
        """
        # How far below the low threshold
        deficit = self.vol_threshold_low - vol_value
        intensity = min(deficit / 0.15, 1.0) * self.reversion_strength

        if self.use_blend:
            blend = self.widen_blend * intensity
            return _blend_toward_uniform(probs, buckets, blend)
        else:
            # Power law: alpha interpolates from 1.0 to widen_alpha
            alpha = 1.0 + (self.widen_alpha - 1.0) * intensity
            return _power_law_adjust(probs, alpha)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "vol_metric": self.vol_metric,
            "vol_threshold_high": self.vol_threshold_high,
            "vol_threshold_low": self.vol_threshold_low,
            "sharpen_alpha": self.sharpen_alpha,
            "widen_alpha": self.widen_alpha,
            "use_blend": self.use_blend,
            "sharpen_blend": self.sharpen_blend,
            "widen_blend": self.widen_blend,
            "extreme_only": self.extreme_only,
            "reversion_strength": self.reversion_strength,
            "market_cap": self.market_cap,
        }

    def get_hyperparameters(self) -> dict:
        return self.get_config()

    def __repr__(self) -> str:
        return (
            "VolatilityRegimeModel(name={!r}, version={!r}, "
            "metric={!r}, high={}, low={}, "
            "sharpen_a={}, widen_a={})".format(
                self.name, self.version,
                self.vol_metric,
                self.vol_threshold_high,
                self.vol_threshold_low,
                self.sharpen_alpha,
                self.widen_alpha,
            )
        )
