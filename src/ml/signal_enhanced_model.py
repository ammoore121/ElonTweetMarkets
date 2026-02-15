"""
Signal-enhanced tail model for Elon Musk tweet count prediction markets.

Builds on the proven TailBoostModel (structural edge: crowd underprices tails)
but dynamically adjusts the tail boost intensity using financial and attention
signals:

1. **Tesla stock volatility** → high TSLA vol = Elon likely tweeting about markets
2. **DOGE momentum** → DOGE pump/dump = Elon crypto tweet storm
3. **Wikipedia pageview spike** → public attention spike = more tweeting
4. **Tesla drawdown** → stock dropping = defensive tweeting (bad_press signal)

The base tail_boost_factor is modulated by a signal_multiplier derived from
these external features. When multiple signals fire simultaneously, the
model applies a stronger tail boost (crowd is even more likely to be wrong
during high-activity periods).

This model uses the "full" feature group to access financial + attention features.
"""

import math
from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.advanced_models import (
    _get_crowd_probs,
    _normalize,
    _compute_implied_ev,
    _shift_distribution,
    PROB_FLOOR,
)
from src.ml.duration_model import _bucket_midpoint, _classify_duration


class SignalEnhancedTailModel(BasePredictionModel):
    """TailBoost with dynamic signal-based tail intensity.

    The tail_boost_factor is computed as:
        effective_boost = base_boost * (1 + signal_weight * composite_signal)

    Where composite_signal combines:
    - Tesla volatility signal (tsla_volatility_5d > threshold → +1)
    - DOGE momentum signal (|doge_pct_change_1d| > threshold → +1)
    - Wikipedia attention spike (wiki_elon_musk_delta > threshold → +1)
    - Tesla drawdown signal (tsla_drawdown_5d < -threshold → +1)

    Each signal adds 0-1 to the composite, capped at max_signal_boost.
    """

    BASE_TAIL_BOOST = 1.25       # Base tail boost (lower than pure TailBoost 1.30)
    TAIL_THRESHOLD_SD = 1.0
    SIGNAL_WEIGHT = 0.08         # Each signal unit adds 8% more boost
    MAX_SIGNAL_BOOST = 0.32      # Cap at 32% additional boost (4 signals * 8%)

    # Signal activation thresholds
    TSLA_VOL_THRESHOLD = 0.03    # 3% daily vol = elevated
    DOGE_MOMENTUM_THRESHOLD = 0.05  # 5% daily move = significant
    WIKI_SPIKE_THRESHOLD = 0.15  # 15% above 7d average = spike
    TSLA_DRAWDOWN_THRESHOLD = -0.05  # 5% drawdown from 5d high

    # Duration-based shrink (from DurationShrinkModel)
    SHORT_SHRINK = 0.18

    def __init__(
        self,
        name: str = "signal_enhanced_tail",
        version: str = "v1",
        base_tail_boost: float = None,
        tail_threshold_sd: float = None,
        signal_weight: float = None,
        short_shrink: float = None,
    ):
        super().__init__(name=name, version=version)
        if base_tail_boost is not None:
            self.BASE_TAIL_BOOST = base_tail_boost
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd
        if signal_weight is not None:
            self.SIGNAL_WEIGHT = signal_weight
        if short_shrink is not None:
            self.SHORT_SHRINK = short_shrink

    def _compute_signal_score(self, features: dict) -> float:
        """Compute composite signal score from financial + attention features.

        Returns a value in [0, 4] where each signal contributes 0-1.
        """
        score = 0.0
        financial = features.get("financial", {})
        attention = features.get("attention", {})

        # Signal 1: Tesla volatility
        tsla_vol = financial.get("tsla_volatility_5d")
        if tsla_vol is not None and tsla_vol > self.TSLA_VOL_THRESHOLD:
            # Proportional activation (capped at 1)
            score += min(1.0, tsla_vol / (self.TSLA_VOL_THRESHOLD * 2))

        # Signal 2: DOGE momentum (absolute value — both pumps and dumps)
        doge_pct = financial.get("doge_pct_change_1d")
        if doge_pct is not None and abs(doge_pct) > self.DOGE_MOMENTUM_THRESHOLD:
            score += min(1.0, abs(doge_pct) / (self.DOGE_MOMENTUM_THRESHOLD * 2))

        # Signal 3: Wikipedia attention spike
        wiki_delta = attention.get("wiki_elon_musk_delta")
        if wiki_delta is not None and wiki_delta > self.WIKI_SPIKE_THRESHOLD:
            score += min(1.0, wiki_delta / (self.WIKI_SPIKE_THRESHOLD * 2))

        # Signal 4: Tesla drawdown (negative = stock falling)
        tsla_dd = financial.get("tsla_drawdown_5d")
        if tsla_dd is not None and tsla_dd < self.TSLA_DRAWDOWN_THRESHOLD:
            score += min(1.0, abs(tsla_dd) / abs(self.TSLA_DRAWDOWN_THRESHOLD * 2))

        return score

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

        probs = dict(crowd_probs)

        # Step 1: Duration-based shrink for short events
        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        if _classify_duration(duration_days) == "short" and self.SHORT_SHRINK > 0:
            probs = _shift_distribution(probs, buckets, -self.SHORT_SHRINK)

        # Step 2: Compute signal-enhanced tail boost
        signal_score = self._compute_signal_score(features)
        signal_boost = min(signal_score * self.SIGNAL_WEIGHT, self.MAX_SIGNAL_BOOST)
        effective_boost = self.BASE_TAIL_BOOST * (1 + signal_boost)

        # Step 3: Apply tail boosting
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
                boosted[label] = p * effective_boost
            else:
                boosted[label] = p

        return _normalize(boosted)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "signal_weight": self.SIGNAL_WEIGHT,
            "max_signal_boost": self.MAX_SIGNAL_BOOST,
            "short_shrink": self.SHORT_SHRINK,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "signal_weight": self.SIGNAL_WEIGHT,
            "short_shrink": self.SHORT_SHRINK,
        }


class SignalEnhancedTailModelV4(SignalEnhancedTailModel):
    """V4: 7-signal composite with VIX, Google Trends, and Crypto Fear & Greed.

    Extends V3's 4-signal approach with 3 new signals from newly added data
    sources. Uses lower per-signal weight (0.05 vs 0.06-0.08) since there
    are now 7 signals instead of 4.

    Signal composite (7 signals, 0-1 each):
        1. Tesla volatility (existing) — tsla_volatility_5d > 0.03
        2. DOGE momentum (existing) — |doge_pct_change_1d| > 0.05
        3. Wikipedia spike (existing) — wiki_elon_musk_delta > 0.15
        4. Tesla drawdown (existing) — tsla_drawdown_5d < -0.05
        5. VIX stress (NEW) — vix_close > 25 OR vix_ma5_ratio > 1.10
        6. Google Trends momentum (NEW) — gt_elon_musk_delta > 0.15
        7. Crypto Fear & Greed extreme (NEW) — fg_value < 25 OR > 75
    """

    # Adjusted for 7 signals
    SIGNAL_WEIGHT = 0.05
    MAX_SIGNAL_BOOST = 0.35  # 7 signals * 0.05

    # New signal thresholds
    VIX_HIGH_THRESHOLD = 25.0
    VIX_MA5_RATIO_THRESHOLD = 1.10
    GT_SPIKE_THRESHOLD = 0.15
    CRYPTO_FG_EXTREME_LOW = 25
    CRYPTO_FG_EXTREME_HIGH = 75

    def __init__(
        self,
        name: str = "signal_enhanced_tail",
        version: str = "v4",
        base_tail_boost: float = None,
        tail_threshold_sd: float = None,
        signal_weight: float = None,
        short_shrink: float = None,
        vix_high_threshold: float = None,
        gt_spike_threshold: float = None,
        crypto_fg_extreme_low: float = None,
        crypto_fg_extreme_high: float = None,
    ):
        super().__init__(
            name=name,
            version=version,
            base_tail_boost=base_tail_boost,
            tail_threshold_sd=tail_threshold_sd,
            signal_weight=signal_weight,
            short_shrink=short_shrink,
        )
        if vix_high_threshold is not None:
            self.VIX_HIGH_THRESHOLD = vix_high_threshold
        if gt_spike_threshold is not None:
            self.GT_SPIKE_THRESHOLD = gt_spike_threshold
        if crypto_fg_extreme_low is not None:
            self.CRYPTO_FG_EXTREME_LOW = crypto_fg_extreme_low
        if crypto_fg_extreme_high is not None:
            self.CRYPTO_FG_EXTREME_HIGH = crypto_fg_extreme_high

    def _compute_signal_score(self, features: dict) -> float:
        """Compute 7-signal composite score.

        Returns a value in [0, 7] where each signal contributes 0-1.
        """
        # Start with the 4 existing signals from parent
        score = super()._compute_signal_score(features)

        financial = features.get("financial", {})
        trends = features.get("trends", {})

        # Signal 5: VIX stress (high absolute level OR spike above MA5)
        vix_close = financial.get("vix_close")
        vix_ma5_ratio = financial.get("vix_ma5_ratio")
        if vix_close is not None:
            if vix_close > self.VIX_HIGH_THRESHOLD:
                # Proportional: VIX 25 → 0, VIX 50 → 1
                score += min(1.0, (vix_close - self.VIX_HIGH_THRESHOLD) / self.VIX_HIGH_THRESHOLD)
            elif vix_ma5_ratio is not None and vix_ma5_ratio > self.VIX_MA5_RATIO_THRESHOLD:
                # Spike above MA5: ratio 1.10 → 0, ratio 1.30 → 1
                score += min(1.0, (vix_ma5_ratio - self.VIX_MA5_RATIO_THRESHOLD) / 0.20)

        # Signal 6: Google Trends momentum (Elon Musk search spike)
        gt_delta = trends.get("gt_elon_musk_delta")
        if gt_delta is not None and gt_delta > self.GT_SPIKE_THRESHOLD:
            score += min(1.0, gt_delta / (self.GT_SPIKE_THRESHOLD * 2))

        # Signal 7: Crypto Fear & Greed extreme (both directions)
        fg_value = financial.get("crypto_fg_value")
        if fg_value is not None:
            if fg_value < self.CRYPTO_FG_EXTREME_LOW:
                # Extreme fear: 25 → 0, 0 → 1
                score += min(1.0, (self.CRYPTO_FG_EXTREME_LOW - fg_value) / self.CRYPTO_FG_EXTREME_LOW)
            elif fg_value > self.CRYPTO_FG_EXTREME_HIGH:
                # Extreme greed: 75 → 0, 100 → 1
                score += min(1.0, (fg_value - self.CRYPTO_FG_EXTREME_HIGH) / (100 - self.CRYPTO_FG_EXTREME_HIGH))

        return score

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "vix_high_threshold": self.VIX_HIGH_THRESHOLD,
            "vix_ma5_ratio_threshold": self.VIX_MA5_RATIO_THRESHOLD,
            "gt_spike_threshold": self.GT_SPIKE_THRESHOLD,
            "crypto_fg_extreme_low": self.CRYPTO_FG_EXTREME_LOW,
            "crypto_fg_extreme_high": self.CRYPTO_FG_EXTREME_HIGH,
        })
        return config

    def get_hyperparameters(self) -> dict:
        params = super().get_hyperparameters()
        params.update({
            "vix_high_threshold": self.VIX_HIGH_THRESHOLD,
            "gt_spike_threshold": self.GT_SPIKE_THRESHOLD,
            "crypto_fg_extreme_low": self.CRYPTO_FG_EXTREME_LOW,
            "crypto_fg_extreme_high": self.CRYPTO_FG_EXTREME_HIGH,
        })
        return params
