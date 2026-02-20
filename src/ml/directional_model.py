"""
Directional signal model for Elon Musk tweet count prediction markets.

Unlike SignalEnhancedTailModel which uses abs() on all signals (symmetric
tail widening), this model uses the SIGN of external signals to predict
whether Elon will tweet MORE or FEWER than crowd consensus, then shifts
the distribution accordingly before applying tail boost.

Directional signals (empirically validated):
  UP (more tweets):
    - DOGE pump (signed, not abs) — crypto excitement = tweeting
    - Tesla drawdown — stock crash = defensive/reactive tweeting
    - Negative media tone (r=-0.51) — bad press = more tweets
    - Wikipedia/Google Trends spike — attention = more tweeting
  DOWN (fewer tweets):
    - SpaceX launches trailing 7d (r=-0.52) — busy launch week = fewer tweets
    - DOGE dump — crypto crash = Elon goes quiet (asymmetric)

Variance signals (symmetric, modulate tail boost intensity):
    - Tesla volatility — high vol = more uncertainty
    - VIX stress — market stress = more uncertainty
    - Crypto Fear & Greed extremes — extreme sentiment = more uncertainty

Flow:
  1. Start from crowd prices
  2. Duration-based short shrink
  3. Compute directional score -> shift distribution UP or DOWN
  4. Compute variance score -> modulate tail boost intensity
  5. Apply tail boost (structural edge)
  6. Normalize and return
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


class DirectionalSignalModel(BasePredictionModel):
    """Directional signal model: shifts distribution UP/DOWN then applies tail boost.

    Hyperparameters:
        directional_strength: Max shift magnitude from directional signals (default 0.12)
        base_tail_boost: Multiplier for tail bucket probabilities (default 1.30)
        tail_threshold_sd: SDs from mean to qualify as "tail" (default 0.9)
        short_shrink: Downward shift for short-duration events (default 0.18)
        variance_weight: Per-signal variance boost weight (default 0.06)
        max_variance_boost: Cap on total variance boost (default 0.24)
    """

    DIRECTIONAL_STRENGTH = 0.12
    BASE_TAIL_BOOST = 1.30
    TAIL_THRESHOLD_SD = 0.9
    SHORT_SHRINK = 0.18
    VARIANCE_WEIGHT = 0.06
    MAX_VARIANCE_BOOST = 0.24

    def __init__(
        self,
        name: str = "directional_signal",
        version: str = "v1",
        directional_strength: float = None,
        base_tail_boost: float = None,
        tail_threshold_sd: float = None,
        short_shrink: float = None,
        variance_weight: float = None,
        max_variance_boost: float = None,
    ):
        super().__init__(name=name, version=version)
        if directional_strength is not None:
            self.DIRECTIONAL_STRENGTH = directional_strength
        if base_tail_boost is not None:
            self.BASE_TAIL_BOOST = base_tail_boost
        if tail_threshold_sd is not None:
            self.TAIL_THRESHOLD_SD = tail_threshold_sd
        if short_shrink is not None:
            self.SHORT_SHRINK = short_shrink
        if variance_weight is not None:
            self.VARIANCE_WEIGHT = variance_weight
        if max_variance_boost is not None:
            self.MAX_VARIANCE_BOOST = max_variance_boost

    def _compute_directional_score(self, features: dict) -> float:
        """Compute directional score in [-1, +1].

        Positive = expect MORE tweets than crowd consensus.
        Negative = expect FEWER tweets than crowd consensus.

        Each signal contributes [-0.5, +0.5], summed and clipped to [-1, +1].
        """
        score = 0.0
        financial = features.get("financial", {})
        media = features.get("media", {})
        attention = features.get("attention", {})
        calendar = features.get("calendar", {})
        trends = features.get("trends", {})

        # Signal 1: DOGE momentum (SIGNED — pump = more tweets, dump = fewer)
        # Unlike SignalEnhanced which uses abs(), we keep the sign
        doge_pct = financial.get("doge_pct_change_1d")
        if doge_pct is not None:
            doge_pct = float(doge_pct)
            # 10% pump → +0.3, 5% dump → -0.15
            # But asymmetric: pumps drive more tweeting than dumps cause silence
            if doge_pct > 0:
                contribution = min(doge_pct * 3.0, 0.5)
            else:
                # Dumps have weaker effect (Elon just goes quiet)
                contribution = max(doge_pct * 1.5, -0.5)
            score += contribution

        # Signal 2: SpaceX launches (busy launch week = fewer tweets, r=-0.52)
        launches = calendar.get("launches_trailing_7d")
        if launches is not None:
            launches = float(launches)
            # 1 launch = neutral, 3+ = significantly fewer tweets
            if launches > 1:
                contribution = -min((launches - 1) * 0.15, 0.5)
                score += contribution

        # Signal 3: Media tone (negative tone = more tweets, r=-0.51)
        tone = media.get("elon_musk_tone_7d")
        if tone is not None:
            tone = float(tone)
            # Negative tone → positive direction (bad press → more tweets)
            # tone is typically in [-1, 1] range
            contribution = max(min(-tone * 1.5, 0.5), -0.5)
            score += contribution

        # Signal 4: Wikipedia attention spike (spike = more tweeting)
        wiki_delta = attention.get("wiki_elon_musk_delta")
        if wiki_delta is not None:
            wiki_delta = float(wiki_delta)
            if wiki_delta > 0.10:
                contribution = min(wiki_delta * 0.8, 0.5)
                score += contribution

        # Signal 5: Google Trends spike (spike = more tweeting)
        gt_delta = trends.get("gt_elon_musk_delta")
        if gt_delta is not None:
            gt_delta = float(gt_delta)
            if gt_delta > 0.10:
                contribution = min(gt_delta * 0.8, 0.5)
                score += contribution

        # Signal 6: Tesla drawdown (crash = defensive tweeting = MORE tweets)
        tsla_dd = financial.get("tsla_drawdown_5d")
        if tsla_dd is not None:
            tsla_dd = float(tsla_dd)
            if tsla_dd < -0.03:
                # Drawdown is negative, more negative = bigger crash = more tweets
                contribution = min(abs(tsla_dd) * 5.0, 0.5)
                score += contribution

        # Signal 7: Trend from temporal features (momentum)
        temporal = features.get("temporal", {})
        trend_7d = temporal.get("trend_7d")
        rolling_avg = temporal.get("rolling_avg_7d")
        if trend_7d is not None and rolling_avg is not None:
            trend_7d = float(trend_7d)
            rolling_avg = float(rolling_avg)
            if rolling_avg > 0:
                # Relative trend: +10% of daily avg is significant
                relative_trend = trend_7d / rolling_avg
                contribution = max(min(relative_trend * 2.0, 0.5), -0.5)
                score += contribution

        # Clip to [-1, +1]
        return max(min(score, 1.0), -1.0)

    def _compute_variance_score(self, features: dict) -> float:
        """Compute variance score in [0, N] for symmetric tail boost modulation.

        These signals indicate higher uncertainty but no directional preference.
        """
        score = 0.0
        financial = features.get("financial", {})

        # Variance signal 1: Tesla volatility
        tsla_vol = financial.get("tsla_volatility_5d")
        if tsla_vol is not None:
            tsla_vol = float(tsla_vol)
            if tsla_vol > 0.03:
                score += min(1.0, tsla_vol / 0.06)

        # Variance signal 2: VIX stress
        vix_close = financial.get("vix_close")
        if vix_close is not None:
            vix_close = float(vix_close)
            if vix_close > 25.0:
                score += min(1.0, (vix_close - 25.0) / 25.0)

        # Variance signal 3: Crypto Fear & Greed extremes (both directions)
        fg_value = financial.get("crypto_fg_value")
        if fg_value is not None:
            fg_value = float(fg_value)
            if fg_value < 25:
                score += min(1.0, (25 - fg_value) / 25)
            elif fg_value > 75:
                score += min(1.0, (fg_value - 75) / 25)

        return score

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
            probs = _shift_distribution(probs, buckets, -self.SHORT_SHRINK)

        # Step 2: DIRECTIONAL shift (the key innovation)
        dir_score = self._compute_directional_score(features)
        if abs(dir_score) > 0.05:
            shift_amount = dir_score * self.DIRECTIONAL_STRENGTH
            probs = _shift_distribution(probs, buckets, shift_amount)

        # Step 3: Compute variance-modulated tail boost
        var_score = self._compute_variance_score(features)
        signal_boost = min(var_score * self.VARIANCE_WEIGHT, self.MAX_VARIANCE_BOOST)
        effective_boost = self.BASE_TAIL_BOOST * (1 + signal_boost)

        # Step 4: Apply tail boost (structural edge, still symmetric)
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
            "directional_strength": self.DIRECTIONAL_STRENGTH,
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "short_shrink": self.SHORT_SHRINK,
            "variance_weight": self.VARIANCE_WEIGHT,
            "max_variance_boost": self.MAX_VARIANCE_BOOST,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "directional_strength": self.DIRECTIONAL_STRENGTH,
            "base_tail_boost": self.BASE_TAIL_BOOST,
            "tail_threshold_sd": self.TAIL_THRESHOLD_SD,
            "short_shrink": self.SHORT_SHRINK,
            "variance_weight": self.VARIANCE_WEIGHT,
            "max_variance_boost": self.MAX_VARIANCE_BOOST,
        }
