"""
Calendar-Media Fusion Model for Elon Musk tweet count prediction markets.

Key insight: Two strongest non-market features are:
    - SpaceX launches_trailing_7d (r=-0.52): busy launch weeks = fewer tweets
    - GDELT elon_musk_tone_7d (r=-0.51): bad press = more tweets

This model starts from market prices (the crowd) and applies targeted
corrections based on calendar and media signals:

    1. Shift LEFT (fewer tweets) when SpaceX launch activity is high
    2. Shift RIGHT (more tweets) when media tone is negative
    3. Optionally combine both via interaction terms

Multiple shift mechanisms are available:
    - Linear: delta = weight * z_score * shift_scale
    - Tanh: delta = tanh(weight * z_score) * max_shift
    - Threshold: only apply when |z_score| > threshold

The shift is applied using the existing _shift_distribution helper
from advanced_models.py.

Usage:
    model = CalendarMediaModel(
        launch_weight=0.08,
        tone_weight=0.08,
        shift_scale=0.15,
    )
    probs = model.predict(features, buckets, context={"entry_prices": {...}})
"""

import math
from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.advanced_models import (
    _normalize,
    _get_crowd_probs,
    _shift_distribution,
    _widen_distribution,
    _uniform,
    PROB_FLOOR,
)


class CalendarMediaModel(BasePredictionModel):
    """Calendar-Media Fusion Model.

    Starts from crowd prices and applies shift corrections based on
    SpaceX launch activity and GDELT media tone.

    Parameters:
        launch_weight: Weight for the launch-activity signal.
        tone_weight: Weight for the negative-tone signal.
        shift_scale: Overall multiplier for the shift magnitude.
        interaction_weight: Weight for the interaction term
            (launch_busy AND negative_tone).
        z_threshold: Minimum |z-score| to trigger a correction (0 = always).
        use_tanh: If True, use tanh(signal) for bounded non-linear shifts.
        max_shift: Maximum shift magnitude (caps the shift amount).
        widen_on_conflict: If launch says "down" and tone says "up",
            widen distribution instead of canceling out.
        base_model: "crowd" uses raw market prices as base.
            "market_adjusted" uses MarketAdjustedModel predictions.
    """

    # Normalization constants for z-scoring (estimated from gold-tier data)
    # launches_trailing_7d: observed range 0-4, mean ~2.0, std ~1.2
    LAUNCH_MEAN = 2.0
    LAUNCH_STD = 1.2
    # elon_musk_tone_7d: observed range -3.0 to -0.3, mean ~ -1.2, std ~ 0.9
    TONE_MEAN = -1.2
    TONE_STD = 0.9

    def __init__(
        self,
        name: str = "calendar_media",
        version: str = "v1",
        launch_weight: float = 0.08,
        tone_weight: float = 0.08,
        shift_scale: float = 0.15,
        interaction_weight: float = 0.0,
        z_threshold: float = 0.0,
        use_tanh: bool = False,
        max_shift: float = 0.25,
        widen_on_conflict: float = 0.0,
        base_model: str = "crowd",
        launch_mean: float = None,
        launch_std: float = None,
        tone_mean: float = None,
        tone_std: float = None,
    ):
        super().__init__(name=name, version=version)
        self.launch_weight = launch_weight
        self.tone_weight = tone_weight
        self.shift_scale = shift_scale
        self.interaction_weight = interaction_weight
        self.z_threshold = z_threshold
        self.use_tanh = use_tanh
        self.max_shift = max_shift
        self.widen_on_conflict = widen_on_conflict
        self.base_model = base_model
        self._adjusted_model = None

        # Override normalization constants if provided
        if launch_mean is not None:
            self.LAUNCH_MEAN = launch_mean
        if launch_std is not None:
            self.LAUNCH_STD = launch_std
        if tone_mean is not None:
            self.TONE_MEAN = tone_mean
        if tone_std is not None:
            self.TONE_STD = tone_std

    def _get_base_probs(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict],
    ) -> Optional[dict[str, float]]:
        """Get base probability distribution to adjust."""
        if self.base_model == "market_adjusted":
            if self._adjusted_model is None:
                from src.ml.advanced_models import MarketAdjustedModel
                self._adjusted_model = MarketAdjustedModel()
            return self._adjusted_model.predict(features, buckets, context)

        # Default: raw crowd prices
        return _get_crowd_probs(buckets, context)

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return probability distribution across buckets."""
        if not buckets:
            return {}

        # Get base distribution
        base_probs = self._get_base_probs(features, buckets, context)
        if base_probs is None:
            # No crowd data: fall back to uniform (cannot do much)
            return _uniform(buckets)

        probs = dict(base_probs)

        # Extract calendar and media features
        calendar = features.get("calendar", {})
        media = features.get("media", {})
        cross = features.get("cross", {})

        # Compute z-scores for launch and tone signals
        launch_z = self._compute_launch_z(calendar)
        tone_z = self._compute_tone_z(media)

        # Compute interaction z-score
        interaction_z = self._compute_interaction_z(cross, calendar, media)

        # Apply z-threshold filtering
        if self.z_threshold > 0:
            if abs(launch_z) < self.z_threshold:
                launch_z = 0.0
            if abs(tone_z) < self.z_threshold:
                tone_z = 0.0

        # Compute raw shift components:
        # launch_z > 0 means MORE launches than usual -> shift LEFT (negative)
        # tone_z > 0 means MORE negative tone than usual -> shift RIGHT (positive)
        launch_shift = -launch_z * self.launch_weight
        tone_shift = tone_z * self.tone_weight
        interaction_shift = interaction_z * self.interaction_weight

        # Combine shifts
        total_signal = launch_shift + tone_shift + interaction_shift

        # Apply non-linearity if requested
        if self.use_tanh:
            shift_amount = math.tanh(total_signal / self.shift_scale) * self.max_shift
        else:
            shift_amount = total_signal * self.shift_scale
            # Clamp
            shift_amount = max(-self.max_shift, min(self.max_shift, shift_amount))

        # Check for conflicting signals -> widen instead
        if self.widen_on_conflict > 0:
            # Conflict: launch says "down" (launch_z > 0) and tone says "up" (tone_z > 0)
            # Both are nonzero and in opposite shift directions
            if launch_shift < -0.005 and tone_shift > 0.005:
                conflict_strength = min(abs(launch_shift), abs(tone_shift))
                widen_factor = 1.0 + conflict_strength * self.widen_on_conflict * 10.0
                widen_factor = min(widen_factor, 1.5)
                probs = _widen_distribution(probs, buckets, widen_factor)

        # Apply the shift
        if abs(shift_amount) > 0.002:
            probs = _shift_distribution(probs, buckets, shift_amount)

        return _normalize(probs)

    def _compute_launch_z(self, calendar: dict) -> float:
        """Compute z-score for SpaceX launch activity.

        Higher z-score = more launches than usual = expect fewer tweets.
        """
        if not calendar:
            return 0.0

        launches = calendar.get("launches_trailing_7d")
        if launches is None:
            return 0.0

        launches = float(launches)
        if self.LAUNCH_STD <= 0:
            return 0.0

        return (launches - self.LAUNCH_MEAN) / self.LAUNCH_STD

    def _compute_tone_z(self, media: dict) -> float:
        """Compute z-score for negative media tone.

        Higher z-score = more negative tone than usual = expect more tweets.
        Note: GDELT tone is negative when press is bad, so we negate it.
        """
        if not media:
            return 0.0

        tone = media.get("elon_musk_tone_7d")
        if tone is None:
            return 0.0

        tone = float(tone)
        if self.TONE_STD <= 0:
            return 0.0

        # Negate: lower tone (more negative) should give higher z-score
        return -(tone - self.TONE_MEAN) / self.TONE_STD

    def _compute_interaction_z(
        self, cross: dict, calendar: dict, media: dict
    ) -> float:
        """Compute interaction signal: launch_busy AND negative_tone.

        This captures the scenario where SpaceX is busy (fewer tweets expected)
        AND press is negative (more tweets expected) -- a conflicting state
        that might resolve in a specific direction.

        Returns a composite z-score. Positive = expect more tweets.
        """
        if not cross and not (calendar and media):
            return 0.0

        # Try the pre-computed cross feature first
        bad_press = cross.get("bad_press_x_low_activity")
        if bad_press is not None and float(bad_press) > 0:
            return float(bad_press)

        # Fallback: compute from raw features
        launch_z = self._compute_launch_z(calendar)
        tone_z = self._compute_tone_z(media)

        # Interaction: only fires when both signals are present
        if launch_z > 0.5 and tone_z > 0.5:
            return launch_z * tone_z
        return 0.0

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "launch_weight": self.launch_weight,
            "tone_weight": self.tone_weight,
            "shift_scale": self.shift_scale,
            "interaction_weight": self.interaction_weight,
            "z_threshold": self.z_threshold,
            "use_tanh": self.use_tanh,
            "max_shift": self.max_shift,
            "widen_on_conflict": self.widen_on_conflict,
            "base_model": self.base_model,
        }

    def get_hyperparameters(self) -> dict:
        return {
            "launch_weight": self.launch_weight,
            "tone_weight": self.tone_weight,
            "shift_scale": self.shift_scale,
            "interaction_weight": self.interaction_weight,
            "z_threshold": self.z_threshold,
            "use_tanh": self.use_tanh,
            "max_shift": self.max_shift,
            "widen_on_conflict": self.widen_on_conflict,
            "base_model": self.base_model,
        }

    def __repr__(self) -> str:
        return (
            "CalendarMediaModel(launch_w={}, tone_w={}, scale={}, "
            "interact={}, base={})".format(
                self.launch_weight,
                self.tone_weight,
                self.shift_scale,
                self.interaction_weight,
                self.base_model,
            )
        )
