"""
Directional consensus ensemble for Elon Musk tweet count prediction markets.

Extends ConsensusEnsembleModel by adding two directional components:

1. **TailBoostModel** (weight 0.20) — structural edge, symmetric variance
2. **PriceDynamicsModel** (weight 0.25) — bucket-level momentum following
3. **SignalEnhancedTailModel v3** (weight 0.20) — signal-modulated variance
4. **MarketAdjustedModel** (weight 0.15) — trend + mean reversion (DIRECTIONAL)
5. **DirectionalSignalModel** (weight 0.20) — multi-signal directional shift

The key difference from ConsensusEnsemble: 35% of ensemble weight now carries
genuine directional signal (MarketAdjusted via trend_7d/mean-reversion +
DirectionalSignal via signed DOGE/launches/tone/attention signals).

The original ConsensusEnsemble was ~60% symmetric variance expansion. This
ensemble is ~45% symmetric + ~55% directionally informed.
"""

from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.duration_model import TailBoostModel
from src.ml.price_dynamics_model import PriceDynamicsModel
from src.ml.signal_enhanced_model import SignalEnhancedTailModel
from src.ml.advanced_models import MarketAdjustedModel, _normalize, PROB_FLOOR
from src.ml.directional_model import DirectionalSignalModel


class DirectionalConsensusEnsemble(BasePredictionModel):
    """5-model ensemble with directional signal integration.

    Hyperparameters:
        weight_tail_boost: Weight for TailBoostModel (default 0.20)
        weight_price_dynamics: Weight for PriceDynamicsModel (default 0.25)
        weight_signal_enhanced: Weight for SignalEnhancedTailModel v3 (default 0.20)
        weight_market_adjusted: Weight for MarketAdjustedModel (default 0.15)
        weight_directional: Weight for DirectionalSignalModel (default 0.20)
    """

    WEIGHT_TAIL_BOOST = 0.20
    WEIGHT_PRICE_DYNAMICS = 0.25
    WEIGHT_SIGNAL_ENHANCED = 0.20
    WEIGHT_MARKET_ADJUSTED = 0.15
    WEIGHT_DIRECTIONAL = 0.20

    def __init__(
        self,
        name: str = "directional_consensus",
        version: str = "v1",
        weight_tail_boost: float = None,
        weight_price_dynamics: float = None,
        weight_signal_enhanced: float = None,
        weight_market_adjusted: float = None,
        weight_directional: float = None,
    ):
        super().__init__(name=name, version=version)
        if weight_tail_boost is not None:
            self.WEIGHT_TAIL_BOOST = weight_tail_boost
        if weight_price_dynamics is not None:
            self.WEIGHT_PRICE_DYNAMICS = weight_price_dynamics
        if weight_signal_enhanced is not None:
            self.WEIGHT_SIGNAL_ENHANCED = weight_signal_enhanced
        if weight_market_adjusted is not None:
            self.WEIGHT_MARKET_ADJUSTED = weight_market_adjusted
        if weight_directional is not None:
            self.WEIGHT_DIRECTIONAL = weight_directional

        # Component models with proven hyperparameters
        self._tail_boost = TailBoostModel(
            name="tail_boost", version="v1",
            tail_boost_factor=1.50,
            tail_threshold_sd=0.8,
        )

        self._price_dynamics = PriceDynamicsModel(
            name="price_dynamics", version="v1",
            base_tail_boost=1.35,
            tail_threshold_sd=0.8,
            momentum_weight=0.15,
            mean_reversion_strength=0.20,
            short_shrink=0.18,
        )

        self._signal_enhanced = SignalEnhancedTailModel(
            name="signal_enhanced_tail", version="v3",
            base_tail_boost=1.40,
            tail_threshold_sd=0.8,
            signal_weight=0.06,
            short_shrink=0.15,
        )

        self._market_adjusted = MarketAdjustedModel(
            name="market_adjusted", version="v1",
            momentum_strength=0.16,
            widen_strength=0.18,
            reversion_strength=0.28,
            regime_correction=0.00,
        )

        self._directional = DirectionalSignalModel(
            name="directional_signal", version="v1",
            directional_strength=0.12,
            base_tail_boost=1.30,
            tail_threshold_sd=0.9,
            short_shrink=0.18,
            variance_weight=0.06,
            max_variance_boost=0.24,
        )

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        # Get predictions from all 5 component models
        probs_tail = self._tail_boost.predict(features, buckets, context)
        probs_dynamics = self._price_dynamics.predict(features, buckets, context)
        probs_signal = self._signal_enhanced.predict(features, buckets, context)
        probs_adjusted = self._market_adjusted.predict(features, buckets, context)
        probs_directional = self._directional.predict(features, buckets, context)

        # Normalize weights
        total_weight = (
            self.WEIGHT_TAIL_BOOST
            + self.WEIGHT_PRICE_DYNAMICS
            + self.WEIGHT_SIGNAL_ENHANCED
            + self.WEIGHT_MARKET_ADJUSTED
            + self.WEIGHT_DIRECTIONAL
        )
        w1 = self.WEIGHT_TAIL_BOOST / total_weight
        w2 = self.WEIGHT_PRICE_DYNAMICS / total_weight
        w3 = self.WEIGHT_SIGNAL_ENHANCED / total_weight
        w4 = self.WEIGHT_MARKET_ADJUSTED / total_weight
        w5 = self.WEIGHT_DIRECTIONAL / total_weight

        # Blend predictions
        ensemble = {}
        for b in buckets:
            label = b["bucket_label"]
            p1 = probs_tail.get(label, PROB_FLOOR)
            p2 = probs_dynamics.get(label, PROB_FLOOR)
            p3 = probs_signal.get(label, PROB_FLOOR)
            p4 = probs_adjusted.get(label, PROB_FLOOR)
            p5 = probs_directional.get(label, PROB_FLOOR)

            ensemble[label] = w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4 + w5 * p5

        return _normalize(ensemble)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "weight_tail_boost": self.WEIGHT_TAIL_BOOST,
            "weight_price_dynamics": self.WEIGHT_PRICE_DYNAMICS,
            "weight_signal_enhanced": self.WEIGHT_SIGNAL_ENHANCED,
            "weight_market_adjusted": self.WEIGHT_MARKET_ADJUSTED,
            "weight_directional": self.WEIGHT_DIRECTIONAL,
            "component_models": [
                self._tail_boost.get_config(),
                self._price_dynamics.get_config(),
                self._signal_enhanced.get_config(),
                self._market_adjusted.get_config(),
                self._directional.get_config(),
            ],
        }

    def get_hyperparameters(self) -> dict:
        return {
            "weight_tail_boost": self.WEIGHT_TAIL_BOOST,
            "weight_price_dynamics": self.WEIGHT_PRICE_DYNAMICS,
            "weight_signal_enhanced": self.WEIGHT_SIGNAL_ENHANCED,
            "weight_market_adjusted": self.WEIGHT_MARKET_ADJUSTED,
            "weight_directional": self.WEIGHT_DIRECTIONAL,
        }
