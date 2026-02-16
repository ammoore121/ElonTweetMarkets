"""
Consensus ensemble model for Elon Musk tweet count prediction markets.

Runs the top 3 proven models and takes a weighted average of their predictions.
This provides diversification across different edge sources:

1. **TailBoostModel** (weight 0.30) -- structural edge, most trusted, 187 bets
2. **PriceDynamicsModel** (weight 0.40) -- best recent performance, +33.1% ROI
3. **SignalEnhancedTailModel v3** (weight 0.30) -- signal-enhanced, highest P&L

The ensemble probability for each bucket is:
    P_ensemble(bucket) = w1 * P_tail(bucket) + w2 * P_dynamics(bucket) + w3 * P_signal(bucket)

Weights are configurable hyperparameters.

Usage:
    model = ConsensusEnsembleModel()
    probs = model.predict(features, buckets, context)
"""

from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.duration_model import TailBoostModel
from src.ml.price_dynamics_model import PriceDynamicsModel
from src.ml.signal_enhanced_model import SignalEnhancedTailModel
from src.ml.advanced_models import _normalize, PROB_FLOOR


class ConsensusEnsembleModel(BasePredictionModel):
    """Weighted consensus of TailBoost + PriceDynamics + SignalEnhanced v3.

    Blends predictions from 3 proven models to reduce variance while
    maintaining the structural tail edge that all 3 share.

    Hyperparameters:
        weight_tail_boost: Weight for TailBoostModel (default 0.30)
        weight_price_dynamics: Weight for PriceDynamicsModel (default 0.40)
        weight_signal_enhanced: Weight for SignalEnhancedTailModel v3 (default 0.30)
    """

    WEIGHT_TAIL_BOOST = 0.30
    WEIGHT_PRICE_DYNAMICS = 0.40
    WEIGHT_SIGNAL_ENHANCED = 0.30

    def __init__(
        self,
        name: str = "consensus_ensemble",
        version: str = "v1",
        weight_tail_boost: float = None,
        weight_price_dynamics: float = None,
        weight_signal_enhanced: float = None,
    ):
        super().__init__(name=name, version=version)
        if weight_tail_boost is not None:
            self.WEIGHT_TAIL_BOOST = weight_tail_boost
        if weight_price_dynamics is not None:
            self.WEIGHT_PRICE_DYNAMICS = weight_price_dynamics
        if weight_signal_enhanced is not None:
            self.WEIGHT_SIGNAL_ENHANCED = weight_signal_enhanced

        # Instantiate component models with their proven hyperparameters
        # TailBoostModel: default params from registry (1.30 boost, 1.0 SD)
        self._tail_boost = TailBoostModel(
            name="tail_boost", version="v1",
            tail_boost_factor=1.50,
            tail_threshold_sd=0.8,
        )

        # PriceDynamicsModel: default params from registry
        self._price_dynamics = PriceDynamicsModel(
            name="price_dynamics", version="v1",
            base_tail_boost=1.35,
            tail_threshold_sd=0.8,
            momentum_weight=0.15,
            mean_reversion_strength=0.20,
            short_shrink=0.18,
        )

        # SignalEnhancedTailModel v3: the best P&L variant
        self._signal_enhanced = SignalEnhancedTailModel(
            name="signal_enhanced_tail", version="v3",
            base_tail_boost=1.40,
            tail_threshold_sd=0.8,
            signal_weight=0.06,
            short_shrink=0.15,
        )

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        if not buckets:
            return {}

        # Get predictions from all 3 component models
        probs_tail = self._tail_boost.predict(features, buckets, context)
        probs_dynamics = self._price_dynamics.predict(features, buckets, context)
        probs_signal = self._signal_enhanced.predict(features, buckets, context)

        # Normalize weights (in case they don't sum to 1.0)
        total_weight = (
            self.WEIGHT_TAIL_BOOST
            + self.WEIGHT_PRICE_DYNAMICS
            + self.WEIGHT_SIGNAL_ENHANCED
        )
        w1 = self.WEIGHT_TAIL_BOOST / total_weight
        w2 = self.WEIGHT_PRICE_DYNAMICS / total_weight
        w3 = self.WEIGHT_SIGNAL_ENHANCED / total_weight

        # Blend predictions
        ensemble = {}
        for b in buckets:
            label = b["bucket_label"]
            p1 = probs_tail.get(label, PROB_FLOOR)
            p2 = probs_dynamics.get(label, PROB_FLOOR)
            p3 = probs_signal.get(label, PROB_FLOOR)

            ensemble[label] = w1 * p1 + w2 * p2 + w3 * p3

        return _normalize(ensemble)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "weight_tail_boost": self.WEIGHT_TAIL_BOOST,
            "weight_price_dynamics": self.WEIGHT_PRICE_DYNAMICS,
            "weight_signal_enhanced": self.WEIGHT_SIGNAL_ENHANCED,
            "component_models": [
                self._tail_boost.get_config(),
                self._price_dynamics.get_config(),
                self._signal_enhanced.get_config(),
            ],
        }

    def get_hyperparameters(self) -> dict:
        return {
            "weight_tail_boost": self.WEIGHT_TAIL_BOOST,
            "weight_price_dynamics": self.WEIGHT_PRICE_DYNAMICS,
            "weight_signal_enhanced": self.WEIGHT_SIGNAL_ENHANCED,
        }
