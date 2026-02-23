"""
Crowd-hedged model for Elon Musk tweet count prediction markets.

Blends crowd consensus (market prices) with ConsensusEnsemble predictions
to create a more conservative model that:

- Anchors heavily to crowd prices (reducing drawdowns on bronze tier
  where the crowd is generally right)
- Retains tail-boost edge from ConsensusEnsemble on gold tier
  (where tails are structurally mispriced)

The blend formula is:
    P_hedged[bucket] = crowd_weight * P_crowd[bucket] + (1 - crowd_weight) * P_consensus[bucket]

Where:
    P_crowd comes from context["entry_prices"] (normalized market prices)
    P_consensus comes from ConsensusEnsembleModel predictions

Usage:
    model = CrowdHedgedModel(crowd_weight=0.5)
    probs = model.predict(features, buckets, context)
"""

from typing import Optional

from src.ml.base_model import BasePredictionModel
from src.ml.baseline_model import CrowdModel
from src.ml.consensus_model import ConsensusEnsembleModel
from src.ml.advanced_models import _normalize, PROB_FLOOR


class CrowdHedgedModel(BasePredictionModel):
    """Weighted blend of crowd prices and ConsensusEnsemble predictions.

    The crowd_weight parameter controls the blend:
        - crowd_weight=0.7: very conservative, mostly follows the crowd
        - crowd_weight=0.5: balanced blend (default)
        - crowd_weight=0.3: more aggressive, mostly follows ConsensusEnsemble

    Hyperparameters:
        crowd_weight: Weight for crowd prices (default 0.50).
                      ConsensusEnsemble gets weight (1 - crowd_weight).
    """

    def __init__(
        self,
        name: str = "crowd_hedged",
        version: str = "v1",
        crowd_weight: float = 0.50,
    ):
        super().__init__(name=name, version=version)
        self.crowd_weight = crowd_weight

        # Component models
        self._crowd = CrowdModel(name="crowd", version="v1")
        self._consensus = ConsensusEnsembleModel(
            name="consensus_ensemble", version="v1",
        )

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return hedged probability distribution across buckets.

        Args:
            features: Feature dict (temporal, media, calendar, etc.)
            buckets: List of bucket dicts with bucket_label, lower_bound, upper_bound.
            context: Must include "entry_prices" for crowd component and
                     "duration_days" for ConsensusEnsemble component.

        Returns:
            dict of bucket_label -> probability (sums to 1.0)
        """
        if not buckets:
            return {}

        # Get crowd predictions (normalized market prices)
        probs_crowd = self._crowd.predict(features, buckets, context)

        # Get ConsensusEnsemble predictions
        probs_consensus = self._consensus.predict(features, buckets, context)

        # Blend
        w_crowd = self.crowd_weight
        w_consensus = 1.0 - w_crowd

        hedged = {}
        for b in buckets:
            label = b["bucket_label"]
            p_crowd = probs_crowd.get(label, PROB_FLOOR)
            p_consensus = probs_consensus.get(label, PROB_FLOOR)

            hedged[label] = w_crowd * p_crowd + w_consensus * p_consensus

        return _normalize(hedged)

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "crowd_weight": self.crowd_weight,
            "component_models": [
                self._crowd.get_config(),
                self._consensus.get_config(),
            ],
        }

    def get_hyperparameters(self) -> dict:
        return {
            "crowd_weight": self.crowd_weight,
        }

    def __repr__(self) -> str:
        return "CrowdHedgedModel(name={!r}, version={!r}, crowd_weight={})".format(
            self.name, self.version, self.crowd_weight
        )
