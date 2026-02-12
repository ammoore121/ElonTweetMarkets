"""
Base class for all tweet count prediction models.

All models must implement predict() which takes features, buckets, and
optional context, and returns a probability distribution across buckets.

Following EsportsBetting's MapWinnerModel pattern but adapted for
categorical (multi-bucket) prediction.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BasePredictionModel(ABC):
    """Base class for all tweet count prediction models.

    All models must implement predict() which takes features, buckets, and
    optional context, and returns a probability distribution across buckets.

    Attributes:
        name: Short identifier for the model (e.g. "naive_negbin").
        version: Version string (e.g. "v1").
        model_id: Combined "{name}_{version}" used as registry key.
    """

    def __init__(self, name: str, version: str = "v1"):
        self.name = name
        self.version = version
        self.model_id = "{}_{}".format(name, version)

    @abstractmethod
    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return probability distribution across buckets.

        Args:
            features: dict with top-level keys from:
                {temporal, gdelt, spacex, market, cross}
            buckets: list of dicts, each with at minimum:
                {"bucket_label": str, "lower_bound": int, "upper_bound": int}
            context: optional dict with:
                - "duration_days": int, number of days in the event window
                - "entry_prices": dict of bucket_label -> market price (0-1)

        Returns:
            dict of bucket_label -> probability.  Values should sum to 1.0.
        """
        ...

    def get_config(self) -> dict:
        """Return model configuration (hyperparameters).

        Subclasses should override this to include their specific
        hyperparameters.
        """
        return {"name": self.name, "version": self.version}

    def get_hyperparameters(self) -> dict:
        """Return hyperparameters dict for registry logging.

        Subclasses should override to expose tunable parameters.
        Returns the full config by default.
        """
        return self.get_config()

    def __repr__(self) -> str:
        return "{}(name={!r}, version={!r})".format(
            self.__class__.__name__, self.name, self.version
        )
