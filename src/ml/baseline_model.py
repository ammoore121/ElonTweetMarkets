"""
Baseline models for Elon Musk tweet count prediction markets.

Two baselines:
1. NaiveBucketModel: Fits a Negative Binomial distribution to trailing XTracker
   daily counts, scales by event duration, then computes per-bucket probabilities.
2. CrowdModel: Uses market prices directly as probability predictions (should
   roughly break even minus slippage -- serves as a sanity check).

Usage:
    model = NaiveBucketModel()
    probs = model.predict(features, buckets, context={"duration_days": 7})

    model = CrowdModel()
    probs = model.predict(features, buckets, context={"entry_prices": {...}})
"""

import math
from typing import Optional

from scipy.stats import nbinom, poisson

from src.ml.base_model import BasePredictionModel


class NaiveBucketModel(BasePredictionModel):
    """Baseline: fit Negative Binomial from trailing XTracker data -> bucket probabilities.

    For events with temporal features (gold tier, post-Oct 2025):
    - Uses rolling_avg and rolling_std to parameterize a Negative Binomial
    - Scales daily rate by event duration to get total count distribution
    - Computes P(count in bucket) for each bucket

    For events WITHOUT temporal features (bronze tier):
    - Falls back to uniform distribution across buckets (no-information prior)
    """

    def __init__(self, name: str = "naive_negbin", version: str = "v1"):
        super().__init__(name=name, version=version)

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return probability distribution across buckets.

        Args:
            features: dict with keys "temporal", "gdelt", "spacex", "market".
                temporal has: rolling_avg_7d, rolling_std_7d, rolling_avg_14d, etc.
            buckets: list of {"bucket_label": str, "lower_bound": int,
                              "upper_bound": int, ...}
            context: optional dict with:
                - "duration_days": int, number of days in the event window.
                  Defaults to 7 if not provided.

        Returns:
            dict of bucket_label -> probability (sums to 1.0)
        """
        if not buckets:
            return {}

        n_buckets = len(buckets)

        # Extract duration from context
        duration_days = 7
        if context and "duration_days" in context:
            dur = context["duration_days"]
            if dur is not None and dur > 0:
                duration_days = dur

        # Try to get daily mean and std from temporal features
        temporal = features.get("temporal", {})
        if not temporal:
            return self._uniform(buckets)

        daily_mean = self._get_daily_mean(temporal)
        daily_std = self._get_daily_std(temporal)

        if daily_mean is None or daily_mean <= 0:
            return self._uniform(buckets)

        # Scale daily stats to event-duration stats.
        # Assuming days are approximately independent:
        #   total_mean = daily_mean * duration_days
        #   total_var  = daily_var  * duration_days
        total_mean = daily_mean * duration_days

        if daily_std is not None and daily_std > 0:
            daily_var = daily_std ** 2
            total_var = daily_var * duration_days
        else:
            # No std available -- fall back to Poisson assumption (var == mean)
            total_var = total_mean * 1.1  # slight inflation to stay NB-valid

        # Ensure overdispersion for Negative Binomial (var > mean required).
        # If variance <= mean the NB degenerates to Poisson which we handle below.
        use_poisson = False
        if total_var <= total_mean:
            use_poisson = True

        if use_poisson:
            probs = self._poisson_bucket_probs(total_mean, buckets)
        else:
            probs = self._negbin_bucket_probs(total_mean, total_var, buckets)

        return probs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_daily_mean(temporal: dict) -> Optional[float]:
        """Extract best available daily mean from temporal features.

        Prefers 7d rolling average, falls back to 14d, then 28d.
        """
        for key in ["rolling_avg_7d", "rolling_avg_14d", "rolling_avg_28d"]:
            val = temporal.get(key)
            if val is not None:
                return float(val)
        return None

    @staticmethod
    def _get_daily_std(temporal: dict) -> Optional[float]:
        """Extract best available daily std dev from temporal features.

        Prefers 7d, falls back to 14d.
        """
        for key in ["rolling_std_7d", "rolling_std_14d"]:
            val = temporal.get(key)
            if val is not None:
                return float(val)
        return None

    @staticmethod
    def _uniform(buckets: list[dict]) -> dict[str, float]:
        """Return uniform distribution across buckets."""
        n = len(buckets)
        if n == 0:
            return {}
        p = 1.0 / n
        return {b["bucket_label"]: p for b in buckets}

    @staticmethod
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

            # Floor to avoid zero probabilities (important for log-scoring)
            probs[bucket["bucket_label"]] = max(prob, 1e-6)

        # Normalize to sum to 1.0
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    @staticmethod
    def _poisson_bucket_probs(
        total_mean: float,
        buckets: list[dict],
    ) -> dict[str, float]:
        """Compute per-bucket probabilities using Poisson CDF.

        Used when variance <= mean (no overdispersion), so Negative Binomial
        is not appropriate.
        """
        mu = max(total_mean, 0.01)

        probs = {}
        for bucket in buckets:
            lower = int(bucket["lower_bound"])
            upper = int(bucket["upper_bound"])

            if upper >= 99999:
                if lower <= 0:
                    prob = 1.0
                else:
                    prob = 1.0 - poisson.cdf(lower - 1, mu)
            elif lower <= 0:
                prob = poisson.cdf(upper, mu)
            else:
                prob = poisson.cdf(upper, mu) - poisson.cdf(lower - 1, mu)

            probs[bucket["bucket_label"]] = max(prob, 1e-6)

        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def get_config(self) -> dict:
        return {"name": self.name, "version": self.version, "distribution": "negative_binomial"}

    def __repr__(self) -> str:
        return "NaiveBucketModel(name={!r}, version={!r})".format(self.name, self.version)


class CrowdModel(BasePredictionModel):
    """Baseline: use market prices as predictions (expects ~0 edge).

    This model returns normalized market prices as its probability predictions.
    Since it agrees with the market, it should produce roughly zero edge --
    useful as a sanity check that the backtest engine penalizes no-edge trades.

    Requires entry_prices to be passed in the context dict:
        context = {"entry_prices": {"<250": 0.05, "250-274": 0.12, ...}}

    If entry_prices are not available, falls back to uniform distribution.
    """

    def __init__(self, name: str = "crowd", version: str = "v1"):
        super().__init__(name=name, version=version)

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Return market-price-derived probability distribution.

        Args:
            features: dict with keys "temporal", "gdelt", "spacex", "market".
                Not directly used by this model -- included for API compatibility.
            buckets: list of {"bucket_label": str, "lower_bound": int,
                              "upper_bound": int, ...}
            context: must include "entry_prices" dict mapping bucket_label to
                     market price (0.0 - 1.0).  Also accepts "duration_days"
                     for API compatibility (ignored).

        Returns:
            dict of bucket_label -> probability (sums to 1.0)
        """
        if not buckets:
            return {}

        # Try to use entry prices from context
        if context and "entry_prices" in context:
            prices = context["entry_prices"]
            if prices and isinstance(prices, dict):
                # Filter to only buckets in our bucket list
                bucket_labels = {b["bucket_label"] for b in buckets}
                filtered = {
                    k: max(float(v), 1e-6)
                    for k, v in prices.items()
                    if k in bucket_labels
                }

                if filtered:
                    # Normalize to sum to 1.0
                    total = sum(filtered.values())
                    if total > 0:
                        normalized = {k: v / total for k, v in filtered.items()}

                        # Ensure all buckets are covered (add floor for missing)
                        for b in buckets:
                            if b["bucket_label"] not in normalized:
                                normalized[b["bucket_label"]] = 1e-6

                        # Re-normalize after adding missing buckets
                        total = sum(normalized.values())
                        return {k: v / total for k, v in normalized.items()}

        # Fallback: uniform distribution
        n = len(buckets)
        p = 1.0 / n
        return {b["bucket_label"]: p for b in buckets}

    def get_config(self) -> dict:
        return {"name": self.name, "version": self.version, "strategy": "market_prices"}

    def __repr__(self) -> str:
        return "CrowdModel(name={!r}, version={!r})".format(self.name, self.version)

