"""
XGBoost bucket-level prediction models for tweet count markets.

Two framings:
1. XGBoostBucketModel: Raw classification — predict P(bucket wins)
2. XGBoostResidualModel: Residual correction — predict edge over crowd

Both use the same (event, bucket) row structure from dataset_builder.py
and subclass BasePredictionModel so they plug into BacktestEngine with
zero changes.

Usage:
    # Framing A: Raw classification
    model = XGBoostBucketModel()
    model.fit(train_events)
    probs = model.predict(features, buckets, context)

    # Framing B: Residual correction
    model = XGBoostResidualModel()
    model.fit(train_events)
    probs = model.predict(features, buckets, context)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml.base_model import BasePredictionModel
from src.ml.dataset_builder import (
    build_bucket_dataset,
    build_single_event_rows,
    get_feature_columns,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PROB_FLOOR = 1e-6


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric in-place. XGBoost requires numeric types.

    Object columns are converted via pd.to_numeric (coerce errors to NaN).
    This preserves column names consistently across train/test sets.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities to sum to 1.0, with floor."""
    if not probs:
        return probs
    probs = {k: max(v, PROB_FLOOR) for k, v in probs.items()}
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


class XGBoostBucketModel(BasePredictionModel):
    """XGBoost classifier predicting P(bucket wins) directly.

    Each (event, bucket) pair is a training example with binary label.
    Features include all 113 event-level features plus per-bucket
    structural features (position, crowd price, heuristic prob, etc.).

    At inference, raw predictions are normalized to sum to 1 across
    all buckets in an event.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": 16.0,  # ~5.8% positive rate
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    def __init__(
        self,
        name: str = "xgb_bucket",
        version: str = "v1",
        model_dir: Optional[str] = None,
        **xgb_params,
    ):
        super().__init__(name=name, version=version)
        self.params = dict(self.DEFAULT_PARAMS)
        self.params.update(xgb_params)
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: Optional[list[str]] = None
        self.is_fitted = False

        # Model artifact directory
        if model_dir:
            self._model_dir = Path(model_dir)
        else:
            self._model_dir = PROJECT_DIR / "models" / "xgb_bucket_v1"

    def fit(
        self,
        train_events: list[dict],
        events_dir: Optional[Path] = None,
    ) -> None:
        """Train the XGBoost classifier on bucket-level data.

        Args:
            train_events: List of event dicts from backtest index.
            events_dir: Optional path to events directory.
        """
        df = build_bucket_dataset(train_events, events_dir=events_dir)
        if df.empty:
            raise ValueError("Empty training dataset")

        self.feature_columns = get_feature_columns(df)
        X = df[self.feature_columns].copy()
        X = _coerce_numeric(X)
        self.feature_columns = list(X.columns)
        y = df["_won"].values

        logger.info(
            "Training XGBoostBucketModel: %d rows, %d features, %.1f%% positive",
            len(X), len(self.feature_columns), 100 * y.mean()
        )

        # Update scale_pos_weight based on actual class balance
        pos_rate = y.mean()
        if pos_rate > 0:
            self.params["scale_pos_weight"] = (1 - pos_rate) / pos_rate

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Predict probability distribution across buckets."""
        if not buckets:
            return {}

        if not self.is_fitted or self.model is None:
            # Try loading from disk
            self._load_model()
            if not self.is_fitted:
                # Fallback: uniform
                n = len(buckets)
                return {b["bucket_label"]: 1.0 / n for b in buckets}

        if context is None:
            context = {}

        # Build metadata dict for build_single_event_rows
        metadata = {
            "event_slug": context.get("event_slug", "inference"),
            "buckets": buckets,
        }

        entry_prices = context.get("entry_prices", {})
        rows_df = build_single_event_rows(
            features=features,
            buckets=buckets,
            context=context,
            metadata=metadata,
            entry_prices=entry_prices,
        )

        # Align feature columns
        X = self._align_features(rows_df)

        # Predict probabilities
        raw_probs = self.model.predict_proba(X)[:, 1]

        probs = {}
        for i, row in rows_df.iterrows():
            label = row["_bucket_label"]
            probs[label] = float(raw_probs[i])

        return _normalize_probs(probs)

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align DataFrame columns to match training feature columns."""
        if self.feature_columns is None:
            return _coerce_numeric(df)

        # Build aligned frame in one shot to avoid fragmentation
        data = {}
        for col in self.feature_columns:
            if col in df.columns:
                data[col] = df[col].values
            else:
                data[col] = np.full(len(df), np.nan)
        result = pd.DataFrame(data, index=df.index)
        return _coerce_numeric(result)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save model artifact and feature config to disk."""
        if path is None:
            path = self._model_dir
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model via joblib
        model_path = path / "model.pkl"
        joblib.dump(self.model, str(model_path))

        # Save feature columns
        config_path = path / "feature_config.json"
        config = {
            "feature_columns": self.feature_columns,
            "params": {
                k: v for k, v in self.params.items()
                if k not in ("n_jobs", "verbosity")
            },
            "model_id": self.model_id,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Model saved to %s", path)
        return path

    def _load_model(self) -> None:
        """Load model from disk if available."""
        model_path = self._model_dir / "model.pkl"
        config_path = self._model_dir / "feature_config.json"

        if not model_path.exists() or not config_path.exists():
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.feature_columns = config["feature_columns"]

            self.model = joblib.load(str(model_path))
            self.is_fitted = True
            logger.info("Model loaded from %s", self._model_dir)
        except Exception as e:
            logger.warning("Failed to load model: %s", e)

    def get_feature_importance(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Return top N features by importance."""
        if not self.is_fitted or self.feature_columns is None:
            return []

        importances = self.model.feature_importances_
        pairs = list(zip(self.feature_columns, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "params": {
                k: v for k, v in self.params.items()
                if k not in ("n_jobs", "verbosity")
            },
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
        }

    def get_hyperparameters(self) -> dict:
        return {
            k: v for k, v in self.params.items()
            if k not in ("n_jobs", "verbosity", "random_state")
        }


class XGBoostResidualModel(BasePredictionModel):
    """XGBoost regressor predicting residual (edge) over crowd prices.

    Instead of predicting P(win) directly, this model predicts
    actual_outcome - crowd_price for each bucket. At inference:
        final_prob = crowd_price + predicted_residual
        clamp to [PROB_FLOOR, 1.0], then normalize

    This leverages the insight that crowd prices are already strong
    (Brier ~0.81) and the model only needs to learn the correction.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": 3,  # Shallower for residual learning
        "learning_rate": 0.03,
        "min_child_weight": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,  # Very aggressive regularization
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    def __init__(
        self,
        name: str = "xgb_residual",
        version: str = "v1",
        model_dir: Optional[str] = None,
        **xgb_params,
    ):
        super().__init__(name=name, version=version)
        self.params = dict(self.DEFAULT_PARAMS)
        self.params.update(xgb_params)
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_columns: Optional[list[str]] = None
        self.is_fitted = False

        if model_dir:
            self._model_dir = Path(model_dir)
        else:
            self._model_dir = PROJECT_DIR / "models" / "xgb_residual_v1"

    def fit(
        self,
        train_events: list[dict],
        events_dir: Optional[Path] = None,
    ) -> None:
        """Train the XGBoost regressor on residuals."""
        df = build_bucket_dataset(train_events, events_dir=events_dir)
        if df.empty:
            raise ValueError("Empty training dataset")

        self.feature_columns = get_feature_columns(df)
        X = df[self.feature_columns].copy()
        X = _coerce_numeric(X)
        self.feature_columns = list(X.columns)

        # Target: actual outcome (0/1) minus crowd price
        y = df["_won"].values - df["crowd_price"].values

        logger.info(
            "Training XGBoostResidualModel: %d rows, %d features, "
            "residual mean=%.4f, std=%.4f",
            len(X), len(self.feature_columns), y.mean(), y.std()
        )

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(
        self,
        features: dict,
        buckets: list[dict],
        context: Optional[dict] = None,
    ) -> dict[str, float]:
        """Predict probability distribution via crowd + residual."""
        if not buckets:
            return {}

        if not self.is_fitted or self.model is None:
            self._load_model()
            if not self.is_fitted:
                n = len(buckets)
                return {b["bucket_label"]: 1.0 / n for b in buckets}

        if context is None:
            context = {}

        metadata = {
            "event_slug": context.get("event_slug", "inference"),
            "buckets": buckets,
        }

        entry_prices = context.get("entry_prices", {})
        rows_df = build_single_event_rows(
            features=features,
            buckets=buckets,
            context=context,
            metadata=metadata,
            entry_prices=entry_prices,
        )

        X = self._align_features(rows_df)

        # Predict residuals
        residuals = self.model.predict(X)

        probs = {}
        for i, row in rows_df.iterrows():
            label = row["_bucket_label"]
            crowd_price = row.get("crowd_price", 0.0)
            if pd.isna(crowd_price):
                crowd_price = 0.0
            # Final prob = crowd + residual, clamped
            final_prob = crowd_price + float(residuals[i])
            final_prob = max(final_prob, PROB_FLOOR)
            final_prob = min(final_prob, 1.0)
            probs[label] = final_prob

        return _normalize_probs(probs)

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align DataFrame columns to match training feature columns."""
        if self.feature_columns is None:
            return _coerce_numeric(df)

        result = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                result[col] = df[col]
            else:
                result[col] = np.nan
        return _coerce_numeric(result)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save model artifact and feature config to disk."""
        if path is None:
            path = self._model_dir
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / "model.pkl"
        joblib.dump(self.model, str(model_path))

        config_path = path / "feature_config.json"
        config = {
            "feature_columns": self.feature_columns,
            "params": {
                k: v for k, v in self.params.items()
                if k not in ("n_jobs", "verbosity")
            },
            "model_id": self.model_id,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Model saved to %s", path)
        return path

    def _load_model(self) -> None:
        """Load model from disk if available."""
        model_path = self._model_dir / "model.pkl"
        config_path = self._model_dir / "feature_config.json"

        if not model_path.exists() or not config_path.exists():
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.feature_columns = config["feature_columns"]

            self.model = joblib.load(str(model_path))
            self.is_fitted = True
            logger.info("Model loaded from %s", self._model_dir)
        except Exception as e:
            logger.warning("Failed to load model: %s", e)

    def get_feature_importance(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Return top N features by importance."""
        if not self.is_fitted or self.feature_columns is None:
            return []

        importances = self.model.feature_importances_
        pairs = list(zip(self.feature_columns, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "params": {
                k: v for k, v in self.params.items()
                if k not in ("n_jobs", "verbosity")
            },
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
        }

    def get_hyperparameters(self) -> dict:
        return {
            k: v for k, v in self.params.items()
            if k not in ("n_jobs", "verbosity", "random_state")
        }
