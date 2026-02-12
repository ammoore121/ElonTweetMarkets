"""
Model and Strategy registries for tracking experiments and deployments.

Follows the EsportsBetting pattern but properly separates concerns:

- **ModelRegistry**: What each model IS (code path, features, hyperparameters).
  Lives at models/model_registry.json.  Backtest results are stored separately
  in per-model folders: models/{model_id}/backtests/{tier}_{date}.json.

- **StrategyRegistry**: How each model is DEPLOYED (filter params, sizing, entry).
  Lives at strategies/strategy_registry.json.  Combines a model_id with trading
  parameters for paper or live betting.

Usage:
    from src.ml.registry import ModelRegistry, StrategyRegistry

    # Models
    mr = ModelRegistry()
    mr.print_comparison()
    model_def = mr.get_model("market_adjusted_v1")

    # Backtests
    mr.log_backtest("market_adjusted_v1", tier="gold", metrics={...})
    backtests = mr.get_backtests("market_adjusted_v1")

    # Strategies
    sr = StrategyRegistry()
    sr.print_strategies()
    strategy = sr.get_strategy("adjusted_conservative")
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_REGISTRY_PATH = PROJECT_DIR / "models" / "model_registry.json"
MODELS_DIR = PROJECT_DIR / "models"
STRATEGY_REGISTRY_PATH = PROJECT_DIR / "strategies" / "strategy_registry.json"


# =========================================================================
# ModelRegistry
# =========================================================================

class ModelRegistry:
    """Track model definitions and their backtest results.

    Storage layout:
        models/model_registry.json           -- model catalog (identity only)
        models/{model_id}/backtests/          -- per-model backtest results
            {tier}_{YYYYMMDD}.json

    Each model entry in the registry contains:
        - model_id, model_name, module_path, class_name
        - version, description, approach
        - feature_group, required_features, hyperparameters
        - data_sources, status, created_at, notes
    """

    VALID_STATUSES = {"active", "deprecated", "production"}

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or MODEL_REGISTRY_PATH
        self._data = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "models" not in data:
                    data["models"] = {}
                if "version" not in data:
                    data["version"] = "2.0"
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": "2.0", "models": {}}

    def _save(self) -> None:
        """Persist registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Model catalog operations
    # ------------------------------------------------------------------

    def log_model(
        self,
        model_id: str,
        model_name: str,
        module_path: str = "",
        class_name: str = "",
        version: str = "v1",
        description: str = "",
        approach: str = "",
        feature_group: str = "base",
        required_features: Optional[list[str]] = None,
        hyperparameters: Optional[dict] = None,
        data_sources: Optional[list[str]] = None,
        status: str = "active",
        notes: str = "",
    ) -> str:
        """Register a model definition (identity only, no metrics).

        If a model with the same model_id already exists it will be
        overwritten.

        Returns the model_id.
        """
        if status not in self.VALID_STATUSES:
            raise ValueError(
                "Invalid status '{}'. Must be one of: {}".format(
                    status, self.VALID_STATUSES
                )
            )

        entry = {
            "model_id": model_id,
            "model_name": model_name,
            "module_path": module_path,
            "class_name": class_name,
            "version": version,
            "description": description,
            "approach": approach,
            "feature_group": feature_group,
            "required_features": required_features or [],
            "hyperparameters": hyperparameters or {},
            "data_sources": data_sources or [],
            "status": status,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "notes": notes,
        }

        self._data["models"][model_id] = entry
        self._save()
        return model_id

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get a single model entry by ID."""
        return self._data["models"].get(model_id)

    def list_models(
        self,
        status: Optional[str] = None,
    ) -> list[dict]:
        """List all models, optionally filtered by status."""
        models = list(self._data["models"].values())
        if status is not None:
            models = [m for m in models if m.get("status") == status]
        return models

    def set_status(self, model_id: str, status: str) -> None:
        """Set model status: active, deprecated, or production."""
        if status not in self.VALID_STATUSES:
            raise ValueError(
                "Invalid status '{}'. Must be one of: {}".format(
                    status, self.VALID_STATUSES
                )
            )
        if model_id not in self._data["models"]:
            raise KeyError("Model '{}' not found in registry".format(model_id))
        self._data["models"][model_id]["status"] = status
        self._save()

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry. Returns True if removed."""
        if model_id in self._data["models"]:
            del self._data["models"][model_id]
            self._save()
            return True
        return False

    def instantiate_model(self, model_id: str):
        """Instantiate a model from registry metadata.

        Dynamically imports the module and class, passes hyperparameters.
        """
        entry = self.get_model(model_id)
        if not entry:
            raise ValueError("Model '{}' not found in registry".format(model_id))

        module_path = entry.get("module_path", "")
        class_name = entry.get("class_name", "")
        if not module_path or not class_name:
            raise ValueError(
                "Model '{}' missing module_path or class_name".format(model_id)
            )

        import importlib
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)

        hyperparams = entry.get("hyperparameters", {})
        version = entry.get("version", "v1")
        if "_v" in model_id:
            short_name = model_id.rsplit("_v", 1)[0]
        else:
            short_name = model_id

        return ModelClass(name=short_name, version=version, **hyperparams)

    # ------------------------------------------------------------------
    # Backtest operations (per-model folder)
    # ------------------------------------------------------------------

    def _backtest_dir(self, model_id: str) -> Path:
        """Return the backtests directory for a model."""
        return MODELS_DIR / model_id / "backtests"

    def log_backtest(
        self,
        model_id: str,
        tier: str,
        metrics: dict,
        config: Optional[str] = None,
        notes: str = "",
        run_date: Optional[str] = None,
    ) -> Path:
        """Save backtest results to models/{model_id}/backtests/{tier}_{date}.json.

        Args:
            model_id: Must exist in the model registry.
            tier: Data tier used (e.g. "gold", "silver", "bronze", "all").
            metrics: Dict with brier_score, log_loss, accuracy, roi_pct, etc.
            config: Config string describing the backtest run.
            notes: Free text.
            run_date: Override date string (YYYYMMDD). Defaults to today.

        Returns:
            Path to the saved backtest file.
        """
        if model_id not in self._data["models"]:
            raise KeyError(
                "Model '{}' not in registry. Register it first.".format(model_id)
            )

        if run_date is None:
            run_date = datetime.now(timezone.utc).strftime("%Y%m%d")

        backtest_dir = self._backtest_dir(model_id)
        backtest_dir.mkdir(parents=True, exist_ok=True)

        filename = "{}_{}.json".format(tier, run_date)
        filepath = backtest_dir / filename

        entry = {
            "model_id": model_id,
            "tier": tier,
            "run_date": run_date,
            "config": config or "",
            "metrics": metrics,
            "notes": notes,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, default=str)

        return filepath

    def get_backtests(
        self,
        model_id: str,
        tier: Optional[str] = None,
    ) -> list[dict]:
        """Return all backtest results for a model, optionally filtered by tier.

        Results are sorted by run_date descending (newest first).
        """
        backtest_dir = self._backtest_dir(model_id)
        if not backtest_dir.exists():
            return []

        results = []
        for fpath in sorted(backtest_dir.glob("*.json"), reverse=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_filepath"] = str(fpath)
                results.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        if tier is not None:
            results = [r for r in results if r.get("tier") == tier]

        return results

    def get_latest_backtest(
        self,
        model_id: str,
        tier: str = "gold",
    ) -> Optional[dict]:
        """Get the most recent backtest for a model and tier."""
        backtests = self.get_backtests(model_id, tier=tier)
        return backtests[0] if backtests else None

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        model_ids: Optional[list[str]] = None,
        tier: str = "gold",
    ) -> list[dict]:
        """Compare models by their latest backtest results on a given tier.

        Returns a list of dicts with model metadata + latest backtest metrics,
        sorted by brier_score ascending (best first).
        """
        if model_ids is not None:
            models = [
                self._data["models"][mid]
                for mid in model_ids
                if mid in self._data["models"]
            ]
        else:
            models = list(self._data["models"].values())

        rows = []
        for m in models:
            mid = m["model_id"]
            bt = self.get_latest_backtest(mid, tier=tier)
            metrics = bt.get("metrics", {}) if bt else {}

            rows.append({
                "model_id": mid,
                "model_name": m.get("model_name", mid),
                "status": m.get("status", "?"),
                "feature_group": m.get("feature_group", "?"),
                "approach": m.get("approach", "?"),
                "brier_score": metrics.get("brier_score"),
                "log_loss": metrics.get("log_loss"),
                "accuracy": metrics.get("accuracy", ""),
                "roi_pct": metrics.get("roi_pct"),
                "n_events": metrics.get("n_events"),
                "n_bets": metrics.get("n_bets"),
                "backtest_tier": tier,
                "backtest_date": bt.get("run_date") if bt else None,
            })

        def sort_key(r):
            brier = r.get("brier_score")
            return brier if brier is not None else float("inf")

        return sorted(rows, key=sort_key)

    def get_best_model(
        self,
        metric: str = "brier_score",
        minimize: bool = True,
        tier: str = "gold",
        status: Optional[str] = None,
    ) -> Optional[dict]:
        """Get the best model by a given metric from latest backtests.

        Args:
            metric: metric key to compare on (must exist in backtest metrics)
            minimize: if True, lower is better (Brier, log loss)
            tier: data tier for backtest results
            status: optionally filter to only models with this status

        Returns:
            The best comparison row dict, or None.
        """
        rows = self.compare_models(tier=tier)
        if status is not None:
            rows = [r for r in rows if r.get("status") == status]
        rows = [r for r in rows if r.get(metric) is not None]
        if not rows:
            return None
        if minimize:
            return min(rows, key=lambda r: r[metric])
        else:
            return max(rows, key=lambda r: r[metric])

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_comparison(
        self,
        model_ids: Optional[list[str]] = None,
        tier: str = "gold",
    ) -> None:
        """Print a formatted comparison table of models."""
        rows = self.compare_models(model_ids, tier=tier)

        if not rows:
            print("No models in registry.")
            return

        sep = "=" * 100
        print(sep)
        print("MODEL COMPARISON  (backtest tier: {})".format(tier))
        print(sep)

        header = "{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s} {:>12s} {:>12s}".format(
            "Model", "Brier", "Log Loss", "Accuracy", "ROI %", "Bets", "Status", "Approach"
        )
        print(header)
        print("-" * 100)

        for r in rows:
            name = str(r.get("model_name", "?"))[:24]
            brier = r.get("brier_score")
            logloss = r.get("log_loss")
            accuracy = r.get("accuracy", "")
            roi = r.get("roi_pct")
            n_bets = r.get("n_bets")
            status = str(r.get("status", "?"))
            approach = str(r.get("approach", "?"))[:12]

            brier_str = "{:.4f}".format(brier) if brier is not None else "-"
            logloss_str = "{:.4f}".format(logloss) if logloss is not None else "-"
            roi_str = "{:+.1f}%".format(roi) if roi is not None else "-"
            acc_str = str(accuracy) if accuracy else "-"
            bets_str = str(n_bets) if n_bets is not None else "-"

            print("{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s} {:>12s} {:>12s}".format(
                name, brier_str, logloss_str, acc_str, roi_str, bets_str, status, approach
            ))

        print(sep)

        # Highlight best model
        best = rows[0] if rows else None
        if best and best.get("brier_score") is not None:
            print("\nBest model (by Brier): {} ({:.4f})".format(
                best.get("model_name", "?"),
                best.get("brier_score", float("inf")),
            ))

    def __repr__(self) -> str:
        n = len(self._data.get("models", {}))
        return "ModelRegistry(path={!r}, n_models={})".format(
            str(self.registry_path), n
        )


# =========================================================================
# StrategyRegistry
# =========================================================================

class StrategyRegistry:
    """Track strategy definitions: model + filters + sizing for deployment.

    Storage: strategies/strategy_registry.json

    Each strategy entry contains:
        - strategy_id, strategy_name, model_id
        - status: paper | live | inactive
        - description, filters, sizing, entry
        - backtest_summary: latest backtest metrics snapshot
        - created_at, notes
    """

    VALID_STATUSES = {"paper", "live", "inactive"}

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or STRATEGY_REGISTRY_PATH
        self._data = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "strategies" not in data:
                    data["strategies"] = {}
                if "version" not in data:
                    data["version"] = "1.0"
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": "1.0", "strategies": {}}

    def _save(self) -> None:
        """Persist registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Strategy operations
    # ------------------------------------------------------------------

    def log_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        model_id: str,
        status: str = "paper",
        description: str = "",
        filters: Optional[dict] = None,
        sizing: Optional[dict] = None,
        entry: Optional[dict] = None,
        backtest_summary: Optional[dict] = None,
        notes: str = "",
    ) -> str:
        """Register or update a strategy definition.

        Returns the strategy_id.
        """
        if status not in self.VALID_STATUSES:
            raise ValueError(
                "Invalid status '{}'. Must be one of: {}".format(
                    status, self.VALID_STATUSES
                )
            )

        entry_config = entry or {
            "hours_before_close": 24,
            "entry_window_hours": 6,
        }

        record = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "model_id": model_id,
            "status": status,
            "description": description,
            "filters": filters or {},
            "sizing": sizing or {},
            "entry": entry_config,
            "backtest_summary": backtest_summary or {},
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "notes": notes,
        }

        self._data["strategies"][strategy_id] = record
        self._save()
        return strategy_id

    def get_strategy(self, strategy_id: str) -> Optional[dict]:
        """Get a single strategy entry by ID."""
        return self._data["strategies"].get(strategy_id)

    def list_strategies(
        self,
        status: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> list[dict]:
        """List all strategies, optionally filtered.

        Args:
            status: filter to strategies with this status
            model_id: filter to strategies using this model

        Returns:
            List of strategy entry dicts.
        """
        strategies = list(self._data["strategies"].values())
        if status is not None:
            strategies = [s for s in strategies if s.get("status") == status]
        if model_id is not None:
            strategies = [s for s in strategies if s.get("model_id") == model_id]
        return strategies

    def set_status(self, strategy_id: str, status: str) -> None:
        """Set strategy status: paper, live, or inactive."""
        if status not in self.VALID_STATUSES:
            raise ValueError(
                "Invalid status '{}'. Must be one of: {}".format(
                    status, self.VALID_STATUSES
                )
            )
        if strategy_id not in self._data["strategies"]:
            raise KeyError(
                "Strategy '{}' not found in registry".format(strategy_id)
            )
        self._data["strategies"][strategy_id]["status"] = status
        self._save()

    def update_backtest_summary(
        self,
        strategy_id: str,
        summary: dict,
    ) -> None:
        """Update the backtest_summary field for a strategy."""
        if strategy_id not in self._data["strategies"]:
            raise KeyError(
                "Strategy '{}' not found in registry".format(strategy_id)
            )
        self._data["strategies"][strategy_id]["backtest_summary"] = summary
        self._save()

    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the registry. Returns True if removed."""
        if strategy_id in self._data["strategies"]:
            del self._data["strategies"][strategy_id]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Engine config builder
    # ------------------------------------------------------------------

    def build_engine_config(self, strategy_id: str) -> dict:
        """Build a BacktestEngine config dict from a strategy definition.

        Returns a dict compatible with BacktestEngine(config=...).
        """
        strategy = self.get_strategy(strategy_id)
        if strategy is None:
            raise KeyError(
                "Strategy '{}' not found in registry".format(strategy_id)
            )

        sizing = strategy.get("sizing", {})
        filters = strategy.get("filters", {})
        entry_cfg = strategy.get("entry", {})

        return {
            "bankroll": sizing.get("bankroll", 1000.0),
            "kelly_fraction": sizing.get("kelly_fraction", 0.25),
            "min_edge": filters.get("min_edge", 0.05),
            "max_bet_pct": sizing.get("max_bet_pct", 0.05),
            "entry_hours_before_close": entry_cfg.get("hours_before_close", 24),
            "entry_window_hours": entry_cfg.get("entry_window_hours", 6),
            "dry_run": False,
        }

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_strategies(
        self,
        status: Optional[str] = None,
    ) -> None:
        """Print a formatted table of strategies."""
        strategies = self.list_strategies(status=status)

        if not strategies:
            print("No strategies in registry.")
            return

        sep = "=" * 110
        print(sep)
        print("STRATEGY REGISTRY")
        print(sep)

        header = "{:<28s} {:>22s} {:>8s} {:>9s} {:>6s} {:>8s} {:>8s} {:>12s}".format(
            "Strategy", "Model", "Status", "Min Edge", "Bets", "Brier", "ROI %", "Kelly Frac"
        )
        print(header)
        print("-" * 110)

        for s in strategies:
            name = str(s.get("strategy_name", "?"))[:27]
            model = str(s.get("model_id", "?"))[:22]
            status_str = str(s.get("status", "?"))
            filters = s.get("filters", {})
            sizing = s.get("sizing", {})
            bt = s.get("backtest_summary", {})

            min_edge = filters.get("min_edge")
            kelly = sizing.get("kelly_fraction")
            n_bets = bt.get("n_bets")
            brier = bt.get("brier")
            roi = bt.get("roi_pct")

            edge_str = "{:.0%}".format(min_edge) if min_edge is not None else "-"
            bets_str = str(n_bets) if n_bets is not None else "-"
            brier_str = "{:.4f}".format(brier) if brier is not None else "-"
            roi_str = "{:+.1f}%".format(roi) if roi is not None else "-"
            kelly_str = "{:.2f}".format(kelly) if kelly is not None else "-"

            print("{:<28s} {:>22s} {:>8s} {:>9s} {:>6s} {:>8s} {:>8s} {:>12s}".format(
                name, model, status_str, edge_str, bets_str, brier_str, roi_str, kelly_str
            ))

        print(sep)

    def __repr__(self) -> str:
        n = len(self._data.get("strategies", {}))
        return "StrategyRegistry(path={!r}, n_strategies={})".format(
            str(self.registry_path), n
        )
