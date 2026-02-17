"""
Walk-forward cross-validation for time-series prediction markets.

Key principle: NEVER use future events for training. Events are sorted
chronologically and we use expanding-window folds where the training set
grows with each fold.

Usage:
    from src.ml.cross_validation import WalkForwardCV

    cv = WalkForwardCV(min_train_events=20, n_folds=5)
    folds = cv.create_folds(events)
    results = cv.evaluate(model_factory, events, engine_config)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.backtesting.engine import BacktestEngine


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


class WalkForwardCV:
    """Walk-forward (expanding window) cross-validation for event-based models.

    Sorts events chronologically by start_date, then creates folds where:
    - Train set: all events before the cutoff
    - Test set: events after the cutoff up to the next cutoff

    This ensures temporal integrity: no future data leaks into training.
    """

    def __init__(
        self,
        min_train_events: int = 20,
        n_folds: int = 5,
    ):
        self.min_train_events = min_train_events
        self.n_folds = n_folds

    def create_folds(
        self, events: list[dict]
    ) -> list[dict]:
        """Create walk-forward folds from a list of events.

        Args:
            events: List of event dicts, each with at least 'start_date'.

        Returns:
            List of fold dicts, each with:
                - fold_idx: int
                - train_events: list[dict]
                - test_events: list[dict]
                - cutoff_date: str (YYYY-MM-DD)
        """
        sorted_events = sorted(
            events, key=lambda e: e.get("start_date") or "9999-99-99"
        )

        n = len(sorted_events)
        if n < self.min_train_events + 1:
            raise ValueError(
                "Need at least {} events for CV, got {}".format(
                    self.min_train_events + 1, n
                )
            )

        # Determine fold boundaries: evenly space cutoffs between
        # min_train_events and total events
        test_pool = n - self.min_train_events
        actual_folds = min(self.n_folds, test_pool)
        if actual_folds < 1:
            actual_folds = 1

        fold_size = test_pool / actual_folds

        folds = []
        for i in range(actual_folds):
            train_end = self.min_train_events + int(i * fold_size)
            test_end = self.min_train_events + int((i + 1) * fold_size)
            if i == actual_folds - 1:
                test_end = n  # last fold gets all remaining

            train_events = sorted_events[:train_end]
            test_events = sorted_events[train_end:test_end]

            if not test_events:
                continue

            cutoff_date = test_events[0].get("start_date", "unknown")

            folds.append({
                "fold_idx": i,
                "train_events": train_events,
                "test_events": test_events,
                "cutoff_date": cutoff_date,
                "n_train": len(train_events),
                "n_test": len(test_events),
            })

        return folds

    def evaluate(
        self,
        model_factory: Callable,
        events: list[dict],
        engine_config: Optional[dict] = None,
    ) -> dict:
        """Run walk-forward CV and return aggregate + per-fold metrics.

        Args:
            model_factory: Callable that returns a model instance.
                For heuristic models: lambda: TailBoostModel()
                For ML models: callable that accepts (train_events,) and
                returns a fitted model.
            events: Full list of events from the backtest index.
            engine_config: Engine config dict (bankroll, kelly, etc.)

        Returns:
            Dict with:
                - folds: list of per-fold results
                - aggregate: overall metrics across all folds
        """
        folds = self.create_folds(events)

        if engine_config is None:
            engine_config = {
                "bankroll": 1000.0,
                "kelly_fraction": 0.25,
                "min_edge": 0.02,
                "max_bet_pct": 0.05,
                "entry_hours_before_close": 24,
                "entry_window_hours": 6,
            }

        fold_results = []

        for fold in folds:
            # Create model -- if factory accepts train_events, pass them
            try:
                model = model_factory(train_events=fold["train_events"])
            except TypeError:
                model = model_factory()

            engine = BacktestEngine(config=engine_config)
            result = engine.run(model, fold["test_events"])

            fold_results.append({
                "fold_idx": fold["fold_idx"],
                "cutoff_date": fold["cutoff_date"],
                "n_train": fold["n_train"],
                "n_test": fold["n_test"],
                "n_bets": result.n_bets,
                "n_wins": result.n_wins,
                "total_wagered": result.total_wagered,
                "total_pnl": result.total_pnl,
                "roi": result.roi,
                "brier_score": result.brier_score,
                "log_loss": result.log_loss,
            })

        # Aggregate across folds
        total_wagered = sum(f["total_wagered"] for f in fold_results)
        total_pnl = sum(f["total_pnl"] for f in fold_results)
        total_bets = sum(f["n_bets"] for f in fold_results)
        total_wins = sum(f["n_wins"] for f in fold_results)

        brier_scores = [
            f["brier_score"] for f in fold_results
            if f["brier_score"] is not None
        ]

        aggregate = {
            "n_folds": len(fold_results),
            "total_events_tested": sum(f["n_test"] for f in fold_results),
            "total_bets": total_bets,
            "total_wins": total_wins,
            "total_wagered": round(total_wagered, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": (
                round(100.0 * total_pnl / total_wagered, 2)
                if total_wagered > 0 else 0.0
            ),
            "mean_brier": (
                round(sum(brier_scores) / len(brier_scores), 6)
                if brier_scores else None
            ),
            "win_rate": (
                round(total_wins / total_bets, 4)
                if total_bets > 0 else 0.0
            ),
        }

        return {
            "folds": fold_results,
            "aggregate": aggregate,
        }

    def print_cv_report(self, cv_result: dict, model_name: str = "") -> None:
        """Print formatted CV report."""
        agg = cv_result["aggregate"]
        folds = cv_result["folds"]

        sep = "=" * 70
        print(sep)
        print("WALK-FORWARD CV REPORT {}".format(
            "(" + model_name + ")" if model_name else ""
        ))
        print(sep)
        print()
        print("Folds: {}  |  Events tested: {}  |  Total bets: {}".format(
            agg["n_folds"], agg["total_events_tested"], agg["total_bets"]
        ))
        print()

        # Per-fold table
        header = "{:>5s} {:>12s} {:>6s} {:>6s} {:>5s} {:>10s} {:>10s} {:>8s} {:>8s}".format(
            "Fold", "Cutoff", "Train", "Test", "Bets",
            "Wagered", "P&L", "ROI%", "Brier"
        )
        print(header)
        print("-" * len(header))

        for f in folds:
            brier_str = "{:.4f}".format(f["brier_score"]) if f["brier_score"] is not None else "N/A"
            print(
                "{:>5d} {:>12s} {:>6d} {:>6d} {:>5d} {:>10s} {:>10s} {:>8s} {:>8s}".format(
                    f["fold_idx"],
                    f["cutoff_date"],
                    f["n_train"],
                    f["n_test"],
                    f["n_bets"],
                    "${:,.2f}".format(f["total_wagered"]),
                    "${:+,.2f}".format(f["total_pnl"]),
                    "{:+.1f}%".format(f["roi"]),
                    brier_str,
                )
            )

        print()
        print("--- Aggregate ---")
        print("  Total wagered:  ${:,.2f}".format(agg["total_wagered"]))
        print("  Total P&L:      ${:+,.2f}".format(agg["total_pnl"]))
        print("  ROI:            {:+.1f}%".format(agg["roi"]))
        if agg["mean_brier"] is not None:
            print("  Mean Brier:     {:.4f}".format(agg["mean_brier"]))
        print("  Win rate:       {:.1%}".format(agg["win_rate"]))
        print(sep)


def load_events_from_index(tier: Optional[str] = None) -> list[dict]:
    """Load events from backtest_index.json, optionally filtering by tier."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "Backtest index not found: {}".format(INDEX_PATH)
        )

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    events = index.get("events", [])
    if tier:
        events = [e for e in events if e.get("ground_truth_tier") == tier]

    return events
