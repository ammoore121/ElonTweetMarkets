"""
Train XGBoost bucket-level models with walk-forward cross-validation.

Trains both framings (classification and residual) and compares against
heuristic baselines. Reports per-fold metrics, feature importance, and
saves the best model.

Usage:
    python scripts/train_xgb_model.py                    # Full training + CV
    python scripts/train_xgb_model.py --framing both     # Both framings (default)
    python scripts/train_xgb_model.py --framing classify  # Classification only
    python scripts/train_xgb_model.py --framing residual  # Residual only
    python scripts/train_xgb_model.py --retrain           # Retrain on ALL data for production
    python scripts/train_xgb_model.py --compare           # Compare vs heuristic baselines
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

from src.ml.cross_validation import WalkForwardCV, load_events_from_index
from src.ml.dataset_builder import build_bucket_dataset, get_feature_columns
from src.ml.gradient_boost_model import XGBoostBucketModel, XGBoostResidualModel
from src.backtesting.engine import BacktestEngine


def train_and_evaluate_cv(
    model_class,
    model_name: str,
    events: list[dict],
    engine_config: dict,
    n_folds: int = 5,
    min_train: int = 20,
) -> dict:
    """Train with walk-forward CV and return results."""
    cv = WalkForwardCV(min_train_events=min_train, n_folds=n_folds)

    def model_factory(train_events=None):
        model = model_class()
        if train_events is not None:
            model.fit(train_events)
        return model

    print("\n--- Walk-Forward CV: {} ---".format(model_name))
    start = time.time()
    result = cv.evaluate(model_factory, events, engine_config)
    elapsed = time.time() - start

    cv.print_cv_report(result, model_name)
    print("  Training time: {:.1f}s".format(elapsed))

    return result


def train_heuristic_cv(
    model_class,
    model_name: str,
    events: list[dict],
    engine_config: dict,
    n_folds: int = 5,
    min_train: int = 20,
) -> dict:
    """Run walk-forward CV for a heuristic (non-ML) model."""
    cv = WalkForwardCV(min_train_events=min_train, n_folds=n_folds)

    def model_factory(train_events=None):
        return model_class()

    print("\n--- Walk-Forward CV: {} ---".format(model_name))
    result = cv.evaluate(model_factory, events, engine_config)
    cv.print_cv_report(result, model_name)
    return result


def analyze_feature_importance(model, top_n: int = 25) -> None:
    """Print feature importance analysis."""
    importance = model.get_feature_importance(top_n=top_n)
    if not importance:
        print("\nNo feature importance available.")
        return

    print("\n--- Top {} Feature Importances ---".format(top_n))
    print("{:<45s} {:>10s}".format("Feature", "Importance"))
    print("-" * 57)
    for feat, imp in importance:
        bar = "#" * int(imp * 200)  # Visual bar
        print("{:<45s} {:>10.4f} {}".format(feat, imp, bar))


def analyze_calibration(model, events: list[dict], engine_config: dict) -> None:
    """Analyze model calibration: predicted P(win) vs actual win rate."""
    print("\n--- Calibration Analysis ---")

    engine = BacktestEngine(config={**engine_config, "dry_run": True})
    result = engine.run(model, events)

    # Collect all (predicted_prob_for_winning_bucket, 1.0) pairs
    bins = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    bin_correct = [0] * (len(bins) - 1)
    bin_total = [0] * (len(bins) - 1)
    bin_sum_pred = [0.0] * (len(bins) - 1)

    for evt in result.per_event:
        if evt.get("skipped"):
            continue
        predicted_probs = evt.get("predicted_probs", {})
        winning = evt.get("winning_bucket")
        if not winning or not predicted_probs:
            continue

        for label, pred_p in predicted_probs.items():
            actual = 1.0 if label == winning else 0.0
            for i in range(len(bins) - 1):
                if bins[i] <= pred_p < bins[i + 1]:
                    bin_total[i] += 1
                    bin_correct[i] += actual
                    bin_sum_pred[i] += pred_p
                    break

    print("{:<20s} {:>8s} {:>8s} {:>12s}".format(
        "Predicted Range", "Count", "Win%", "Avg Pred"
    ))
    print("-" * 52)
    for i in range(len(bins) - 1):
        n = bin_total[i]
        if n == 0:
            continue
        actual_rate = bin_correct[i] / n
        avg_pred = bin_sum_pred[i] / n
        print("{:<20s} {:>8d} {:>8.1%} {:>12.4f}".format(
            "[{:.2f}, {:.2f})".format(bins[i], bins[i + 1]),
            n, actual_rate, avg_pred,
        ))


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost bucket models")
    parser.add_argument(
        "--framing",
        choices=["both", "classify", "residual"],
        default="both",
        help="Which framing to train (default: both)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain on ALL data (no CV) for production deployment",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run CV on heuristic baselines for comparison",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=20,
        help="Minimum training events per fold (default: 20)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum edge for betting (default: 0.02)",
    )
    args = parser.parse_args()

    # Load all events
    print("Loading events from backtest index...")
    events = load_events_from_index()
    print("  {} events loaded".format(len(events)))

    engine_config = {
        "bankroll": 1000.0,
        "kelly_fraction": 0.25,
        "min_edge": args.min_edge,
        "max_bet_pct": 0.05,
        "entry_hours_before_close": 24,
        "entry_window_hours": 6,
    }

    results = {}

    # ---------------------------------------------------------------
    # Walk-Forward CV: XGBoost models
    # ---------------------------------------------------------------
    if not args.retrain:
        if args.framing in ("both", "classify"):
            results["xgb_bucket"] = train_and_evaluate_cv(
                XGBoostBucketModel, "XGBoostBucketModel",
                events, engine_config,
                n_folds=args.n_folds, min_train=args.min_train,
            )

        if args.framing in ("both", "residual"):
            results["xgb_residual"] = train_and_evaluate_cv(
                XGBoostResidualModel, "XGBoostResidualModel",
                events, engine_config,
                n_folds=args.n_folds, min_train=args.min_train,
            )

        # ---------------------------------------------------------------
        # Compare vs heuristic baselines
        # ---------------------------------------------------------------
        if args.compare:
            from src.ml.duration_model import TailBoostModel
            from src.ml.signal_enhanced_model import SignalEnhancedTailModel
            from src.ml.consensus_model import ConsensusEnsembleModel

            results["tail_boost"] = train_heuristic_cv(
                TailBoostModel, "TailBoostModel",
                events, engine_config,
                n_folds=args.n_folds, min_train=args.min_train,
            )

            results["signal_enhanced_v3"] = train_heuristic_cv(
                lambda: SignalEnhancedTailModel(version="v3"),
                "SignalEnhanced v3",
                events, engine_config,
                n_folds=args.n_folds, min_train=args.min_train,
            )

            results["consensus"] = train_heuristic_cv(
                ConsensusEnsembleModel, "ConsensusEnsemble",
                events, engine_config,
                n_folds=args.n_folds, min_train=args.min_train,
            )

        # ---------------------------------------------------------------
        # Summary comparison table
        # ---------------------------------------------------------------
        if len(results) > 1:
            print("\n" + "=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)
            header = "{:<25s} {:>8s} {:>8s} {:>10s} {:>10s} {:>8s}".format(
                "Model", "Bets", "Wins", "P&L", "ROI%", "Brier"
            )
            print(header)
            print("-" * len(header))
            for name, r in sorted(
                results.items(),
                key=lambda x: x[1]["aggregate"]["roi"],
                reverse=True,
            ):
                agg = r["aggregate"]
                brier_str = (
                    "{:.4f}".format(agg["mean_brier"])
                    if agg["mean_brier"] is not None else "N/A"
                )
                print("{:<25s} {:>8d} {:>8d} {:>10s} {:>10s} {:>8s}".format(
                    name,
                    agg["total_bets"],
                    agg["total_wins"],
                    "${:+,.2f}".format(agg["total_pnl"]),
                    "{:+.1f}%".format(agg["roi"]),
                    brier_str,
                ))

    # ---------------------------------------------------------------
    # Retrain on ALL data for production
    # ---------------------------------------------------------------
    if args.retrain:
        print("\n" + "=" * 70)
        print("RETRAINING ON ALL {} EVENTS FOR PRODUCTION".format(len(events)))
        print("=" * 70)

        if args.framing in ("both", "classify"):
            print("\nTraining XGBoostBucketModel on all data...")
            model = XGBoostBucketModel()
            model.fit(events)
            save_path = model.save_model()
            print("  Model saved to: {}".format(save_path))
            analyze_feature_importance(model)
            analyze_calibration(model, events, engine_config)

        if args.framing in ("both", "residual"):
            print("\nTraining XGBoostResidualModel on all data...")
            model = XGBoostResidualModel()
            model.fit(events)
            save_path = model.save_model()
            print("  Model saved to: {}".format(save_path))
            analyze_feature_importance(model)
            analyze_calibration(model, events, engine_config)

    # ---------------------------------------------------------------
    # Final feature importance (from last fold of CV)
    # ---------------------------------------------------------------
    if not args.retrain and args.framing in ("both", "classify"):
        # Train one final model on most data to get feature importance
        print("\nTraining final model for feature importance analysis...")
        model = XGBoostBucketModel()
        model.fit(events)
        analyze_feature_importance(model, top_n=30)


if __name__ == "__main__":
    main()
