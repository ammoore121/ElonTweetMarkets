"""Grid search over PerBucketModel hyperparameters on gold-tier events.

Tests momentum_alpha, reversion_strength, and widen_strength combinations
to find the optimal configuration for gold-tier Brier score.
"""

import json
import sys
from pathlib import Path
from itertools import product

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtesting.engine import BacktestEngine
from src.ml.per_bucket_model import PerBucketModel
from src.ml.advanced_models import MarketAdjustedModel
from src.ml.baseline_model import CrowdModel

BACKTEST_DIR = PROJECT / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


def load_gold_events():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    return [e for e in index["events"] if e.get("ground_truth_tier") == "gold"]


def evaluate_model(model, events, config=None):
    engine = BacktestEngine(config=config or {"min_edge": 0.02})
    result = engine.run(model, events)
    return result.brier_score, result.accuracy, result.n_bets, result.roi


def main():
    events = load_gold_events()
    print("Gold-tier events: {}".format(len(events)))
    print()

    # Baselines
    print("=" * 70)
    print("BASELINES")
    print("=" * 70)

    crowd = CrowdModel()
    brier, acc, bets, roi = evaluate_model(crowd, events)
    print("CrowdModel:          Brier={:.4f}  Acc={:.3f}  Bets={}  ROI={:.1f}%".format(
        brier, acc, bets, roi))

    adjusted = MarketAdjustedModel()
    brier, acc, bets, roi = evaluate_model(adjusted, events, {"min_edge": 0.02})
    print("MarketAdjusted:      Brier={:.4f}  Acc={:.3f}  Bets={}  ROI={:.1f}%".format(
        brier, acc, bets, roi))

    print()
    print("=" * 70)
    print("GRID SEARCH: PerBucketModel")
    print("=" * 70)

    # Grid search
    alpha_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    reversion_values = [0.0, 0.14, 0.28, 0.42]
    widen_values = [0.0, 0.10, 0.15, 0.20]

    best_brier = 999
    best_params = {}
    results = []

    # First: test alpha alone (no reversion or widening)
    print()
    print("--- Phase 1: Momentum Alpha only (no reversion/widening) ---")
    for alpha in alpha_values:
        model = PerBucketModel()
        model.MOMENTUM_ALPHA = alpha
        model.REVERSION_STRENGTH = 0.0
        model.WIDEN_STRENGTH = 0.0

        brier, acc, bets, roi = evaluate_model(model, events)
        results.append({
            "alpha": alpha, "reversion": 0.0, "widen": 0.0,
            "brier": brier, "acc": acc, "bets": bets
        })
        marker = " ***" if brier < best_brier else ""
        print("  alpha={:.2f}: Brier={:.4f}  Acc={:.3f}{}".format(alpha, brier, acc, marker))
        if brier < best_brier:
            best_brier = brier
            best_params = {"alpha": alpha, "reversion": 0.0, "widen": 0.0}

    # Phase 2: Best alpha + reversion sweep
    print()
    print("--- Phase 2: Best alpha + reversion sweep ---")
    best_alpha_phase1 = best_params["alpha"]
    for rev in reversion_values:
        model = PerBucketModel()
        model.MOMENTUM_ALPHA = best_alpha_phase1
        model.REVERSION_STRENGTH = rev
        model.WIDEN_STRENGTH = 0.0

        brier, acc, bets, roi = evaluate_model(model, events)
        results.append({
            "alpha": best_alpha_phase1, "reversion": rev, "widen": 0.0,
            "brier": brier, "acc": acc, "bets": bets
        })
        marker = " ***" if brier < best_brier else ""
        print("  alpha={:.2f} rev={:.2f}: Brier={:.4f}  Acc={:.3f}{}".format(
            best_alpha_phase1, rev, brier, acc, marker))
        if brier < best_brier:
            best_brier = brier
            best_params = {"alpha": best_alpha_phase1, "reversion": rev, "widen": 0.0}

    # Phase 3: Best (alpha, reversion) + widen sweep
    print()
    print("--- Phase 3: Best (alpha, reversion) + widen sweep ---")
    best_rev = best_params["reversion"]
    for w in widen_values:
        model = PerBucketModel()
        model.MOMENTUM_ALPHA = best_alpha_phase1
        model.REVERSION_STRENGTH = best_rev
        model.WIDEN_STRENGTH = w

        brier, acc, bets, roi = evaluate_model(model, events)
        results.append({
            "alpha": best_alpha_phase1, "reversion": best_rev, "widen": w,
            "brier": brier, "acc": acc, "bets": bets
        })
        marker = " ***" if brier < best_brier else ""
        print("  alpha={:.2f} rev={:.2f} widen={:.2f}: Brier={:.4f}  Acc={:.3f}{}".format(
            best_alpha_phase1, best_rev, w, brier, acc, marker))
        if brier < best_brier:
            best_brier = brier
            best_params = {"alpha": best_alpha_phase1, "reversion": best_rev, "widen": w}

    # Phase 4: Fine-tune around best alpha
    print()
    print("--- Phase 4: Fine-tune alpha around best ---")
    fine_alphas = [max(0, best_alpha_phase1 - 0.03), best_alpha_phase1 - 0.01,
                   best_alpha_phase1 + 0.01, best_alpha_phase1 + 0.03]
    for alpha in fine_alphas:
        model = PerBucketModel()
        model.MOMENTUM_ALPHA = alpha
        model.REVERSION_STRENGTH = best_params["reversion"]
        model.WIDEN_STRENGTH = best_params["widen"]

        brier, acc, bets, roi = evaluate_model(model, events)
        marker = " ***" if brier < best_brier else ""
        print("  alpha={:.3f} rev={:.2f} widen={:.2f}: Brier={:.4f}  Acc={:.3f}{}".format(
            alpha, best_params["reversion"], best_params["widen"], brier, acc, marker))
        if brier < best_brier:
            best_brier = brier
            best_params["alpha"] = alpha

    # Also test with min_edge=0.02 for ROI
    print()
    print("=" * 70)
    print("BEST CONFIG: alpha={:.3f}, reversion={:.2f}, widen={:.2f}".format(
        best_params["alpha"], best_params["reversion"], best_params["widen"]))
    print("Best Brier: {:.4f}".format(best_brier))
    print("=" * 70)

    # Final evaluation with ROI at different min_edge thresholds
    print()
    print("--- ROI at different min_edge thresholds ---")
    for min_e in [0.02, 0.03, 0.05, 0.08, 0.10]:
        model = PerBucketModel()
        model.MOMENTUM_ALPHA = best_params["alpha"]
        model.REVERSION_STRENGTH = best_params["reversion"]
        model.WIDEN_STRENGTH = best_params["widen"]

        brier, acc, bets, roi = evaluate_model(model, events, {"min_edge": min_e})
        print("  min_edge={:.0%}: Brier={:.4f} Bets={} ROI={:+.1f}%".format(
            min_e, brier, bets, roi))


if __name__ == "__main__":
    main()
