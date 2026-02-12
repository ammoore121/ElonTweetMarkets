"""Test hybrid model: MarketAdjusted + z-score momentum correction.

Runs MarketAdjusted's shift/reversion/widening first, then applies
z-score-weighted momentum correction on top.
"""

import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtesting.engine import BacktestEngine
from src.ml.advanced_models import MarketAdjustedModel, _normalize, _get_crowd_probs, PROB_FLOOR
from src.ml.per_bucket_model import _bucket_midpoint
from src.ml.base_model import BasePredictionModel
from src.ml.baseline_model import CrowdModel
from typing import Optional

BACKTEST_DIR = PROJECT / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


class HybridModel(BasePredictionModel):
    """MarketAdjusted + z-score momentum correction."""

    def __init__(self, alpha=0.15, name="hybrid", version="v1"):
        super().__init__(name=name, version=version)
        self.alpha = alpha
        self.base_model = MarketAdjustedModel()

    def predict(self, features, buckets, context=None):
        # Get MarketAdjusted's prediction first
        base_probs = self.base_model.predict(features, buckets, context)
        if not base_probs:
            return base_probs

        # Apply z-score momentum correction on top
        market = features.get("market", {})
        temporal = features.get("temporal", {})

        crowd_ev = market.get("crowd_implied_ev")
        crowd_std = market.get("crowd_std_dev")

        # Get momentum signal
        momentum = self._get_momentum(temporal, market)
        if momentum is None or crowd_ev is None or crowd_std is None or crowd_std <= 0:
            return base_probs

        probs = {}
        for b in buckets:
            label = b["bucket_label"]
            base_p = base_probs.get(label, PROB_FLOOR)
            mid = _bucket_midpoint(b, buckets)
            z = (mid - crowd_ev) / crowd_std

            correction = 1.0 + self.alpha * z * momentum
            correction = max(min(correction, 1.3), 0.7)  # cap
            probs[label] = max(base_p * correction, PROB_FLOOR)

        return _normalize(probs)

    def _get_momentum(self, temporal, market):
        if temporal:
            trend = temporal.get("trend_7d")
            if trend is not None:
                avg = temporal.get("rolling_avg_7d")
                if avg and avg > 0:
                    return float(trend) / float(avg)
                return float(trend) / 100.0
        if market:
            shift = market.get("price_shift_24h")
            if shift is not None:
                ev = market.get("crowd_implied_ev", 100)
                if ev and ev > 0:
                    return float(shift) / float(ev)
        return None


def load_gold_events():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    return [e for e in index["events"] if e.get("ground_truth_tier") == "gold"]


def evaluate(model, events, min_edge=0.02):
    engine = BacktestEngine(config={"min_edge": min_edge})
    result = engine.run(model, events)
    return result.brier_score, result.accuracy, result.n_bets, result.roi, result.total_pnl


def main():
    events = load_gold_events()
    print("Gold events: {}".format(len(events)))
    print()

    # Baselines
    print("BASELINES:")
    for min_e in [0.02, 0.03, 0.05]:
        brier, acc, bets, roi, pnl = evaluate(CrowdModel(), events, min_e)
        print("  Crowd     (edge>={:.0%}): Brier={:.4f} Acc={:.3f} Bets={:>2d} ROI={:>+6.1f}% PnL=${:>+7.2f}".format(
            min_e, brier, acc, bets, roi, pnl))

    for min_e in [0.02, 0.03, 0.05]:
        brier, acc, bets, roi, pnl = evaluate(MarketAdjustedModel(), events, min_e)
        print("  Adjusted  (edge>={:.0%}): Brier={:.4f} Acc={:.3f} Bets={:>2d} ROI={:>+6.1f}% PnL=${:>+7.2f}".format(
            min_e, brier, acc, bets, roi, pnl))

    # Hybrid model grid search
    print()
    print("HYBRID (MarketAdjusted + z-score momentum):")
    print()

    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

    best_brier = 999
    best_alpha = 0

    for alpha in alphas:
        model = HybridModel(alpha=alpha)
        brier, acc, bets, roi, pnl = evaluate(model, events, 0.02)
        marker = " ***" if brier < best_brier else ""
        print("  alpha={:.2f}: Brier={:.4f} Acc={:.3f} Bets={:>2d} ROI={:>+6.1f}%{}".format(
            alpha, brier, acc, bets, roi, marker))
        if brier < best_brier:
            best_brier = brier
            best_alpha = alpha

    print()
    print("Best alpha: {:.2f} (Brier={:.4f})".format(best_alpha, best_brier))

    # Fine tune
    print()
    print("Fine-tuning around alpha={:.2f}:".format(best_alpha))
    for delta in [-0.04, -0.02, -0.01, 0.01, 0.02, 0.04]:
        alpha = best_alpha + delta
        if alpha < 0:
            continue
        model = HybridModel(alpha=alpha)
        brier, acc, bets, roi, pnl = evaluate(model, events, 0.02)
        marker = " ***" if brier < best_brier else ""
        print("  alpha={:.3f}: Brier={:.4f} Acc={:.3f} Bets={:>2d} ROI={:>+6.1f}%{}".format(
            alpha, brier, acc, bets, roi, marker))
        if brier < best_brier:
            best_brier = brier
            best_alpha = alpha

    # Best model at various edge thresholds
    print()
    print("=" * 70)
    print("BEST HYBRID (alpha={:.3f}) at various edge thresholds:".format(best_alpha))
    for min_e in [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
        model = HybridModel(alpha=best_alpha)
        brier, acc, bets, roi, pnl = evaluate(model, events, min_e)
        print("  edge>={:.0%}: Brier={:.4f} Bets={:>2d} ROI={:>+6.1f}% PnL=${:>+7.2f}".format(
            min_e, brier, bets, roi, pnl))

    # Also try: what if momentum only on events WITH temporal features?
    print()
    print("=" * 70)
    print("ANALYSIS: How many gold events have temporal features?")
    n_temporal = 0
    for evt in events:
        slug = evt["event_slug"]
        feat_path = BACKTEST_DIR / "events" / slug / "features.json"
        if feat_path.exists():
            with open(feat_path, "r") as f:
                feat = json.load(f)
            if feat.get("temporal", {}).get("rolling_avg_7d") is not None:
                n_temporal += 1
    print("  {} / {} gold events have temporal features".format(n_temporal, len(events)))


if __name__ == "__main__":
    main()
