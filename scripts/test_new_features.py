"""Cross-tier backtest: compare baseline models vs signal-enhanced model.

Tests the new SignalEnhancedTailModel (using financial + attention features)
against the proven TailBoost and DurationShrink models across all tiers.
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtesting.engine import BacktestEngine
from src.ml.baseline_model import CrowdModel
from src.ml.duration_model import (
    TailBoostModel, DurationTailModel, DurationShrinkModel,
)
from src.ml.signal_enhanced_model import SignalEnhancedTailModel

INDEX_PATH = PROJECT / "data" / "backtest" / "backtest_index.json"


def main():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        idx = json.load(f)

    events = idx["events"]
    gold = [e for e in events if e.get("ground_truth_tier") == "gold"]
    silver = [e for e in events if e.get("ground_truth_tier") == "silver"]
    bronze = [e for e in events if e.get("ground_truth_tier") == "bronze"]
    all_events = gold + silver + bronze

    for lst in [gold, silver, bronze, all_events]:
        lst.sort(key=lambda e: e.get("start_date", "") or "")

    print("Events: {} gold, {} silver, {} bronze = {} total".format(
        len(gold), len(silver), len(bronze), len(all_events)))
    print()

    models = {
        "CrowdBaseline": CrowdModel(),
        "TailBoost": TailBoostModel(
            tail_boost_factor=1.50, tail_threshold_sd=0.8, center_power=1.0),
        "DurationShrink": DurationShrinkModel(
            short_shrink=0.25, medium_momentum=0.15, long_widen=1.20),
        "DurationTail": DurationTailModel(
            short_shrink=0.20, medium_momentum=0.15, long_widen=1.20,
            tail_boost_factor=1.30, tail_threshold_sd=0.8),
        "SignalEnhanced": SignalEnhancedTailModel(
            base_tail_boost=1.25, tail_threshold_sd=0.8,
            signal_weight=0.08, short_shrink=0.18),
        "SignalEnhanced_v2": SignalEnhancedTailModel(
            name="signal_enhanced_tail_v2", version="v2",
            base_tail_boost=1.35, tail_threshold_sd=0.9,
            signal_weight=0.10, short_shrink=0.20),
        "SignalEnhanced_v3": SignalEnhancedTailModel(
            name="signal_enhanced_tail_v3", version="v3",
            base_tail_boost=1.40, tail_threshold_sd=0.8,
            signal_weight=0.06, short_shrink=0.15),
    }

    tiers = [
        ("Gold", gold),
        ("Silver", silver),
        ("Bronze", bronze),
        ("ALL", all_events),
    ]

    min_edges = [0.02, 0.05]

    for min_edge in min_edges:
        print("=" * 90)
        print(f"  MIN_EDGE = {min_edge:.0%}")
        print("=" * 90)
        print("{:<22s} {:<8s} {:>7s} {:>8s} {:>5s} {:>5s} {:>10s} {:>10s}".format(
            "Model", "Tier", "Brier", "ROI%", "Bets", "Wins", "P&L", "Wagered"))
        print("-" * 90)

        for model_name, model in models.items():
            for tier_name, tier_events in tiers:
                if not tier_events:
                    continue

                config = {
                    "bankroll": 1000.0,
                    "kelly_fraction": 0.25,
                    "min_edge": min_edge,
                    "max_bet_pct": 0.10,
                    "entry_hours_before_close": 24,
                    "entry_window_hours": 6,
                    "dry_run": False,
                }

                engine = BacktestEngine(config=config)
                result = engine.run(model, tier_events)

                print("{:<22s} {:<8s} {:>7.4f} {:>7.1f}% {:>5d} {:>5d} {:>+10.1f} {:>10.1f}".format(
                    model_name,
                    tier_name,
                    result.brier_score,
                    result.roi,
                    result.n_bets,
                    result.n_wins,
                    result.total_pnl,
                    result.total_wagered,
                ))

            print()  # Blank line between models
        print()


if __name__ == "__main__":
    main()
