"""Run all profitable models across all tiers (gold/silver/bronze/all).

Quick script to see how models perform on extended history.
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtesting.engine import BacktestEngine
from src.ml.baseline_model import CrowdModel
from src.ml.per_bucket_model import PerBucketModel
from src.ml.duration_model import (
    TailBoostModel, DurationTailModel, DurationShrinkModel,
    DurationOverlayModel, PowerLawTailModel,
)
from src.ml.asymmetric_model import AsymmetricPerBucketModel

INDEX_PATH = PROJECT / "data" / "backtest" / "backtest_index.json"


def main():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        idx = json.load(f)

    events = idx["events"]
    gold = [e for e in events if e.get("ground_truth_tier") == "gold"]
    silver = [e for e in events if e.get("ground_truth_tier") == "silver"]
    bronze = [e for e in events if e.get("ground_truth_tier") == "bronze"]
    all_events = gold + silver + bronze

    # Sort by start date
    for lst in [gold, silver, bronze, all_events]:
        lst.sort(key=lambda e: e.get("start_date", "") or "")

    print("Events: {} gold, {} silver, {} bronze = {} total".format(
        len(gold), len(silver), len(bronze), len(all_events)))
    if bronze:
        print("Bronze range: {} to {}".format(bronze[0].get("start_date"), bronze[-1].get("end_date")))
    if gold:
        print("Gold range:   {} to {}".format(gold[0].get("start_date"), gold[-1].get("end_date")))
    print()

    models = {
        "CrowdModel": CrowdModel(),
        "PerBucket": PerBucketModel(),
        "TailBoost": TailBoostModel(
            tail_boost_factor=1.50, tail_threshold_sd=0.8, center_power=1.0),
        "DurationTail": DurationTailModel(
            short_shrink=0.20, medium_momentum=0.15, long_widen=1.20,
            tail_boost_factor=1.30, tail_threshold_sd=0.8),
        "DurationShrink": DurationShrinkModel(
            short_shrink=0.25, medium_momentum=0.15, long_widen=1.20),
        "DurationOverlay": DurationOverlayModel(
            short_shrink=0.20, long_widen=1.10,
            tail_boost_factor=1.30, tail_threshold_sd=0.8),
        "AsymPerBucket": AsymmetricPerBucketModel(
            alpha_down=0.36, alpha_up=0.12, max_correction=0.40,
            tail_cap_multiplier=3.5),
    }

    tiers = [
        ("Gold (38)", gold),
        ("Silver (9)", silver),
        ("Bronze (112)", bronze),
        ("ALL (159)", all_events),
    ]

    print("{:<18s} {:<14s} {:>7s} {:>8s} {:>5s} {:>5s} {:>10s}".format(
        "Model", "Tier", "Brier", "ROI", "Bets", "Wins", "P&L"))
    print("-" * 75)

    for tier_name, tier_events in tiers:
        if not tier_events:
            print("  {} -- no events".format(tier_name))
            print()
            continue

        for model_name, model in models.items():
            engine = BacktestEngine(config={"min_edge": 0.02})
            result = engine.run(model, tier_events)
            brier_str = "{:.4f}".format(result.brier_score) if result.brier_score else "N/A"
            print("{:<18s} {:<14s} {:>7s} {:>+7.1f}% {:>5d} {:>5d} {:>10s}".format(
                model_name, tier_name, brier_str, result.roi, result.n_bets,
                result.n_wins, "${:+,.0f}".format(result.total_pnl)))
        print()


if __name__ == "__main__":
    main()
