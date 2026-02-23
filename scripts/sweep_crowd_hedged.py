"""Sweep CrowdHedged model weights on GOLD tier only (trusted ground truth)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.crowd_hedged_model import CrowdHedgedModel
from src.ml.consensus_model import ConsensusEnsembleModel
from src.ml.duration_model import TailBoostModel
from src.ml.price_dynamics_model import PriceDynamicsModel
from src.ml.directional_model import DirectionalSignalModel
from src.backtesting.engine import BacktestEngine

BACKTEST_DIR = ROOT / "data" / "backtest"
INDEX_FILE = BACKTEST_DIR / "backtest_index.json"


def load_events(tier=None):
    with open(INDEX_FILE) as f:
        index = json.load(f)
    events = index.get("events", [])
    if tier:
        events = [e for e in events if e.get("ground_truth_tier") == tier]
    return events


def run_one(model, events, min_edge):
    config = {
        "min_edge": min_edge,
        "kelly_fraction": 0.25,
        "max_bet_pct": 0.10,
        "bankroll": 1000.0,
        "entry_hours_before_close": 24,
        "entry_window_hours": 6,
    }
    engine = BacktestEngine(config=config)
    return engine.run(model, events)


def fmt(label, r):
    if r.n_bets == 0:
        return f"{label:<35} {r.n_bets:>4} bets   {'---':>8}   {'---':>8}   {'---':>7}   {r.brier_score:.4f}   {r.accuracy_str}"
    return (
        f"{label:<35} {r.n_bets:>4} bets   "
        f"${r.total_wagered:>7.0f}   ${r.total_pnl:>+7.0f}   "
        f"{r.roi:>+6.1f}%   {r.brier_score:.4f}   {r.accuracy_str}"
    )


def main():
    gold_events = load_events("gold")
    silver_events = load_events("silver")
    print(f"Gold events: {len(gold_events)}, Silver events: {len(silver_events)}")

    print("=" * 95)
    print("  GOLD TIER ONLY -- Validated Ground Truth (38 events)")
    print("=" * 95)

    for min_edge in [0.02, 0.05]:
        print(f"\n{'-'*95}")
        print(f"  min_edge = {min_edge:.0%}")
        print(f"{'-'*95}")
        header = f"{'Model':<35} {'Bets':>4}       {'Wagered':>8}   {'P&L':>8}   {'ROI':>7}   {'Brier':>6}   {'Acc'}"
        print(header)
        print("-" * 95)

        # Baselines
        for ModelClass, name in [
            (ConsensusEnsembleModel, "ConsensusEnsemble"),
            (TailBoostModel, "TailBoost"),
            (PriceDynamicsModel, "PriceDynamics"),
            (DirectionalSignalModel, "DirectionalSignal"),
        ]:
            r = run_one(ModelClass(), gold_events, min_edge)
            print(fmt(name, r))

        print()

        # CrowdHedged sweep
        for w in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
            model = CrowdHedgedModel(crowd_weight=w)
            r = run_one(model, gold_events, min_edge)
            print(fmt(f"CrowdHedged w={w:.2f}", r))

    # Silver for reference
    print(f"\n{'='*95}")
    print(f"  SILVER TIER (9 events) -- min_edge=2%")
    print(f"{'='*95}")
    header = f"{'Model':<35} {'Bets':>4}       {'Wagered':>8}   {'P&L':>8}   {'ROI':>7}   {'Brier':>6}   {'Acc'}"
    print(header)
    print("-" * 95)

    r = run_one(ConsensusEnsembleModel(), silver_events, 0.02)
    print(fmt("ConsensusEnsemble", r))
    for w in [0.20, 0.30, 0.40, 0.50]:
        model = CrowdHedgedModel(crowd_weight=w)
        r = run_one(model, silver_events, 0.02)
        print(fmt(f"CrowdHedged w={w:.2f}", r))


if __name__ == "__main__":
    main()
