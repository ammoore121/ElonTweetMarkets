"""Sweep bet sizing for CrowdHedged w=0.20 vs ConsensusEnsemble on GOLD tier.

Tests: what happens when we size up on the higher-conviction filtered bets?
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.crowd_hedged_model import CrowdHedgedModel
from src.ml.consensus_model import ConsensusEnsembleModel
from src.backtesting.engine import BacktestEngine

BACKTEST_DIR = ROOT / "data" / "backtest"
INDEX_FILE = BACKTEST_DIR / "backtest_index.json"


def load_events(tier):
    with open(INDEX_FILE) as f:
        index = json.load(f)
    return [e for e in index["events"] if e.get("ground_truth_tier") == tier]


def run_one(model, events, kelly, max_bet_pct, bankroll=1000.0, min_edge=0.02):
    config = {
        "min_edge": min_edge,
        "kelly_fraction": kelly,
        "max_bet_pct": max_bet_pct,
        "bankroll": bankroll,
        "entry_hours_before_close": 24,
        "entry_window_hours": 6,
    }
    engine = BacktestEngine(config=config)
    return engine.run(model, events)


def fmt(label, r):
    if r.n_bets == 0:
        return f"{label:<45} {r.n_bets:>3} bets   {'---':>7}   {'---':>7}   {'---':>6}"
    # Compute max drawdown from per-event P&L
    max_dd = 0.0
    cum = 0.0
    peak = 0.0
    for ev in r.per_event:
        cum += ev.get("pnl", 0.0)
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return (
        f"{label:<45} {r.n_bets:>3} bets   "
        f"${r.total_wagered:>7.0f}   ${r.total_pnl:>+7.0f}   "
        f"{r.roi:>+6.1f}%   ${max_dd:>5.0f}"
    )


def main():
    gold = load_events("gold")
    print(f"Gold events: {len(gold)}")
    print()

    header = f"{'Configuration':<45} {'Bets':>3}       {'Wagered':>8}   {'P&L':>8}   {'ROI':>6}   {'MaxDD':>5}"
    sep = "-" * 95

    # --- Section 1: ConsensusEnsemble sizing sweep ---
    print("=" * 95)
    print("  ConsensusEnsemble -- Sizing Sweep (GOLD, min_edge=2%)")
    print("=" * 95)
    print(header)
    print(sep)
    for kelly in [0.25, 0.35, 0.50, 0.75, 1.00]:
        for max_pct in [0.10, 0.15]:
            label = f"CE  kelly={kelly:.2f}  max_bet={max_pct:.0%}"
            r = run_one(ConsensusEnsembleModel(), gold, kelly, max_pct)
            print(fmt(label, r))
        if kelly < 1.0:
            print()

    # --- Section 2: CrowdHedged w=0.20 sizing sweep ---
    print()
    print("=" * 95)
    print("  CrowdHedged w=0.20 -- Sizing Sweep (GOLD, min_edge=2%)")
    print("=" * 95)
    print(header)
    print(sep)
    for kelly in [0.25, 0.35, 0.50, 0.75, 1.00]:
        for max_pct in [0.10, 0.15]:
            label = f"CH20  kelly={kelly:.2f}  max_bet={max_pct:.0%}"
            r = run_one(CrowdHedgedModel(crowd_weight=0.20), gold, kelly, max_pct)
            print(fmt(label, r))
        if kelly < 1.0:
            print()

    # --- Section 3: Head-to-head at matched wagered amounts ---
    print()
    print("=" * 95)
    print("  Head-to-Head: Same Capital Deployed (~$500 wagered)")
    print("=" * 95)
    print(header)
    print(sep)
    # CE at 0.25 kelly wagers ~$519
    r = run_one(ConsensusEnsembleModel(), gold, 0.25, 0.10)
    print(fmt("CE  kelly=0.25 (baseline)", r))
    # CH20 at 0.25 kelly wagers ~$309 -- need ~0.42 kelly to match $519
    for kelly in [0.35, 0.40, 0.45, 0.50]:
        r = run_one(CrowdHedgedModel(crowd_weight=0.20), gold, kelly, 0.10)
        print(fmt(f"CH20  kelly={kelly:.2f}", r))

    # --- Section 4: CrowdHedged w=0.10 (lighter hedge) ---
    print()
    print("=" * 95)
    print("  CrowdHedged w=0.10 -- Sizing Sweep (GOLD, min_edge=2%)")
    print("=" * 95)
    print(header)
    print(sep)
    for kelly in [0.25, 0.35, 0.50, 0.75]:
        label = f"CH10  kelly={kelly:.2f}  max_bet=10%"
        r = run_one(CrowdHedgedModel(crowd_weight=0.10), gold, kelly, 0.10)
        print(fmt(label, r))


if __name__ == "__main__":
    main()
