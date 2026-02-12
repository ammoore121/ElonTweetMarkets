"""
Grid search and evaluation for the Contrarian Price-Reversal Model.

Tests multiple configurations against gold-tier events:
1. Pure price-reversal (various strengths)
2. Pure skewness correction
3. Combined reversal + skewness
4. Threshold-gated (only large shifts)
5. MarketAdjusted base + reversal

Also sweeps min_edge to find the best trading threshold.

Usage:
    python scripts/test_contrarian.py

Requires:
    - Backtest dataset at data/backtest/ (run build_backtest_dataset.py first)
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.backtesting.engine import BacktestEngine
from src.ml.contrarian_model import ContrarianModel
from src.ml.advanced_models import MarketAdjustedModel
from src.ml.baseline_model import CrowdModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gold_events():
    """Load gold-tier events from the backtest index."""
    if not INDEX_PATH.exists():
        print("ERROR: Backtest index not found at {}".format(INDEX_PATH))
        sys.exit(1)

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    events = index.get("events", [])
    gold = [e for e in events if e.get("ground_truth_tier") == "gold"]
    print("Loaded {} gold-tier events".format(len(gold)))
    return gold


def run_config(name, model, events, min_edge=0.02):
    """Run a single configuration and return summary dict."""
    engine = BacktestEngine(config={
        "bankroll": 1000.0,
        "kelly_fraction": 0.25,
        "min_edge": min_edge,
        "max_bet_pct": 0.10,
        "entry_hours_before_close": 24,
        "entry_window_hours": 6,
    })

    result = engine.run(model, events)

    summary = {
        "config_name": name,
        "min_edge": min_edge,
        "n_events": result.n_events,
        "n_bets": result.n_bets,
        "n_wins": result.n_wins,
        "brier": result.brier_score,
        "log_loss": result.log_loss,
        "accuracy": result.accuracy_str,
        "total_wagered": result.total_wagered,
        "total_pnl": result.total_pnl,
        "roi": result.roi,
    }

    return summary


def print_result_row(s):
    """Print a single result row."""
    brier_str = "{:.4f}".format(s["brier"]) if s["brier"] is not None else "  -   "
    log_str = "{:.4f}".format(s["log_loss"]) if s["log_loss"] is not None else "  -   "
    print(
        "  {:<40s} edge={:.0%} | Brier={} LL={} Acc={:<6s} | "
        "Bets={:>3d} W={:>2d} Wager=${:>7.0f} PnL=${:>+8.2f} ROI={:>+6.1f}%".format(
            s["config_name"][:40],
            s["min_edge"],
            brier_str,
            log_str,
            s["accuracy"],
            s["n_bets"],
            s["n_wins"],
            s["total_wagered"],
            s["total_pnl"],
            s["roi"],
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print("CONTRARIAN MODEL GRID SEARCH")
    print("=" * 110)
    print()

    events = load_gold_events()
    if not events:
        print("No gold events found.")
        return

    all_results = []

    # -----------------------------------------------------------------------
    # Phase 0: Baselines (CrowdModel and MarketAdjustedModel)
    # -----------------------------------------------------------------------
    print("--- Phase 0: Baselines ---")

    for min_edge in [0.01, 0.02, 0.03, 0.05]:
        # CrowdModel baseline
        s = run_config("Crowd (baseline)", CrowdModel(), events, min_edge=min_edge)
        all_results.append(s)
        print_result_row(s)

    print()
    for min_edge in [0.01, 0.02, 0.03, 0.05]:
        s = run_config("MarketAdjusted (baseline)", MarketAdjustedModel(), events, min_edge=min_edge)
        all_results.append(s)
        print_result_row(s)

    print()

    # -----------------------------------------------------------------------
    # Phase 1: Pure price-reversal (sweep reversal_strength)
    # -----------------------------------------------------------------------
    print("--- Phase 1: Pure Price-Reversal ---")

    reversal_strengths = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    shift_thresholds = [0.0, 5.0, 10.0, 15.0, 20.0]

    best_phase1 = None
    for rev in reversal_strengths:
        for thresh in shift_thresholds:
            model = ContrarianModel(
                reversal_strength=rev,
                shift_threshold=thresh,
                skew_correction=0.0,
                entropy_correction=0.0,
                use_adjusted_base=False,
            )
            config_name = "Rev={:.2f} Thresh={:.0f}".format(rev, thresh)
            s = run_config(config_name, model, events, min_edge=0.02)
            all_results.append(s)
            print_result_row(s)

            if best_phase1 is None or (s["brier"] is not None and s["brier"] < best_phase1["brier"]):
                best_phase1 = s

    print()
    if best_phase1:
        print("  >> Best Phase 1: {}  Brier={:.4f} ROI={:+.1f}%".format(
            best_phase1["config_name"], best_phase1["brier"], best_phase1["roi"]))
    print()

    # -----------------------------------------------------------------------
    # Phase 2: Pure skewness correction
    # -----------------------------------------------------------------------
    print("--- Phase 2: Pure Skewness Correction ---")

    skew_corrections = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    for skew in skew_corrections:
        model = ContrarianModel(
            reversal_strength=0.0,
            shift_threshold=0.0,
            skew_correction=skew,
            entropy_correction=0.0,
            use_adjusted_base=False,
        )
        config_name = "Skew={:.2f}".format(skew)
        s = run_config(config_name, model, events, min_edge=0.02)
        all_results.append(s)
        print_result_row(s)

    print()

    # -----------------------------------------------------------------------
    # Phase 3: Combined reversal + skewness
    # -----------------------------------------------------------------------
    print("--- Phase 3: Combined Reversal + Skewness ---")

    combos = [
        (0.10, 10.0, 0.05),
        (0.15, 10.0, 0.05),
        (0.20, 10.0, 0.05),
        (0.10, 5.0, 0.08),
        (0.15, 5.0, 0.08),
        (0.20, 5.0, 0.10),
        (0.10, 15.0, 0.05),
        (0.20, 15.0, 0.10),
        (0.30, 10.0, 0.05),
        (0.15, 10.0, 0.10),
    ]

    best_phase3 = None
    for rev, thresh, skew in combos:
        model = ContrarianModel(
            reversal_strength=rev,
            shift_threshold=thresh,
            skew_correction=skew,
            entropy_correction=0.0,
            use_adjusted_base=False,
        )
        config_name = "Rev={:.2f} T={:.0f} Skew={:.2f}".format(rev, thresh, skew)
        s = run_config(config_name, model, events, min_edge=0.02)
        all_results.append(s)
        print_result_row(s)

        if best_phase3 is None or (s["brier"] is not None and s["brier"] < best_phase3["brier"]):
            best_phase3 = s

    print()
    if best_phase3:
        print("  >> Best Phase 3: {}  Brier={:.4f} ROI={:+.1f}%".format(
            best_phase3["config_name"], best_phase3["brier"], best_phase3["roi"]))
    print()

    # -----------------------------------------------------------------------
    # Phase 4: Entropy correction (standalone and combined)
    # -----------------------------------------------------------------------
    print("--- Phase 4: Entropy Correction ---")

    entropy_combos = [
        (0.0, 0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 2.0),
        (0.15, 10.0, 0.05, 0.5),
        (0.15, 10.0, 0.05, 1.0),
        (0.20, 10.0, 0.05, 1.0),
    ]

    for rev, thresh, skew, ent in entropy_combos:
        model = ContrarianModel(
            reversal_strength=rev,
            shift_threshold=thresh,
            skew_correction=skew,
            entropy_correction=ent,
            use_adjusted_base=False,
        )
        config_name = "Rev={:.2f} Skew={:.2f} Ent={:.1f}".format(rev, skew, ent)
        s = run_config(config_name, model, events, min_edge=0.02)
        all_results.append(s)
        print_result_row(s)

    print()

    # -----------------------------------------------------------------------
    # Phase 5: MarketAdjusted base + contrarian corrections
    # -----------------------------------------------------------------------
    print("--- Phase 5: MarketAdjusted Base + Contrarian ---")

    adjusted_combos = [
        (0.10, 10.0, 0.0, 0.0),
        (0.15, 10.0, 0.0, 0.0),
        (0.20, 10.0, 0.0, 0.0),
        (0.10, 10.0, 0.05, 0.0),
        (0.15, 10.0, 0.05, 0.0),
        (0.20, 10.0, 0.05, 0.0),
        (0.15, 5.0, 0.08, 0.0),
        (0.20, 5.0, 0.10, 0.0),
        (0.15, 10.0, 0.05, 1.0),
        (0.20, 10.0, 0.10, 1.0),
    ]

    best_phase5 = None
    for rev, thresh, skew, ent in adjusted_combos:
        model = ContrarianModel(
            reversal_strength=rev,
            shift_threshold=thresh,
            skew_correction=skew,
            entropy_correction=ent,
            use_adjusted_base=True,
        )
        config_name = "Adj+Rev={:.2f} T={:.0f} S={:.2f} E={:.1f}".format(
            rev, thresh, skew, ent
        )
        s = run_config(config_name, model, events, min_edge=0.02)
        all_results.append(s)
        print_result_row(s)

        if best_phase5 is None or (s["brier"] is not None and s["brier"] < best_phase5["brier"]):
            best_phase5 = s

    print()
    if best_phase5:
        print("  >> Best Phase 5: {}  Brier={:.4f} ROI={:+.1f}%".format(
            best_phase5["config_name"], best_phase5["brier"], best_phase5["roi"]))
    print()

    # -----------------------------------------------------------------------
    # Phase 6: Best configs from above, sweep min_edge
    # -----------------------------------------------------------------------
    print("--- Phase 6: Min-Edge Sweep on Best Configs ---")

    # Collect the top 5 configs by Brier score
    scored = [r for r in all_results if r["brier"] is not None and r["min_edge"] == 0.02]
    scored.sort(key=lambda x: x["brier"])
    top_configs = scored[:5]

    print("  Top 5 configs (by Brier at min_edge=2%):")
    for i, s in enumerate(top_configs):
        print("    {}. {} Brier={:.4f} ROI={:+.1f}%".format(
            i + 1, s["config_name"], s["brier"], s["roi"]))
    print()

    # Rebuild and sweep min_edge for each top config
    # We need to reconstruct the model from the config_name...
    # Instead, let's just run a focused sweep on interesting configs
    best_configs_for_sweep = [
        # (name, model_kwargs)
        ("BestRev", {
            "reversal_strength": 0.15, "shift_threshold": 10.0,
            "skew_correction": 0.0, "entropy_correction": 0.0,
            "use_adjusted_base": False,
        }),
        ("BestCombo", {
            "reversal_strength": 0.15, "shift_threshold": 10.0,
            "skew_correction": 0.05, "entropy_correction": 0.0,
            "use_adjusted_base": False,
        }),
        ("BestAdj", {
            "reversal_strength": 0.15, "shift_threshold": 10.0,
            "skew_correction": 0.05, "entropy_correction": 0.0,
            "use_adjusted_base": True,
        }),
        ("AggressiveRev", {
            "reversal_strength": 0.30, "shift_threshold": 10.0,
            "skew_correction": 0.05, "entropy_correction": 0.0,
            "use_adjusted_base": False,
        }),
        ("LowThresh", {
            "reversal_strength": 0.20, "shift_threshold": 5.0,
            "skew_correction": 0.10, "entropy_correction": 0.0,
            "use_adjusted_base": False,
        }),
    ]

    min_edges = [0.01, 0.02, 0.03, 0.05]
    sweep_results = []

    for cfg_name, kwargs in best_configs_for_sweep:
        for min_edge in min_edges:
            model = ContrarianModel(**kwargs)
            full_name = "{} (edge={:.0%})".format(cfg_name, min_edge)
            s = run_config(full_name, model, events, min_edge=min_edge)
            sweep_results.append(s)
            print_result_row(s)
        print()

    all_results.extend(sweep_results)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 110)
    print("SUMMARY: Best Configurations")
    print("=" * 110)
    print()

    # Best by Brier
    scored_all = [r for r in all_results if r["brier"] is not None]
    scored_all.sort(key=lambda x: x["brier"])

    print("Top 10 by Brier Score:")
    print("-" * 110)
    for i, s in enumerate(scored_all[:10]):
        print_result_row(s)

    print()

    # Best by ROI (only among configs with positive ROI and >= 5 bets)
    profitable = [r for r in all_results if r["roi"] > 0 and r["n_bets"] >= 3]
    profitable.sort(key=lambda x: x["roi"], reverse=True)

    if profitable:
        print("Top 10 by ROI (min 3 bets, positive ROI):")
        print("-" * 110)
        for i, s in enumerate(profitable[:10]):
            print_result_row(s)
    else:
        print("No configurations achieved positive ROI with >= 3 bets.")

    print()

    # Best by combined metric: Brier < crowd AND ROI > 0
    crowd_brier = None
    for r in all_results:
        if r["config_name"] == "Crowd (baseline)" and r["min_edge"] == 0.02:
            crowd_brier = r["brier"]
            break

    if crowd_brier is not None:
        beats_crowd = [
            r for r in all_results
            if r["brier"] is not None
            and r["brier"] < crowd_brier
        ]
        beats_crowd.sort(key=lambda x: x["brier"])

        print("Configs that BEAT crowd Brier ({:.4f}):".format(crowd_brier))
        print("-" * 110)
        if beats_crowd:
            for s in beats_crowd[:15]:
                print_result_row(s)
        else:
            print("  None found.")

    print()
    print("=" * 110)
    print("DONE. Total configurations tested: {}".format(len(all_results)))
    print("=" * 110)


if __name__ == "__main__":
    main()
