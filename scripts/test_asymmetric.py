"""
Test and grid-search both asymmetric model variants on gold-tier events.

Tests:
    1. AsymmetricMomentumModel - direct on crowd prices
    2. AsymmetricPerBucketModel - MarketAdjusted base + asymmetric z-score

Searches over alpha_down, alpha_up, and supporting parameters.
Tests at min_edge thresholds: 0.01, 0.02, 0.03, 0.05.

Reports Brier score, ROI, number of bets for each configuration.

Usage:
    set PYTHONIOENCODING=utf-8
    python scripts/test_asymmetric.py
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
from src.ml.asymmetric_model import AsymmetricMomentumModel, AsymmetricPerBucketModel

# ---------------------------------------------------------------------------
# Load gold events
# ---------------------------------------------------------------------------
INDEX_PATH = PROJECT_DIR / "data" / "backtest" / "backtest_index.json"


def load_gold_events():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    gold = [e for e in index["events"] if e.get("ground_truth_tier") == "gold"]
    print("Loaded {} gold-tier events".format(len(gold)))
    return gold


# ---------------------------------------------------------------------------
# Run a single configuration
# ---------------------------------------------------------------------------

def run_model(model, events, min_edge):
    """Run backtest for a model at a given min_edge."""
    engine = BacktestEngine(config={
        "min_edge": min_edge,
        "bankroll": 1000.0,
        "kelly_fraction": 0.25,
        "max_bet_pct": 0.05,
        "entry_hours_before_close": 24,
        "entry_window_hours": 6,
    })
    return engine.run(model, events)


# ---------------------------------------------------------------------------
# Test a batch of configs
# ---------------------------------------------------------------------------

def test_configs(configs, events, min_edges, section_label):
    """Run all configs at all min_edge levels and collect results."""
    results = []
    best_roi = -9999
    best_label = None
    best_edge = None

    print()
    print("=" * 90)
    print(section_label)
    print("=" * 90)

    for cfg in configs:
        label = cfg["label"]
        model = cfg["model"]

        print()
        print("-" * 90)
        print("Testing: {}".format(label))
        print("  Hyperparameters: {}".format(model.get_hyperparameters()))
        print()

        for me in min_edges:
            result = run_model(model, events, me)

            brier = result.brier_score if result.brier_score is not None else float("nan")
            roi = result.roi
            n_bets = result.n_bets
            n_wins = result.n_wins
            pnl = result.total_pnl
            wagered = result.total_wagered

            results.append({
                "label": label,
                "min_edge": me,
                "brier": brier,
                "roi": roi,
                "n_bets": n_bets,
                "n_wins": n_wins,
                "pnl": pnl,
                "wagered": wagered,
                "section": section_label,
            })

            roi_marker = " ***" if roi > 0 and n_bets > 0 else ""
            print(
                "  min_edge={:.0%}: Brier={:.4f}, ROI={:+.1f}%, bets={}, "
                "wins={}, P&L=${:+,.2f}, wagered=${:,.2f}{}".format(
                    me, brier, roi, n_bets, n_wins, pnl, wagered, roi_marker
                )
            )

            if roi > best_roi and n_bets > 0:
                best_roi = roi
                best_label = label
                best_edge = me

    return results, best_label, best_edge, best_roi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    events = load_gold_events()
    if not events:
        print("No gold events found. Check backtest index.")
        sys.exit(1)

    min_edges = [0.01, 0.02, 0.03, 0.05]
    all_results = []

    # ===================================================================
    # SECTION 1: AsymmetricMomentumModel (direct on crowd)
    # ===================================================================
    direct_configs = [
        {
            "label": "Direct C1: Strong asym (4:1), moderate reversion",
            "model": AsymmetricMomentumModel(
                alpha_down=0.40, alpha_up=0.10, max_correction=0.35,
                reversion_down=0.25, reversion_up=0.08,
                widen_on_downswing=0.15, widen_base=0.10,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "Direct C2: Moderate asym (3:1), strong reversion",
            "model": AsymmetricMomentumModel(
                alpha_down=0.30, alpha_up=0.10, max_correction=0.30,
                reversion_down=0.35, reversion_up=0.10,
                widen_on_downswing=0.12, widen_base=0.08,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "Direct C3: Very strong asym (6:1), minimal reversion",
            "model": AsymmetricMomentumModel(
                alpha_down=0.50, alpha_up=0.08, max_correction=0.40,
                reversion_down=0.15, reversion_up=0.05,
                widen_on_downswing=0.20, widen_base=0.12,
                tail_cap_multiplier=3.5,
            ),
        },
        {
            "label": "Direct C4: Conservative (2:1), tight cap",
            "model": AsymmetricMomentumModel(
                alpha_down=0.20, alpha_up=0.10, max_correction=0.20,
                reversion_down=0.20, reversion_up=0.10,
                widen_on_downswing=0.08, widen_base=0.06,
                tail_cap_multiplier=2.5,
            ),
        },
        {
            "label": "Direct C5: Down-only (alpha_up=0)",
            "model": AsymmetricMomentumModel(
                alpha_down=0.35, alpha_up=0.00, max_correction=0.30,
                reversion_down=0.28, reversion_up=0.00,
                widen_on_downswing=0.18, widen_base=0.10,
                tail_cap_multiplier=3.0,
            ),
        },
    ]

    r1, best1_label, best1_edge, best1_roi = test_configs(
        direct_configs, events, min_edges,
        "SECTION 1: AsymmetricMomentumModel (direct on crowd)"
    )
    all_results.extend(r1)

    # ===================================================================
    # SECTION 2: AsymmetricPerBucketModel (MarketAdjusted base)
    # ===================================================================
    perbucket_configs = [
        {
            "label": "PerBucket C1: Mild asym (2:1) -- alpha_down=0.32, up=0.16",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.32, alpha_up=0.16, max_correction=0.30,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "PerBucket C2: Strong asym (3:1) -- alpha_down=0.36, up=0.12",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.36, alpha_up=0.12, max_correction=0.30,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "PerBucket C3: Extreme asym (5:1) -- alpha_down=0.45, up=0.09",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.45, alpha_up=0.09, max_correction=0.35,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "PerBucket C4: Down-heavy (alpha_down=0.40, up=0.05)",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.40, alpha_up=0.05, max_correction=0.35,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "PerBucket C5: Symmetric baseline (alpha=0.24 both)",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.24, alpha_up=0.24, max_correction=0.30,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "PerBucket C6: Gentle asym (1.5:1) -- alpha_down=0.28, up=0.18",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.28, alpha_up=0.18, max_correction=0.30,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "PerBucket C7: Strong down, zero up -- alpha_down=0.35, up=0.00",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.35, alpha_up=0.00, max_correction=0.30,
                tail_cap_multiplier=3.0,
            ),
        },
        {
            "label": "PerBucket C8: Higher cap -- alpha_down=0.36, up=0.12, cap=0.40",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.36, alpha_up=0.12, max_correction=0.40,
                tail_cap_multiplier=3.5,
            ),
        },
        {
            "label": "PerBucket C9: Lower cap -- alpha_down=0.36, up=0.12, cap=0.20",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.36, alpha_up=0.12, max_correction=0.20,
                tail_cap_multiplier=2.5,
            ),
        },
        {
            "label": "PerBucket C10: Very strong down=0.50, up=0.10, cap=0.35",
            "model": AsymmetricPerBucketModel(
                alpha_down=0.50, alpha_up=0.10, max_correction=0.35,
                tail_cap_multiplier=3.0,
            ),
        },
    ]

    r2, best2_label, best2_edge, best2_roi = test_configs(
        perbucket_configs, events, min_edges,
        "SECTION 2: AsymmetricPerBucketModel (MarketAdjusted base)"
    )
    all_results.extend(r2)

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print()
    print()
    print("=" * 100)
    print("FINAL SUMMARY - ALL RESULTS SORTED BY ROI")
    print("=" * 100)
    print()

    with_bets = [r for r in all_results if r["n_bets"] > 0]
    with_bets.sort(key=lambda r: r["roi"], reverse=True)

    print("{:<60s} {:>5s} {:>8s} {:>7s} {:>5s} {:>5s} {:>12s}".format(
        "Config", "Edge", "Brier", "ROI", "Bets", "Wins", "P&L"
    ))
    print("-" * 110)

    for r in with_bets:
        print("{:<60s} {:>4.0%} {:>8.4f} {:>+6.1f}% {:>5d} {:>5d} {:>12s}".format(
            r["label"][:60],
            r["min_edge"],
            r["brier"],
            r["roi"],
            r["n_bets"],
            r["n_wins"],
            "${:+,.2f}".format(r["pnl"]),
        ))

    # Brier comparison
    print()
    print("=" * 100)
    print("BRIER SCORE COMPARISON (one per config, independent of min_edge)")
    print("=" * 100)
    seen = set()
    for r in all_results:
        if r["label"] not in seen:
            seen.add(r["label"])
            print("  {}: Brier={:.4f}".format(r["label"], r["brier"]))

    print()
    print("Reference baselines:")
    print("  CrowdModel:        Brier=0.8137, ROI=0%")
    print("  MarketAdjusted:    Brier=0.8004, ROI=-10.3% at 2% edge")
    print("  PerBucketModel:    Brier=0.7993, ROI=+67.6% at 2% edge")

    # Overall best
    print()
    print("=" * 100)
    print("OVERALL BEST")
    print("=" * 100)
    if with_bets:
        best = with_bets[0]
        print("  Config:   {}".format(best["label"]))
        print("  Min edge: {:.0%}".format(best["min_edge"]))
        print("  Brier:    {:.4f}".format(best["brier"]))
        print("  ROI:      {:+.1f}%".format(best["roi"]))
        print("  Bets:     {}".format(best["n_bets"]))
        print("  Wins:     {}".format(best["n_wins"]))
        print("  P&L:      ${:+,.2f}".format(best["pnl"]))
        print("  Wagered:  ${:,.2f}".format(best["wagered"]))
    else:
        print("  No configurations produced any bets.")

    return all_results


if __name__ == "__main__":
    main()
