"""
Test script for VolatilityRegimeModel.

Grid-searches over hyperparameters and reports Brier, ROI, and bet counts
across gold-tier backtest events.

Usage:
    set PYTHONIOENCODING=utf-8
    python scripts/test_volatility.py
"""

import json
import sys
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.backtesting.engine import BacktestEngine
from src.ml.volatility_model import VolatilityRegimeModel
from src.ml.advanced_models import MarketAdjustedModel

# ---------------------------------------------------------------------------
# Load gold events
# ---------------------------------------------------------------------------
INDEX_PATH = PROJECT_DIR / "data" / "backtest" / "backtest_index.json"

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    index = json.load(f)

gold_events = [e for e in index["events"] if e.get("ground_truth_tier") == "gold"]
print("Gold-tier events: {}".format(len(gold_events)))
print()

# ---------------------------------------------------------------------------
# Baseline: MarketAdjustedModel at various min_edge
# ---------------------------------------------------------------------------
print("=" * 80)
print("BASELINE: MarketAdjustedModel")
print("=" * 80)

for min_edge in [0.01, 0.02, 0.03, 0.05]:
    engine = BacktestEngine(config={"min_edge": min_edge})
    baseline = MarketAdjustedModel()
    result = engine.run(baseline, gold_events)
    print(
        "  min_edge={:.2f}  Brier={:.4f}  ROI={:+.1f}%  bets={}  wagered=${:.0f}  pnl=${:+.0f}  wins={}/{}  acc={}".format(
            min_edge,
            result.brier_score if result.brier_score else 0.0,
            result.roi,
            result.n_bets,
            result.total_wagered,
            result.total_pnl,
            result.n_wins,
            result.n_bets,
            result.accuracy_str,
        )
    )

print()

# ---------------------------------------------------------------------------
# Helper to run one configuration
# ---------------------------------------------------------------------------
def run_config(config_dict, min_edge_values, label=""):
    """Run a single model config across multiple min_edge values."""
    model = VolatilityRegimeModel(**config_dict)

    results = []
    for min_edge in min_edge_values:
        engine = BacktestEngine(config={"min_edge": min_edge})
        result = engine.run(model, gold_events)
        results.append({
            "min_edge": min_edge,
            "brier": result.brier_score,
            "roi": result.roi,
            "n_bets": result.n_bets,
            "n_wins": result.n_wins,
            "total_wagered": result.total_wagered,
            "total_pnl": result.total_pnl,
            "accuracy_str": result.accuracy_str,
        })

    # Print results
    if label:
        print("  --- {} ---".format(label))
    print("  Config: {}".format(
        {k: v for k, v in config_dict.items() if k not in ("name", "version")}
    ))
    for r in results:
        print(
            "    min_edge={:.2f}  Brier={:.4f}  ROI={:+.1f}%  bets={}  wagered=${:.0f}  pnl=${:+.0f}  wins={}/{}".format(
                r["min_edge"],
                r["brier"] if r["brier"] else 0.0,
                r["roi"],
                r["n_bets"],
                r["total_wagered"],
                r["total_pnl"],
                r["n_wins"],
                r["n_bets"],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Common min_edge values
# ---------------------------------------------------------------------------
MIN_EDGES = [0.01, 0.02, 0.03, 0.05]


# ===========================================================================
# ITERATION 1: Power-law with cv_14d, standard thresholds
# ===========================================================================
print("=" * 80)
print("ITERATION 1: Power-law sharpening/widening with cv_14d")
print("=" * 80)

configs_1 = []
for sharpen_a in [1.1, 1.2, 1.3, 1.5]:
    for widen_a in [0.7, 0.8, 0.9]:
        for high_th in [0.40, 0.45, 0.50]:
            for low_th in [0.30, 0.35]:
                configs_1.append({
                    "vol_metric": "cv_14d",
                    "vol_threshold_high": high_th,
                    "vol_threshold_low": low_th,
                    "sharpen_alpha": sharpen_a,
                    "widen_alpha": widen_a,
                    "use_blend": False,
                    "reversion_strength": 1.0,
                    "market_cap": 3.0,
                })

print("Testing {} configurations...".format(len(configs_1)))

best_1 = {"roi": -999, "config": None, "results": None}
for i, cfg in enumerate(configs_1):
    # Quick test at min_edge=0.02 first
    model = VolatilityRegimeModel(**cfg)
    engine = BacktestEngine(config={"min_edge": 0.02})
    result = engine.run(model, gold_events)

    if result.roi > best_1["roi"]:
        best_1["roi"] = result.roi
        best_1["config"] = cfg
        best_1["brier"] = result.brier_score
        best_1["n_bets"] = result.n_bets

    if (i + 1) % 20 == 0:
        print("  ... tested {}/{}, best ROI so far: {:+.1f}%".format(
            i + 1, len(configs_1), best_1["roi"]
        ))

print()
print("BEST from Iteration 1 (at min_edge=0.02):")
print("  ROI={:+.1f}%, Brier={:.4f}, bets={}".format(
    best_1["roi"], best_1["brier"] if best_1["brier"] else 0, best_1["n_bets"]
))
print("  Config: {}".format(best_1["config"]))

# Run best across all min_edges
print()
print("Best config across all min_edge values:")
run_config(best_1["config"], MIN_EDGES, label="Iter1 Best")
print()


# ===========================================================================
# ITERATION 2: rolling_std_7d (normalized) instead of cv_14d
# ===========================================================================
print("=" * 80)
print("ITERATION 2: Power-law with rolling_std_7d (normalized by avg)")
print("=" * 80)

configs_2 = []
for sharpen_a in [1.1, 1.2, 1.3, 1.5]:
    for widen_a in [0.7, 0.8, 0.9]:
        for high_th in [0.35, 0.45, 0.55]:
            for low_th in [0.20, 0.25, 0.30]:
                configs_2.append({
                    "vol_metric": "rolling_std_7d",
                    "vol_threshold_high": high_th,
                    "vol_threshold_low": low_th,
                    "sharpen_alpha": sharpen_a,
                    "widen_alpha": widen_a,
                    "use_blend": False,
                    "reversion_strength": 1.0,
                    "market_cap": 3.0,
                })

print("Testing {} configurations...".format(len(configs_2)))

best_2 = {"roi": -999, "config": None}
for i, cfg in enumerate(configs_2):
    model = VolatilityRegimeModel(**cfg)
    engine = BacktestEngine(config={"min_edge": 0.02})
    result = engine.run(model, gold_events)

    if result.roi > best_2["roi"]:
        best_2["roi"] = result.roi
        best_2["config"] = cfg
        best_2["brier"] = result.brier_score
        best_2["n_bets"] = result.n_bets

    if (i + 1) % 20 == 0:
        print("  ... tested {}/{}, best ROI so far: {:+.1f}%".format(
            i + 1, len(configs_2), best_2["roi"]
        ))

print()
print("BEST from Iteration 2 (at min_edge=0.02):")
print("  ROI={:+.1f}%, Brier={:.4f}, bets={}".format(
    best_2["roi"], best_2["brier"] if best_2["brier"] else 0, best_2["n_bets"]
))
print("  Config: {}".format(best_2["config"]))

print()
print("Best config across all min_edge values:")
run_config(best_2["config"], MIN_EDGES, label="Iter2 Best")
print()


# ===========================================================================
# ITERATION 3: Blend-toward-uniform (alternative to power-law)
# ===========================================================================
print("=" * 80)
print("ITERATION 3: Blend-toward-uniform with cv_14d")
print("=" * 80)

configs_3 = []
for sharpen_b in [-0.05, -0.10, -0.15, -0.20, -0.30]:
    for widen_b in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for high_th in [0.40, 0.45, 0.50]:
            for low_th in [0.30, 0.35]:
                configs_3.append({
                    "vol_metric": "cv_14d",
                    "vol_threshold_high": high_th,
                    "vol_threshold_low": low_th,
                    "sharpen_alpha": 1.3,  # unused
                    "widen_alpha": 0.8,    # unused
                    "use_blend": True,
                    "sharpen_blend": sharpen_b,
                    "widen_blend": widen_b,
                    "reversion_strength": 1.0,
                    "market_cap": 3.0,
                })

print("Testing {} configurations...".format(len(configs_3)))

best_3 = {"roi": -999, "config": None}
for i, cfg in enumerate(configs_3):
    model = VolatilityRegimeModel(**cfg)
    engine = BacktestEngine(config={"min_edge": 0.02})
    result = engine.run(model, gold_events)

    if result.roi > best_3["roi"]:
        best_3["roi"] = result.roi
        best_3["config"] = cfg
        best_3["brier"] = result.brier_score
        best_3["n_bets"] = result.n_bets

    if (i + 1) % 20 == 0:
        print("  ... tested {}/{}, best ROI so far: {:+.1f}%".format(
            i + 1, len(configs_3), best_3["roi"]
        ))

print()
print("BEST from Iteration 3 (at min_edge=0.02):")
print("  ROI={:+.1f}%, Brier={:.4f}, bets={}".format(
    best_3["roi"], best_3["brier"] if best_3["brier"] else 0, best_3["n_bets"]
))
print("  Config: {}".format(best_3["config"]))

print()
print("Best config across all min_edge values:")
run_config(best_3["config"], MIN_EDGES, label="Iter3 Best")
print()


# ===========================================================================
# ITERATION 4: Extreme-only (top/bottom quartile of vol)
# ===========================================================================
print("=" * 80)
print("ITERATION 4: Extreme-only regimes with power-law")
print("=" * 80)

configs_4 = []
# More extreme thresholds for quartile-based application
for sharpen_a in [1.2, 1.3, 1.5, 1.8]:
    for widen_a in [0.6, 0.7, 0.8]:
        for high_th in [0.50, 0.55, 0.60]:
            for low_th in [0.25, 0.30]:
                configs_4.append({
                    "vol_metric": "cv_14d",
                    "vol_threshold_high": high_th,
                    "vol_threshold_low": low_th,
                    "sharpen_alpha": sharpen_a,
                    "widen_alpha": widen_a,
                    "use_blend": False,
                    "extreme_only": True,
                    "reversion_strength": 1.0,
                    "market_cap": 3.0,
                })

print("Testing {} configurations...".format(len(configs_4)))

best_4 = {"roi": -999, "config": None}
for i, cfg in enumerate(configs_4):
    model = VolatilityRegimeModel(**cfg)
    engine = BacktestEngine(config={"min_edge": 0.02})
    result = engine.run(model, gold_events)

    if result.roi > best_4["roi"]:
        best_4["roi"] = result.roi
        best_4["config"] = cfg
        best_4["brier"] = result.brier_score
        best_4["n_bets"] = result.n_bets

    if (i + 1) % 20 == 0:
        print("  ... tested {}/{}, best ROI so far: {:+.1f}%".format(
            i + 1, len(configs_4), best_4["roi"]
        ))

print()
print("BEST from Iteration 4 (at min_edge=0.02):")
print("  ROI={:+.1f}%, Brier={:.4f}, bets={}".format(
    best_4["roi"], best_4["brier"] if best_4["brier"] else 0, best_4["n_bets"]
))
print("  Config: {}".format(best_4["config"]))

print()
print("Best config across all min_edge values:")
run_config(best_4["config"], MIN_EDGES, label="Iter4 Best")
print()


# ===========================================================================
# ITERATION 5: Variable reversion_strength + fine-tuning best approach
# ===========================================================================
print("=" * 80)
print("ITERATION 5: Fine-tuning best approach with reversion_strength")
print("=" * 80)

# Determine which iteration was best
iter_bests = [
    (1, best_1),
    (2, best_2),
    (3, best_3),
    (4, best_4),
]
overall_best_iter, overall_best = max(iter_bests, key=lambda x: x[1]["roi"])
print("Overall best so far: Iteration {} with ROI {:+.1f}%".format(
    overall_best_iter, overall_best["roi"]
))

# Fine-tune around the best config
base_cfg = dict(overall_best["config"])

configs_5 = []
# Try varying reversion_strength and market_cap
for rev_str in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]:
    for mcap in [2.0, 2.5, 3.0, 5.0]:
        cfg = dict(base_cfg)
        cfg["reversion_strength"] = rev_str
        cfg["market_cap"] = mcap
        configs_5.append(cfg)

# Also try the best config with very aggressive settings
if not base_cfg.get("use_blend", False):
    for sa in [base_cfg.get("sharpen_alpha", 1.3) - 0.1,
               base_cfg.get("sharpen_alpha", 1.3),
               base_cfg.get("sharpen_alpha", 1.3) + 0.1,
               base_cfg.get("sharpen_alpha", 1.3) + 0.2]:
        for wa in [base_cfg.get("widen_alpha", 0.8) - 0.1,
                   base_cfg.get("widen_alpha", 0.8),
                   base_cfg.get("widen_alpha", 0.8) + 0.1]:
            for rev_str in [0.5, 0.7, 1.0]:
                cfg = dict(base_cfg)
                cfg["sharpen_alpha"] = max(1.01, sa)
                cfg["widen_alpha"] = min(0.99, max(0.5, wa))
                cfg["reversion_strength"] = rev_str
                configs_5.append(cfg)

print("Testing {} fine-tuning configurations...".format(len(configs_5)))

best_5 = {"roi": -999, "config": None}
for i, cfg in enumerate(configs_5):
    model = VolatilityRegimeModel(**cfg)
    engine = BacktestEngine(config={"min_edge": 0.02})
    result = engine.run(model, gold_events)

    if result.roi > best_5["roi"]:
        best_5["roi"] = result.roi
        best_5["config"] = cfg
        best_5["brier"] = result.brier_score
        best_5["n_bets"] = result.n_bets

    if (i + 1) % 20 == 0:
        print("  ... tested {}/{}, best ROI so far: {:+.1f}%".format(
            i + 1, len(configs_5), best_5["roi"]
        ))

print()
print("BEST from Iteration 5 (at min_edge=0.02):")
print("  ROI={:+.1f}%, Brier={:.4f}, bets={}".format(
    best_5["roi"], best_5["brier"] if best_5["brier"] else 0, best_5["n_bets"]
))
print("  Config: {}".format(best_5["config"]))

print()
print("Best config across all min_edge values:")
run_config(best_5["config"], MIN_EDGES, label="Iter5 Best")
print()


# ===========================================================================
# ITERATION 6: Asymmetric application (only sharpen OR only widen)
# ===========================================================================
print("=" * 80)
print("ITERATION 6: Asymmetric -- sharpen-only and widen-only")
print("=" * 80)

# Sharpen-only: set low threshold very low so widen never triggers
print("  -- Sharpen-only (vol mean-reversion: high vol -> calm next period) --")
configs_6a = []
for sharpen_a in [1.1, 1.15, 1.2, 1.3, 1.5]:
    for high_th in [0.35, 0.40, 0.45, 0.50]:
        for rev_str in [0.5, 0.7, 1.0]:
            configs_6a.append({
                "vol_metric": "cv_14d",
                "vol_threshold_high": high_th,
                "vol_threshold_low": 0.0,  # never widen
                "sharpen_alpha": sharpen_a,
                "widen_alpha": 1.0,  # no-op
                "use_blend": False,
                "reversion_strength": rev_str,
                "market_cap": 3.0,
            })

best_6a = {"roi": -999, "config": None}
for cfg in configs_6a:
    model = VolatilityRegimeModel(**cfg)
    engine = BacktestEngine(config={"min_edge": 0.02})
    result = engine.run(model, gold_events)
    if result.roi > best_6a["roi"]:
        best_6a["roi"] = result.roi
        best_6a["config"] = cfg
        best_6a["brier"] = result.brier_score
        best_6a["n_bets"] = result.n_bets

print("  BEST sharpen-only: ROI={:+.1f}%, Brier={:.4f}, bets={}".format(
    best_6a["roi"], best_6a["brier"] if best_6a["brier"] else 0, best_6a["n_bets"]
))
run_config(best_6a["config"], MIN_EDGES, label="Sharpen-only Best")

# Widen-only: set high threshold very high so sharpen never triggers
print()
print("  -- Widen-only (vol mean-reversion: low vol -> wild next period) --")
configs_6b = []
for widen_a in [0.6, 0.7, 0.8, 0.9]:
    for low_th in [0.30, 0.35, 0.40, 0.45]:
        for rev_str in [0.5, 0.7, 1.0]:
            configs_6b.append({
                "vol_metric": "cv_14d",
                "vol_threshold_high": 99.0,  # never sharpen
                "vol_threshold_low": low_th,
                "sharpen_alpha": 1.0,  # no-op
                "widen_alpha": widen_a,
                "use_blend": False,
                "reversion_strength": rev_str,
                "market_cap": 3.0,
            })

best_6b = {"roi": -999, "config": None}
for cfg in configs_6b:
    model = VolatilityRegimeModel(**cfg)
    engine = BacktestEngine(config={"min_edge": 0.02})
    result = engine.run(model, gold_events)
    if result.roi > best_6b["roi"]:
        best_6b["roi"] = result.roi
        best_6b["config"] = cfg
        best_6b["brier"] = result.brier_score
        best_6b["n_bets"] = result.n_bets

print("  BEST widen-only: ROI={:+.1f}%, Brier={:.4f}, bets={}".format(
    best_6b["roi"], best_6b["brier"] if best_6b["brier"] else 0, best_6b["n_bets"]
))
run_config(best_6b["config"], MIN_EDGES, label="Widen-only Best")
print()


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

all_bests = [
    ("Iter1 (power-law cv_14d)", best_1),
    ("Iter2 (power-law std_7d)", best_2),
    ("Iter3 (blend cv_14d)", best_3),
    ("Iter4 (extreme-only)", best_4),
    ("Iter5 (fine-tuned)", best_5),
    ("Iter6a (sharpen-only)", best_6a),
    ("Iter6b (widen-only)", best_6b),
]

print("{:<30s} {:>8s} {:>8s} {:>6s}".format("Config", "ROI%", "Brier", "Bets"))
print("-" * 56)
for label, b in all_bests:
    print("{:<30s} {:>+7.1f}% {:>8.4f} {:>6d}".format(
        label,
        b["roi"],
        b["brier"] if b["brier"] else 0,
        b["n_bets"],
    ))

# Find overall best
overall_label, overall_best_final = max(all_bests, key=lambda x: x[1]["roi"])
print()
print("OVERALL BEST: {} with ROI {:+.1f}%".format(overall_label, overall_best_final["roi"]))
print("  Config: {}".format(overall_best_final["config"]))
print()

# Run overall best across all min_edges with full detail
print("Detailed results for overall best:")
run_config(overall_best_final["config"], MIN_EDGES, label="OVERALL BEST")

# Also print per-event detail for the best
print()
print("--- Per-event detail for best config at min_edge=0.02 ---")
model = VolatilityRegimeModel(**overall_best_final["config"])
engine = BacktestEngine(config={"min_edge": 0.02})
result = engine.run(model, gold_events)

for evt in result.per_event:
    if evt.get("skipped"):
        continue
    slug = evt.get("event_slug", "")[:50]
    brier = evt.get("brier_score", 0)
    n_bets = evt.get("n_bets", 0)
    pnl = evt.get("total_pnl", 0)
    winner = evt.get("winning_bucket", "?")
    wp = evt.get("winning_bucket_prob", 0)
    print("  {:<50s}  Brier={:.3f}  bets={}  pnl=${:+.0f}  winner={}  P(win)={:.3f}".format(
        slug, brier, n_bets, pnl, winner, wp
    ))
