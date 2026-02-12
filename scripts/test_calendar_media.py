"""
Test script for CalendarMediaModel: grid-search over hyperparameters
and report best configurations.

Runs multiple model configurations against gold-tier backtest events,
testing at various min_edge thresholds.

Usage:
    python scripts/test_calendar_media.py
"""

import json
import sys
from pathlib import Path

# Path setup
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.backtesting.engine import BacktestEngine
from src.ml.calendar_media_model import CalendarMediaModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


def load_gold_events():
    """Load gold-tier events from the backtest index."""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    events = index.get("events", [])
    gold = [e for e in events if e.get("ground_truth_tier") == "gold"]
    return gold


def run_config(config, events, min_edges):
    """Run a single model configuration at multiple min_edge levels.

    Returns list of result dicts, one per min_edge.
    """
    results = []
    for me in min_edges:
        model = CalendarMediaModel(**config)
        engine = BacktestEngine(config={
            "bankroll": 1000.0,
            "kelly_fraction": 0.25,
            "min_edge": me,
            "max_bet_pct": 0.05,
            "entry_hours_before_close": 24,
            "entry_window_hours": 6,
        })
        result = engine.run(model, events)
        results.append({
            "min_edge": me,
            "brier": result.brier_score,
            "log_loss": result.log_loss,
            "accuracy": result.accuracy_str,
            "n_bets": result.n_bets,
            "n_wins": result.n_wins,
            "wagered": result.total_wagered,
            "pnl": result.total_pnl,
            "roi": result.roi,
        })
    return results


def format_config(config):
    """Format a config dict as a compact string."""
    parts = []
    for k, v in config.items():
        if k in ("name", "version"):
            continue
        if isinstance(v, float):
            parts.append("{}={:.3f}".format(k, v))
        elif isinstance(v, bool):
            parts.append("{}={}".format(k, v))
        elif isinstance(v, str):
            parts.append("{}={}".format(k, v))
        else:
            parts.append("{}={}".format(k, v))
    return ", ".join(parts)


def print_results_table(config_name, config, results):
    """Print results for one configuration across all min_edges."""
    print("\n" + "=" * 80)
    print("CONFIG: {}".format(config_name))
    print("  {}".format(format_config(config)))
    print("-" * 80)
    print("{:>8s}  {:>7s}  {:>8s}  {:>8s}  {:>6s}  {:>5s}  {:>10s}  {:>10s}  {:>8s}".format(
        "MinEdge", "Brier", "LogLoss", "Accuracy", "Bets", "Wins", "Wagered", "P&L", "ROI"))
    print("-" * 80)
    for r in results:
        brier_str = "{:.4f}".format(r["brier"]) if r["brier"] is not None else "N/A"
        ll_str = "{:.4f}".format(r["log_loss"]) if r["log_loss"] is not None else "N/A"
        print("{:>7.0%}  {:>7s}  {:>8s}  {:>8s}  {:>6d}  {:>5d}  {:>10s}  {:>10s}  {:>7.1f}%".format(
            r["min_edge"],
            brier_str,
            ll_str,
            r["accuracy"],
            r["n_bets"],
            r["n_wins"],
            "${:,.2f}".format(r["wagered"]),
            "${:+,.2f}".format(r["pnl"]),
            r["roi"],
        ))


def analyze_feature_distributions(gold_events):
    """Pre-scan gold events to compute actual feature statistics.

    Returns dict with mean/std for key features used by CalendarMediaModel.
    """
    import math as m
    EVENTS_DIR = BACKTEST_DIR / "events"

    launches = []
    tones = []
    interactions = []

    for evt in gold_events:
        slug = evt["event_slug"]
        feat_path = EVENTS_DIR / slug / "features.json"
        if not feat_path.exists():
            continue
        with open(feat_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        cal = features.get("calendar", {})
        med = features.get("media", {})
        crs = features.get("cross", {})

        lt7 = cal.get("launches_trailing_7d")
        if lt7 is not None:
            launches.append(float(lt7))

        tone7 = med.get("elon_musk_tone_7d")
        if tone7 is not None:
            tones.append(float(tone7))

        bp = crs.get("bad_press_x_low_activity")
        if bp is not None:
            interactions.append(float(bp))

    def stats(vals):
        if not vals:
            return None, None, None, None
        n = len(vals)
        mean = sum(vals) / n
        std = m.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
        return mean, std, min(vals), max(vals)

    launch_stats = stats(launches)
    tone_stats = stats(tones)
    interact_stats = stats(interactions)

    print("\n--- Feature Distribution Analysis ({} gold events) ---".format(
        len(gold_events)))
    print("  launches_trailing_7d:      mean={:.2f}, std={:.2f}, min={}, max={}".format(
        *launch_stats) if launch_stats[0] is not None else "  launches_trailing_7d: N/A")
    print("  elon_musk_tone_7d:         mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
        *tone_stats) if tone_stats[0] is not None else "  elon_musk_tone_7d: N/A")
    print("  bad_press_x_low_activity:  mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}".format(
        *interact_stats) if interact_stats[0] is not None else "  bad_press_x_low_activity: N/A")
    print()

    return {
        "launch_mean": launch_stats[0],
        "launch_std": launch_stats[1],
        "tone_mean": tone_stats[0],
        "tone_std": tone_stats[1],
    }


def main():
    print("=" * 80)
    print("CALENDAR-MEDIA FUSION MODEL - GRID SEARCH")
    print("=" * 80)

    # Load events
    gold_events = load_gold_events()
    print("\nLoaded {} gold-tier events".format(len(gold_events)))

    # Analyze feature distributions
    feat_stats = analyze_feature_distributions(gold_events)

    min_edges = [0.01, 0.02, 0.03, 0.05]

    # Use computed feature stats for z-scoring
    z_params = {}
    if feat_stats.get("launch_mean") is not None:
        z_params["launch_mean"] = feat_stats["launch_mean"]
        z_params["launch_std"] = feat_stats["launch_std"]
        z_params["tone_mean"] = feat_stats["tone_mean"]
        z_params["tone_std"] = feat_stats["tone_std"]
        print("Using computed z-scoring: launch({:.2f}+/-{:.2f}), tone({:.2f}+/-{:.2f})".format(
            z_params["launch_mean"], z_params["launch_std"],
            z_params["tone_mean"], z_params["tone_std"],
        ))
    else:
        print("WARNING: Could not compute feature stats, using defaults")

    # -----------------------------------------------------------------------
    # Define configurations to test
    # -----------------------------------------------------------------------
    configs = {}

    # Config 1: Basic linear shift (conservative)
    configs["1_basic_linear"] = {
        "launch_weight": 0.06,
        "tone_weight": 0.06,
        "shift_scale": 0.12,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.20,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 2: Higher weights, still linear
    configs["2_moderate_linear"] = {
        "launch_weight": 0.10,
        "tone_weight": 0.10,
        "shift_scale": 0.18,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.25,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 3: With interaction term
    configs["3_with_interaction"] = {
        "launch_weight": 0.08,
        "tone_weight": 0.08,
        "shift_scale": 0.15,
        "interaction_weight": 0.05,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.25,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 4: Tanh non-linear with moderate weights
    configs["4_tanh_nonlinear"] = {
        "launch_weight": 0.12,
        "tone_weight": 0.12,
        "shift_scale": 0.15,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": True,
        "max_shift": 0.20,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 5: Z-threshold filtering (only act on strong signals)
    configs["5_z_threshold_1.0"] = {
        "launch_weight": 0.12,
        "tone_weight": 0.12,
        "shift_scale": 0.20,
        "interaction_weight": 0.0,
        "z_threshold": 1.0,
        "use_tanh": False,
        "max_shift": 0.25,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 6: MarketAdjusted base (stack on top of existing best model)
    configs["6_on_mkt_adjusted"] = {
        "launch_weight": 0.06,
        "tone_weight": 0.06,
        "shift_scale": 0.10,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.15,
        "widen_on_conflict": 0.0,
        "base_model": "market_adjusted",
        **z_params,
    }

    # Config 7: Widen on conflicting signals
    configs["7_widen_on_conflict"] = {
        "launch_weight": 0.08,
        "tone_weight": 0.08,
        "shift_scale": 0.15,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.20,
        "widen_on_conflict": 0.5,
        "base_model": "crowd",
        **z_params,
    }

    # Config 8: Tone-only (ablation: is tone the real edge?)
    configs["8_tone_only"] = {
        "launch_weight": 0.0,
        "tone_weight": 0.12,
        "shift_scale": 0.20,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.25,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 9: Launch-only (ablation: is launch the real edge?)
    configs["9_launch_only"] = {
        "launch_weight": 0.12,
        "tone_weight": 0.0,
        "shift_scale": 0.20,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.25,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 10: Aggressive with tanh + interaction + z-threshold
    configs["10_aggressive_combo"] = {
        "launch_weight": 0.15,
        "tone_weight": 0.15,
        "shift_scale": 0.20,
        "interaction_weight": 0.08,
        "z_threshold": 0.5,
        "use_tanh": True,
        "max_shift": 0.30,
        "widen_on_conflict": 0.3,
        "base_model": "crowd",
        **z_params,
    }

    # Config 11: Conservative on MarketAdjusted + z-threshold
    configs["11_mkt_adj_z_thresh"] = {
        "launch_weight": 0.08,
        "tone_weight": 0.08,
        "shift_scale": 0.12,
        "interaction_weight": 0.03,
        "z_threshold": 0.8,
        "use_tanh": False,
        "max_shift": 0.15,
        "widen_on_conflict": 0.0,
        "base_model": "market_adjusted",
        **z_params,
    }

    # Config 12: Very subtle shifts (micro-edge hunting)
    configs["12_micro_edge"] = {
        "launch_weight": 0.04,
        "tone_weight": 0.04,
        "shift_scale": 0.08,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.10,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 13: Asymmetric weights -- tone stronger than launches
    # (motivated by crowd capturing only 48% of downswings)
    configs["13_tone_heavy"] = {
        "launch_weight": 0.04,
        "tone_weight": 0.15,
        "shift_scale": 0.18,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": False,
        "max_shift": 0.25,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 14: Launch-heavy with tanh (SpaceX busy -> fewer tweets is cleaner signal)
    configs["14_launch_heavy_tanh"] = {
        "launch_weight": 0.18,
        "tone_weight": 0.05,
        "shift_scale": 0.15,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": True,
        "max_shift": 0.20,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 15: Strong z-threshold (1.5) -- only fire on extreme signals
    configs["15_z_thresh_1.5"] = {
        "launch_weight": 0.15,
        "tone_weight": 0.15,
        "shift_scale": 0.25,
        "interaction_weight": 0.0,
        "z_threshold": 1.5,
        "use_tanh": False,
        "max_shift": 0.30,
        "widen_on_conflict": 0.0,
        "base_model": "crowd",
        **z_params,
    }

    # Config 16: MarketAdjusted + strong tone + tanh (layer refinement)
    configs["16_mktadj_tone_tanh"] = {
        "launch_weight": 0.04,
        "tone_weight": 0.12,
        "shift_scale": 0.12,
        "interaction_weight": 0.0,
        "z_threshold": 0.0,
        "use_tanh": True,
        "max_shift": 0.15,
        "widen_on_conflict": 0.0,
        "base_model": "market_adjusted",
        **z_params,
    }

    # -----------------------------------------------------------------------
    # Run all configs
    # -----------------------------------------------------------------------
    all_results = {}
    best_roi = -999.0
    best_config_name = None
    best_min_edge = None

    for config_name, config in configs.items():
        print("\nRunning config: {} ...".format(config_name))
        results = run_config(config, gold_events, min_edges)
        all_results[config_name] = results
        print_results_table(config_name, config, results)

        # Track best
        for r in results:
            if r["n_bets"] > 0 and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_config_name = config_name
                best_min_edge = r["min_edge"]

    # -----------------------------------------------------------------------
    # Also run the CrowdModel baseline for comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BASELINE: CrowdModel (market prices as-is)")
    print("-" * 80)
    from src.ml.baseline_model import CrowdModel
    crowd_model = CrowdModel()
    baseline_results = []
    for me in min_edges:
        engine = BacktestEngine(config={
            "bankroll": 1000.0,
            "kelly_fraction": 0.25,
            "min_edge": me,
            "max_bet_pct": 0.05,
            "entry_hours_before_close": 24,
            "entry_window_hours": 6,
        })
        result = engine.run(crowd_model, gold_events)
        baseline_results.append({
            "min_edge": me,
            "brier": result.brier_score,
            "log_loss": result.log_loss,
            "accuracy": result.accuracy_str,
            "n_bets": result.n_bets,
            "n_wins": result.n_wins,
            "wagered": result.total_wagered,
            "pnl": result.total_pnl,
            "roi": result.roi,
        })

    print_results_table("BASELINE_crowd", {"model": "CrowdModel"}, baseline_results)

    # Also run MarketAdjusted baseline
    print("\n" + "=" * 80)
    print("BASELINE: MarketAdjustedModel")
    print("-" * 80)
    from src.ml.advanced_models import MarketAdjustedModel
    ma_model = MarketAdjustedModel()
    ma_results = []
    for me in min_edges:
        engine = BacktestEngine(config={
            "bankroll": 1000.0,
            "kelly_fraction": 0.25,
            "min_edge": me,
            "max_bet_pct": 0.05,
            "entry_hours_before_close": 24,
            "entry_window_hours": 6,
        })
        result = engine.run(ma_model, gold_events)
        ma_results.append({
            "min_edge": me,
            "brier": result.brier_score,
            "log_loss": result.log_loss,
            "accuracy": result.accuracy_str,
            "n_bets": result.n_bets,
            "n_wins": result.n_wins,
            "wagered": result.total_wagered,
            "pnl": result.total_pnl,
            "roi": result.roi,
        })

    print_results_table("BASELINE_mkt_adjusted", {"model": "MarketAdjustedModel"}, ma_results)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: BEST RESULTS BY ROI (with bets > 0)")
    print("=" * 80)

    # Collect all (config, min_edge, result) tuples with bets
    all_combos = []
    for config_name, results in all_results.items():
        for r in results:
            if r["n_bets"] > 0:
                all_combos.append((config_name, r))

    # Sort by ROI descending
    all_combos.sort(key=lambda x: x[1]["roi"], reverse=True)

    print("\n{:<25s} {:>8s} {:>7s} {:>8s} {:>6s} {:>5s} {:>10s} {:>8s}".format(
        "Config", "MinEdge", "Brier", "Accuracy", "Bets", "Wins", "P&L", "ROI"))
    print("-" * 90)
    for config_name, r in all_combos[:20]:
        brier_str = "{:.4f}".format(r["brier"]) if r["brier"] is not None else "N/A"
        print("{:<25s} {:>7.0%} {:>7s} {:>8s} {:>6d} {:>5d} {:>10s} {:>7.1f}%".format(
            config_name,
            r["min_edge"],
            brier_str,
            r["accuracy"],
            r["n_bets"],
            r["n_wins"],
            "${:+,.2f}".format(r["pnl"]),
            r["roi"],
        ))

    # Also show Brier ranking
    print("\n" + "=" * 80)
    print("SUMMARY: BEST RESULTS BY BRIER SCORE (lower is better)")
    print("=" * 80)

    # All configs at min_edge 0.01 (captures calibration before filtering)
    brier_combos = []
    for config_name, results in all_results.items():
        for r in results:
            if r["min_edge"] == 0.01 and r["brier"] is not None:
                brier_combos.append((config_name, r))

    brier_combos.sort(key=lambda x: x[1]["brier"])

    print("\n{:<25s} {:>7s} {:>8s} {:>8s} {:>6s} {:>10s} {:>8s}".format(
        "Config", "Brier", "LogLoss", "Accuracy", "Bets", "P&L", "ROI"))
    print("-" * 80)

    # Add baselines to the Brier ranking
    for r in baseline_results:
        if r["min_edge"] == 0.01 and r["brier"] is not None:
            brier_combos.append(("BASELINE_crowd", r))
    for r in ma_results:
        if r["min_edge"] == 0.01 and r["brier"] is not None:
            brier_combos.append(("BASELINE_mkt_adj", r))

    brier_combos.sort(key=lambda x: x[1]["brier"])

    for config_name, r in brier_combos:
        brier_str = "{:.4f}".format(r["brier"]) if r["brier"] is not None else "N/A"
        ll_str = "{:.4f}".format(r["log_loss"]) if r["log_loss"] is not None else "N/A"
        print("{:<25s} {:>7s} {:>8s} {:>8s} {:>6d} {:>10s} {:>7.1f}%".format(
            config_name,
            brier_str,
            ll_str,
            r["accuracy"],
            r["n_bets"],
            "${:+,.2f}".format(r["pnl"]),
            r["roi"],
        ))

    print("\n" + "=" * 80)
    if best_config_name:
        print("BEST CONFIG: {} at min_edge={:.0%} -> ROI={:+.1f}%".format(
            best_config_name, best_min_edge, best_roi))
    else:
        print("NO CONFIGS GENERATED ANY BETS")
    print("=" * 80)


if __name__ == "__main__":
    main()
