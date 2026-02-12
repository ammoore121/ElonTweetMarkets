"""Grid search over Duration-Adaptive Tail Model variants on gold-tier events.

Tests 5 model variants across multiple hyperparameter configs and min_edge
thresholds to find the best duration-aware prediction approach.

Usage:
    python scripts/test_duration.py
"""

import json
import sys
from pathlib import Path
from itertools import product

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.backtesting.engine import BacktestEngine
from src.ml.baseline_model import CrowdModel
from src.ml.advanced_models import MarketAdjustedModel
from src.ml.per_bucket_model import PerBucketModel
from src.ml.duration_model import (
    DurationShrinkModel,
    TailBoostModel,
    DurationTailModel,
    PowerLawTailModel,
    DurationOverlayModel,
)

BACKTEST_DIR = PROJECT / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


def load_gold_events():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    return [e for e in index["events"] if e.get("ground_truth_tier") == "gold"]


def evaluate_model(model, events, min_edge=0.02):
    engine = BacktestEngine(config={"min_edge": min_edge})
    result = engine.run(model, events)
    return {
        "brier": result.brier_score,
        "acc": result.accuracy,
        "acc_str": result.accuracy_str,
        "bets": result.n_bets,
        "roi": result.roi,
        "pnl": result.total_pnl,
        "wagered": result.total_wagered,
        "n_wins": result.n_wins,
    }


def fmt(r, label="", width=25):
    """Format a result dict into a one-line summary."""
    s = "{:<{w}s} Brier={:.4f}  Acc={:>5s}  Bets={:>3d}  ROI={:>+7.1f}%  PnL=${:>+8.2f}".format(
        label, r["brier"], r["acc_str"], r["bets"], r["roi"], r["pnl"], w=width
    )
    return s


def main():
    events = load_gold_events()
    print("Gold-tier events: {}".format(len(events)))

    # Classify events by duration for analysis
    short_events = [e for e in events if e.get("market_type") in ("daily", "short")]
    medium_events = [e for e in events if e.get("market_type") == "weekly"]
    long_events = [e for e in events if e.get("market_type") == "monthly"]
    print("  Short (daily/short): {}".format(len(short_events)))
    print("  Medium (weekly):     {}".format(len(medium_events)))
    print("  Long (monthly):      {}".format(len(long_events)))
    print()

    # ===================================================================
    # BASELINES
    # ===================================================================
    print("=" * 85)
    print("BASELINES (min_edge=2%)")
    print("=" * 85)

    crowd = CrowdModel()
    r = evaluate_model(crowd, events, min_edge=0.02)
    print(fmt(r, "CrowdModel"))

    adjusted = MarketAdjustedModel()
    r_adj = evaluate_model(adjusted, events, min_edge=0.02)
    print(fmt(r_adj, "MarketAdjusted"))

    perbucket = PerBucketModel()
    r_pb = evaluate_model(perbucket, events, min_edge=0.02)
    print(fmt(r_pb, "PerBucket"))

    best_brier = r.get("brier", 999)
    best_roi = r_pb.get("roi", 0)
    print()

    # Track all results for final comparison
    all_results = []
    all_results.append(("CrowdModel", r, {}))
    all_results.append(("MarketAdjusted", r_adj, {}))
    all_results.append(("PerBucket", r_pb, {}))

    # ===================================================================
    # CONFIG 1: Duration-Based Shrink Only
    # ===================================================================
    print("=" * 85)
    print("CONFIG 1: DurationShrinkModel (shrink for short events)")
    print("=" * 85)

    shrink_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    momentum_values = [0.05, 0.10, 0.15]
    widen_values = [1.10, 1.20, 1.30, 1.50]

    best_shrink_brier = 999
    best_shrink_cfg = {}

    # Phase 1a: Shrink sweep (medium_momentum=0.10, long_widen=1.20)
    print("\n--- Phase 1a: short_shrink sweep ---")
    for shrink in shrink_values:
        model = DurationShrinkModel(
            short_shrink=shrink, medium_momentum=0.10, long_widen=1.20
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_shrink_brier else ""
        print("  shrink={:.2f}: {}{}".format(shrink, fmt(r, "", 1), marker))
        if r["brier"] < best_shrink_brier:
            best_shrink_brier = r["brier"]
            best_shrink_cfg = {"short_shrink": shrink, "medium_momentum": 0.10, "long_widen": 1.20}
            best_shrink_r = r

    # Phase 1b: Medium momentum sweep with best shrink
    print("\n--- Phase 1b: medium_momentum sweep (shrink={:.2f}) ---".format(
        best_shrink_cfg.get("short_shrink", 0.20)))
    for mom in momentum_values:
        model = DurationShrinkModel(
            short_shrink=best_shrink_cfg["short_shrink"],
            medium_momentum=mom,
            long_widen=1.20,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_shrink_brier else ""
        print("  momentum={:.2f}: {}{}".format(mom, fmt(r, "", 1), marker))
        if r["brier"] < best_shrink_brier:
            best_shrink_brier = r["brier"]
            best_shrink_cfg["medium_momentum"] = mom
            best_shrink_r = r

    # Phase 1c: Long widen sweep
    print("\n--- Phase 1c: long_widen sweep ---")
    for wid in widen_values:
        model = DurationShrinkModel(
            short_shrink=best_shrink_cfg["short_shrink"],
            medium_momentum=best_shrink_cfg["medium_momentum"],
            long_widen=wid,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_shrink_brier else ""
        print("  widen={:.2f}: {}{}".format(wid, fmt(r, "", 1), marker))
        if r["brier"] < best_shrink_brier:
            best_shrink_brier = r["brier"]
            best_shrink_cfg["long_widen"] = wid
            best_shrink_r = r

    print("\n  BEST DurationShrink: Brier={:.4f} cfg={}".format(
        best_shrink_brier, best_shrink_cfg))
    all_results.append(("DurationShrink", best_shrink_r, best_shrink_cfg))

    # ===================================================================
    # CONFIG 2: Tail Boost Only
    # ===================================================================
    print()
    print("=" * 85)
    print("CONFIG 2: TailBoostModel (redistribute center->tails)")
    print("=" * 85)

    boost_values = [1.10, 1.20, 1.30, 1.50, 2.00]
    threshold_values = [0.5, 0.8, 1.0, 1.2, 1.5]
    center_power_values = [1.0, 1.05, 1.10, 1.15]

    best_tail_brier = 999
    best_tail_cfg = {}
    best_tail_r = {}

    # Phase 2a: boost factor sweep
    print("\n--- Phase 2a: tail_boost_factor sweep (threshold=1.0) ---")
    for boost in boost_values:
        model = TailBoostModel(
            tail_boost_factor=boost, tail_threshold_sd=1.0, center_power=1.0
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_tail_brier else ""
        print("  boost={:.2f}: {}{}".format(boost, fmt(r, "", 1), marker))
        if r["brier"] < best_tail_brier:
            best_tail_brier = r["brier"]
            best_tail_cfg = {"boost": boost, "threshold": 1.0, "center_power": 1.0}
            best_tail_r = r

    # Phase 2b: threshold sweep
    print("\n--- Phase 2b: tail_threshold_sd sweep (boost={:.2f}) ---".format(
        best_tail_cfg.get("boost", 1.30)))
    for thresh in threshold_values:
        model = TailBoostModel(
            tail_boost_factor=best_tail_cfg["boost"],
            tail_threshold_sd=thresh,
            center_power=1.0,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_tail_brier else ""
        print("  threshold={:.1f}: {}{}".format(thresh, fmt(r, "", 1), marker))
        if r["brier"] < best_tail_brier:
            best_tail_brier = r["brier"]
            best_tail_cfg["threshold"] = thresh
            best_tail_r = r

    # Phase 2c: center power sweep
    print("\n--- Phase 2c: center_power sweep ---")
    for cp in center_power_values:
        model = TailBoostModel(
            tail_boost_factor=best_tail_cfg["boost"],
            tail_threshold_sd=best_tail_cfg["threshold"],
            center_power=cp,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_tail_brier else ""
        print("  center_power={:.2f}: {}{}".format(cp, fmt(r, "", 1), marker))
        if r["brier"] < best_tail_brier:
            best_tail_brier = r["brier"]
            best_tail_cfg["center_power"] = cp
            best_tail_r = r

    print("\n  BEST TailBoost: Brier={:.4f} cfg={}".format(
        best_tail_brier, best_tail_cfg))
    all_results.append(("TailBoost", best_tail_r, best_tail_cfg))

    # ===================================================================
    # CONFIG 3: Combined Duration + Tail
    # ===================================================================
    print()
    print("=" * 85)
    print("CONFIG 3: DurationTailModel (combined duration + tail)")
    print("=" * 85)

    # Use best params from Config 1 + Config 2 as starting point, plus variations
    dt_shrink_vals = [0.05, 0.10, 0.15, 0.20]
    dt_boost_vals = [1.0, 1.10, 1.20, 1.30]
    dt_thresh_vals = [0.8, 1.0, 1.2]

    best_dt_brier = 999
    best_dt_cfg = {}
    best_dt_r = {}

    print("\n--- Grid search: shrink x boost x threshold ---")
    for shrink, boost, thresh in product(dt_shrink_vals, dt_boost_vals, dt_thresh_vals):
        model = DurationTailModel(
            short_shrink=shrink,
            medium_momentum=best_shrink_cfg.get("medium_momentum", 0.10),
            long_widen=best_shrink_cfg.get("long_widen", 1.20),
            tail_boost_factor=boost,
            tail_threshold_sd=thresh,
        )
        r = evaluate_model(model, events)
        if r["brier"] < best_dt_brier:
            best_dt_brier = r["brier"]
            best_dt_cfg = {
                "shrink": shrink, "boost": boost, "thresh": thresh,
                "momentum": best_shrink_cfg.get("medium_momentum", 0.10),
                "widen": best_shrink_cfg.get("long_widen", 1.20),
            }
            best_dt_r = r

    print("  BEST DurationTail: Brier={:.4f} cfg={}".format(best_dt_brier, best_dt_cfg))
    print("  {}".format(fmt(best_dt_r, "DurationTail")))
    all_results.append(("DurationTail", best_dt_r, best_dt_cfg))

    # ===================================================================
    # CONFIG 4: Power-Law Tail Adjustment
    # ===================================================================
    print()
    print("=" * 85)
    print("CONFIG 4: PowerLawTailModel (P^alpha transformation)")
    print("=" * 85)

    tail_alpha_vals = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    center_alpha_vals = [1.0, 1.05, 1.10, 1.15, 1.20]
    pl_thresh_vals = [0.5, 0.8, 1.0, 1.2, 1.5]

    best_pl_brier = 999
    best_pl_cfg = {}
    best_pl_r = {}

    # Phase 4a: tail_alpha sweep
    print("\n--- Phase 4a: tail_alpha sweep (center=1.0, thresh=1.0) ---")
    for ta in tail_alpha_vals:
        model = PowerLawTailModel(tail_alpha=ta, center_alpha=1.0, tail_threshold_sd=1.0)
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_pl_brier else ""
        print("  tail_alpha={:.2f}: {}{}".format(ta, fmt(r, "", 1), marker))
        if r["brier"] < best_pl_brier:
            best_pl_brier = r["brier"]
            best_pl_cfg = {"tail_alpha": ta, "center_alpha": 1.0, "threshold": 1.0}
            best_pl_r = r

    # Phase 4b: center_alpha sweep
    print("\n--- Phase 4b: center_alpha sweep (tail={:.2f}) ---".format(
        best_pl_cfg.get("tail_alpha", 0.80)))
    for ca in center_alpha_vals:
        model = PowerLawTailModel(
            tail_alpha=best_pl_cfg["tail_alpha"],
            center_alpha=ca,
            tail_threshold_sd=1.0,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_pl_brier else ""
        print("  center_alpha={:.2f}: {}{}".format(ca, fmt(r, "", 1), marker))
        if r["brier"] < best_pl_brier:
            best_pl_brier = r["brier"]
            best_pl_cfg["center_alpha"] = ca
            best_pl_r = r

    # Phase 4c: threshold sweep
    print("\n--- Phase 4c: threshold sweep ---")
    for thresh in pl_thresh_vals:
        model = PowerLawTailModel(
            tail_alpha=best_pl_cfg["tail_alpha"],
            center_alpha=best_pl_cfg["center_alpha"],
            tail_threshold_sd=thresh,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_pl_brier else ""
        print("  threshold={:.1f}: {}{}".format(thresh, fmt(r, "", 1), marker))
        if r["brier"] < best_pl_brier:
            best_pl_brier = r["brier"]
            best_pl_cfg["threshold"] = thresh
            best_pl_r = r

    print("\n  BEST PowerLawTail: Brier={:.4f} cfg={}".format(best_pl_brier, best_pl_cfg))
    all_results.append(("PowerLawTail", best_pl_r, best_pl_cfg))

    # ===================================================================
    # CONFIG 5: MarketAdjusted Base + Duration Overlay
    # ===================================================================
    print()
    print("=" * 85)
    print("CONFIG 5: DurationOverlayModel (MarketAdjusted + duration + tails)")
    print("=" * 85)

    do_shrink_vals = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    do_widen_vals = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30]
    do_boost_vals = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30]
    do_thresh_vals = [0.8, 1.0, 1.2, 1.5]

    best_do_brier = 999
    best_do_cfg = {}
    best_do_r = {}

    # Phase 5a: short_shrink sweep (no tail boost)
    print("\n--- Phase 5a: short_shrink sweep (no tail boost) ---")
    for shrink in do_shrink_vals:
        model = DurationOverlayModel(
            short_shrink=shrink, long_widen=1.0,
            tail_boost_factor=1.0, tail_threshold_sd=1.0,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_do_brier else ""
        print("  shrink={:.2f}: {}{}".format(shrink, fmt(r, "", 1), marker))
        if r["brier"] < best_do_brier:
            best_do_brier = r["brier"]
            best_do_cfg = {"shrink": shrink, "widen": 1.0, "boost": 1.0, "thresh": 1.0}
            best_do_r = r

    # Phase 5b: long_widen sweep
    print("\n--- Phase 5b: long_widen sweep (shrink={:.2f}) ---".format(
        best_do_cfg.get("shrink", 0.12)))
    for wid in do_widen_vals:
        model = DurationOverlayModel(
            short_shrink=best_do_cfg["shrink"], long_widen=wid,
            tail_boost_factor=1.0, tail_threshold_sd=1.0,
        )
        r = evaluate_model(model, events)
        marker = " ***" if r["brier"] < best_do_brier else ""
        print("  widen={:.2f}: {}{}".format(wid, fmt(r, "", 1), marker))
        if r["brier"] < best_do_brier:
            best_do_brier = r["brier"]
            best_do_cfg["widen"] = wid
            best_do_r = r

    # Phase 5c: tail_boost sweep
    print("\n--- Phase 5c: tail_boost sweep ---")
    for boost in do_boost_vals:
        for thresh in do_thresh_vals:
            model = DurationOverlayModel(
                short_shrink=best_do_cfg["shrink"],
                long_widen=best_do_cfg["widen"],
                tail_boost_factor=boost,
                tail_threshold_sd=thresh,
            )
            r = evaluate_model(model, events)
            marker = " ***" if r["brier"] < best_do_brier else ""
            if marker or boost in (1.0, 1.10, 1.20):
                print("  boost={:.2f} thresh={:.1f}: {}{}".format(
                    boost, thresh, fmt(r, "", 1), marker))
            if r["brier"] < best_do_brier:
                best_do_brier = r["brier"]
                best_do_cfg["boost"] = boost
                best_do_cfg["thresh"] = thresh
                best_do_r = r

    print("\n  BEST DurationOverlay: Brier={:.4f} cfg={}".format(
        best_do_brier, best_do_cfg))
    all_results.append(("DurationOverlay", best_do_r, best_do_cfg))

    # ===================================================================
    # ROI SWEEP: Test best configs at different min_edge thresholds
    # ===================================================================
    print()
    print("=" * 85)
    print("ROI SWEEP: Best config from each variant at min_edge=[0.01, 0.02, 0.03, 0.05]")
    print("=" * 85)

    min_edges = [0.01, 0.02, 0.03, 0.05]

    # Rebuild best models
    best_models = {
        "DurationShrink": DurationShrinkModel(
            short_shrink=best_shrink_cfg.get("short_shrink", 0.20),
            medium_momentum=best_shrink_cfg.get("medium_momentum", 0.10),
            long_widen=best_shrink_cfg.get("long_widen", 1.20),
        ),
        "TailBoost": TailBoostModel(
            tail_boost_factor=best_tail_cfg.get("boost", 1.30),
            tail_threshold_sd=best_tail_cfg.get("threshold", 1.0),
            center_power=best_tail_cfg.get("center_power", 1.0),
        ),
        "DurationTail": DurationTailModel(
            short_shrink=best_dt_cfg.get("shrink", 0.15),
            medium_momentum=best_dt_cfg.get("momentum", 0.10),
            long_widen=best_dt_cfg.get("widen", 1.20),
            tail_boost_factor=best_dt_cfg.get("boost", 1.20),
            tail_threshold_sd=best_dt_cfg.get("thresh", 1.0),
        ),
        "PowerLawTail": PowerLawTailModel(
            tail_alpha=best_pl_cfg.get("tail_alpha", 0.80),
            center_alpha=best_pl_cfg.get("center_alpha", 1.0),
            tail_threshold_sd=best_pl_cfg.get("threshold", 1.0),
        ),
        "DurationOverlay": DurationOverlayModel(
            short_shrink=best_do_cfg.get("shrink", 0.12),
            long_widen=best_do_cfg.get("widen", 1.0),
            tail_boost_factor=best_do_cfg.get("boost", 1.0),
            tail_threshold_sd=best_do_cfg.get("thresh", 1.0),
        ),
    }

    # Add baselines
    best_models["CrowdModel"] = CrowdModel()
    best_models["MarketAdjusted"] = MarketAdjustedModel()
    best_models["PerBucket"] = PerBucketModel()

    for me in min_edges:
        print("\n--- min_edge = {:.0%} ---".format(me))
        for name, model in best_models.items():
            r = evaluate_model(model, events, min_edge=me)
            print("  {}".format(fmt(r, name, 25)))

    # ===================================================================
    # BY DURATION TYPE: Test best models on each event type separately
    # ===================================================================
    print()
    print("=" * 85)
    print("BY DURATION: Best overlay model on short/medium/long separately (min_edge=2%)")
    print("=" * 85)

    for subset_name, subset_events in [
        ("Short (daily/short)", short_events),
        ("Medium (weekly)", medium_events),
        ("Long (monthly)", long_events),
    ]:
        if not subset_events:
            print("\n  {} -- No events".format(subset_name))
            continue
        print("\n  {} ({} events)".format(subset_name, len(subset_events)))
        for name in ["CrowdModel", "MarketAdjusted", "PerBucket", "DurationOverlay", "DurationTail"]:
            model = best_models[name]
            r = evaluate_model(model, subset_events, min_edge=0.02)
            print("    {}".format(fmt(r, name, 25)))

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print()
    print("=" * 85)
    print("FINAL SUMMARY (min_edge=2%)")
    print("=" * 85)

    # Sort by Brier
    all_results.sort(key=lambda x: x[1].get("brier", 999))

    for name, r, cfg in all_results:
        print("  {}".format(fmt(r, name, 25)))
        if cfg:
            print("    cfg={}".format(cfg))

    print()
    print("DONE")


if __name__ == "__main__":
    main()
