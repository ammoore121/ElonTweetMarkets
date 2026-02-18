"""
Backtest the strategy bands meta-strategy.

Simulates what generate_band_signals.py does in production, but applied
to the full historical backtest dataset (159 events).

Band logic:
  - Band 1 (high):   4+ strategies agree on same bucket -> 0.30 Kelly, max_bet_pct 0.06
  - Band 2 (medium): 2-3 strategies agree on same bucket -> 0.25 Kelly, max_bet_pct 0.05

For each event:
1. Run all active paper base strategies (excluding band strategies) through their models
2. Apply each strategy's filters to determine if it would signal, and which bucket is best
3. Group by (event, best_bucket) to count agreement
4. Apply band logic: bet the consensus bucket with band-appropriate sizing
5. Compute P&L: if consensus bucket == winning bucket, payout = wager/market_price

Usage:
    python scripts/backtest_bands.py
    python scripts/backtest_bands.py --tier gold
    python scripts/backtest_bands.py --min-edge 0.03
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ml.registry import ModelRegistry, StrategyRegistry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"
EVENTS_DIR = BACKTEST_DIR / "events"

# ---------------------------------------------------------------------------
# Band configuration (matches generate_band_signals.py)
# ---------------------------------------------------------------------------
BAND_CONFIG = {
    1: {
        "min_strategies": 4,
        "kelly_fraction": 0.30,
        "max_bet_pct": 0.06,
        "label": "Band1 (4+)",
    },
    2: {
        "min_strategies": 2,
        "kelly_fraction": 0.25,
        "max_bet_pct": 0.05,
        "label": "Band2 (2-3)",
    },
}

# Base strategies to include (paper status, NOT band strategies)
# Excluded: band_1_aggressive, band_2_moderate (they are the meta-strategies we're building)
# Excluded: crowd_baseline (control, never bets by design)
# Excluded: asymmetric_selective, per_bucket_proven, cross_market_arb_gold (inactive)
# Excluded: xgb_bucket_primary (inactive), xgb_residual_primary (needs training artifact)
# Excluded: signal_enhanced_v6_reddit (inactive)
BASE_STRATEGY_IDS = [
    "tail_boost_primary",
    "duration_tail_robust",
    "duration_shrink_simple",
    "signal_enhanced_volume",          # signal_enhanced_tail_v3
    "signal_enhanced_selective",       # signal_enhanced_tail_v1
    "signal_enhanced_v4_volume",       # signal_enhanced_tail_v4
    "signal_enhanced_v5_volume",       # signal_enhanced_tail_v5
    "price_dynamics_primary",
    "consensus_primary",
]

# MODEL_ID -> short key for run_backtest.py MODEL_MAP
MODEL_ID_TO_KEY = {
    "naive_negbin_v1": "naive",
    "crowd_v1": "crowd",
    "regime_aware_v1": "regime",
    "market_adjusted_v1": "adjusted",
    "ensemble_v1": "ensemble",
    "per_bucket_v1": "perbucket",
    "tail_boost_v1": "signal_enhanced",     # TailBoostModel is loaded via duration_model
    "duration_tail_v1": "signal_enhanced",  # will instantiate via registry
    "duration_shrink_v1": "signal_enhanced",
    "signal_enhanced_tail_v1": "signal_enhanced",
    "signal_enhanced_tail_v3": "signal_enhanced",
    "signal_enhanced_tail_v4": "signal_enhanced_v4",
    "signal_enhanced_tail_v5": "signal_enhanced_v5",
    "signal_enhanced_tail_v6": "signal_enhanced_v6",
    "price_dynamics_v1": "price_dynamics",
    "cross_market_arb_v1": "cross_market_arb",
    "consensus_ensemble_v1": "consensus_ensemble",
    "xgb_bucket_v1": "xgb_bucket",
    "xgb_residual_v1": "xgb_residual",
}


def load_index():
    """Load the master backtest index."""
    if not INDEX_PATH.exists():
        print("ERROR: Backtest index not found at {}".format(INDEX_PATH))
        print("       Run: python scripts/build_backtest_dataset.py")
        sys.exit(1)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_event_data(event_slug: str):
    """Load metadata, features, and prices for a single event.

    Returns (metadata, features, prices_df) where prices_df may be None.
    """
    import pandas as pd

    event_dir = EVENTS_DIR / event_slug

    meta_path = event_dir / "metadata.json"
    if not meta_path.exists():
        return None, None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feat_path = event_dir / "features.json"
    features = {}
    if feat_path.exists():
        with open(feat_path, "r", encoding="utf-8") as f:
            features = json.load(f)

    prices_df = None
    prices_path = event_dir / "prices.parquet"
    if prices_path.exists():
        try:
            prices_df = pd.read_parquet(prices_path)
        except Exception:
            pass

    return metadata, features, prices_df


def get_entry_prices(metadata: dict, prices_df, entry_hours: int = 24,
                     window_hours: int = 6) -> dict:
    """Get market prices at entry time (T-entry_hours before close).

    Returns dict of {bucket_label: price}.
    Matches BacktestEngine._get_entry_prices() logic exactly.
    """
    import pandas as pd

    buckets = metadata.get("buckets", [])
    bucket_prices = {}

    # Try price history first
    if prices_df is not None and not prices_df.empty:
        try:
            end_str = metadata.get("end_date")
            if end_str:
                end_dt = pd.Timestamp(end_str, tz="UTC")
                target_time = end_dt - pd.Timedelta(hours=entry_hours)
                window_start = target_time - pd.Timedelta(hours=window_hours)
                window_end = target_time + pd.Timedelta(hours=window_hours)

                ts_col = "timestamp"
                if ts_col in prices_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(prices_df[ts_col]):
                        prices_df = prices_df.copy()
                        prices_df[ts_col] = pd.to_datetime(prices_df[ts_col], utc=True)

                    windowed = prices_df[
                        (prices_df[ts_col] >= window_start)
                        & (prices_df[ts_col] <= window_end)
                    ]

                    if not windowed.empty:
                        windowed = windowed.copy()
                        windowed["_diff"] = (windowed[ts_col] - target_time).abs()
                        closest = windowed.sort_values("_diff").drop_duplicates(
                            subset=["bucket_label"], keep="first"
                        )
                        for _, row in closest.iterrows():
                            bucket_prices[str(row["bucket_label"])] = float(row["price"])

                        if bucket_prices:
                            return bucket_prices
        except Exception:
            pass

    # Fallback: use final market prices from metadata
    for bucket in buckets:
        label = bucket["bucket_label"]
        price = bucket.get("price_yes")
        bucket_prices[label] = float(price) if price is not None else 0.0

    return bucket_prices


def apply_strategy_filter(strategy_def: dict, metadata: dict, features: dict) -> bool:
    """Check if this event passes the strategy's filters (market type, duration, etc.).

    Returns True if the event should be considered for this strategy.
    """
    filters = strategy_def.get("filters", {})

    # Market type filter
    allowed_types = filters.get("market_types", [])
    if allowed_types:
        market_type = metadata.get("market_type", "")
        if market_type not in allowed_types:
            return False

    # Duration filter
    min_dur = filters.get("min_duration_days", 1)
    duration = metadata.get("duration_days", 7)
    if duration < min_dur:
        return False

    # Temporal features filter
    require_temporal = filters.get("require_temporal_features", False)
    if require_temporal:
        temporal = features.get("temporal", {})
        if not temporal or temporal.get("rolling_avg_7d") is None:
            return False

    return True


def compute_strategy_signal(
    model,
    strategy_def: dict,
    metadata: dict,
    features: dict,
    bucket_prices: dict,
) -> dict | None:
    """Run one strategy against one event.

    Returns a signal dict with best_bucket and edge, or None if no signal.

    Signal dict: {
        'best_bucket': str,
        'edge': float,
        'model_prob': float,
        'market_price': float,
        'predicted_probs': dict,
    }
    """
    buckets = metadata.get("buckets", [])
    if not buckets:
        return None

    filters = strategy_def.get("filters", {})
    min_edge = filters.get("min_edge", 0.02)
    max_edge = filters.get("max_edge", 0.3)
    min_price = filters.get("min_bucket_price", 0.02)
    max_price = filters.get("max_bucket_price", 0.5)

    duration_days = metadata.get("duration_days", 7)
    context = {
        "duration_days": duration_days,
        "entry_prices": bucket_prices,
    }

    # Get model predictions
    try:
        predicted_probs = model.predict(features, buckets, context=context)
    except Exception as e:
        return None

    if not predicted_probs:
        return None

    # Find best bucket: largest positive edge that passes all filters
    best_bucket = None
    best_edge = -float("inf")
    best_model_prob = 0.0
    best_market_price = 0.0

    for bucket in buckets:
        label = bucket["bucket_label"]
        model_prob = predicted_probs.get(label, 0.0)
        market_price = bucket_prices.get(label, 0.0)

        # Price filter
        if market_price < min_price or market_price > max_price:
            continue
        if market_price <= 0 or market_price >= 1:
            continue

        edge = model_prob - market_price

        # Edge filter
        if edge < min_edge or edge > max_edge:
            continue

        if edge > best_edge:
            best_edge = edge
            best_bucket = label
            best_model_prob = model_prob
            best_market_price = market_price

    if best_bucket is None:
        return None

    return {
        "best_bucket": best_bucket,
        "edge": best_edge,
        "model_prob": best_model_prob,
        "market_price": best_market_price,
        "predicted_probs": predicted_probs,
    }


def determine_band_level(n_strategies: int) -> int:
    """Return band level (1 or 2) or 0 if below threshold."""
    if n_strategies >= BAND_CONFIG[1]["min_strategies"]:
        return 1
    elif n_strategies >= BAND_CONFIG[2]["min_strategies"]:
        return 2
    return 0


def average_distributions(signals_list: list) -> dict:
    """Element-wise mean of predicted probability distributions, normalized."""
    if not signals_list:
        return {}

    bucket_sums = defaultdict(float)
    n = len(signals_list)

    for sig in signals_list:
        for label, prob in sig["predicted_probs"].items():
            bucket_sums[label] += prob

    averaged = {label: total / n for label, total in bucket_sums.items()}
    total_mass = sum(averaged.values())
    if total_mass > 0:
        averaged = {label: p / total_mass for label, p in averaged.items()}

    return averaged


def kelly_wager(
    edge: float,
    market_price: float,
    kelly_fraction: float,
    max_bet_pct: float,
    bankroll: float,
    min_bet: float = 10.0,
) -> float:
    """Compute Kelly wager for a band signal."""
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return 0.0

    kelly_f = edge / (1.0 - market_price)
    wager = bankroll * kelly_f * kelly_fraction
    wager = max(min_bet, wager)
    wager = min(bankroll * max_bet_pct, wager)
    return wager


def run_bands_backtest(
    events: list,
    strategy_models: list,  # list of (strategy_id, strategy_def, model)
    bankroll: float = 1000.0,
    verbose: bool = False,
) -> dict:
    """Run the full bands backtest.

    Returns results dict keyed by tier with per-event detail.
    """
    per_event_results = []

    for evt in events:
        slug = evt["event_slug"]
        metadata, features, prices_df = load_event_data(slug)
        if metadata is None:
            continue

        tier = metadata.get("ground_truth_tier", "unknown")
        winning_bucket = metadata.get("winning_bucket")
        buckets = metadata.get("buckets", [])

        if not buckets or not winning_bucket:
            continue

        # Get entry prices (shared across all strategies for this event)
        bucket_prices = get_entry_prices(metadata, prices_df)

        # Run each strategy and collect signals
        # Group by best_bucket to count agreement
        bucket_signals = defaultdict(list)  # best_bucket -> list of signal dicts

        for strategy_id, strategy_def, model in strategy_models:
            # Check strategy event-level filters
            if not apply_strategy_filter(strategy_def, metadata, features):
                continue

            signal = compute_strategy_signal(
                model, strategy_def, metadata, features, bucket_prices
            )
            if signal is None:
                continue

            signal["strategy_id"] = strategy_id
            bucket_signals[signal["best_bucket"]].append(signal)

        # For each agreed bucket, determine band level and compute P&L
        event_bets = []

        for best_bucket, group_signals in bucket_signals.items():
            n_strategies = len(group_signals)
            band_level = determine_band_level(n_strategies)

            if band_level == 0:
                continue

            band_cfg = BAND_CONFIG[band_level]
            contributing_ids = sorted(set(s["strategy_id"] for s in group_signals))

            # Average predicted distributions
            avg_probs = average_distributions(group_signals)

            # Consensus edge = avg model prob for the agreed bucket - market price
            avg_model_prob = avg_probs.get(best_bucket, 0.0)
            avg_market_price = sum(s["market_price"] for s in group_signals) / n_strategies
            consensus_edge = avg_model_prob - avg_market_price

            if consensus_edge <= 0:
                if verbose:
                    print("    [skip] {}/{}: {} agree, consensus_edge={:.3f} <= 0".format(
                        slug[:30], best_bucket, n_strategies, consensus_edge))
                continue

            # Compute wager
            min_bet = 10.0 if band_level == 1 else 5.0
            wager = kelly_wager(
                edge=consensus_edge,
                market_price=avg_market_price,
                kelly_fraction=band_cfg["kelly_fraction"],
                max_bet_pct=band_cfg["max_bet_pct"],
                bankroll=bankroll,
                min_bet=min_bet,
            )

            if wager <= 0:
                continue

            # Settle
            won = (best_bucket == winning_bucket)
            if won and avg_market_price > 0:
                payout = wager / avg_market_price
                pnl = payout - wager
            else:
                payout = 0.0
                pnl = -wager

            event_bets.append({
                "best_bucket": best_bucket,
                "winning_bucket": winning_bucket,
                "band_level": band_level,
                "n_strategies": n_strategies,
                "contributing_ids": contributing_ids,
                "avg_model_prob": round(avg_model_prob, 4),
                "avg_market_price": round(avg_market_price, 4),
                "consensus_edge": round(consensus_edge, 4),
                "wager": round(wager, 2),
                "won": won,
                "pnl": round(pnl, 2),
            })

            if verbose:
                print("    [{}] {}/{}: {} agree ({}), edge={:.3f}, wager=${:.2f}, won={}, pnl=${:+.2f}".format(
                    band_cfg["label"], slug[:30], best_bucket, n_strategies,
                    ",".join(contributing_ids), consensus_edge, wager, won, pnl))

        per_event_results.append({
            "event_slug": slug,
            "event_title": metadata.get("event_title", ""),
            "tier": tier,
            "market_type": metadata.get("market_type", ""),
            "winning_bucket": winning_bucket,
            "n_bets": len(event_bets),
            "total_wagered": round(sum(b["wager"] for b in event_bets), 2),
            "total_pnl": round(sum(b["pnl"] for b in event_bets), 2),
            "bets": event_bets,
        })

    return per_event_results


def compute_tier_stats(per_event_results: list, tier: str | None = None) -> dict:
    """Compute aggregate stats for a given tier (or all tiers if tier is None)."""
    if tier is not None:
        events = [r for r in per_event_results if r["tier"] == tier]
    else:
        events = per_event_results

    n_events = len(events)
    n_events_with_bets = sum(1 for r in events if r["n_bets"] > 0)

    all_bets = [bet for r in events for bet in r["bets"]]
    n_bets = len(all_bets)
    total_wagered = sum(b["wager"] for b in all_bets)
    total_pnl = sum(b["pnl"] for b in all_bets)
    n_wins = sum(1 for b in all_bets if b["won"])
    roi = 100.0 * total_pnl / total_wagered if total_wagered > 0 else 0.0

    band1_bets = [b for b in all_bets if b["band_level"] == 1]
    band2_bets = [b for b in all_bets if b["band_level"] == 2]

    return {
        "tier": tier or "all",
        "n_events": n_events,
        "n_events_with_bets": n_events_with_bets,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": round(100.0 * n_wins / n_bets, 1) if n_bets > 0 else 0.0,
        "total_wagered": round(total_wagered, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi, 2),
        "band1_n_bets": len(band1_bets),
        "band1_wagered": round(sum(b["wager"] for b in band1_bets), 2),
        "band1_pnl": round(sum(b["pnl"] for b in band1_bets), 2),
        "band1_roi": round(
            100.0 * sum(b["pnl"] for b in band1_bets) / sum(b["wager"] for b in band1_bets), 2
        ) if band1_bets and sum(b["wager"] for b in band1_bets) > 0 else 0.0,
        "band2_n_bets": len(band2_bets),
        "band2_wagered": round(sum(b["wager"] for b in band2_bets), 2),
        "band2_pnl": round(sum(b["pnl"] for b in band2_bets), 2),
        "band2_roi": round(
            100.0 * sum(b["pnl"] for b in band2_bets) / sum(b["wager"] for b in band2_bets), 2
        ) if band2_bets and sum(b["wager"] for b in band2_bets) > 0 else 0.0,
    }


def print_report(per_event_results: list, strategy_models: list):
    """Print the full backtest report."""
    sep = "=" * 75

    print()
    print(sep)
    print("STRATEGY BANDS BACKTEST REPORT")
    print(sep)
    print()
    print("Base strategies used ({} total):".format(len(strategy_models)))
    for sid, sdef, _ in strategy_models:
        model_id = sdef["model_id"]
        min_edge = sdef.get("filters", {}).get("min_edge", 0.02)
        print("  - {} (model: {}, min_edge: {:.0%})".format(sid, model_id, min_edge))

    print()
    print("Band configuration:")
    print("  Band 1 (4+ strategies agree): Kelly={}, max_bet_pct={}".format(
        BAND_CONFIG[1]["kelly_fraction"], BAND_CONFIG[1]["max_bet_pct"]))
    print("  Band 2 (2-3 strategies agree): Kelly={}, max_bet_pct={}".format(
        BAND_CONFIG[2]["kelly_fraction"], BAND_CONFIG[2]["max_bet_pct"]))

    print()
    print(sep)
    print("RESULTS BY TIER")
    print(sep)

    tiers = ["gold", "silver", "bronze", None]
    tier_labels = ["Gold", "Silver", "Bronze", "ALL"]
    n_tiers_total = {"gold": 38, "silver": 9, "bronze": 112, None: 159}

    for tier, label in zip(tiers, tier_labels):
        stats = compute_tier_stats(per_event_results, tier)
        n_total = n_tiers_total.get(tier, stats["n_events"])

        print()
        print("--- {} Tier ({} events) ---".format(label, stats["n_events"]))
        print("  Total bets:       {:>6d}".format(stats["n_bets"]))
        print("  Total wagered:    {:>8.2f}".format(stats["total_wagered"]))
        print("  Total P&L:        {:>+8.2f}".format(stats["total_pnl"]))
        print("  ROI:              {:>+8.2f}%".format(stats["roi_pct"]))
        print("  Win rate:         {:>7.1f}%  ({}/{} bets)".format(
            stats["win_rate"], stats["n_wins"], stats["n_bets"]))
        print()
        print("  Band 1 (4+):  {:>3d} bets, ${:>7.2f} wagered, ${:>+7.2f} P&L, {:>+.1f}% ROI".format(
            stats["band1_n_bets"], stats["band1_wagered"],
            stats["band1_pnl"], stats["band1_roi"]))
        print("  Band 2 (2-3): {:>3d} bets, ${:>7.2f} wagered, ${:>+7.2f} P&L, {:>+.1f}% ROI".format(
            stats["band2_n_bets"], stats["band2_wagered"],
            stats["band2_pnl"], stats["band2_roi"]))

    print()
    print(sep)
    print("COMPARISON VS REFERENCE MODELS (ALL tiers, 159 events, 2% min_edge)")
    print(sep)
    print()
    all_stats = compute_tier_stats(per_event_results, None)

    print("{:<30s} {:>6s} {:>10s} {:>10s} {:>8s}".format(
        "Model/Strategy", "Bets", "Wagered", "P&L", "ROI%"))
    print("-" * 70)

    # Reference baselines from CLAUDE.md / MEMORY.md
    references = [
        ("ConsensusEnsemble",       132, 1539, 594, 37.0),
        ("XGBoost Residual (CV)",   161, 2908, 826, 28.4),
        ("SignalEnhanced v3",       188, 2237, 395, 17.7),
        ("TailBoost",               187, 2180, 303, 13.9),
        ("PriceDynamics",           116, 1706, 564, 33.1),
    ]
    for name, bets, wagered, pnl, roi in references:
        print("{:<30s} {:>6d} {:>10.0f} {:>+10.0f} {:>+8.1f}%".format(
            name, bets, wagered, pnl, roi))

    print()
    band_wagered = all_stats["total_wagered"]
    band_pnl = all_stats["total_pnl"]
    band_roi = all_stats["roi_pct"]
    band_bets = all_stats["n_bets"]

    print("{:<30s} {:>6d} {:>10.2f} {:>+10.2f} {:>+8.2f}%  <-- BANDS".format(
        "StrategyBands (B1+B2)", band_bets, band_wagered, band_pnl, band_roi))
    print("{:<30s} {:>6d} {:>10.2f} {:>+10.2f} {:>+8.2f}%".format(
        "  Band1 only (4+)",
        all_stats["band1_n_bets"], all_stats["band1_wagered"],
        all_stats["band1_pnl"], all_stats["band1_roi"]))
    print("{:<30s} {:>6d} {:>10.2f} {:>+10.2f} {:>+8.2f}%".format(
        "  Band2 only (2-3)",
        all_stats["band2_n_bets"], all_stats["band2_wagered"],
        all_stats["band2_pnl"], all_stats["band2_roi"]))

    print()
    print(sep)

    # Per-event detail (only events with bets)
    print("PER-EVENT RESULTS (events with bets)")
    print(sep)
    events_with_bets = [r for r in per_event_results if r["n_bets"] > 0]
    header = "{:<52s} {:>6s} {:>5s} {:>7s} {:>7s} {:>5s}".format(
        "Event Slug", "Tier", "Bets", "Wagered", "P&L", "ROI%")
    print(header)
    print("-" * 85)

    for r in events_with_bets:
        slug = r["event_slug"][:51]
        tier = r["tier"][:6]
        n_bets = r["n_bets"]
        wagered = r["total_wagered"]
        pnl = r["total_pnl"]
        roi = 100.0 * pnl / wagered if wagered > 0 else 0.0
        print("{:<52s} {:>6s} {:>5d} {:>7.2f} {:>+7.2f} {:>+5.1f}%".format(
            slug, tier, n_bets, wagered, pnl, roi))

    print()
    print("Events with bets: {} / {}".format(
        len(events_with_bets), len(per_event_results)))


def main():
    parser = argparse.ArgumentParser(
        description="Backtest strategy bands meta-strategy"
    )
    parser.add_argument(
        "--tier",
        choices=["gold", "silver", "bronze"],
        default=None,
        help="Filter to a specific tier (default: all tiers)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-bet detail during backtest",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Bankroll for wager sizing (default: 1000)",
    )
    args = parser.parse_args()

    # Load backtest index
    index = load_index()
    events = index.get("events", [])

    # Filter by tier if specified
    if args.tier:
        events = [e for e in events if e.get("ground_truth_tier") == args.tier]

    print("Loaded {} events from backtest index".format(len(events)))

    # Load all base strategies and their models
    sr = StrategyRegistry()
    mr = ModelRegistry()

    strategy_models = []
    print()
    print("Loading base strategies...")

    for strategy_id in BASE_STRATEGY_IDS:
        strategy_def = sr.get_strategy(strategy_id)
        if strategy_def is None:
            print("  WARNING: Strategy '{}' not found in registry, skipping".format(
                strategy_id))
            continue

        status = strategy_def.get("status", "inactive")
        model_id = strategy_def["model_id"]

        # Skip band strategies (they use strategy_bands_v1 which has no class)
        if model_id == "strategy_bands_v1":
            print("  SKIP (meta): {}".format(strategy_id))
            continue

        # Try to instantiate the model via registry (uses correct hyperparameters)
        try:
            model = mr.instantiate_model(model_id)
            strategy_models.append((strategy_id, strategy_def, model))
            print("  OK [{}]: {} (model: {})".format(status, strategy_id, model_id))
        except Exception as e:
            print("  FAIL: {} - {} (reason: {})".format(strategy_id, model_id, e))
            continue

    if not strategy_models:
        print("ERROR: No base strategies could be loaded.")
        sys.exit(1)

    print()
    print("Running bands backtest on {} events with {} base strategies...".format(
        len(events), len(strategy_models)))
    print()

    # Run the backtest
    per_event_results = run_bands_backtest(
        events=events,
        strategy_models=strategy_models,
        bankroll=args.bankroll,
        verbose=args.verbose,
    )

    # Print report
    print_report(per_event_results, strategy_models)


if __name__ == "__main__":
    main()
