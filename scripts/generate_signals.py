"""Generate trading signals from active strategies and optionally place paper trades.

Loads active strategies, runs model predictions against latest odds,
applies strategy filters, and saves signals. If auto_bet is enabled
in config, also creates betslips and fills.

Usage:
    python scripts/generate_signals.py
    python scripts/generate_signals.py --dry-run
    python scripts/generate_signals.py --strategy tail_boost_primary
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ml.registry import ModelRegistry, StrategyRegistry
from src.features.feature_builder import TweetFeatureBuilder
from src.paper_trading.schemas import Signal
from src.paper_trading.tracker import PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "market_catalog.parquet"
CONFIG_PATH = PROJECT_DIR / "config" / "pipeline.json"


def load_config() -> dict:
    """Load pipeline config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"auto_bet": {"enabled": False, "mode": "paper"}}


def load_active_events() -> pd.DataFrame:
    """Load active events from market catalog."""
    if not CATALOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(CATALOG_PATH)
    now = datetime.now(timezone.utc)
    if "is_resolved" in df.columns:
        df = df[df["is_resolved"] == False]
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
        df = df[df["end_date"] > now]
    return df


def apply_filters(edge: float, market_price: float, market_type: str,
                  filters: dict) -> bool:
    """Check if a signal meets strategy filter criteria."""
    if edge < filters.get("min_edge", 0.02):
        return False
    if edge > filters.get("max_edge", 1.0):
        return False
    if market_price < filters.get("min_bucket_price", 0):
        return False
    if market_price > filters.get("max_bucket_price", 1.0):
        return False
    allowed_types = filters.get("market_types", ["weekly", "daily", "short", "monthly"])
    if market_type not in allowed_types:
        return False
    return True


def compute_kelly_wager(edge: float, market_price: float, sizing: dict) -> tuple:
    """Compute Kelly-criterion wager. Returns (wager, kelly_fraction)."""
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return 0.0, 0.0

    bankroll = sizing.get("bankroll", 1000.0)
    kelly_frac = sizing.get("kelly_fraction", 0.25)
    min_bet = sizing.get("min_bet", 5.0)
    max_bet_pct = sizing.get("max_bet_pct", 0.05)

    kelly_f = edge / (1.0 - market_price)
    wager = bankroll * kelly_f * kelly_frac
    wager = max(min_bet, wager)
    wager = min(bankroll * max_bet_pct, wager)

    return wager, kelly_f


def compute_implied_ev(predicted_probs: dict, buckets_df: pd.DataFrame) -> float:
    """Compute model's predicted expected tweet count."""
    ev = 0.0
    for _, row in buckets_df.iterrows():
        label = row["bucket_label"]
        prob = predicted_probs.get(label, 0.0)
        lower = int(row["lower_bound"])
        upper = int(row["upper_bound"])
        if upper >= 99999:
            widths = [int(r2["upper_bound"]) - int(r2["lower_bound"])
                      for _, r2 in buckets_df.iterrows()
                      if int(r2["upper_bound"]) < 99999]
            typical = sum(widths) / len(widths) if widths else 25
            mid = lower + typical / 2
        elif lower <= 0:
            mid = upper / 2
        else:
            mid = (lower + upper) / 2
        ev += prob * mid
    return ev


def main():
    parser = argparse.ArgumentParser(description="Generate trading signals")
    parser.add_argument("--dry-run", action="store_true", help="Generate signals but skip betslip creation")
    parser.add_argument("--strategy", type=str, default=None, help="Run only this strategy")
    args = parser.parse_args()

    config = load_config()
    auto_bet = config.get("auto_bet", {})
    auto_bet_enabled = auto_bet.get("enabled", False) and not args.dry_run

    # Load registries
    model_registry = ModelRegistry()
    strategy_registry = StrategyRegistry()

    # Get active strategies
    strategies = strategy_registry.list_strategies(status="paper")
    if args.strategy:
        strategies = [s for s in strategies if s["strategy_id"] == args.strategy]

    if not strategies:
        logger.info("No active (paper) strategies found.")
        return

    logger.info("Active strategies: %s", [s["strategy_id"] for s in strategies])

    # Load active events
    active_df = load_active_events()
    if active_df.empty:
        logger.info("No active events found.")
        return

    event_groups = active_df.groupby("event_slug")
    logger.info("Found %d active events", len(event_groups))

    # Initialize
    builder = TweetFeatureBuilder(feature_group="market_adjusted")
    tracker = PerformanceTracker(
        odds_dir=str(PROJECT_DIR / "data" / "odds"),
        signals_dir=str(PROJECT_DIR / "data" / "signals"),
    )

    total_signals = 0
    actionable_signals = 0
    betslips_placed = 0

    for event_slug, buckets_df in event_groups:
        first = buckets_df.iloc[0]
        event_id = str(first.get("event_id", ""))
        market_type = str(first.get("market_type", ""))
        start_date = str(first.get("start_date", ""))[:10] if first.get("start_date") else ""
        end_date = str(first.get("end_date", ""))[:10] if first.get("end_date") else ""

        # Get latest odds for this event
        latest_odds = tracker.get_latest_odds(str(event_slug))
        if latest_odds is None:
            logger.warning("No odds recorded for %s, skipping (run fetch_current_odds.py first)", event_slug)
            continue

        # Build features
        event_metadata = {
            "event_id": event_id,
            "start_date": start_date,
            "end_date": end_date,
        }
        try:
            features = builder.build_features(str(event_slug), event_metadata)
        except Exception as e:
            logger.warning("Feature build failed for %s: %s", event_slug, e)
            continue

        # Build bucket list for model (matching backtest engine format)
        buckets = []
        for _, row in buckets_df.sort_values("lower_bound").iterrows():
            buckets.append({
                "bucket_label": row["bucket_label"],
                "lower_bound": int(row["lower_bound"]),
                "upper_bound": int(row["upper_bound"]),
            })

        # Compute duration
        duration_days = 7
        if start_date and end_date:
            try:
                sd = datetime.strptime(start_date, "%Y-%m-%d")
                ed = datetime.strptime(end_date, "%Y-%m-%d")
                duration_days = max(1, (ed - sd).days)
            except ValueError:
                pass

        # Build context
        context = {
            "duration_days": duration_days,
            "entry_prices": latest_odds.bucket_prices,
        }

        logger.info("Processing %s (%s, %d buckets, %d days)",
                     event_slug, market_type, len(buckets), duration_days)

        for strategy in strategies:
            strategy_id = strategy["strategy_id"]
            model_id = strategy["model_id"]
            filters = strategy.get("filters", {})
            sizing = strategy.get("sizing", {})

            # Instantiate model
            try:
                model = model_registry.instantiate_model(model_id)
            except Exception as e:
                logger.warning("Failed to instantiate %s: %s", model_id, e)
                continue

            # Get predictions
            try:
                predicted_probs = model.predict(features, buckets, context=context)
            except Exception as e:
                logger.warning("Prediction failed for %s on %s: %s", model_id, event_slug, e)
                continue

            if not predicted_probs:
                continue

            # Find best bucket (highest edge)
            best_bucket = None
            best_edge = -1.0
            best_model_prob = 0.0
            best_market_price = 0.0
            n_with_edge = 0
            min_edge = filters.get("min_edge", 0.02)

            for label, model_prob in predicted_probs.items():
                market_price = latest_odds.bucket_prices.get(label, 0.0)
                if market_price <= 0 or market_price >= 1:
                    continue
                edge = model_prob - market_price
                if edge >= min_edge:
                    n_with_edge += 1
                if edge > best_edge:
                    best_edge = edge
                    best_bucket = label
                    best_model_prob = model_prob
                    best_market_price = market_price

            if best_bucket is None:
                continue

            # Apply strategy filters
            meets = apply_filters(best_edge, best_market_price, market_type, filters)

            # Compute Kelly sizing
            wager, kelly_f = compute_kelly_wager(best_edge, best_market_price, sizing)

            # Compute model EV
            predicted_ev = compute_implied_ev(predicted_probs, buckets_df)

            # Build feature summary (subset for audit trail)
            feature_summary = {}
            temporal = features.get("temporal", {})
            for k in ["rolling_avg_7d", "trend_7d", "cv_14d"]:
                if k in temporal:
                    feature_summary[k] = temporal[k]

            signal = Signal(
                odds_id=latest_odds.odds_id,
                model_id=model_id,
                strategy_id=strategy_id,
                predicted_probs=predicted_probs,
                predicted_ev=predicted_ev,
                best_bucket=best_bucket,
                best_bucket_edge=best_edge,
                best_bucket_model_prob=best_model_prob,
                best_bucket_market_price=best_market_price,
                meets_criteria=meets,
                n_buckets_with_edge=n_with_edge,
                kelly_fraction=min(kelly_f, 1.0),
                recommended_wager=wager if meets else 0.0,
                strategy_ids=strategy_id,
                n_strategies=1,
                feature_summary=feature_summary,
            )

            signal_id = tracker.record_signal(signal)
            total_signals += 1

            status = "MEETS CRITERIA" if meets else "below threshold"
            logger.info("  [%s] %s -> %s edge=%.3f wager=$%.2f (%s)",
                         strategy_id, event_slug, best_bucket, best_edge,
                         wager if meets else 0, status)

            if meets:
                actionable_signals += 1

                # Check for existing position
                if tracker.has_open_position(str(event_slug), best_bucket):
                    logger.info("  Already have open position on %s/%s, skipping betslip",
                                event_slug, best_bucket)
                    continue

                # Place paper trade if auto_bet enabled
                if auto_bet_enabled:
                    try:
                        betslip_id = tracker.create_betslip_from_signal(signal_id, placed_by="paper")
                        tracker.add_fill(betslip_id, price=best_market_price, amount=wager)
                        betslips_placed += 1
                        logger.info("  Placed paper bet: %s ($%.2f on %s @ %.3f)",
                                     betslip_id, wager, best_bucket, best_market_price)
                    except Exception as e:
                        logger.error("  Betslip creation failed: %s", e)

    # Summary
    print()
    print("=" * 55)
    print("  SIGNAL GENERATION SUMMARY")
    print("=" * 55)
    print("  Strategies run:     {:>4d}".format(len(strategies)))
    print("  Events processed:   {:>4d}".format(len(event_groups)))
    print("  Signals generated:  {:>4d}".format(total_signals))
    print("  Meet criteria:      {:>4d}".format(actionable_signals))
    print("  Betslips placed:    {:>4d}".format(betslips_placed))
    if args.dry_run:
        print("  (dry-run mode — no betslips created)")
    elif not auto_bet_enabled:
        print("  (auto_bet disabled in config/pipeline.json)")
    print("=" * 55)


if __name__ == "__main__":
    main()
