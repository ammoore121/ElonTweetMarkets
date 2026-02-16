"""
Run backtest with a specified model or strategy against the backtest dataset.

Thin CLI wrapper around src.backtesting.BacktestEngine.

Usage (model mode -- original):
    python scripts/run_backtest.py                          # NaiveBucketModel on gold events
    python scripts/run_backtest.py --model crowd            # CrowdModel baseline
    python scripts/run_backtest.py --model adjusted         # MarketAdjustedModel
    python scripts/run_backtest.py --tier gold              # Only gold tier
    python scripts/run_backtest.py --all                    # All tiers

Usage (strategy mode -- NEW):
    python scripts/run_backtest.py --strategy adjusted_conservative --tier gold
    python scripts/run_backtest.py --strategy adjusted_aggressive --tier gold

The --strategy flag loads model_id, filters, sizing, and entry config from
strategies/strategy_registry.json.  It overrides --model, --kelly, --min-edge,
--bankroll, and --entry-hours.

Requires:
    - Backtest dataset at data/backtest/ (run build_backtest_dataset.py first)
    - src/ml/baseline_model.py (NaiveBucketModel, CrowdModel)
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.backtesting.engine import BacktestEngine
from src.ml.baseline_model import NaiveBucketModel, CrowdModel
from src.ml.advanced_models import RegimeAwareModel, MarketAdjustedModel, EnsembleModel
from src.ml.per_bucket_model import PerBucketModel
from src.ml.signal_enhanced_model import SignalEnhancedTailModel, SignalEnhancedTailModelV4, SignalEnhancedTailModelV5
from src.ml.price_dynamics_model import PriceDynamicsModel
from src.ml.cross_market_model import CrossMarketArbModel
from src.ml.consensus_model import ConsensusEnsembleModel
from src.ml.registry import ModelRegistry, StrategyRegistry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "naive": NaiveBucketModel,
    "crowd": CrowdModel,
    "regime": RegimeAwareModel,
    "adjusted": MarketAdjustedModel,
    "ensemble": EnsembleModel,
    "perbucket": PerBucketModel,
    "signal_enhanced": SignalEnhancedTailModel,
    "signal_enhanced_v4": SignalEnhancedTailModelV4,
    "signal_enhanced_v5": SignalEnhancedTailModelV5,
    "price_dynamics": PriceDynamicsModel,
    "cross_market_arb": CrossMarketArbModel,
    "consensus_ensemble": ConsensusEnsembleModel,
}

# Maps model_id from registry to MODEL_MAP key for instantiation
MODEL_ID_TO_KEY = {
    "naive_negbin_v1": "naive",
    "crowd_v1": "crowd",
    "regime_aware_v1": "regime",
    "market_adjusted_v1": "adjusted",
    "ensemble_v1": "ensemble",
    "per_bucket_v1": "perbucket",
    "signal_enhanced_tail_v1": "signal_enhanced",
    "signal_enhanced_tail_v2": "signal_enhanced",
    "signal_enhanced_tail_v3": "signal_enhanced",
    "signal_enhanced_tail_v4": "signal_enhanced_v4",
    "signal_enhanced_tail_v5": "signal_enhanced_v5",
    "price_dynamics_v1": "price_dynamics",
    "cross_market_arb_v1": "cross_market_arb",
    "consensus_ensemble_v1": "consensus_ensemble",
}


def load_backtest_index():
    """Load the master backtest index."""
    if not INDEX_PATH.exists():
        print("ERROR: Backtest index not found at {}".format(INDEX_PATH))
        print("       Run: python scripts/build_backtest_dataset.py")
        sys.exit(1)

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_name):
    """Instantiate a model by short name (naive, crowd, etc.)."""
    cls = MODEL_MAP.get(model_name)
    if cls is None:
        print("Unknown model: {}. Available: {}".format(
            model_name, list(MODEL_MAP.keys())))
        sys.exit(1)
    return cls()


def load_model_from_id(model_id):
    """Instantiate a model from a registry model_id."""
    key = MODEL_ID_TO_KEY.get(model_id)
    if key is None:
        # Try the short name directly (e.g. if model_id is "adjusted")
        key = model_id
    return load_model(key)


def filter_events(events, tier=None, all_tiers=False, event_slug=None):
    """Filter events from the index by tier and/or slug."""
    if not all_tiers and tier:
        events = [e for e in events if e.get("ground_truth_tier") == tier]
    if event_slug:
        events = [e for e in events if e.get("event_slug") == event_slug]
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run backtest with model or strategy"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        default="naive",
        help="Model to use (default: naive). Ignored if --strategy is set.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy ID from strategy_registry.json. Overrides --model and sizing args.",
    )
    parser.add_argument(
        "--tier",
        default="gold",
        help="Ground truth tier filter (default: gold)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all tiers",
    )
    parser.add_argument(
        "--entry-hours",
        type=int,
        default=24,
        help="Hours before close to enter (default: 24)",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll (default: 1000)",
    )
    parser.add_argument(
        "--kelly",
        type=float,
        default=0.25,
        help="Kelly fraction (default: 0.25)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge to bet (default: 0.05)",
    )
    parser.add_argument(
        "--max-bet-pct",
        type=float,
        default=0.10,
        help="Max bet as pct of bankroll (default: 0.10)",
    )
    parser.add_argument(
        "--event-slug",
        type=str,
        default=None,
        help="Run on a single event",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show predictions without betting",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip logging results to model registry",
    )
    args = parser.parse_args()

    # Load index
    index = load_backtest_index()
    if index is None:
        return

    events = index.get("events", [])

    # ---------------------------------------------------------------
    # Resolve strategy OR model
    # ---------------------------------------------------------------
    strategy_def = None
    strategy_id = args.strategy

    if strategy_id is not None:
        # Strategy mode: load from strategy registry
        sr = StrategyRegistry()
        strategy_def = sr.get_strategy(strategy_id)
        if strategy_def is None:
            print("ERROR: Strategy '{}' not found in strategy registry.".format(
                strategy_id))
            print("       Available: {}".format(
                [s["strategy_id"] for s in sr.list_strategies()]))
            sys.exit(1)

        # Get model from strategy — use ModelRegistry.instantiate_model()
        # so hyperparameters from the registry are injected correctly
        # (e.g. signal_enhanced_tail_v2 vs v3 have different params)
        model_id = strategy_def["model_id"]
        try:
            mr = ModelRegistry()
            model = mr.instantiate_model(model_id)
        except (ValueError, KeyError):
            # Fallback to MODEL_MAP if not in registry
            model = load_model_from_id(model_id)

        # Build engine config from strategy
        engine_config = sr.build_engine_config(strategy_id)
        engine_config["dry_run"] = args.dry_run

        # Use strategy's model short name for display
        model_short = MODEL_ID_TO_KEY.get(model_id, model_id)

        print(
            "Running backtest: strategy={}, model={}, {} events, "
            "min_edge={:.0%}, kelly={}, bankroll=${:,.0f}".format(
                strategy_id,
                model_short,
                len(filter_events(events, tier=args.tier, all_tiers=args.all,
                                  event_slug=args.event_slug)),
                engine_config["min_edge"],
                engine_config["kelly_fraction"],
                engine_config["bankroll"],
            )
        )
    else:
        # Model mode: original behavior
        model = load_model(args.model)

        engine_config = {
            "bankroll": args.bankroll,
            "kelly_fraction": args.kelly,
            "min_edge": args.min_edge,
            "max_bet_pct": 0.05,
            "entry_hours_before_close": args.entry_hours,
            "entry_window_hours": 6,
            "dry_run": args.dry_run,
        }

        print(
            "Running backtest: model={}, {} events, entry=T-{}h, "
            "bankroll=${:,.0f}".format(
                args.model,
                len(filter_events(events, tier=args.tier, all_tiers=args.all,
                                  event_slug=args.event_slug)),
                args.entry_hours, args.bankroll,
            )
        )

    print()

    # Filter events
    events = filter_events(
        events,
        tier=args.tier,
        all_tiers=args.all,
        event_slug=args.event_slug,
    )

    if not events:
        print("No events match the filter criteria.")
        return

    # Run backtest
    engine = BacktestEngine(config=engine_config)
    result = engine.run(model, events)

    # Report
    engine.print_report(result)

    # Save results to data/results/ (legacy location)
    save_path = engine.save_result(result)
    print("\nResults saved to: {}".format(save_path))

    # Log to model registry (per-model backtest folder)
    if not args.no_log:
        _log_to_registry(model, result, args, strategy_def)


def _log_to_registry(model, result, args, strategy_def=None):
    """Log backtest results to the ModelRegistry per-model backtest folder.

    Also updates the strategy's backtest_summary if running in strategy mode.
    """
    try:
        registry = ModelRegistry()

        # Get model_id
        model_id = getattr(model, "model_id", "{}_v1".format(model.name))

        # Determine tier
        tier = "all" if args.all else args.tier

        # Build metrics dict from BacktestResult
        metrics = {
            "brier_score": result.brier_score,
            "log_loss": result.log_loss,
            "accuracy": result.accuracy_str,
            "roi_pct": result.roi,
            "n_events": result.n_events,
            "n_bets": result.n_bets,
            "total_pnl": result.total_pnl,
            "total_wagered": result.total_wagered,
        }

        # Build config string
        if strategy_def is not None:
            config_str = (
                "strategy={}, model={}, bankroll=${:,.0f}, kelly={}, "
                "min_edge={:.0%}, entry=T-{}h".format(
                    strategy_def["strategy_id"],
                    model_id,
                    result.config.get("bankroll", 0),
                    result.config.get("kelly_fraction", 0),
                    result.config.get("min_edge", 0),
                    result.config.get("entry_hours_before_close", 0),
                )
            )
        else:
            config_str = (
                "model={}, bankroll=${:,.0f}, kelly={}, "
                "min_edge={:.0%}, entry=T-{}h".format(
                    args.model,
                    args.bankroll,
                    args.kelly,
                    args.min_edge,
                    args.entry_hours,
                )
            )

        # Check model exists in registry (it should for all known models)
        if registry.get_model(model_id) is None:
            print("WARNING: Model '{}' not in registry, skipping backtest log.".format(
                model_id))
            return

        # Save to per-model backtest folder
        bt_path = registry.log_backtest(
            model_id=model_id,
            tier=tier,
            metrics=metrics,
            config=config_str,
            notes="Auto-logged from run_backtest.py",
        )

        print("Backtest logged: {} -> {}".format(model_id, bt_path))

        # If running with a strategy, update its backtest_summary
        if strategy_def is not None:
            try:
                sr = StrategyRegistry()
                summary = {
                    "tier": tier,
                    "n_events": result.n_events,
                    "n_bets": result.n_bets,
                    "brier": result.brier_score,
                    "roi_pct": result.roi,
                    "notes": "Auto-updated from run_backtest.py",
                }
                sr.update_backtest_summary(strategy_def["strategy_id"], summary)
                print("Strategy '{}' backtest_summary updated.".format(
                    strategy_def["strategy_id"]))
            except Exception as e:
                print("WARNING: Failed to update strategy summary: {}".format(e))

    except Exception as e:
        print("WARNING: Failed to log to registry: {}".format(e))


if __name__ == "__main__":
    main()
