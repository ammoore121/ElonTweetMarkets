"""Signal pipeline — runs every 6 hours.

Fetches only time-sensitive data (live odds, current tweet counts, order book),
then generates signals and settles bets. Relies on daily feature data already
on disk from cron_daily.py.

Steps:
    1. Fetch XTracker history (today's tweet count for temporal features)
    2. Refresh market catalog (new markets / resolved markets)
    3. Fetch current odds (live CLOB bucket prices)
    4. Fetch order book depth (liquidity snapshots)
    5. Generate signals + place paper trades
    6. Settle completed events
    7. Print performance summary

Usage:
    python scripts/cron_signal.py
    python scripts/cron_signal.py --dry-run
    python scripts/cron_signal.py --skip-xtracker
    python scripts/cron_signal.py --skip-odds
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_filename = "signal_{}.log".format(
    datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
)
log_path = LOG_DIR / log_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def run_step(name: str, func, argv_override: list | None = None, **kwargs) -> bool:
    """Run a pipeline step with error handling. Returns True on success."""
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("=" * 60)
    saved_argv = sys.argv
    if argv_override is not None:
        sys.argv = argv_override
    try:
        func(**kwargs)
        logger.info("STEP %s: SUCCESS", name)
        return True
    except SystemExit as e:
        if e.code == 0 or e.code is None:
            logger.info("STEP %s: SUCCESS (exit 0)", name)
            return True
        logger.error("STEP %s: FAILED (exit %s)", name, e.code)
        return False
    except Exception:
        logger.error("STEP %s: FAILED", name)
        logger.error(traceback.format_exc())
        return False
    finally:
        sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(description="Signal pipeline (every 6h)")
    parser.add_argument("--skip-xtracker", action="store_true",
                        help="Skip XTracker history fetch")
    parser.add_argument("--skip-odds", action="store_true",
                        help="Skip odds fetch (use cached)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals but don't place bets")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc)
    logger.info("Signal pipeline started at %s", start_time.isoformat())
    logger.info("Log file: %s", log_path)

    results = {}

    # Step 1: Fetch XTracker history (today's tweet count for temporal features)
    if not args.skip_xtracker:
        try:
            from scripts.fetch_xtracker_history import main as fetch_xtracker_main
            results["xtracker"] = run_step(
                "Fetch XTracker History", fetch_xtracker_main,
                argv_override=["fetch_xtracker_history.py"],
            )
        except ImportError:
            logger.warning("fetch_xtracker_history.py not importable, skipping")
            results["xtracker"] = False
    else:
        logger.info("Skipping XTracker fetch (--skip-xtracker)")
        results["xtracker"] = True

    # Step 2: Refresh market catalog (new/resolved markets)
    if not args.skip_odds:
        try:
            from scripts.fetch_elon_markets import main as fetch_markets_main
            results["markets"] = run_step(
                "Fetch Fresh Markets", fetch_markets_main,
                argv_override=["fetch_elon_markets.py"],
            )
        except ImportError:
            logger.warning("fetch_elon_markets.py not importable, skipping")
            results["markets"] = False

        try:
            from scripts.build_market_catalog import main as build_catalog_main
            results["catalog"] = run_step(
                "Build Market Catalog", build_catalog_main,
                argv_override=["build_market_catalog.py"],
            )
        except ImportError:
            logger.warning("build_market_catalog.py not importable, skipping")
            results["catalog"] = False
    else:
        logger.info("Skipping market refresh (--skip-odds)")
        results["markets"] = True
        results["catalog"] = True

    # Step 3: Fetch current odds (live CLOB bucket prices)
    if not args.skip_odds:
        try:
            from scripts.fetch_current_odds import main as fetch_odds_main
            results["odds"] = run_step(
                "Fetch Current Odds", fetch_odds_main,
                argv_override=["fetch_current_odds.py"],
            )
        except ImportError:
            logger.warning("fetch_current_odds.py not importable, skipping")
            results["odds"] = False
    else:
        logger.info("Skipping odds fetch (--skip-odds)")
        results["odds"] = True

    # Step 4: Fetch order book depth
    if not args.skip_odds:
        try:
            from scripts.fetch_orderbook import main as fetch_orderbook_main
            results["orderbook"] = run_step(
                "Fetch Order Book", fetch_orderbook_main,
                argv_override=["fetch_orderbook.py"],
            )
        except ImportError:
            logger.warning("fetch_orderbook.py not importable, skipping")
            results["orderbook"] = False
        except Exception:
            logger.warning("Fetch Order Book failed (non-critical), continuing")
            logger.warning(traceback.format_exc())
            results["orderbook"] = False

    # Step 5: Generate signals
    signal_argv = ["generate_signals.py"]
    if args.dry_run:
        signal_argv.append("--dry-run")
    try:
        from scripts.generate_signals import main as generate_signals_main
        results["signals"] = run_step(
            "Generate Signals", generate_signals_main,
            argv_override=signal_argv,
        )
    except ImportError:
        logger.warning("generate_signals.py not importable, skipping")
        results["signals"] = False

    # Step 6: Settle bets
    try:
        from scripts.settle_bets import main as settle_main
        results["settlement"] = run_step(
            "Settle Bets", settle_main,
            argv_override=["settle_bets.py"],
        )
    except ImportError:
        logger.warning("settle_bets.py not importable, skipping")
        results["settlement"] = False

    # Step 7: Performance summary + strategy health monitor
    try:
        from src.paper_trading.tracker import PerformanceTracker
        tracker = PerformanceTracker(
            odds_dir=str(PROJECT_DIR / "data" / "odds"),
            signals_dir=str(PROJECT_DIR / "data" / "signals"),
        )
        tracker.print_performance()

        # Post-settlement health check
        from src.monitoring.strategy_monitor import StrategyMonitor
        monitor = StrategyMonitor()
        alerts = monitor.check_strategy_health(tracker)
        if alerts:
            logger.warning("=" * 55)
            logger.warning("  STRATEGY HEALTH ALERTS")
            logger.warning("=" * 55)
            for alert in alerts:
                severity = alert["severity"].upper()
                logger.log(
                    logging.CRITICAL if severity == "CRITICAL" else logging.WARNING,
                    "[%s] %s: %s -- %s",
                    severity, alert["strategy_id"],
                    alert["message"], alert["recommendation"],
                )
            logger.warning("=" * 55)
        else:
            logger.info("Strategy health check: all strategies OK")
    except Exception:
        logger.error("Failed to print performance summary")
        logger.error(traceback.format_exc())

    # Final summary
    end_time = datetime.now(timezone.utc)
    elapsed = (end_time - start_time).total_seconds()

    n_success = sum(1 for v in results.values() if v)
    n_total = len(results)

    print()
    print("=" * 55)
    print("  SIGNAL PIPELINE SUMMARY")
    print("=" * 55)
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        print("  {:<25s} {}".format(step, status))
    print()
    print("  Steps: {}/{} succeeded".format(n_success, n_total))
    print("  Elapsed: {:.1f}s".format(elapsed))
    print("  Log: {}".format(log_path))
    print("=" * 55)

    if n_success == n_total:
        sys.exit(0)
    elif n_success > 0:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
