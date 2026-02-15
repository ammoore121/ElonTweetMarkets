"""Full paper trading pipeline orchestrator.

Runs all pipeline steps in order:
    1. Fetch XTracker history (daily tweet counts)
    1b. Fetch market data (Tesla stock + crypto prices)
    1c. Fetch Wikipedia pageviews (attention signals)
    2. Refresh market catalog (fresh token IDs from Gamma API)
    3. Fetch current market odds (live bucket prices from CLOB)
    4. Generate signals + place paper trades
    5. Settle completed events
    6. Print performance summary

Designed for scheduled execution (e.g., every 6 hours via Task Scheduler or cron).

Usage:
    python scripts/cron_pipeline.py
    python scripts/cron_pipeline.py --skip-xtracker
    python scripts/cron_pipeline.py --dry-run
"""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Set up logging to both console and file
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_filename = "pipeline_{}.log".format(
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
    """Run a pipeline step with error handling. Returns True on success.

    argv_override: temporarily replace sys.argv so sub-scripts' argparse
    doesn't see the parent's flags.
    """
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
        # Scripts may call sys.exit(0) on success
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
    parser = argparse.ArgumentParser(description="Full paper trading pipeline")
    parser.add_argument("--skip-xtracker", action="store_true",
                        help="Skip XTracker history fetch")
    parser.add_argument("--skip-odds", action="store_true",
                        help="Skip odds fetch (use cached)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals but don't place bets")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc)
    logger.info("Pipeline started at %s", start_time.isoformat())
    logger.info("Log file: %s", log_path)

    results = {}

    # Step 1: Fetch XTracker history
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

    # Step 1b: Fetch market data (Tesla stock + crypto prices)
    try:
        from scripts.fetch_market_data import main as fetch_market_data_main
        results["market_data"] = run_step(
            "Fetch Market Data", fetch_market_data_main,
            argv_override=["fetch_market_data.py"],
        )
    except ImportError:
        logger.warning("fetch_market_data.py not importable, skipping")
        results["market_data"] = False
    except Exception:
        logger.warning("Fetch Market Data failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["market_data"] = False

    # Step 1c: Fetch Wikipedia pageviews
    try:
        from scripts.fetch_wikipedia_pageviews import main as fetch_wikipedia_main
        results["wikipedia"] = run_step(
            "Fetch Wikipedia Pageviews", fetch_wikipedia_main,
            argv_override=["fetch_wikipedia_pageviews.py"],
        )
    except ImportError:
        logger.warning("fetch_wikipedia_pageviews.py not importable, skipping")
        results["wikipedia"] = False
    except Exception:
        logger.warning("Fetch Wikipedia Pageviews failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["wikipedia"] = False

    # Step 1d: Fetch VIX data
    try:
        from scripts.fetch_vix_data import main as fetch_vix_main
        results["vix"] = run_step(
            "Fetch VIX Data", fetch_vix_main,
            argv_override=["fetch_vix_data.py"],
        )
    except ImportError:
        logger.warning("fetch_vix_data.py not importable, skipping")
        results["vix"] = False
    except Exception:
        logger.warning("Fetch VIX Data failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["vix"] = False

    # Step 1e: Fetch Crypto Fear & Greed Index
    try:
        from scripts.fetch_crypto_fg import main as fetch_crypto_fg_main
        results["crypto_fg"] = run_step(
            "Fetch Crypto Fear & Greed", fetch_crypto_fg_main,
            argv_override=["fetch_crypto_fg.py"],
        )
    except ImportError:
        logger.warning("fetch_crypto_fg.py not importable, skipping")
        results["crypto_fg"] = False
    except Exception:
        logger.warning("Fetch Crypto Fear & Greed failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["crypto_fg"] = False

    # Step 1f: Fetch Google Trends (rate-limited, may take ~20min)
    try:
        from scripts.fetch_google_trends import main as fetch_trends_main
        results["google_trends"] = run_step(
            "Fetch Google Trends", fetch_trends_main,
            argv_override=["fetch_google_trends.py"],
        )
    except ImportError:
        logger.warning("fetch_google_trends.py not importable, skipping")
        results["google_trends"] = False
    except Exception:
        logger.warning("Fetch Google Trends failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["google_trends"] = False

    # Step 2: Refresh market catalog (fresh token IDs for active events)
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

    # Step 3: Fetch current odds
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

    # Step 4: Generate signals
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

    # Step 5: Settle bets
    try:
        from scripts.settle_bets import main as settle_main
        results["settlement"] = run_step(
            "Settle Bets", settle_main,
            argv_override=["settle_bets.py"],
        )
    except ImportError:
        logger.warning("settle_bets.py not importable, skipping")
        results["settlement"] = False

    # Step 6: Performance summary
    try:
        from src.paper_trading.tracker import PerformanceTracker
        tracker = PerformanceTracker(
            odds_dir=str(PROJECT_DIR / "data" / "odds"),
            signals_dir=str(PROJECT_DIR / "data" / "signals"),
        )
        tracker.print_performance()
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
    print("  PIPELINE SUMMARY")
    print("=" * 55)
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        print("  {:<25s} {}".format(step, status))
    print()
    print("  Steps: {}/{} succeeded".format(n_success, n_total))
    print("  Elapsed: {:.1f}s".format(elapsed))
    print("  Log: {}".format(log_path))
    print("=" * 55)

    # Exit code
    if n_success == n_total:
        sys.exit(0)
    elif n_success > 0:
        sys.exit(1)  # Partial failure
    else:
        sys.exit(2)  # Critical failure


if __name__ == "__main__":
    main()
