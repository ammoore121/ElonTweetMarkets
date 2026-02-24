"""Daily data pipeline — runs once per day in the AM.

Fetches slow-moving data sources that update at most once per day.
These populate the parquet files that the signal pipeline reads as features.

Steps:
    1.  Tesla stock + Crypto prices (yfinance)
    2.  VIX volatility index (yfinance)
    3.  Crypto Fear & Greed index (alternative.me)
    4.  Wikipedia pageviews (Wikimedia API, ~2 day lag)
    5.  GDELT news volumes/tone
    6.  SpaceX launch calendar
    7.  Government calendar (Federal Register + GovTrack)
    8.  Corporate events (yfinance + SEC EDGAR)
    9.  Google Trends (~20 min, rate limited)
    10. Reddit activity (daily mode)
    11. CLOB price history backfill (--daily mode)

Usage:
    python scripts/cron_daily.py
    python scripts/cron_daily.py --skip-trends     # Skip Google Trends (slow)
    python scripts/cron_daily.py --skip-clob        # Skip CLOB history backfill
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

log_filename = "daily_{}.log".format(
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
    parser = argparse.ArgumentParser(description="Daily data pipeline (AM)")
    parser.add_argument("--skip-trends", action="store_true",
                        help="Skip Google Trends fetch (~20 min)")
    parser.add_argument("--skip-clob", action="store_true",
                        help="Skip CLOB price history backfill")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc)
    logger.info("Daily pipeline started at %s", start_time.isoformat())
    logger.info("Log file: %s", log_path)

    results = {}

    # Step 1: Tesla stock + Crypto prices
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

    # Step 2: VIX volatility index
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

    # Step 3: Crypto Fear & Greed
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

    # Step 4: Wikipedia pageviews
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

    # Step 5: GDELT news
    try:
        from scripts.fetch_gdelt_news import main as fetch_gdelt_main
        results["gdelt"] = run_step(
            "Fetch GDELT News", fetch_gdelt_main,
            argv_override=["fetch_gdelt_news.py"],
        )
    except ImportError:
        logger.warning("fetch_gdelt_news.py not importable, skipping")
        results["gdelt"] = False
    except Exception:
        logger.warning("Fetch GDELT News failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["gdelt"] = False

    # Step 6: SpaceX launches
    try:
        from scripts.fetch_spacex_launches import main as fetch_spacex_main
        results["spacex"] = run_step(
            "Fetch SpaceX Launches", fetch_spacex_main,
            argv_override=["fetch_spacex_launches.py"],
        )
    except ImportError:
        logger.warning("fetch_spacex_launches.py not importable, skipping")
        results["spacex"] = False
    except Exception:
        logger.warning("Fetch SpaceX Launches failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["spacex"] = False

    # Step 7: Government calendar
    try:
        from scripts.fetch_government_calendar import main as fetch_govt_main
        results["government"] = run_step(
            "Fetch Government Calendar", fetch_govt_main,
            argv_override=["fetch_government_calendar.py"],
        )
    except ImportError:
        logger.warning("fetch_government_calendar.py not importable, skipping")
        results["government"] = False
    except Exception:
        logger.warning("Fetch Government Calendar failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["government"] = False

    # Step 8: Corporate events
    try:
        from scripts.fetch_corporate_events import main as fetch_corp_main
        results["corporate"] = run_step(
            "Fetch Corporate Events", fetch_corp_main,
            argv_override=["fetch_corporate_events.py"],
        )
    except ImportError:
        logger.warning("fetch_corporate_events.py not importable, skipping")
        results["corporate"] = False
    except Exception:
        logger.warning("Fetch Corporate Events failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["corporate"] = False

    # Step 9: Google Trends (~20 min, rate limited)
    if not args.skip_trends:
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
    else:
        logger.info("Skipping Google Trends (--skip-trends)")
        results["google_trends"] = True

    # Step 10: Reddit activity
    try:
        from scripts.fetch_reddit_activity import main as fetch_reddit_main
        results["reddit"] = run_step(
            "Fetch Reddit Activity", fetch_reddit_main,
            argv_override=["fetch_reddit_activity.py", "--mode", "daily"],
        )
    except ImportError:
        logger.warning("fetch_reddit_activity.py not importable, skipping")
        results["reddit"] = False
    except Exception:
        logger.warning("Fetch Reddit Activity failed (non-critical), continuing")
        logger.warning(traceback.format_exc())
        results["reddit"] = False

    # Step 11: CLOB price history backfill
    if not args.skip_clob:
        try:
            from scripts.fetch_clob_prices import main as fetch_clob_main
            results["clob_history"] = run_step(
                "Fetch CLOB Price History", fetch_clob_main,
                argv_override=["fetch_clob_prices.py", "--daily"],
            )
        except ImportError:
            logger.warning("fetch_clob_prices.py not importable, skipping")
            results["clob_history"] = False
        except Exception:
            logger.warning("Fetch CLOB Price History failed (non-critical), continuing")
            logger.warning(traceback.format_exc())
            results["clob_history"] = False
    else:
        logger.info("Skipping CLOB history backfill (--skip-clob)")
        results["clob_history"] = True

    # Final summary
    end_time = datetime.now(timezone.utc)
    elapsed = (end_time - start_time).total_seconds()

    n_success = sum(1 for v in results.values() if v)
    n_total = len(results)

    print()
    print("=" * 55)
    print("  DAILY PIPELINE SUMMARY")
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
