# Elon Musk Tweet Count Prediction Markets

**Goal**: ML models to predict Elon Musk's daily/weekly tweet counts for Polymarket betting.
**Current State**: Paper trading phase -- 19 models registered, 7 profitable, pipeline runs every 6h.
**Market**: Categorical prediction (multiple buckets of tweet count ranges).
**Backtest**: 159 events across 3 tiers (gold 38, silver 9, bronze 112), 21 months of data.

---

## For AI Agents: Key Context

1. **Check [task.md](task.md)** for active tasks and blockers
2. **This is NOT binary prediction**: Markets have 10-30 outcome buckets (tweet count ranges)
3. **XTracker is the oracle**: All markets resolve via https://xtracker.polymarket.com
4. **11 data sources active**: XTracker, Kaggle, GDELT, SpaceX, Polymarket CLOB, Tesla (yfinance), Crypto (yfinance), Wikipedia pageviews, VIX (yfinance), Crypto Fear & Greed (alternative.me), Google Trends (pytrends)
5. **91 features across 8 categories**: temporal (12), media (21), calendar (6), market (8), cross (5), financial (21), attention (8), trends (10)
6. **Twitter API is NOT needed**: Free tier is write-only. Use Kaggle + XTracker instead.
7. **See [docs/RESEARCH.md](docs/RESEARCH.md)** for detailed data source documentation

---

## Quick Commands

```bash
# Data collection
python scripts/fetch_xtracker_history.py           # XTracker daily tweet counts
python scripts/fetch_elon_markets.py                # Polymarket Elon tweet markets
python scripts/fetch_clob_prices.py --daily         # CLOB price history (daily mode)
python scripts/fetch_market_data.py                 # Tesla stock + crypto (yfinance)
python scripts/fetch_wikipedia_pageviews.py         # Wikipedia pageviews (Wikimedia API)
python scripts/fetch_gdelt_news.py                  # GDELT news volumes/tone
python scripts/fetch_spacex_launches.py             # SpaceX launch calendar
python scripts/fetch_vix_data.py                    # VIX volatility index (yfinance)
python scripts/fetch_crypto_fg.py                   # Crypto Fear & Greed (alternative.me)
python scripts/fetch_google_trends.py               # Google Trends (~20 min, rate limited)

# Data processing
python scripts/build_market_catalog.py              # Build market catalog parquet
python scripts/build_xtracker_mapping.py            # Map events to ground truth tiers
python scripts/build_backtest_dataset.py            # Build unified backtest dataset (159 events)

# Backtesting
python scripts/run_backtest.py --model naive --tier gold     # Single model backtest
python scripts/run_backtest.py --strategy tail_boost_primary  # Strategy backtest
python scripts/run_backtest.py --all                          # All tiers
python scripts/test_new_features.py                           # Cross-tier model comparison

# Paper trading (production)
python scripts/cron_pipeline.py                     # Full automated pipeline (every 6h)
python scripts/fetch_current_odds.py                # Capture live bucket prices
python scripts/generate_signals.py                  # Generate betting signals
python scripts/settle_bets.py                       # Settle completed events
```

---

## CRITICAL GOTCHAS (DO NOT SKIP)

### 1. Distributional Prediction, Not Binary

Markets have multiple outcome buckets (e.g., 0-19, 20-39, 40-59, ..., 740+).
Must predict probability distribution across all buckets, not a single number.

**Proven approaches**: Tail-redistribution models (move mass from center to tails),
duration-aware shrinkage, signal-enhanced crowd adjustment. See `src/ml/` for implementations.

### 2. XTracker is the Resolution Oracle

What it counts: original posts, reposts, quote tweets.
What it doesn't: replies (unless on main feed), deleted posts (if deleted within ~5 min).

**ALWAYS validate against XTracker**, not raw Twitter data.

### 3. Temporal Leakage

Only use data from BEFORE the prediction window. No same-day features.

### 4. Market Structure Varies

- **2-3 day markets**: ~10 buckets
- **Weekly markets**: ~27-32 buckets
- **Monthly markets**: Many more buckets

Parse market question carefully to extract bucket boundaries.

### 5. High Variance = The Structural Edge

Crowd underprices tails -- this is STRUCTURAL (holds across 21 months of data).
Crowd captures only 48% of downswings but 105% of upswings (asymmetric adjustment).
Daily markets: crowd overestimates by +27% on average.
Z-score momentum is REGIME-SPECIFIC (fails pre-Oct 2025) -- do not rely on it alone.

### 6. XTracker Zero-Count Days

30/103 XTracker records show count=0 due to noon-to-noon tracking window boundaries.
These are NOT actual zero-tweet days. Handle accordingly in feature engineering.

---

## Data Sources (All Free, 11 Active)

| Source | Script | Storage | Features |
|--------|--------|---------|----------|
| **XTracker** | fetch_xtracker_history.py | data/sources/xtracker/ | 12 temporal features |
| **Kaggle tweets** | process_kaggle_tweets.py | data/processed/daily_counts.json | Merged with XTracker for unified counts |
| **GDELT** | fetch_gdelt_news.py | data/sources/gdelt/ | 21 media features |
| **SpaceX LL2** | fetch_spacex_launches.py | data/sources/calendar/ | 6 calendar features |
| **Polymarket CLOB** | fetch_clob_prices.py | data/sources/polymarket/prices/ | 8 market features |
| **Tesla (yfinance)** | fetch_market_data.py | data/sources/market/tesla_daily.parquet | 6 financial features |
| **Crypto (yfinance)** | fetch_market_data.py | data/sources/market/crypto_daily.parquet | 6 financial features |
| **Wikipedia** | fetch_wikipedia_pageviews.py | data/sources/wikipedia/pageviews.json | 8 attention features |
| **VIX (yfinance)** | fetch_vix_data.py | data/sources/market/vix_daily.parquet | 5 financial features |
| **Crypto F&G** | fetch_crypto_fg.py | data/sources/market/crypto_fear_greed.parquet | 4 financial features |
| **Google Trends** | fetch_google_trends.py | data/sources/trends/google_trends.parquet | 10 trends features |

**Not used**: SEC EDGAR (low signal), ElonJet Mastodon (24h delay too stale), Twitter API (free tier is write-only).

---

## TRADING CONSTRAINTS

### Categorical Market Structure

Markets have 10-30 buckets. Liquidity is split across all buckets.
With $50K total market liquidity and 10 buckets: ~$5K per bucket.

**Require min $5K per bucket for acceptable slippage on $100 bets.**

### Kelly for Multi-Outcome

```python
# If model predicts P(bucket_i) = 0.35, market price = 0.25
edge = 0.35 - 0.25  # 10% edge
kelly_fraction = edge / (1 - market_prob)
wager = bankroll * kelly_fraction * 0.25  # Quarter Kelly
```

---

## Model Performance (Cross-Tier Validated)

Top strategies at 2% min_edge, backtested on 159 events across 21 months:

| Strategy | Gold ROI | Gold Bets | All-Tier ROI | All-Tier P&L | Notes |
|----------|----------|-----------|--------------|-------------|-------|
| **SignalEnhanced v3** | +66.3% | 61 | Best overall | +$395 | Highest P&L across all tiers |
| **DurationTail** | +67.7% | 43 | +13.8% | +$199 | Strong gold performance |
| **DurationShrink** | +49.7% | 23 | +21.5% | +$112 | Fewest bets, high conviction |
| **TailBoost** | +44.4% | 60 | +13.9% | +$303 | Most trusted, structural edge, 187 bets all tiers |
| **SignalEnhanced v4** | +65.4% | 61 | ~+13% | +$389 | 7 signals (VIX+Trends+F&G), 189 bets all tiers |

**Key insight**: Only tail-redistribution models survive out-of-sample. Z-score / momentum
models overfit to the current regime and fail on bronze tier (pre-Oct 2025 data).

---

## Pipeline Refresh Order

```bash
# 1. Fetch all data sources
python scripts/fetch_xtracker_history.py         # Resolution data (oracle)
python scripts/fetch_elon_markets.py             # Polymarket event catalog
python scripts/fetch_clob_prices.py --daily      # Bucket price history
python scripts/fetch_market_data.py              # Tesla + crypto
python scripts/fetch_wikipedia_pageviews.py      # Wikipedia attention
python scripts/fetch_gdelt_news.py               # News volumes/tone
python scripts/fetch_spacex_launches.py          # Launch calendar
python scripts/fetch_vix_data.py                 # VIX volatility index
python scripts/fetch_crypto_fg.py                # Crypto Fear & Greed
python scripts/fetch_google_trends.py            # Google Trends (~20 min)

# 2. Build processed datasets
python scripts/build_market_catalog.py           # Market catalog parquet
python scripts/build_xtracker_mapping.py         # Event-to-ground-truth mapping
python scripts/build_backtest_dataset.py         # Unified backtest dataset

# 3. Backtest models
python scripts/run_backtest.py --all             # All models, all tiers
python scripts/test_new_features.py              # Cross-tier comparison

# 4. Paper trading (automated via cron)
python scripts/cron_pipeline.py                  # Runs fetch_current_odds + generate_signals + settle_bets
```

---

## Architecture

```
src/
  features/
    feature_builder.py    # TweetFeatureBuilder - unified feature pipeline
    extractors.py         # 7 feature categories (temporal, media, calendar, market, cross, financial, attention)
    factor_registry.py    # 36+ documented factors
  ml/
    base_model.py         # BasePredictionModel ABC
    baseline_model.py     # NaiveModel, HistoricalAvgModel
    advanced_models.py    # TailBoostModel, ContrarianModel, VolatilityModel
    per_bucket_model.py   # PerBucketModel (z-score, regime-specific)
    duration_model.py     # DurationTailModel, DurationShrinkModel, DurationOverlayModel
    asymmetric_model.py   # AsymPerBucketModel
    signal_enhanced_model.py  # SignalEnhancedTail v3 + v4 (top performers)
    registry.py           # ModelRegistry + StrategyRegistry
  backtesting/
    engine.py             # BacktestEngine (cross-tier validation)
    schemas.py            # Trade, Settlement, BacktestResult
  paper_trading/
    schemas.py            # MarketOdds -> Signal -> Betslip -> Fill -> Settlement
    tracker.py            # PerformanceTracker (parquet persistence)
    validators.py         # Signal validation + risk checks
  data_sources/
    polymarket/           # client.py, orders.py, auth.py
models/
  model_registry.json     # 14+ registered models (identity only, no metrics)
  {model_id}/backtests/   # Per-model backtest results by tier
strategies/
  strategy_registry.json  # 7 strategies (model + filters + sizing -> deployment)
```

---

## Relationship to EsportsBetting Project

This project shares **betting infrastructure** (paper trading, Polymarket clients)
copied from `EsportsBetting/BettingMarkets`. No linkages -- independent copies.

**Shared**: schemas.py, tracker.py, validators.py, polymarket client/order/auth
**Different**: Data sources, features, models, strategies

---

*Last updated: 2026-02-15*
