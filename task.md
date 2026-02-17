# Elon Musk Tweet Markets - Task Tracker

## Current Phase: Data Processing & Backtest Pipeline

---

### Phase 1: Data Collection ✅ (Mostly Complete)

- [x] **Task 1: Fetch XTracker historical data**
  - 103 daily records (Oct 30, 2025 - Feb 9, 2026) in `data/sources/xtracker/`
  - 48 tracking periods with cumulative counts
  - Note: 30/103 records show count=0 (noon-to-noon boundary artifact, not real zeros)

- [x] **Task 2: Download Kaggle tweet dataset** ✅ (Completed in Task 15)
  - 3 CSVs merged → `data/processed/daily_counts.json` (3053 days, 2010-2026)
  - Gap: 2025-04-13 to 2025-10-30 (~200 days, Kaggle ends before XTracker starts)

- [x] **Task 3: Fetch Polymarket Elon tweet markets**
  - `tweet_events_comprehensive.json` (10.8MB, 125 events) - PRIMARY source
  - `elon_tweet_clob_markets.json` (5.5MB, 2008 markets) - token IDs + winners
  - `elon_tweet_markets_full.json` (3.4MB, 40 events + 382 markets) - early markets
  - Combined: ~133 unique events spanning Apr 2024 - Feb 2026

- [x] **Task 4: Fetch GDELT news data**
  - 4 entities (Elon Musk, Tesla, SpaceX, Neuralink) × 3 modes
  - Coverage: Jan 2024 - Feb 2026 (~750 data points per entity)
  - xAI and DOGE queries return empty (GDELT limitation)
  - Script: `scripts/fetch_gdelt_news.py`

- [x] **Task 5: Fetch SpaceX launch calendar**
  - 323 historical launches (2024-2026) + upcoming schedule
  - Script: `scripts/fetch_spacex_launches.py`

- [x] **Task 6: CLOB API credentials configured**
  - `POLYMARKET_PRIVATE_KEY` in `.env`
  - Auth client, order client, read-only client all in `src/data_sources/polymarket/`

---

### Phase 2: Parse & Unify Market Data ✅

- [x] **Task 7: Build market catalog**
  - Script: `scripts/build_market_catalog.py`
  - Output: `data/processed/market_catalog.parquet` (2,929 rows, 169 events)
  - All events are bucketed categorical (7-51 buckets each, no binary/outright)
  - 4 market types: weekly (120), daily (33), monthly (6), short (6)

- [x] **Task 8: Build XTracker-to-market mapping**
  - Script: `scripts/build_xtracker_mapping.py`
  - Output: `data/processed/xtracker_mapping.parquet` (159 events)
  - Ground truth tiers: 38 gold, 9 silver, 112 bronze
  - 94.7% validation rate for gold tier

---

### Phase 3: Fetch CLOB Historical Prices ✅

- [x] **Task 9: Fetch CLOB price histories for resolved markets**
  - Script: `scripts/fetch_clob_prices.py` (backfill + daily modes)
  - Consolidation: `scripts/consolidate_prices.py`
  - Output: `data/sources/polymarket/prices/price_history.parquet` (549,246 rows, 157 events, 2.5MB)
  - Per-event backfill artifacts: 543 files in `data/sources/polymarket/prices/events/`

---

### Phase 4: Build Unified Backtest Dataset ✅

- [x] **Task 10: Build backtest dataset**
  - Script: `scripts/build_backtest_dataset.py` (1013 lines, 23 functions)
  - Per event: `data/backtest/events/{event_slug}/metadata.json`, `prices.parquet`, `features.json`
  - 38 features across 4 categories: temporal (11), GDELT (18), SpaceX (4), market (5)
  - Coverage: 159 events, 157 with prices, 42 with temporal features, 37 gold fully-featured
  - Master index: `data/backtest/backtest_index.json`

---

### Phase 5: Backtest Engine & Baseline Model ✅

- [x] **Task 11: Build backtest engine**
  - Self-contained in `scripts/run_backtest.py` (engine logic inlined, ~570 lines)
  - Configurable entry time, Kelly sizing, min edge threshold
  - Output: per-event trades, aggregate P&L, ROI, Brier score, accuracy
  - Results saved to `data/results/backtest_{model}_{timestamp}.json`

- [x] **Task 12: Implement baseline models**
  - Module: `src/ml/baseline_model.py` (314 lines)
  - NaiveBucketModel: Negative Binomial from trailing XTracker data → bucket probs
  - CrowdModel: uses market prices as predictions (sanity-check baseline)
  - **Results on gold tier (38 events):**
    - Naive NegBin: Brier 1.05, log loss 4.62, accuracy 3/38, ROI -92.6%
    - Crowd: Brier 0.81, log loss 1.80, accuracy 5/38, ROI 0% (no bets, as expected)

---

### Phase 6: Per-Bucket Model Optimization ✅

- [x] **Task 13: Per-bucket zone analysis (35 variants)**
  - 7 approaches × 5 zones (low-tail, below-center, center, above-center, high-tail)
  - Scripts: `scripts/zone_1_analysis.py` through `zone_5_analysis.py`
  - Finding: market extremely well-calibrated, Zone 4 momentum only meaningful signal
  - Dataset: `data/analysis/bucket_dataset.json` (2475 bucket-event pairs)

- [x] **Task 14: PerBucketModel (hybrid z-score momentum)**
  - Module: `src/ml/per_bucket_model.py`
  - Hybrid: MarketAdjusted base + z-score-dependent momentum correction (alpha=0.24)
  - **Results on gold tier: Brier 0.7993 (BEST), ROI +67.6% at 2% edge**
  - Optimizer scripts: `scripts/optimize_hybrid.py`, `scripts/optimize_per_bucket.py`

---

### Phase 7: Model Search & Optimization ✅

- [x] **Task 15: Download Kaggle tweets**
  - 3 CSVs merged → `data/processed/daily_counts.json` (3053 days, 2010-2026)
  - Gap: 2025-04-13 to 2025-10-30 (~200 days, Kaggle ends before XTracker starts)
  - Script: `scripts/process_kaggle_tweets.py`

- [x] **Task 16: Creative model search (5+ positive ROI models)**
  - 10 model architectures tested via parallel subagent grid searches
  - **7 positive ROI models found** at 2% min_edge on gold tier (38 events):
    1. TailBoost: ROI +44.4%, 60 bets, Brier 0.8022 ← **MOST TRUSTED (highest volume)**
    2. DurationTail: ROI +67.7%, 43 bets, Brier 0.8016 ← Most robust (highest P&L)
    3. PerBucket: ROI +67.6%, 12 bets, Brier 0.7993 ← Proven baseline
    4. AsymmetricPerBucket: ROI +152.6%, 9 bets, Brier 0.7992 ← Highest ROI
    5. DurationShrink: ROI +49.7%, 23 bets, Brier 0.8095 ← Simplest
    6. DurationOverlay: ROI +8.3%, 46 bets, Brier 0.7945 ← Best calibration
    7. PowerLawTail: ROI +299.1%, 4 bets, Brier 0.8033 ← Too few bets
  - **Failed approaches**: CalendarMedia (-10.1%), Volatility (-1.5%), Contrarian (+30.1% but 3 bets)
  - All models registered in `models/model_registry.json` (v3.0, 14 models)
  - 5 strategies in `strategies/strategy_registry.json` (v2.0)

- [x] **Task 16b: Cross-tier validation (159 events, 21 months)**
  - Ran all 7 models on gold (38) + silver (9) + bronze (112) = 159 events
  - **TailBoost is the ONLY model profitable across all tiers**: 187 bets, +13.9% ROI, +$303
  - DurationShrink also generalizes: 47 bets, +21.5% ROI combined
  - DurationTail generalizes with weaker edge: 133 bets, +13.8% ROI combined
  - **Z-score models (PerBucket, AsymPerBucket) FAIL on bronze**: -92% and -91% ROI
  - Key insight: Tail redistribution = structural edge. Z-score momentum = regime-specific
  - Checkpoint: `data/results/checkpoint_2026-02-11_cross_tier_validation.json`

---

### Phase 8: Paper Trading & Live Deployment ✅ (Mostly Complete)

- [x] **Task 17: Adapt paper trading for categorical markets**
  - `scripts/fetch_current_odds.py` (197 lines) — captures live bucket prices as MarketOdds snapshots
  - `scripts/generate_signals.py` (345 lines) — runs models via strategy registry, applies filters, Kelly sizing
  - `scripts/settle_bets.py` (228 lines) — resolves completed events using XTracker as oracle
  - `src/ml/registry.py` updated with `instantiate_model()` factory for dynamic model loading
  - `src/paper_trading/tracker.py` updated with configurable odds_dir/signals_dir storage paths

- [x] **Task 18: Build cron pipeline** (`scripts/cron_pipeline.py`)
  - 193 lines, orchestrates: XTracker refresh → fetch odds → generate signals → settle bets → summary
  - Flags: `--dry-run`, `--skip-xtracker`
  - Designed for Task Scheduler / cron (every 6 hours)

- [x] **Task 19: Daily data refresh**
  - XTracker refresh integrated into cron pipeline (step 1)
  - CLOB price refresh via existing `scripts/fetch_clob_prices.py --daily`

- [x] **Task 20: End-to-end validation**
  - Full pipeline passes: 4/4 steps OK in 27.7s (dry-run mode)
  - Fixed: `PerBucketModel` and `MarketAdjustedModel` `__init__` signatures now accept registry hyperparameters
  - 404s on delisted tokens: expected behavior (Polymarket delists zero-probability buckets mid-event), handled gracefully
  - Feature builder performance: fast (~0.3s per event), no bottleneck
  - 3 active events, 5 strategies, 15 signals generated, 3 actionable (TailBoost + DurationTail)
  - 66 total signals accumulated across test runs

- [ ] **Task 21: Live signal dashboard** — compare model signals across strategies

---

### Phase 9: Signal-Enhanced Models ✅

- [x] **Task 22: Download financial market data**
  - Script: `scripts/fetch_market_data.py` (Tesla OHLCV + DOGE-USD + BTC-USD via yfinance)
  - Output: `data/sources/market/tesla_daily.parquet` (532 rows), `crypto_daily.parquet` (775 rows)
  - Coverage: Jan 2024 - Feb 2026

- [x] **Task 23: Download Wikipedia pageview data**
  - Script: `scripts/fetch_wikipedia_pageviews.py` (Wikimedia REST API)
  - Articles: Elon_Musk, Tesla_Inc, SpaceX, Dogecoin, Department_of_Government_Efficiency
  - Output: `data/sources/wikipedia/pageviews.json` (776 days each)
  - Elon Musk avg: 61.5K pageviews/day

- [x] **Task 24: Integrate new features into pipeline**
  - 12 financial features (Tesla: 6, Crypto: 6) in `src/features/extractors.py`
  - 8 attention features (Wikipedia pageviews) in `src/features/extractors.py`
  - Feature builder updated: 72 total features across 7 categories
  - Coverage: 157/159 events (98.7%) for financial + attention

- [x] **Task 25: Build SignalEnhancedTailModel**
  - Module: `src/ml/signal_enhanced_model.py`
  - 3 variants (v1, v2, v3) with different hyperparameters
  - Dynamic tail boost modulated by 4 signals: TSLA vol, DOGE momentum, Wiki attention spike, TSLA drawdown
  - v3 new best at 2% min_edge: +$395 P&L across all 159 events (188 bets, +17.7% ROI)
  - v1 best at 5% min_edge: +$381 P&L (22 bets, +84.4% ROI)

- [x] **Task 26: Register models and strategies**
  - 3 model entries in model_registry.json (v1, v2, v3)
  - 2 strategy entries: signal_enhanced_volume (v3, 2% edge), signal_enhanced_selective (v1, 5% edge)
  - Fixed run_backtest.py to use ModelRegistry.instantiate_model() for correct hyperparameter injection

---

## MVP Acceptance Criteria ✅ ALL MET

Pipeline is "working" when:
1. ✅ Backtest runs on 16+ resolved weekly events (XTracker overlap) → **38 gold events tested**
2. ✅ Per-event output: model bucket probabilities vs market bucket prices
3. ✅ Aggregate output: total P&L, ROI, Brier score, log loss, accuracy
4. ✅ Zero temporal leakage verified (features use only pre-event data)
5. ✅ CrowdModel sanity check: 0 bets placed (agrees with market = no edge found)
6. ✅ Multiple profitable models with diverse edge sources (7 models, 4-60 bets each)

---

## Ground Truth Tiers

| Tier | Period | Source | Events | Quality |
|------|--------|--------|--------|---------|
| Gold | Oct 30, 2025+ | XTracker daily_metrics (exact daily counts) | 38 | Exact count, full temporal features |
| Silver | Sept-Oct 2025 | XTracker tracking_details (cumulative totals) | 9 | Total count per tracking period |
| Bronze | Pre-Sept 2025 | Market resolution (winner flag = winning bucket) | 112 | Count range only (bucket, not exact) |

---

## Status Log

| Date | Update |
|------|--------|
| 2026-02-09 | Project created, initial structure scaffolded |
| 2026-02-09 | Phase 1 data collection: XTracker (103 days), Polymarket (125 events), GDELT (4 entities, 25 months), SpaceX (323 launches) |
| 2026-02-09 | CLOB API credentials configured in .env |
| 2026-02-09 | Analysis complete: identified ~60 categorical events, 3-tier ground truth strategy, backtesting pipeline designed |
| 2026-02-10 | Phase 2-5 complete: market catalog (169 events), XTracker mapping (38 gold), CLOB prices (549K rows), backtest dataset (159 events), baseline models tested |
| 2026-02-10 | Baseline results: Naive NegBin ROI -92.6% (terrible), Crowd Brier 0.81 (market efficiency baseline to beat) |
| 2026-02-11 | Per-bucket zone analysis: 35 variants (7 approaches × 5 zones), market extremely well-calibrated |
| 2026-02-11 | PerBucketModel (hybrid z-score momentum): Brier 0.7993, ROI +67.6% — new best model |
| 2026-02-11 | Kaggle tweets integrated: 3053 days of daily counts merged with XTracker |
| 2026-02-11 | Model search complete: 10 architectures tested, 7 positive ROI models found |
| 2026-02-11 | TailBoost designated most trusted: 60 bets at +44.4% ROI (strongest statistical confidence) |
| 2026-02-11 | Registry v3.0: 14 models, 5 strategies, 3 deprecated approaches |
| 2026-02-11 | **Cross-tier validation**: TailBoost only model profitable across all 159 events (21 months). Z-score models fail on bronze. |
| 2026-02-11 | Model confidence tiers established: Structural (TailBoost, DurationShrink) > Robust Recent (DurationTail) > Regime-Specific (PerBucket, Asym) |
| 2026-02-11 | Paper trading pipeline built: fetch_current_odds, generate_signals, settle_bets, cron_pipeline (2,363 lines) |
| 2026-02-12 | Task list updated to reflect Phase 8 completion. End-to-end pipeline validated: 4/4 steps pass. |
| 2026-02-12 | Fixed PerBucketModel + MarketAdjustedModel init signatures for registry hyperparameter injection. |
| 2026-02-15 | Phase 9: Signal-enhanced models. 3 new data sources (Tesla/TSLA, Crypto DOGE+BTC, Wikipedia pageviews) integrated. 72 features across 7 categories. SignalEnhancedTail v3 new best model (+$395 P&L across all tiers). 2 new strategies registered. |
| 2026-02-15 | Phase 10 planned: 3 new approaches (intra-market arb, price dynamics, cross-market consistency), 6 new data sources, 20+ new features. Goal: high volume + positive expectancy. |
| 2026-02-15 | **Phase 10 complete**: 3 approaches validated (1 FAIL, 2 PASS). 5 new data fetchers built. ~98 features across 10 categories. 4 new models: PriceDynamics (+33.1% ROI), CrossMarketArb (+55.4% gold), SignalEnhanced v5 (+7.5%), **ConsensusEnsemble (+37.0% ROI, $594 P&L, 132 bets = NEW BEST)**. Registry: 23 models, 11 strategies. |
| 2026-02-16 | **Phase 11 complete**: Real ML via XGBoost. Walk-forward CV infra, bucket-level dataset builder, 2 framings (classification + residual). **XGBoostResidualModel: +28.4% ROI on walk-forward CV (161 bets, $826 P&L)**. Top features: crowd_price (12.2%), heuristic_prob (6.3%), elon_musk_vol_3d (3.4%). Registry: 26 models, 16 strategies. |
| 2026-02-17 | Activated 10 paper strategies for live pipeline (deactivated 5 negative-ROI strategies). Pipeline run: 17/17 steps OK, 50 signals, 3 new paper bets. Cumulative paper P&L: +$191.63 (+68.5% ROI on 26 bets). |

---

### Phase 10: Volume Scaling — New Edges, Data Sources & Features ✅

**Goal**: Break the volume-ROI tradeoff. Current best: 188 bets at +17.7% ROI. Target: 250+ bets at 25%+ ROI by stacking independent edges.
**Result**: ConsensusEnsemble achieves 132 bets at +37.0% ROI ($594 P&L). PriceDynamics achieves 116 bets at +33.1% ROI ($564 P&L). Both exceed ROI target.

---

#### 10A: New Approaches (Fail Fast Validation) ✅

##### Approach 1: Intra-Market Arbitrage (Overround Mispricing) — FAIL ❌

- **Verdict**: FAIL — mathematically invalid. `mispricing[i] = price[i] * overround/(1+overround)` is a constant fraction for all buckets. Overround distributes uniformly; no exploitable differential.
- Mean overround: 2.51% (median 0.97%). Winners are slightly overpriced (+0.006), not underpriced.
- Script: `scripts/analyze_overround.py`
- [x] Analysis complete — no model built (signal doesn't exist)

##### Approach 2: Price Dynamics Trading (Intra-Market Momentum) — PASS ✅

- **Verdict**: PASS — strong momentum-following signal. Winners have +0.07 avg momentum vs losers -0.009. Cohen's d = 0.89 (large effect). Rising buckets win 26.4% vs falling 7.4%.
- **Key insight**: Signal is momentum-FOLLOWING (not contrarian). Crowd adjustments are correct in direction but lag the true probability.
- Model: `src/ml/price_dynamics_model.py` (PriceDynamicsModel)
- Features: `price_dynamics` category saved per-event in features.json
- **Results**: Gold +98.4% (41 bets), Silver +221.6% (5), Bronze -9.7% (70). **ALL +33.1% ROI, $564 P&L, 116 bets.**
- [x] CLOB data has full multi-snapshot coverage (409 events with ≥48h span)
- [x] Momentum features computed and saved for 137 events
- [x] PriceDynamicsModel built and backtested
- [x] Registered in model_registry.json

##### Approach 3: Cross-Market Consistency Arbitrage (Daily vs Weekly) — PASS (Recent Only) ✅

- **Verdict**: PASS for gold/silver (recent period with daily markets), FAIL on bronze.
- 73 overlapping daily-weekly pairs found. 39.5% of buckets have >5% divergence.
- Cheap side wins at 10.8% vs avg price 5.7% (87.9% edge above price).
- Model: `src/ml/cross_market_model.py` (CrossMarketArbModel)
- **Results**: Gold +55.4% (32 bets), Silver +384.4% (4), Bronze -60.0% (60). Falls back to tail boost on bronze.
- [x] Analysis: `scripts/analyze_cross_market.py`
- [x] CrossMarketArbModel built and backtested
- [x] Registered in model_registry.json

---

#### 10B: New Data Sources ✅

- [x] **DS1: Order Book Depth** — `scripts/fetch_orderbook.py`. Tested: 121 buckets across 4 active events. Saved to `data/sources/polymarket/orderbook/`.
- [x] **DS3: Government Calendar** — `scripts/fetch_government_calendar.py`. 142 events (10 exec orders, 112 rules, 19 notices). Federal Register + GovTrack APIs. → `data/sources/government/events.parquet`
- [x] **DS4: Reddit Activity** — `scripts/fetch_reddit_activity.py`. Arctic Shift API works. 5 subreddits (elonmusk, teslamotors, SpaceX, dogecoin, WSB). → `data/sources/reddit/daily_activity.parquet`. Historical backfill needs extended run.
- [x] **DS5: Trade History** — `scripts/fetch_trade_history.py`. 56,723 trades from 2 active events ($5.5M volume). CLOB trades API works. → `data/sources/polymarket/trades/`
- [x] **DS6: Corporate Events** — `scripts/fetch_corporate_events.py`. 64 events (Tesla 48, xAI 7, SpaceX 5, Neuralink 4). yfinance earnings + SEC EDGAR + manual curation. → `data/sources/calendar/corporate_events.parquet`
- [ ] **DS2: Social Blade** — Deferred (scraping risk, lower priority)

---

#### 10C: New Features ✅

~98 scalar features across 10 categories (was 91 across 8). Added government and corporate as new top-level categories.

##### From Existing Data — DONE ✅

- [x] `market_overround` — sum(prices) - 1.0, measures vig
- [x] `bucket_relative_mispricing` — per-bucket deviation from fair-share
- [x] `bucket_position_normalized` — bucket index / total (0=lowest, 1=highest)
- [x] `price_momentum_24h/48h` — bucket price changes (saved in price_dynamics features)
- [x] `price_acceleration` — momentum_24h - momentum_48h
- [x] `bucket_price_volatility_24h` — std of bucket price over 24h
- [x] `cross_market_daily_weekly_div` — daily vs weekly divergence (cross category)
- [x] `gdelt_entity_divergence` — Elon vs Tesla news volume divergence (media category)
- [x] `wikipedia_entity_divergence` — Elon vs Tesla pageview ratio (attention category)
- [x] `day_of_week_sin/cos` — cyclical day-of-week encoding (temporal category)
- [x] `hours_until_resolution` — already existed in market features

##### From New Data Sources — DONE ✅

- [x] `govt_event_flag_7d` — binary: any govt event in next 7 days (government category)
- [x] `govt_event_count_trailing_7d` — count of govt events in past 7 days
- [x] `govt_exec_order_flag_7d` — binary: executive order in next 7 days
- [x] `corporate_event_flag_7d` — binary: any corporate event in next 7 days (corporate category)
- [x] `corporate_event_count_7d` — count of corporate events in window
- [x] `tesla_earnings_flag_14d` — binary: Tesla earnings within 14 days
- [ ] Order book features (bid_ask_spread, book_imbalance, total_book_depth) — deferred, no historical data for backtest
- [ ] Reddit features (reddit_volume_24h, reddit_volume_delta) — deferred, insufficient backfill
- [ ] Trade history features (whale_trade_count, whale_net_direction, etc.) — deferred, no historical data for backtest

##### Updated Category Counts

- temporal: 14 (+day_of_week_sin, day_of_week_cos)
- media: 22 (+gdelt_entity_divergence)
- calendar: 6 (unchanged)
- **government: 3 (NEW category)**
- **corporate: 3 (NEW category)**
- market: 10 (+market_overround, bucket_relative_mispricings)
- cross: 6 (+cross_market_daily_weekly_div)
- financial: 21 (unchanged)
- attention: 9 (+wikipedia_entity_divergence)
- trends: 10 (unchanged)
- **price_dynamics: per-bucket features (momentum, acceleration, volatility) — stored separately**

---

#### 10D: Execution Order ✅

**Wave 1 — Validate new approaches** ✅
- [x] Approach 1: Intra-Market Arb — FAIL (overround is uniform, no signal)
- [x] Approach 2: Price Dynamics — PASS (+33.1% ROI, 116 bets)
- [x] Approach 3: Cross-Market Consistency — PASS (gold/silver only, +55.4% gold)

**Wave 2 — New data sources** ✅ (5/6, Social Blade deferred)
- [x] DS1: Order book depth (fetcher built, 121 buckets tested)
- [x] DS4: Reddit activity (fetcher built, Arctic Shift API)
- [x] DS5: Trade history (fetcher built, 56K trades)
- [x] DS3: Government calendar (142 events fetched)
- [x] DS6: Corporate events (64 events fetched)
- [ ] DS2: Social Blade — deferred (scraping risk)

**Wave 3 — Feature engineering** ✅
- [x] Built 12 features from existing data (extractors.py)
- [x] Built 6 features from new data (govt + corporate)
- [x] Integrated into feature_builder.py (government + corporate categories)
- [x] Rebuilt backtest dataset (159 events, ~98 scalar features)

**Wave 4 — Model integration** ✅
- [x] ConsensusEnsembleModel: TailBoost(0.30) + PriceDynamics(0.40) + SignalEnhanced v3(0.30) — **NEW BEST: +37.0% ROI, $594 P&L, 132 bets**
- [x] SignalEnhanced v5: 10 signals (v4 + govt, corporate, momentum) — +7.5% ROI, 119 bets
- [x] Cross-tier backtested all 4 new models
- [x] Registered: 23 models (v5.0), 11 strategies (v4.0)

---

### Phase 11: Real ML Models (XGBoost Bucket-Level) ✅

- [x] **Task 27: Walk-forward cross-validation infrastructure**
  - Module: `src/ml/cross_validation.py` (WalkForwardCV class)
  - Expanding-window folds, temporal integrity (never trains on future events)
  - Min 20 training events per fold, configurable n_folds
  - Integrated into `run_backtest.py` via `--cv` and `--cv-folds` flags

- [x] **Task 28: Bucket-level dataset builder**
  - Module: `src/ml/dataset_builder.py`
  - Converts 159 events → ~2,725 (event, bucket) training rows
  - 125 features: 113 event-level (flattened from TweetFeatureBuilder) + 12 per-bucket structural
  - Per-bucket features: position_normalized, midpoint, width, crowd_price, is_tail, distance_from_ev, heuristic_prob, heuristic_edge
  - Handles open-ended buckets, object→numeric coercion, null handling

- [x] **Task 29: XGBoost models (two framings)**
  - Module: `src/ml/gradient_boost_model.py`
  - **Framing A (XGBoostBucketModel)**: Raw P(bucket wins) classification. -20.4% ROI on CV — overfits.
  - **Framing B (XGBoostResidualModel)**: Predicts residual = actual - crowd_price. **+28.4% ROI on walk-forward CV, 161 bets, $826 P&L. Beats all heuristics.**
  - Aggressive regularization: max_depth=3-4, reg_lambda=5-10, min_child_weight=10-15
  - Both save/load via joblib. Plugs into BacktestEngine with zero changes.

- [x] **Task 30: Training script + comparison**
  - Script: `scripts/train_xgb_model.py`
  - Walk-forward CV with per-fold metrics
  - `--compare` flag runs heuristic baselines for head-to-head comparison
  - `--retrain` retrains on all data for production deployment
  - Feature importance analysis, calibration analysis

- [x] **Task 31: Integration & registration**
  - XGBoost models added to `MODEL_MAP` in run_backtest.py
  - 2 models registered in model_registry.json (26 models total)
  - 2 strategies registered in strategy_registry.json (16 strategies total, 10 paper, 6 inactive)
  - `--cv` flag added to run_backtest.py for walk-forward CV mode

**Walk-Forward CV Results (5 folds, 139 test events, 2% min_edge):**

| Model | Bets | P&L | ROI | Brier |
|-------|------|-----|-----|-------|
| **XGBoost Residual** | 161 | **+$826** | **+28.4%** | 0.5965 |
| ConsensusEnsemble | 159 | +$377 | +21.7% | 0.5945 |
| SignalEnhanced v3 | 105 | +$22 | +1.9% | 0.6031 |
| XGBoost Bucket | 267 | -$639 | -20.4% | 0.6360 |
| TailBoost | 67 | -$233 | -36.1% | 0.5971 |

**Key findings:**
- Residual framing >> raw classification. Crowd is strong baseline; learning corrections is easier.
- Top features: crowd_price (12.2%), heuristic_prob (6.3%), elon_musk_vol_3d (3.4%), heuristic_edge (3.1%)
- XGBoost Bucket model is well-calibrated (predicted P(win) matches actual win rates) but bets too aggressively
- XGBoost Residual uses all 113 features — first model to actually leverage the full feature set

---

### Phase 12: Strategy Bands & New Institutional Data Sources

- [ ] **Task 32: Build strategy bands (confidence-based sizing)**
  - Design and implement a multi-tier betting strategy that varies position size based on model agreement / confidence level
  - "Bands": e.g., Band 1 (high confidence, 3+ models agree, full Kelly), Band 2 (moderate, 2 models, half Kelly), Band 3 (low, 1 model, quarter Kelly)
  - Compare different band definitions (by model count, edge magnitude, signal overlap) on walk-forward CV
  - Integration with existing strategy registry — add band-based strategies
  - Should work with both heuristic and ML models
  - Goal: maximize Sharpe ratio / risk-adjusted returns, not just raw ROI

- [ ] **Task 33: Fetch SEC 13F institutional holdings data (TSLA)**
  - Build fetcher: `scripts/fetch_13f_holdings.py`
  - Source: SEC EDGAR API (free, no auth) — 13F-HR filings for TSLA
  - Extract: top 50 institutional holders, quarterly position changes (bought/sold/new/exited)
  - Compute features: net_institutional_flow, pct_held_by_institutions, top10_net_change, n_new_positions, n_exited_positions
  - Output: `data/sources/sec/tsla_13f_holdings.parquet`
  - Integrate into feature_builder.py as "institutional" category
  - Key signal: large institutional sells may correlate with Elon sentiment / tweet activity
  - Note: 13F filings are quarterly (~45 day lag), so features will be point-in-time from last filing date

- [ ] **Task 34: Fetch TSLA insider transactions**
  - Build fetcher: `scripts/fetch_insider_transactions.py`
  - Source: SEC EDGAR XBRL API (Form 4 filings) or yfinance insider data
  - Extract: Elon Musk + other TSLA officers/directors transactions (buys, sells, option exercises)
  - Compute features: insider_net_shares_30d, insider_sell_count_30d, elon_sold_flag_30d, insider_buy_sell_ratio
  - Output: `data/sources/sec/tsla_insider_transactions.parquet`
  - Integrate into feature_builder.py as "insider" or "institutional" category
  - Key signal: Elon selling TSLA shares often correlates with high tweet activity (managing narrative)
  - Note: Form 4 filings typically within 2 business days of transaction — much more timely than 13F
