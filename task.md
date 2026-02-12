# Elon Musk Tweet Markets - Task Tracker

## Current Phase: Data Processing & Backtest Pipeline

---

### Phase 1: Data Collection ✅ (Mostly Complete)

- [x] **Task 1: Fetch XTracker historical data**
  - 103 daily records (Oct 30, 2025 - Feb 9, 2026) in `data/sources/xtracker/`
  - 48 tracking periods with cumulative counts
  - Note: 30/103 records show count=0 (noon-to-noon boundary artifact, not real zeros)

- [ ] **Task 2: Download Kaggle tweet dataset** ⛔ BLOCKED
  - Needs manual Kaggle API key at `~/.kaggle/kaggle.json`
  - NOT blocking backtest (XTracker + market resolution data sufficient for MVP)
  - Script ready: `scripts/fetch_tweet_history.py`

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
