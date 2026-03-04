# ElonTweetMarkets: Categorical Prediction Market Alpha via Agentic Quant Research

**Status**: Paper trading live (10 strategies, 6-hour cron cycle)
**Timeline**: Feb 9-17, 2026 (8 days from idea to live paper trading)
**Built with**: Claude Code (agentic AI development — iterative research, model search, pipeline automation)

---

## 1. What Is This?

An end-to-end quantitative prediction system for **Polymarket's Elon Musk tweet count markets** — categorical markets with 10-30 outcome buckets (e.g., "200-219 tweets", "220-239 tweets") that resolve via XTracker oracle.

**The edge**: The crowd structurally underprices tail outcomes. They anchor on recent history, concentrate probability mass too tightly around the mode, and underestimate variance. We exploit this with distributional models that redistribute probability from center to tails.

**Scale**: $5-20M weekly volume across ~24 simultaneous markets. 50-100x more liquid than niche prediction categories.

**Result**: 27 registered models, 18 strategies (10 paper), 88 documented factors across 11 data sources, all orchestrated by an automated pipeline.

---

## 2. Why This Project Is Interesting

### System at a Glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (11 sources)                        │
│                                                                       │
│  XTracker ──┐   Tesla/Crypto ──┐   GDELT ──┐   Wikipedia ──┐         │
│  Kaggle ────┤   VIX ───────────┤   SpaceX ──┤   Reddit ────┤         │
│  Polymarket ┤   Crypto F&G ────┘   Gov/Corp ┘   Trends ────┘         │
│             │                                                         │
│             ▼                                                         │
│  data/sources/{source}/     ← versioned folders, schema.md, logs     │
│  data/datacatalog.csv       ← central registry (20 entries)          │
│  data/processed/            ← market_catalog, xtracker_mapping       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       FEATURE PIPELINE                                 │
│                                                                        │
│  extractors.py (11 categories) → feature_builder.py → 113 features    │
│  factor_registry.py (88 documented factors with formulas + rationale)  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                                    │
│                                                                        │
│  26 models (src/ml/)                                                   │
│  ├─ Heuristic: TailBoost, PriceDynamics, SignalEnhanced, Duration...   │
│  ├─ ML: XGBoostResidual (+28% CV), XGBoostBucket                      │
│  └─ Ensemble: ConsensusEnsemble (+37% backtest)                       │
│                                                                        │
│  Validation: BacktestEngine (159 events, 3 tiers, walk-forward CV)     │
│  Registry:   models/model_registry.json (identity + hyperparams)       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      STRATEGY + EXECUTION                              │
│                                                                        │
│  strategies/strategy_registry.json (10 paper, 6 inactive)              │
│  ├─ Filters: min_edge, bucket_price, market_type                       │
│  ├─ Sizing: quarter-Kelly, bankroll, max_bet_pct                       │
│  └─ Entry: hours_before_close, entry_window                            │
│                                                                        │
│  cron_pipeline.py (every 6h): fetch → build → signal → settle → report│
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   PAPER TRADING + FEEDBACK                             │
│                                                                        │
│  Signal → Betslip → Fill → Settlement (XTracker oracle)                │
│  data/paper_trading/ (parquet-backed P&L tracking)                     │
│                                                                        │
│  Feedback loop:                                                        │
│    Post-mortems (docs/POSTMORTEM_*.md) → CLAUDE.md gotchas → retrain  │
│    Checkpoints (docs/CHECKPOINT_*.md) → strategy retire/iterate       │
│    MEMORY.md (cross-session learnings) ← agent references on startup  │
└──────────────────────────────────────────────────────────────────────────┘
```

### A. Agentic, Iterative Workflow

This wasn't built linearly. It was built through **rapid hypothesis testing with AI agents** — Claude Code agents running parallel model searches, validating approaches fail-fast, and documenting everything.

**The development loop**:
```
Hypothesis → Data Collection → Feature Engineering → Model Build →
Cross-Tier Backtest → Kill or Register → Ensemble → Paper Trade
```

Each phase produced concrete deliverables documented in `task.md` (466 lines of structured progress tracking):

| Phase | What Happened | Days |
|-------|--------------|------|
| 1-5 | Data pipeline: 5 sources, 549K price rows, 159 events | 1.5 |
| 6 | Per-bucket zone analysis: 35 variants tested | 0.5 |
| 7 | Creative model search: 10 architectures, 7 profitable | 0.5 |
| 8 | Paper trading infra: odds → signals → settlement | 0.5 |
| 9 | Signal-enhanced models: +3 data sources, +20 features | 1.0 |
| 10 | Volume scaling: 3 new approaches, +5 data sources, +26 features | 1.5 |
| 11 | XGBoost ML: walk-forward CV, residual beats heuristics | 1.0 |
| 12 | Strategy bands: meta-strategy, confidence-based sizing | 1.0 |

#### Notetaking, Memory & Decision Records

The agentic workflow relies on a layered documentation system that ensures continuity across sessions and prevents repeated mistakes:

- **`MEMORY.md`** (persistent cross-session memory): Loaded into every agent conversation on startup. Contains API endpoint gotchas, data quality traps, model performance summaries, and architecture notes — all accumulated and refined across dozens of sessions. The agent updates this after each session with stable findings, and removes entries that turn out to be wrong.

- **`CLAUDE.md`** (project instructions): The canonical reference document. Contains critical gotchas (distributional prediction, XTracker zero-count days, temporal leakage), pipeline commands, architecture overview, and model performance tables. Agents are instructed to check this first and adhere to its constraints. When a checkpoint or post-mortem surfaces a new gotcha, it gets promoted into CLAUDE.md as a permanent guideline.

- **Checkpoints** (`docs/CHECKPOINT_*.md`): Triggered when P&L crosses a threshold or a pattern emerges. Structured analysis: trigger → executive summary → findings → actions taken. Example: the Feb 27 strategy bleed checkpoint diagnosed that 12/14 strategies were correlated duplicates, leading to 8 retirements and a restriction of duration models to daily/short markets only.

- **Post-mortems** (`docs/POSTMORTEM_*.md`): Written after significant losses. Root-cause analysis with numbered issues, evidence, fixes, and monitoring. Example: the Feb 23 post-mortem found 3 bugs (position limit bypass, feature loading, duplicate bets) plus a regime shift, leading to code fixes and new validation checks.

- **`task.md`** (progress tracker): Living log of every phase, model, and validation result with dates. Acts as the project's chronological record — 466 lines covering data collection through live deployment.

This system creates a feedback loop: live results → post-mortem/checkpoint → CLAUDE.md gotchas → agent behavior change → better decisions next session.

### B. Data Sourcing Was Iterative, Not Predetermined

We didn't start with 11 data sources. We started with 2 (XTracker + Kaggle), discovered what signals existed, and added sources to fill gaps:

**Wave 1** (Day 1): XTracker (oracle), Kaggle (historical tweets), Polymarket CLOB (market prices), GDELT (news), SpaceX (calendar)
**Wave 2** (Day 6): Tesla stock, Crypto (DOGE/BTC), Wikipedia pageviews — added because signal-enhanced models showed financial/attention signals modulate tail behavior
**Wave 3** (Day 6): VIX, Crypto Fear & Greed, Google Trends — added to extend signal-enhanced models from 4 to 7 features
**Wave 4** (Day 7): Government calendar, Corporate events, Reddit, Order book, Trade history — added for volume scaling and new edge discovery

Each source was **validated before integration**: if it didn't add signal, it was dropped (SEC EDGAR = low signal, ElonJet Mastodon = 24h delay too stale, Social Blade = scraping risk).

#### Data Catalog & Source Management

Every data source lives in a versioned subdirectory under `data/sources/` with its own documentation:

```
data/sources/
  xtracker/       schema.md, daily_metrics_full.json, hourly_metrics_full.json, ...
  polymarket/     schema.md, versions.md, DATA_INVENTORY.md, prices/, orderbook/, trades/
  gdelt/          schema.md, gdelt_all_combined.json, per-entity-mode JSONs
  calendar/       schema.md, spacex_launches_historical.json, corporate_events.parquet
  market/         tesla_daily.parquet, crypto_daily.parquet, vix_daily.parquet, crypto_fear_greed.parquet
  trends/         google_trends.parquet
  wikipedia/      pageviews.json
  reddit/         daily_activity.parquet, fetch_log.json
  government/     events.parquet, fetch_log.json
```

Each source subdirectory maintains **schema documentation** (field definitions, types, gotchas) and **version history** (what changed per collection run). For example, `polymarket/versions.md` tracks that v1.0 had 125 events, documents known gaps (Sept 13-27 2024), and notes parsing quirks (double-encoded JSON fields). `xtracker/schema.md` documents the noon-to-noon counting window that causes 30/103 zero-count artifacts.

The central **`data/datacatalog.csv`** (20 entries) ties everything together — each row records the source ID, storage location, fetch script, record count, date range, and downstream features produced. This makes provenance traceable end-to-end: raw API response → versioned source folder → fetch script → feature extractor → factor registry → model → strategy → paper trade.

### C. Models Flow from Creation → Registry → Strategy → Deployment

```
Model Code (src/ml/)
    ↓
Cross-Tier Backtest (159 events, 3 tiers)
    ↓
Model Registry (models/model_registry.json) — 26 entries
    ↓
Strategy Definition (strategies/strategy_registry.json) — filters, sizing, entry rules
    ↓
Signal Generation (scripts/generate_signals.py) — runs registered strategies
    ↓
Paper Trading (data/paper_trading/) — fills, settlements, P&L tracking
    ↓
Cron Pipeline (scripts/cron_pipeline.py) — 17-step automated loop every 6h
```

Every model gets a registry entry with: module path, class name, hyperparameters, feature group, backtest results, and status (active/paper/deprecated/inactive). This means **zero code changes to deploy a new model** — just register it and assign a strategy.

**Sample model registry entry** (`models/model_registry.json` — 27 models total):
```json
"consensus_ensemble_v1": {
  "model_name": "ConsensusEnsembleModel",
  "module_path": "src.ml.consensus_model",
  "class_name": "ConsensusEnsembleModel",
  "approach": "ensemble",
  "feature_group": "full",
  "required_features": ["price_dynamics", "financial", "attention"],
  "hyperparameters": {
    "weight_tail_boost": 0.3,
    "weight_price_dynamics": 0.4,
    "weight_signal_enhanced": 0.3
  },
  "data_sources": ["xtracker", "polymarket_clob", "yfinance", "wikipedia", "gdelt", "crypto_fg"],
  "status": "production",
  "backtest_all": { "roi_pct": 37.0, "n_bets": 132, "n_events": 159, "pnl": 594 },
  "cross_tier_confidence": "HIGH — best all-tier ROI (+37.0%) and most robust across tiers."
}
```

**Sample strategy registry entry** (`strategies/strategy_registry.json` — 18 strategies total):
```json
"consensus_primary": {
  "strategy_name": "Consensus Ensemble (Primary)",
  "model_id": "consensus_ensemble_v1",
  "status": "paper",
  "filters": {
    "min_edge": 0.02, "max_edge": 0.3,
    "min_bucket_price": 0.02, "max_bucket_price": 0.5,
    "market_types": ["weekly", "daily", "short"]
  },
  "sizing": {
    "kelly_fraction": 0.25, "max_bet_pct": 0.05,
    "min_bet": 5.0, "bankroll": 1000.0
  },
  "entry": { "hours_before_close": 24, "entry_window_hours": 6 },
  "backtest_summary": { "n_events": 159, "n_bets": 132, "roi_pct": 37.0, "total_pnl": 594.26 }
}
```

The strategy decouples **what** to predict (model) from **how** to trade (filters, sizing, entry timing). This lets us run the same model with different risk profiles, or swap models without touching trading logic.

**Sample factor registry entry** (`src/features/factor_registry.py` — 88 factors across 11 categories):
```python
Factor(
    name="reddit_elonmusk_posts_7d",
    category="reddit",
    description="Posts in r/elonmusk (trailing 7d avg)",
    formula="mean(r/elonmusk daily posts) for [t-7, t)",
    rationale="Direct Elon community activity level.",
    data_source="Reddit API (data/sources/reddit/daily_activity.parquet)",
    coverage_pct=95.0,
    expected_range="0-50",
    existing=True,
    priority="high",
)
```

Each factor documents its formula, data source, coverage, and rationale — serving as a single source of truth for the feature pipeline.

### D. Data Catalog — Single Source of Truth for All Sources

Every data source is tracked in `data/datacatalog.csv` (20 entries) with fetch script, record counts, date ranges, and the features it produces. Three derived datasets sit on top:

**Sample from `data/datacatalog.csv`** (20 sources total):

| source_id | type | records | date range | features | description |
|-----------|------|---------|------------|----------|-------------|
| `xtracker_daily` | api | 103 | 2025-10 → 2026-02 | 14 temporal | XTracker oracle daily counts |
| `polymarket_clob` | api | 549,246 | 2024-04 → 2026-02 | 10 market | Hourly bucket prices |
| `gdelt` | api | 948 | 2024-01 → 2026-02 | 22 media | 4 entities x 3 modes (vol/tone/timeline) |
| `tesla_stock` | api | 532 | 2024-01 → 2026-02 | 6 financial | Daily OHLCV (yfinance) |
| `google_trends` | api | 777 | 2024-01 → 2026-02 | 10 trends | 5 queries, 85-day overlapping windows |
| `reddit` | api | 3,885 | 2024-01 → 2026-02 | 7 reddit | 5 subreddits daily |
| `market_catalog` | derived | 3,029 | 2024-04 → 2026-02 | — | 173 events, parsed bucket boundaries |
| `xtracker_mapping` | derived | 159 | 2024-04 → 2026-02 | — | 3-tier ground truth mapping |

**Sample from `market_catalog.parquet`** — one resolved short event (Feb 5-7, 2026):

| bucket_label | lower | upper | is_winner | price_yes |
|-------------|-------|-------|-----------|-----------|
| <40 | 0 | 39 | | 0.00 |
| 40-64 | 40 | 64 | **WIN** | 1.00 |
| 65-89 | 65 | 89 | | 0.00 |
| 90-114 | 90 | 114 | | 0.00 |
| 115-139 | 115 | 139 | | 0.00 |
| 140-164 | 140 | 164 | | 0.00 |

**Sample from `xtracker_mapping.parquet`** — ground truth linkage:

| event_slug | type | tier | xtracker_count | winning_bucket |
|-----------|------|------|----------------|----------------|
| elon-musk-of-tweets-december-2025 | monthly | gold | 1,654 | 1400+ |
| elon-musk-of-tweets-november-2025 | monthly | gold | 923 | 920-959 |
| elon-musk-of-tweets-november-14-21 | weekly | gold | 165 | 160-179 |

The data catalog tracks **provenance end-to-end**: raw API → fetch script → storage location → feature extractor → factor registry → model → strategy → paper trade. Every number in a backtest can be traced back to its source.

### E. The Notetaking / Memory System

The project uses **persistent AI memory** (`MEMORY.md`) that accumulates across sessions:
- API endpoint gotchas (XTracker needs `/api` prefix — learned the hard way)
- Data quality issues (30/103 XTracker zero-count days = noon-to-noon artifact, NOT real zeros)
- Model performance summaries updated after each phase
- Feature analysis insights (crowd captures 48% of downswings, 105% of upswings)

Plus `task.md` (466 lines) serves as the living progress tracker — every phase, every model, every validation result is recorded with dates.

---

## 3. What Made This Project Successful

### 3a. Structural Edge Discovery, Not Curve Fitting

The core edge — **tail underpricing** — is structural and market-microstructure-driven, not statistical artifact:
- The crowd anchors on recent mode and builds narrow distributions
- In categorical markets with 10-30 buckets, this systematically underprices tails
- **Evidence**: Holds across 21 months, 159 events, 3 ground-truth quality tiers
- **Mechanism**: TailBoost (redistribute 20-40% of mass from center to tails) is the only model profitable on ALL tiers

### 3b. Cross-Tier Validation Killed Overfitting Early

Instead of one backtest set, we have **three quality tiers**:

| Tier | Events | Period | Quality | Purpose |
|------|--------|--------|---------|---------|
| Gold | 38 | Oct 2025+ | Exact daily counts | Primary validation |
| Silver | 9 | Sep-Oct 2025 | Cumulative counts | Transition period |
| Bronze | 112 | Pre-Sep 2025 | Bucket-range only | Out-of-sample stress test |

Models that looked amazing on Gold (PerBucket +67.6%, Asymmetric +152.6%) **died on Bronze** (-92%, -91%). Only structural models (TailBoost, DurationShrink) survived. This saved us from deploying regime-specific models at scale.

### 3c. Fail-Fast Approach Validation

Phase 10 tested 3 new approaches before writing any model code:

1. **Intra-Market Arbitrage (Overround)** → Mathematically invalid. Overround distributes uniformly. Killed in 2 hours.
2. **Price Dynamics (Momentum)** → Strong signal (Cohen's d = 0.89). Built model same day.
3. **Cross-Market Consistency** → Works on gold/silver, fails bronze. Built with fallback.

### 3d. Ensemble > Individual

The **ConsensusEnsemble** (TailBoost 0.30 + PriceDynamics 0.40 + SignalEnhanced 0.30) beats every individual model:

| Model | Bets | ROI | P&L |
|-------|------|-----|-----|
| ConsensusEnsemble | 132 | **+37.0%** | **$594** |
| PriceDynamics (solo) | 116 | +33.1% | $564 |
| SignalEnhanced v3 (solo) | 188 | +17.7% | $395 |
| TailBoost (solo) | 187 | +13.9% | $303 |

Complementary edges (structural tail + momentum-following + financial signals) diversify better than any single approach.

### 3e. ML Residual Framing > Raw Classification

The crowd is a strong baseline (Brier 0.81). Our XGBoost models showed:

| Framing | Bets | ROI | Why |
|---------|------|-----|-----|
| Residual (correct crowd) | 161 | **+28.4%** | Learning small corrections to a good baseline is tractable |
| Classification (from scratch) | 267 | -20.4% | Too many false edges, overfits |

Top features in XGBoost Residual: crowd_price (12.2%), heuristic_prob (6.3%), elon_musk_vol_3d (3.4%), heuristic_edge (3.1%). The model learned **when the crowd is wrong**, not how to predict from scratch.

---

## 4. Key Learnings & Gotchas

### Data Sourcing
- **XTracker zero-count days** were a trap: 30/103 records show 0 tweets due to noon-to-noon tracking windows, not actual silence. Took a full investigation to discover.
- **GDELT gaps**: June 15 - July 1, 2025 missing across all queries. Had to handle gracefully.
- **Free data is sufficient**: All 11 sources cost $0. Twitter API ($5K/mo) is NOT needed — XTracker + Kaggle covers everything.
- **Google Trends rate limiting**: Requires 85-day overlapping windows with 15s sleep. ~20 min per full fetch. Built into pipeline.

### Market Microstructure
- **All markets are bucketed categorical**: No binary/outright bets exist. Everything is distributional.
- **Crowd overestimates short events by ~27%**: Systematic bias in daily markets.
- **Momentum is FOLLOWING, not contrarian**: Rising buckets win 26.4% vs falling 7.4%. The crowd adjusts correctly but lags.
- **Overround mispricing is NOT exploitable**: mispricing[i] = price[i] × overround/(1+overround) is constant fraction across all buckets.

### Modeling
- **Z-score momentum is regime-specific**: Works brilliantly post-Oct 2025 (+67% gold), catastrophically pre-Sep 2025 (-92% bronze). Do not deploy without regime detection.
- **Feature importance is dominated by market data**: crowd_price alone is 12.2% of XGBoost importance. Most "alternative data" features add marginal lift individually.
- **Aggressive regularization required for ML**: max_depth=3-4, reg_lambda=5-10, min_child_weight=10-15. The market signal is subtle; complex trees overfit.

### What We're Doing About IS/OS Concerns
- **Walk-forward CV** (expanding window, temporal integrity, never trains on future) is our honest evaluation method
- **Three-tier validation** catches regime-specific overfitting that single-tier backtest misses
- **Paper trading live** since Feb 17 (cumulative: +$191.63, +68.5% ROI on 26 bets — early, small sample)
- **Strategy bands** aggregate model agreement for position sizing confidence

---

## 5. System Architecture (Quick Reference)

```
src/
  features/
    feature_builder.py      # TweetFeatureBuilder — 11 extractors, lazy-loaded, PIT-correct
    extractors.py           # 113 features across 11 categories
    factor_registry.py      # 88 documented factors across 11 categories
  ml/
    base_model.py           # ABC: predict(features, buckets, context) → probabilities
    baseline_model.py       # NaiveModel, CrowdModel (baselines)
    advanced_models.py      # TailBoostModel — the structural workhorse
    signal_enhanced_model.py  # v1-v6 (4 to 12 signal variants)
    price_dynamics_model.py # Momentum-following tail boost
    consensus_model.py      # Weighted ensemble of top 3 models
    gradient_boost_model.py # XGBoostBucketModel, XGBoostResidualModel
    cross_validation.py     # WalkForwardCV (expanding window)
    dataset_builder.py      # 159 events → 2,725 bucket-level rows
    registry.py             # ModelRegistry + StrategyRegistry (factory pattern)
  backtesting/
    engine.py               # BacktestEngine — cross-tier, Kelly sizing
  paper_trading/
    tracker.py              # Parquet-backed P&L tracking
    validators.py           # Signal validation + risk checks

models/model_registry.json       # 27 models (identity, hyperparams, data_sources, backtest results)
strategies/strategy_registry.json # 18 strategies (10 paper, 8 inactive)

scripts/
  cron_pipeline.py          # 17-step automated pipeline (every 6h)
  train_xgb_model.py        # ML training + walk-forward CV + comparison
  run_backtest.py           # Backtest any model/strategy on any tier
  test_new_features.py      # Cross-tier model comparison grid
  fetch_*.py                # 11 data source fetchers
  build_*.py                # Dataset builders (catalog, mapping, backtest)
```

---

## 6. Live Pipeline Flow

```
Every 6 hours (cron_pipeline.py):

Step 1: Refresh 11 data sources
  → XTracker, Tesla, Crypto, Wikipedia, VIX, Crypto F&G,
    Google Trends, GDELT, SpaceX, Government, Corporate, Reddit

Step 2: Rebuild market catalog (fresh token IDs from Gamma API)

Step 3: Fetch live bucket prices from CLOB + order book snapshots

Step 4: Generate signals
  → For each active strategy:
      Load model from registry → Build features → Predict distribution
      → Compare to market prices → Kelly sizing → Filter by edge/liquidity
      → Emit signal (or skip)

Step 5: Settle completed events via XTracker oracle

Step 6: Print performance summary
```

---

## 7. Numbers at a Glance

| Metric | Value |
|--------|-------|
| Models registered | 27 |
| Active paper strategies | 10 |
| Features | 88 registered factors, 113 scalar + per-bucket dynamics |
| Data sources | 11 (all free) |
| Backtest events | 159 (21 months) |
| Ground truth tiers | 3 (Gold 38, Silver 9, Bronze 112) |
| Best single-pass ROI | +37.0% (ConsensusEnsemble, 132 bets) |
| Best walk-forward CV ROI | +28.4% (XGBoost Residual, 161 bets) |
| Paper trading P&L | +$191.63 (+68.5% ROI on 26 bets, early) |
| Pipeline cadence | Every 6 hours |
| Lines of Python | ~15,000+ |
| Development time | 8 days (Feb 9-17) |

---

## 8. Getting Started

### Prerequisites

```bash
pip install -r requirements.txt   # pandas, pyarrow, xgboost, scikit-learn, requests, pytrends, yfinance, praw
```

### Quick Start

```bash
# 1. Fetch data (all free, no API keys required for core sources)
python scripts/fetch_xtracker_history.py
python scripts/fetch_elon_markets.py
python scripts/fetch_clob_prices.py --daily
python scripts/fetch_market_data.py

# 2. Build datasets
python scripts/build_market_catalog.py
python scripts/build_xtracker_mapping.py
python scripts/build_backtest_dataset.py

# 3. Run a backtest
python scripts/run_backtest.py --model tail_boost --all        # Heuristic model
python scripts/run_backtest.py --strategy consensus_primary    # Full strategy

# 4. Train & evaluate ML models
python scripts/train_xgb_model.py              # Walk-forward CV
python scripts/train_xgb_model.py --compare    # Compare vs heuristics

# 5. Run the live pipeline (paper trading)
python scripts/cron_pipeline.py                # Full 17-step cycle
```

### For AI Agents

This project is designed for **agentic development with Claude Code**. If you're an agent working on this codebase:

1. **Start with `CLAUDE.md`** — contains critical gotchas, architecture, and pipeline commands
2. **Check `task.md`** — living progress tracker with active tasks and blockers
3. **Read `MEMORY.md`** — cross-session learnings (API quirks, data traps, model insights)
4. **Document your work**:
   - Significant findings → update `MEMORY.md` with stable patterns
   - New gotchas discovered → propose addition to `CLAUDE.md`
   - Performance regressions → write a checkpoint (`docs/CHECKPOINT_YYYY-MM-DD_topic.md`)
   - Bugs or losses → write a post-mortem (`docs/POSTMORTEM_YYYY-MM-DD.md`)
5. **Data sources** live in `data/sources/{source}/` with `schema.md` files — read these before modifying fetch scripts
6. **Never skip cross-tier validation** — a model that wins on Gold but dies on Bronze is overfitting

---

*Generated 2026-02-20. Updated 2026-03-03. For technical details see `task.md`, `docs/RESEARCH.md`, `CLAUDE.md`.*
