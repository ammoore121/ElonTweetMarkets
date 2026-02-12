# Elon Musk Tweet Count Prediction Markets

**Goal**: ML models to predict Elon Musk's daily/weekly tweet counts for Polymarket betting.
**Current State**: Data collection phase - building initial datasets.
**Market**: Categorical prediction (multiple buckets of tweet count ranges).

---

## For AI Agents: Key Context

1. **Check [task.md](task.md)** for active tasks and blockers
2. **This is NOT binary prediction**: Markets have 10-30 outcome buckets (tweet count ranges)
3. **XTracker is the oracle**: All markets resolve via https://xtracker.polymarket.com
4. **Data sources**: XTracker API (free), Kaggle tweets (free), GDELT (free), SpaceX launches (free)
5. **Twitter API is NOT needed**: Free tier is write-only. Use Kaggle + XTracker instead.
6. **See [docs/RESEARCH.md](docs/RESEARCH.md)** for detailed data source documentation

---

## Quick Commands

```bash
python scripts/fetch_xtracker_history.py           # Fetch tweet metrics from XTracker
python scripts/fetch_elon_markets.py                # Fetch Polymarket Elon tweet markets
python scripts/fetch_tweet_history.py               # Process Kaggle tweet dataset
python scripts/train_model.py                       # Train prediction model
python scripts/generate_signals.py                  # Generate betting signals
python scripts/cron_pipeline.py                     # Full automated pipeline
```

---

## CRITICAL GOTCHAS (DO NOT SKIP)

### 1. Distributional Prediction, Not Binary

Markets have multiple outcome buckets (e.g., 0-19, 20-39, 40-59, ..., 740+).
Must predict probability distribution across all buckets, not a single number.

**Approach options**: Poisson regression, quantile regression, ordinal regression,
histogram-based gradient boosting.

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

### 5. High Variance = The Edge

Musk's weekly tweet rate recently: 380, 280, 130. The crowd anchors on recent
averages and underestimates variance. A model that detects regime changes has edge.

---

## Data Sources (All Free)

| Source | URL | What | Cost |
|--------|-----|------|------|
| **XTracker** | xtracker.polymarket.com | Daily tweet counts (oracle) | Free |
| **Kaggle** | kaggle.com/datasets/... | Historical tweets 2010-2025 | Free |
| **GDELT** | gdeltproject.org | News events mentioning Musk | Free |
| **Launch Library 2** | ll.thespacedevs.com | SpaceX launch schedule | Free |
| **SEC EDGAR** | sec.gov | Tesla filings | Free |
| **ElonJet Mastodon** | mastodon.social/@elonjet | Flight tracking (24h delay) | Free |
| **Polymarket Data API** | data-api.polymarket.com | Leaderboard, trades, positions | Free |

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

## Pipeline Refresh Order

```bash
# 1. Fetch XTracker history (resolution data)
python scripts/fetch_xtracker_history.py

# 2. Process tweet history (feature engineering data)
python scripts/fetch_tweet_history.py

# 3. Fetch Polymarket markets
python scripts/fetch_elon_markets.py

# 4. Train model
python scripts/train_model.py

# 5. Generate signals
python scripts/generate_signals.py
```

---

## Relationship to EsportsBetting Project

This project shares **betting infrastructure** (paper trading, Polymarket clients)
copied from `EsportsBetting/BettingMarkets`. No linkages — independent copies.

**Shared**: schemas.py, tracker.py, validators.py, polymarket client/order/auth
**Different**: Data sources, features, models, strategies

---

*Last updated: 2026-02-09*
