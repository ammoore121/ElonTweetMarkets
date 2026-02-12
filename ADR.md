# Architectural Decision Records (ADR)

This document tracks architectural decisions and learned knowledge for the Elon Musk Tweet Count Prediction Markets project.

---

## ADR-001: Project Initialization and Structure

**Date**: 2026-02-09
**Status**: Accepted

### Context

Starting a prediction market trading project focused on Elon Musk tweet count markets on Polymarket. Need to establish project structure, data strategy, and modeling approach. This project borrows betting infrastructure from `EsportsBetting/BettingMarkets` but is otherwise independent.

### Decision

- Modular directory structure: `data/`, `scripts/`, `src/`, `strategies/`, `models/`, `config/`, `docs/`
- Centralized task tracking in `task.md`, architectural decisions in `ADR.md`
- Reuse paper trading and Polymarket client code from EsportsBetting project (independent copies, no linkage)
- Python-first stack with pandas, scikit-learn for modeling

### Consequences

- Clear separation between data collection, feature engineering, modeling, and trading
- Shared infrastructure avoids reinventing Polymarket integration
- Independent copies mean no cross-project dependency issues

---

## ADR-002: XTracker as Single Source of Truth

**Date**: 2026-02-09
**Status**: Accepted

### Context

Multiple data sources exist for Elon Musk tweet counts: Twitter/X API, Kaggle datasets, XTracker, and web scrapers. Need to decide which is authoritative for model training and validation.

### Decision

Use **XTracker** (https://xtracker.polymarket.com) as the single source of truth for tweet counts.

### Rationale

- XTracker is Polymarket's official resolution oracle -- markets settle against its numbers
- Eliminates data matching ambiguity (replies excluded, reposts included, deletion rules defined)
- Free, no authentication required
- Any model must be validated against XTracker counts, not raw Twitter data

### Consequences

**Positive:**
- Zero resolution ambiguity (unlike CS2 esports team name matching)
- Ground truth is identical to what markets pay out on
- Free and reliable

**Negative:**
- Historical coverage may be limited vs. Kaggle datasets
- Must cross-validate Kaggle data against XTracker for overlapping periods
- Rate limits undocumented -- must use conservative request patterns

---

## ADR-003: No Twitter API -- Kaggle + XTracker for All Data

**Date**: 2026-02-09
**Status**: Accepted

### Context

Twitter/X API pricing:
- Free tier: write-only (cannot read tweets)
- Basic: $200/month for 15,000 reads
- Pro: $5,000/month for full archive

Need tweet history for feature engineering without recurring costs.

### Decision

Do NOT use Twitter API for reads. Use Kaggle datasets for historical data and XTracker for ground truth counts.

### Rationale

- Kaggle provides 2010-2025 tweet data for free
- Daily-updated Kaggle dataset keeps data current
- XTracker provides the exact counts markets resolve against
- $0/month vs $200-5,000/month with equivalent or better coverage

### Backup Plan

If Kaggle + XTracker become insufficient:
- `twikit` or `twscrape` for scraping (violates ToS, use as last resort)

### Consequences

- $0 ongoing data cost
- Must filter Kaggle data to match XTracker counting rules (exclude replies)
- Cross-validation between sources required during feature engineering

---

## ADR-004: Distributional Prediction, Not Point Estimation

**Date**: 2026-02-09
**Status**: Accepted

### Context

Polymarket tweet markets have 10-30 outcome buckets (e.g., "0-19", "20-39", ..., "740+"). A single point prediction (e.g., "250 tweets this week") is insufficient.

### Decision

Frame as a **distributional forecast problem**. Model must output probability for each bucket, not a single number.

### Candidate Models (Priority Order)

| Model | Why | Status |
|-------|-----|--------|
| Negative binomial regression | Handles overdispersion, natural for count data | Baseline |
| Histogram-based gradient boosting | Flexible, handles mixed features | Primary candidate |
| Mixture models | Captures regime structure (high/low output) | If regime detection proves important |

### Consequences

- Must calibrate probability outputs, not just predict counts
- Evaluation metric: log-likelihood of correct bucket (not MAE/RMSE)
- Kelly criterion applied per-bucket against market prices
- More complex than binary prediction (EsportsBetting) but also more opportunity per market

---

## ADR-005: Regime Detection as Core Edge

**Date**: 2026-02-09
**Status**: Accepted

### Context

Research shows the crowd consistently misprices tweet count markets during regime changes:
- Three consecutive weeks: 380 -> 280 -> 130 weekly tweets
- Market predicted 220-239 for the 130-tweet week
- A trader named Max made $490K+ fading consensus predictions

The crowd anchors on recent averages and underestimates variance.

### Decision

Regime change detection is the primary source of edge. Features should focus on:
1. Rolling statistics (mean, std, trend over 3/7/14/30 days)
2. Calendar events (Tesla earnings, SpaceX launches, DOGE meetings)
3. News cycle intensity (GDELT article volume and tone)
4. Flight activity (travel correlates with tweeting changes)
5. Intra-window momentum (early tweet rate predicts final count)

### Consequences

- Model complexity justified by high-variance target
- Need multiple external data sources (GDELT, flight tracking, SpaceX schedule)
- Calendar features are forward-looking and known in advance -- low-hanging fruit
- Must avoid overfitting to small number of regime change events

---

## ADR-006: Temporal Discipline -- No Lookahead Bias

**Date**: 2026-02-09
**Status**: Accepted

### Context

Lesson learned from EsportsBetting project: temporal leakage in features destroys model validity. Must be rigorous about point-in-time feature computation.

### Decision

- ALL features computed using only data available BEFORE the prediction window
- No same-day features for market-open predictions
- Temporal train/test splits only (never random splits)
- Feature timestamps must be validated against market window start times

### Consequences

- Feature engineering is more complex (must track what was known when)
- Backtesting is slower but trustworthy
- Prevents the false confidence that destroyed high-edge bets in EsportsBetting (ADR-008 in that project)

---

## ADR-007: Shared Infrastructure from EsportsBetting

**Date**: 2026-02-09
**Status**: Accepted

### Context

EsportsBetting/BettingMarkets already has working Polymarket integration:
- `schemas.py`: Bet schemas and data models
- `tracker.py`: Paper trading tracker
- `validators.py`: Bet validation
- Polymarket client, order, and auth modules

### Decision

Copy (not link) shared infrastructure from EsportsBetting into this project's `src/` directory.

### Files Copied

| Source (EsportsBetting) | Destination (ElonTweetMarkets) |
|------------------------|-------------------------------|
| `src/paper_trading/schemas.py` | `src/paper_trading/schemas.py` |
| `src/paper_trading/tracker.py` | `src/paper_trading/tracker.py` |
| `src/paper_trading/validators.py` | `src/paper_trading/validators.py` |
| `src/data_sources/polymarket/` | `src/data_sources/polymarket/` |

### Consequences

**Positive:**
- No cross-project dependencies
- Can modify freely without affecting EsportsBetting
- Already tested and working

**Negative:**
- Duplicate code -- fixes in one project must be manually applied to the other
- May diverge over time as each project evolves

---

## Learned Knowledge

### 2026-02-09: Project Setup

#### Market Structure Insights
- Weekly markets: ~27-32 buckets, Mon noon ET to Mon noon ET
- 2-3 day markets: ~10 buckets
- Monthly markets: Many more buckets
- Volume: $5-20M per week across all Elon tweet markets
- Liquidity: $500K-2M per market (50-100x more than CS2 esports)

#### XTracker Counting Rules
- Counts: original posts, reposts, quote tweets
- Does NOT count: replies (unless on main feed)
- Deleted posts: counted if captured within ~5 minutes
- Resolution windows: 12:00 PM ET to 12:00 PM ET (noon-to-noon)

#### Key Statistics
- 2025 daily average: 93.8 tweets/day (median: 97.5)
- Recent weekly totals: 380, 280, 130 (extreme variance)
- Peak hours: 9am-6pm EST (~36 tweets during this window)
- Weekdays significantly exceed weekends

#### Advantages Over EsportsBetting
- 100x more liquidity
- No data matching problem (XTracker is oracle)
- Free data (no scraping needed)
- Higher frequency (weekly markets)
- Crowd is demonstrably wrong (Feb 3-10 miss: predicted 220-239, actual 130)

---

## Future Decisions Queue

- [ ] Specific model architecture selection (after baseline results)
- [ ] Kelly sizing parameters for multi-outcome markets
- [ ] Position limits per bucket
- [ ] Cron pipeline scheduling (how often to refresh data)
- [ ] Alerting thresholds for signal generation
- [ ] Live trading activation criteria (what paper trading results justify going live)
