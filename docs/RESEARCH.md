# Elon Musk Tweet Prediction Markets -- Research Findings

**Project**: Predict Elon Musk tweet counts for Polymarket categorical markets.
**Status**: Research complete. Ready for data pipeline and modeling.
**Last updated**: 2026-02-15

---

## Table of Contents

1. [Market Overview](#1-market-overview)
2. [Edge Thesis](#2-edge-thesis)
3. [Data Sources](#3-data-sources)
4. [Competitor and Inspiration Tools](#4-competitor-and-inspiration-tools)
5. [Key Statistics](#5-key-statistics)
6. [Modeling Approach](#6-modeling-approach)
7. [Comparison to CS2 Esports Project](#7-comparison-to-cs2-esports-project)

---

## 1. Market Overview

### Market Format

Polymarket runs **categorical prediction markets** on how many posts Elon Musk will make on X (formerly Twitter) within a defined time window. Each market presents **~10-30 buckets** of tweet count ranges (e.g., "200-219", "220-239", "240-259", etc.). Traders buy shares in the bucket they believe will contain the actual count. Shares in the winning bucket pay out $1; all others pay $0.

### Volume and Liquidity

- **Weekly volume**: $5-20M per week across all active Elon tweet markets
- **Liquidity per market**: $500K-2M (AMM-provided)
- **Active markets simultaneously**: ~24 at any given time, covering weekly, 2-3 day, and monthly windows
- This is **50-100x more liquid** than CS2 esports markets, meaning larger position sizes are viable with minimal slippage

### Resolution Mechanics

- **Resolution oracle**: XTracker (https://xtracker.polymarket.com) -- Polymarket's own tracking service
- **Resolution windows**: Typically **12:00 PM ET to 12:00 PM ET** (noon-to-noon)
- **What counts as a post**:
  - Main feed posts (original tweets) -- YES
  - Quote posts (quote retweets) -- YES
  - Reposts (retweets) -- YES
  - Replies -- **NO** (do not count)
  - Deleted posts -- **YES**, if captured by XTracker within approximately 5 minutes of posting
- **Data integrity**: Because XTracker is Polymarket's own oracle, there is no ambiguity about resolution. The XTracker count is final. This eliminates the "data matching" problem that plagues other prediction market categories.

### Market Lifecycle

Markets are created in advance with fixed time windows. Typical patterns:
- **Weekly markets**: Monday noon ET to Monday noon ET (7-day window)
- **2-3 day markets**: Shorter windows, often mid-week to weekend or weekend to mid-week
- **Monthly markets**: Full calendar month windows

---

## 2. Edge Thesis

### The Crowd Has Been Badly Wrong

The market consistently misprices Musk tweet counts, particularly during regime changes:

- **Feb 3-10, 2026**: Market consensus centered on 220-239 tweets. Actual result: **130 tweets**. The modal bucket was off by nearly 100 tweets.
- **Three consecutive weeks** saw weekly totals of **380 -> 280 -> 130**, a massive swing that the market failed to anticipate at each step.

### Why the Crowd Gets It Wrong

1. **Anchoring bias**: Traders anchor heavily on the most recent week's count. If Musk tweeted 380 times last week, the market assumes something close to 380 next week. This is a poor heuristic given the high variance.

2. **Underestimating variance**: Musk's tweeting behavior has extremely high variance. The standard deviation of weekly counts is large relative to the mean. The crowd prices narrow distributions when the true distribution is wide.

3. **Regime blindness**: Musk's tweeting follows distinct regimes -- high-output weeks (often correlated with political events, product launches, or public controversies) and low-output weeks (travel, meetings, focused work periods). The crowd is slow to detect regime shifts.

### Sources of Predictive Signal

1. **Time-of-day and day-of-week patterns**: Musk tweets disproportionately during US business hours (~36 tweets between 9am-6pm EST on a typical day). Weekdays significantly outpace weekends. These patterns are consistent and under-weighted by the crowd.

2. **Calendar events**: Known upcoming events strongly predict tweeting behavior:
   - **Tesla earnings calls** (quarterly) -- Musk tends to tweet more in the days surrounding earnings
   - **SpaceX launches** -- Musk live-tweets launches, especially crewed missions or Starship tests
   - **DOGE (Department of Government Efficiency) meetings** -- government-related activity correlates with political tweeting bursts
   - **Court dates, regulatory hearings** -- sometimes reduce tweeting (lawyer-imposed silence)

3. **Flight schedule**: Musk's private jet (N628TS, Gulfstream G650) activity correlates with tweeting patterns. During long flights, tweeting decreases (Starlink connectivity is inconsistent). Travel to certain locations (e.g., Boca Chica, Fremont, Washington DC) predicts different tweeting regimes.

4. **News cycle reactivity**: Major news events mentioning Musk, Tesla, SpaceX, or xAI tend to trigger tweeting bursts. GDELT provides real-time news event data that can proxy this.

5. **Intra-window momentum**: Once the market window opens, the tweet rate in the first hours/day is highly predictive of the final count. Early counting gives a major information advantage over traders who check sporadically.

### Proven Profitability

A Dutch trader named **Max** reportedly made **$490K+** betting against consensus Musk tweet predictions on Polymarket. His general approach was fading the crowd when prices implied overconfidence in a narrow range. This validates that there is real, extractable edge in these markets.

---

## 3. Data Sources

| Source | URL / Library | What | Cost |
|--------|---------------|------|------|
| **XTracker** | xtracker.polymarket.com | Daily tweet counts (oracle) | Free |
| **Kaggle** | kaggle.com/datasets/... | Historical tweets 2010-2025 | Free |
| **Twitter/X API** | developer.x.com | Live tweets (NOT recommended) | $0-5000/mo |
| **GDELT** | gdeltproject.org | News events mentioning Musk | Free |
| **SpaceX Launch Library 2** | ll.thespacedevs.com | SpaceX launch schedule | Free |
| **SEC EDGAR** | sec.gov | Tesla filings | Free |
| **Flight Tracking** | Various (ADS-B, OpenSky, ElonJet) | Musk jet movements | Free-$10/mo |
| **Polymarket Data API** | data-api.polymarket.com | Market prices, trades, positions | Free |
| **Tesla Stock (yfinance)** | yfinance library | TSLA OHLCV + volatility | Free |
| **Crypto (yfinance)** | yfinance library (DOGE-USD, BTC-USD) | Crypto prices + momentum | Free |
| **Wikipedia Pageviews** | wikimedia.org/api/rest_v1 | Public attention proxy (5 articles) | Free |
| **VIX (yfinance)** | yfinance library (^VIX) | Market fear gauge / volatility index | Free |
| **Crypto Fear & Greed** | api.alternative.me | Crypto sentiment (0-100 daily) | Free |
| **Google Trends** | pytrends library | Search interest for Musk-related queries | Free |

### 3.1 XTracker API (Resolution Oracle)

| Property | Value |
|----------|-------|
| **URL** | https://xtracker.polymarket.com |
| **Documentation** | https://xtracker.polymarket.com/docs |
| **Cost** | FREE |
| **Authentication** | None required |
| **Rate limits** | Rate limited (specific limits undocumented; assume conservative usage) |

**Purpose**: This is Polymarket's official resolution oracle. It provides the definitive tweet counts that markets settle against. It is our ground truth data source.

**Key Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/users/{handle}` | GET | Get user profile and userId for a given X handle (e.g., `elonmusk`) |
| `/metrics/{userId}?type=daily&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD` | GET | Get daily post counts for a user over a date range |
| `/users/{handle}/trackings` | GET | Get active tracking windows and current counts for a user |

**Data provided**:
- Daily post counts (the exact counts Polymarket uses for resolution)
- Active tracking windows with live counts
- User profile metadata

**Critical importance**: Because this is the resolution source, any model we build must be validated against XTracker counts, not raw Twitter API counts (which may differ due to deletion timing, reply exclusion rules, etc.).

---

### 3.2 Kaggle Datasets (Historical Training Data)

| Property | Value |
|----------|-------|
| **Cost** | FREE |
| **Authentication** | Kaggle account + API key (`~/.kaggle/kaggle.json`) |
| **Rate limits** | Standard Kaggle download limits |

**Dataset 1: "Elon Musk Tweets 2010-2025 (April)"**
- **URL**: https://www.kaggle.com/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march
- **Coverage**: 2010 through April 2025
- **Fields**: `createdAt` (timestamp), `id` (tweet ID), `fullText` (tweet content), engagement metrics (likes, retweets, replies, views)
- **Use**: Primary historical training data for establishing long-term patterns

**Dataset 2: "Elon Musk Tweets (Daily Updated)"**
- **URL**: https://www.kaggle.com/datasets/aryansingh0909/elon-musk-tweets-updated-daily
- **Coverage**: Continuously updated (daily refresh)
- **Fields**: Same schema as Dataset 1
- **Use**: Keeps training data current without requiring Twitter API access

**Important notes**:
- Kaggle datasets may include replies, which XTracker excludes. During feature engineering, filter to match XTracker's counting rules (exclude replies, include reposts and quote posts).
- Cross-validate Kaggle daily counts against XTracker daily counts for the overlapping period to quantify any discrepancies.

---

### 3.3 Twitter/X API

| Property | Value |
|----------|-------|
| **Cost** | $0 (Free), $200/mo (Basic), $5,000/mo (Pro) |
| **Authentication** | OAuth 2.0 / Bearer Token |

**Tier breakdown**:

| Tier | Price | Read Access | Write Access |
|------|-------|-------------|--------------|
| Free | $0 | **NONE** (write only) | Yes |
| Basic | $200/mo | 15,000 reads/month | Yes |
| Pro | $5,000/mo | Full archive search | Yes |

**Recommendation**: **DO NOT USE for reads**. The free tier cannot read tweets at all, Basic is expensive for limited reads, and Pro is prohibitively expensive. Instead, use Kaggle datasets + XTracker for all historical and current data.

**Backup scraping options** (use only if Kaggle/XTracker become insufficient):
- **twikit**: https://github.com/d60/twikit -- Python library for scraping X without API access
- **twscrape**: https://github.com/vladkens/twscrape -- Another scraping library, supports multiple accounts for rate limit distribution

**Warning**: Scraping violates X's Terms of Service. Use only as a last resort, and be prepared for breakage as X frequently changes their frontend.

---

### 3.4 GDELT (News and Event Data)

| Property | Value |
|----------|-------|
| **URL** | https://gdeltproject.org |
| **Cost** | FREE |
| **Authentication** | None (direct download) or Google Cloud account (BigQuery) |
| **Update frequency** | Every 15 minutes |
| **Historical coverage** | Back to 1979 |

**Purpose**: GDELT monitors news sources worldwide and categorizes events. We use it to detect news events that are likely to trigger Musk tweeting bursts.

**Access methods**:
1. **Google BigQuery** (recommended): Free tier includes 1TB/month of queries. GDELT tables are publicly available.
2. **Direct download**: CSV files available at http://data.gdeltproject.org
3. **Python package**: `pip install gdelt`

**Relevant queries**:
- Filter for events/articles mentioning "Elon Musk", "Tesla", "SpaceX", "DOGE", "xAI", "Neuralink", "Boring Company"
- GDELT provides 300+ event categories (CAMEO codes) with tone/sentiment scores
- Track article volume as a proxy for news intensity

**Feature engineering**: Daily/hourly article counts and average tone scores for Musk-related entities. Spikes in article volume often precede or coincide with tweeting bursts.

---

### 3.5 SpaceX Launch Library 2

| Property | Value |
|----------|-------|
| **URL** | https://ll.thespacedevs.com/2.0.0 |
| **Cost** | FREE |
| **Authentication** | None required |
| **Rate limit** | 15 requests/hour (free tier) |

**Purpose**: SpaceX launch schedule. Launches are known events that predictably affect Musk's tweeting behavior.

**Key endpoint**:
- `GET /launch/?lsp__name=SpaceX` -- Returns SpaceX launches with dates, status, mission details

**Alternative source**:
- **r/SpaceX API**: https://github.com/r-spacex/SpaceX-API -- Community-maintained, includes historical launch data and upcoming schedule

**Feature engineering**: Binary flags for "launch within X days", launch type (Starship vs Falcon 9 vs crewed), and launch success/failure outcomes.

---

### 3.6 SEC EDGAR (Tesla Filings)

| Property | Value |
|----------|-------|
| **URL** | https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1318605 |
| **Cost** | FREE |
| **Authentication** | None required (but must include User-Agent header with contact email) |
| **Tesla CIK** | 1318605 |

**Purpose**: Tesla earnings dates and material event filings. Earnings calls are known catalysts for tweeting behavior.

**Relevant filing types**:
- **8-K**: Material events (earnings releases, executive changes, acquisitions)
- **10-Q**: Quarterly earnings reports
- **10-K**: Annual reports

**Feature engineering**: Binary flags for "earnings within X days", "8-K filed in last 24 hours". Earnings dates are known in advance and can be hardcoded or fetched from financial calendars.

---

### 3.7 Flight Tracking (Musk's Private Jet)

| Property | Value |
|----------|-------|
| **Aircraft** | N628TS (Gulfstream G650) |
| **ICAO hex** | a835af |
| **Cost** | $0-10/month depending on source |

**Purpose**: Musk's physical location and travel status correlate with tweeting patterns. In-flight periods tend to have reduced tweeting. Travel to specific locations (Boca Chica = SpaceX activity, Washington DC = political activity) predicts tweeting topics and intensity.

**Data sources**:

| Source | URL | Cost | Notes |
|--------|-----|------|-------|
| ADS-B Exchange | https://globe.adsbexchange.com | ~$10/mo via RapidAPI | Real-time tracking, full history, API access |
| OpenSky Network | https://opensky-network.org | Free | Free API with 1-hour history; full archive available for academic use |
| ElonJet (Mastodon) | https://mastodon.social/@elonjet | Free | Automated posts with 24-hour delay; accessible via Mastodon API |
| plane-notify | https://github.com/Jxck-S/plane-notify | Free (self-hosted) | Bot that sends alerts on aircraft movements; can be self-hosted |
| CelebrityPrivateJetTracker | https://celebrityprivatejettracker.com | Free | Monthly flight history for N628TS; manual data collection |

**Feature engineering**: Binary "in flight" indicator, hours since last landing, destination city, days since last long-haul flight.

---

### 3.8 Polymarket Data API (Market Prices and Whale Activity)

| Property | Value |
|----------|-------|
| **Cost** | FREE |
| **Authentication** | None required |
| **Rate limits** | Reasonable; no hard documentation |

**Purpose**: Current market prices (for detecting mispricing), historical trade data (for understanding market dynamics), and whale tracking (for detecting informed flow).

**Key endpoints**:

| Endpoint | Description |
|----------|-------------|
| `GET https://gamma-api.polymarket.com/markets` | List markets, filter by tag/keyword |
| `GET https://data-api.polymarket.com/v1/leaderboard` | Top traders by P&L |
| `GET https://data-api.polymarket.com/trades?user={wallet}` | Trades for a specific wallet |
| `GET https://data-api.polymarket.com/positions?user={wallet}` | Current positions for a wallet |

**Use cases**:
- Fetch current bucket prices for all active Elon tweet markets
- Track whale movements into specific buckets (contrarian signal)
- Monitor price movements over time for backtesting
- Identify consistently profitable traders to copy/fade

---

### 3.9 Tesla Stock Data (yfinance)

| Property | Value |
|----------|-------|
| **Library** | yfinance (pip install yfinance) |
| **Cost** | FREE |
| **Authentication** | None required |
| **Coverage** | Jan 2024 - present |

**Purpose**: Tesla stock price movements correlate with Musk's tweeting behavior. High volatility or drawdowns trigger defensive/reactive tweeting.

**Data stored**: `data/sources/market/tesla_daily.parquet` -- daily OHLCV + derived features (pct_change, volatility_5d, volume_ma5, gap).

**Feature engineering**: 6 features -- tsla_pct_change_1d, tsla_pct_change_5d, tsla_volatility_5d, tsla_volume_ratio, tsla_drawdown_5d, tsla_gap_1d.

---

### 3.10 Cryptocurrency Data (yfinance)

| Property | Value |
|----------|-------|
| **Library** | yfinance (DOGE-USD, BTC-USD) |
| **Cost** | FREE |
| **Authentication** | None required |
| **Coverage** | Jan 2024 - present |

**Purpose**: DOGE price movements are strongly associated with Musk tweeting activity. BTC provides broader crypto market context.

**Data stored**: `data/sources/market/crypto_daily.parquet` -- daily close prices + pct_change for DOGE and BTC.

**Feature engineering**: 6 features -- doge_pct_change_1d, doge_pct_change_5d, doge_volatility_5d, btc_pct_change_1d, btc_pct_change_5d, btc_volatility_5d.

---

### 3.11 Wikipedia Pageviews (Wikimedia REST API)

| Property | Value |
|----------|-------|
| **URL** | https://wikimedia.org/api/rest_v1/ |
| **Cost** | FREE |
| **Authentication** | None required |
| **Rate limits** | 200 req/sec |
| **Coverage** | Jan 2024 - present |

**Purpose**: Wikipedia pageview spikes are a proxy for public attention and interest. When the Elon_Musk article gets more views, he's in the public eye and likely tweeting more.

**Data stored**: `data/sources/wikipedia/pageviews.json` -- daily views for 5 articles (Elon_Musk, Tesla_Inc, SpaceX, Dogecoin, Department_of_Government_Efficiency).

**Feature engineering**: 8 features -- wiki_elon_musk_7d, wiki_elon_musk_delta, wiki_tesla_7d, wiki_tesla_delta, wiki_doge_7d, wiki_doge_delta, wiki_total_7d, wiki_attention_concentration.

---

### 3.12 VIX (CBOE Volatility Index via yfinance)

| Property | Value |
|----------|-------|
| **URL** | yfinance library (`^VIX`) |
| **Cost** | FREE |
| **Authentication** | None required |
| **Rate limits** | Standard yfinance limits |
| **Coverage** | Jan 2024 - present (532 daily rows) |

**Purpose**: The VIX is the market's "fear gauge." When VIX is elevated (>25), markets are stressed and Elon may tweet more about markets/economy. VIX spikes above its 5-day moving average (ratio >1.10) signal acute stress events.

**Data stored**: `data/sources/market/vix_daily.parquet` -- daily OHLCV plus derived features (pct_change, pct_change_5d, ma5, ma5_ratio, level_category).

**Feature engineering**: 5 features -- vix_close, vix_pct_change_1d, vix_pct_change_5d, vix_level_category, vix_ma5_ratio.

---

### 3.13 Crypto Fear & Greed Index (alternative.me)

| Property | Value |
|----------|-------|
| **URL** | https://api.alternative.me/fng/?limit=0&format=json |
| **Cost** | FREE |
| **Authentication** | None required |
| **Rate limits** | Generous (no documented limits) |
| **Coverage** | Jan 2024 - present (776 daily rows) |

**Purpose**: The Crypto Fear & Greed Index (0-100) measures overall crypto market sentiment. Extreme fear (<25) or extreme greed (>75) often trigger Musk crypto-related tweets. The index combines volatility, market momentum, social media, surveys, BTC dominance, and Google Trends data.

**Data stored**: `data/sources/market/crypto_fear_greed.parquet` -- daily values with 7-day and 3-day rolling averages, delta, and category labels.

**Feature engineering**: 4 features -- crypto_fg_value, crypto_fg_7d_avg, crypto_fg_delta, crypto_fg_category.

---

### 3.14 Google Trends (pytrends)

| Property | Value |
|----------|-------|
| **URL** | trends.google.com (via pytrends library) |
| **Cost** | FREE |
| **Authentication** | None required |
| **Rate limits** | ~5 requests/minute (15s sleep between windows) |
| **Coverage** | Jan 2024 - present (777 daily rows) |

**Purpose**: Google search interest for Musk-related queries provides an independent public attention signal. Complementary to Wikipedia pageviews (different audience). Search interest spikes for "Elon Musk" with >15% delta above 7-day average signal breaking stories.

**Queries**: `["Elon Musk", "Tesla", "SpaceX", "Dogecoin", "DOGE government"]`

**Data collection**: Uses overlapping 85-day windows (max for daily granularity) stitched with overlap normalization. Takes ~20 minutes due to rate limiting.

**Data stored**: `data/sources/trends/google_trends.parquet` -- daily relative interest (0-100) for each query.

**Feature engineering**: 10 features -- gt_{query}_7d, gt_{query}_delta for 4 queries, plus gt_total_7d, gt_concentration (HHI).

---

## 4. Competitor and Inspiration Tools

### Tracking and Analytics Platforms

| Tool | URL | What It Does | Relevance |
|------|-----|-------------|-----------|
| Polymarket Analytics | https://polymarketanalytics.com | Leaderboard, P&L tracking, filter by category (including SPORTS) | Identify top Elon tweet market traders |
| Polywhaler | https://polywhaler.com | Whale tracking, alerts for $10K+ trades | Detect informed flow into specific buckets |
| PolyWatch | https://polywatch.tech | Free whale tracker with Telegram bot alerts | Real-time whale monitoring without cost |
| Stand.trade | https://stand.trade | Copy trading platform for Polymarket | Could be used to auto-follow top tweet traders |
| Hashdive | https://hashdive.com | "Smart scores" for markets, analytics dashboard | Alternative market sentiment data |

### Open Source Tools

| Tool | URL | What It Does |
|------|-----|-------------|
| poly_data | https://github.com/warproxxx/poly_data | Open source Polymarket data pipeline; fetches markets, trades, and CLOB data |

### Notable Mentions

- **"Wenger" intelligence board**: Referenced in a YouTube video about Polymarket trading strategies. Not found publicly available. May be a private/internal tool, or may have been renamed. Worth monitoring for if it surfaces.
- The **Dutch trader Max** ($490K+ profit) did not appear to use any publicly available tool; his edge was fundamental analysis of Musk's behavioral patterns.

---

## 5. Key Statistics

### Long-Term Averages

- **2025 average**: 93.8 tweets/day (median: 97.5 tweets/day)
- **Tweeted at least once** on all but 14 days in 2021 (near-daily consistency)

### Recent Weekly Totals (2026)

| Week | Total Posts | Market Prediction |
|------|------------|-------------------|
| Jan 20-27 | ~380 | Unknown |
| Jan 27 - Feb 3 | ~280 | Unknown |
| Feb 3-10 | ~130 | 220-239 (badly wrong) |

The three-week sequence of **380 -> 280 -> 130** illustrates the extreme variance. This is a roughly 3x swing in weekly output over just three weeks.

### Intra-Day Distribution

- **Peak hours**: US business hours, approximately 9am-6pm EST
- **Peak output**: ~36 tweets during the 9am-6pm EST window on a typical weekday
- **Weekdays vs weekends**: Weekday tweet counts significantly exceed weekend counts
- **Late night**: Low but non-zero activity (Musk is known to tweet at 2-3am)

### Variance Characteristics

The high variance is the defining characteristic of this prediction problem. Key observations:
- Weekly standard deviation is large relative to the mean (coefficient of variation likely > 0.3)
- Distribution is likely right-skewed (occasional extremely high-output weeks)
- Serial autocorrelation is moderate -- knowing this week's count helps predict next week's, but not enough for the crowd to price correctly
- Regime shifts (from high-output to low-output or vice versa) are the primary source of market mispricing

### Proven Model Performance (159 events, 21 months)

| Model | Gold ROI | All-Tier ROI | Bets | Key Edge |
|-------|----------|--------------|------|----------|
| SignalEnhanced v3 | +66.3% | +17.7% | 188 | Signal-modulated tail boost |
| TailBoost | +44.4% | +13.9% | 187 | Structural tail underpricing |
| DurationTail | +67.7% | +13.8% | 133 | Duration-aware tail boost |
| DurationShrink | +49.7% | +21.5% | 47 | Short event EV shrinkage |

---

## 6. Modeling Approach

### Problem Framing

This is a **distributional forecast problem**, NOT a binary classification or simple point prediction.

We need to predict the **probability of each bucket** (tweet count range) being the correct one. The market prices each bucket independently, and our edge comes from identifying buckets that are over- or under-priced.

For example, if the market prices the "120-139" bucket at $0.05 (5% implied probability) but our model assigns it 15% probability, we have a +10% edge on that bucket.

### Candidate Models

| Model | Why It Fits | Limitations |
|-------|-------------|-------------|
| **Poisson regression** | Natural model for count data; single parameter (lambda) | Assumes mean = variance; real data is overdispersed |
| **Negative binomial regression** | Handles overdispersion (variance > mean) | Still assumes unimodal distribution |
| **Quantile regression** | Directly predicts percentiles of the distribution | Doesn't give bucket probabilities directly; requires post-processing |
| **Ordinal regression** | Models ordered categories (buckets are ordered) | May not capture the shape of the distribution well |
| **Histogram-based gradient boosting** | Flexible, handles mixed feature types, fast | Requires careful calibration to output probabilities |
| **Zero-inflated / mixture models** | Captures regime structure (high/low output states) | More complex to fit and validate |

**Recommended approach**: Start with negative binomial regression as a baseline (interpretable, naturally suited to count data). Then layer on gradient boosting with calibrated probability outputs. Use a mixture model if regime detection proves important.

### Feature Groups

**1. Temporal Features (Rolling Statistics)**
- Rolling 7-day, 14-day, 30-day mean and standard deviation of daily tweet counts
- Day-of-week encoding (one-hot or cyclic)
- Hour-of-day patterns (for intra-window prediction)
- Trend (is daily count increasing or decreasing over the last N days?)
- Days since last "quiet day" (< 50 tweets) or "burst day" (> 150 tweets)

**2. Calendar Features (Known Events)**
- Tesla earnings date proximity (days until/since)
- SpaceX launch date proximity
- DOGE meeting schedule
- US federal holidays
- Major tech conferences (CES, GTC, etc.)
- Day of week, weekend flag

**3. News Features (GDELT)**
- Daily article count mentioning Musk-related entities
- Average tone/sentiment of articles
- Spike detection (article count > 2 standard deviations above rolling mean)
- Topic distribution (politics vs technology vs business)

**4. Flight Features**
- Binary: is Musk currently in flight?
- Hours since last landing
- Destination type (SpaceX site, Tesla factory, Washington DC, international)
- Total flight hours in last 7 days (proxy for travel intensity)

**5. Market Features (Polymarket Prices)**
- Current bucket prices (crowd's implied distribution)
- Price momentum (how prices have moved since market opened)
- Whale activity (large trades into specific buckets)
- Spread between our model and market for each bucket

### Backtesting Strategy

1. **Collect historical data**: Merge Kaggle tweet history with XTracker daily counts
2. **Reconstruct historical markets**: For each past market window, compute the actual bucket outcome
3. **Train on pre-window data**: For each market, train model only on data available before the window opened
4. **Compare predictions to market prices**: If we had historical Polymarket prices, compare our predicted bucket probabilities to market-implied probabilities. Without historical prices, compare our predictions to a naive baseline (e.g., always predicting the most recent week's distribution).
5. **Simulate P&L**: For each market, calculate what our P&L would have been if we bet on buckets where our model probability exceeded market price by more than a threshold (e.g., 5%, 10%, 15% edge).

### Point-in-Time Discipline

Following lessons learned from the CS2 Esports project: **ALL features must be computed using only data available at prediction time.** No lookahead bias. Use `train_test_split_temporal()` equivalent -- never random splits.

---

## 7. Comparison to CS2 Esports Project

| Dimension | CS2 Esports | Elon Tweets |
|-----------|-------------|-------------|
| **Volume/week** | $50-200K | $5-20M |
| **Liquidity/market** | $5-50K | $500K-2M |
| **Simultaneous markets** | Varies (match schedule) | ~24 (rolling windows) |
| **Data matching rate** | 41% (team alias hell) | 100% (XTracker is oracle) |
| **Prediction type** | Binary (team A wins vs team B) | Distributional (which bucket?) |
| **Edge source** | Elo ratings + form features | Temporal patterns + events + regime detection |
| **Data cost** | Free (HLTV scraping) | Free (Kaggle + XTracker) |
| **Resolution ambiguity** | Low (match results are clear) | None (XTracker is definitive) |
| **Position sizing** | Limited by $5-50K liquidity | $500K-2M allows meaningful size |
| **Frequency** | Depends on match schedule | Weekly+ (continuous markets) |
| **Competitive advantage** | Better Elo/form models | Better regime detection + event features |
| **Main risk** | Low liquidity, stale odds | Musk behavior is inherently unpredictable |

### Key Advantages Over CS2

1. **100x more liquidity**: Can take larger positions without moving the market
2. **No data matching problem**: XTracker eliminates the team name alias nightmare
3. **Free data**: No scraping required; Kaggle and XTracker provide everything needed
4. **Higher frequency**: Markets resolve weekly, providing rapid feedback loops
5. **Crowd is demonstrably wrong**: The Feb 3-10 miss (predicted 220-239, actual 130) shows large, exploitable mispricing

### Key Risks

1. **Single-person dependency**: All prediction hinges on one individual's behavior, which can change without warning
2. **Black swan events**: Musk could get banned from X, sell the platform, go to prison, etc.
3. **Market efficiency improvement**: If our model works, others will develop similar approaches
4. **Regime detection is hard**: The core edge (detecting regime shifts before the crowd) is inherently difficult
5. **XTracker counting rule changes**: If Polymarket changes what counts as a "post", historical data becomes less relevant

---

## Appendix: Research Sources

- Polymarket Elon tweet markets (live market observation)
- XTracker documentation (https://xtracker.polymarket.com/docs)
- Kaggle dataset pages (URLs in Section 3)
- YouTube video on Polymarket trading strategies (referenced "Wenger" tool and Max's profits)
- X API pricing page (https://developer.x.com/en/docs/twitter-api)
- GDELT documentation (https://gdeltproject.org)
- Launch Library 2 documentation (https://thespacedevs.com/llapi)
- OpenSky Network documentation (https://openskynetwork.github.io/opensky-api/)
- ADS-B Exchange (https://www.adsbexchange.com)
- Various Polymarket analytics tools (URLs in Section 4)

---

*This document represents the state of research as of 2026-02-15. Data sources and market conditions may change.*
