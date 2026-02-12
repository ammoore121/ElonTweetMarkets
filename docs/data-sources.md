# Data Sources -- Quick Reference

Concise setup and usage guide for every data source in the Elon Tweet Markets project.

**Last updated**: 2026-02-09

---

## 1. XTracker API

| Property | Value |
|----------|-------|
| **URL** | https://xtracker.polymarket.com |
| **Docs** | https://xtracker.polymarket.com/docs |
| **Provides** | Official tweet counts (Polymarket resolution oracle) |
| **Auth** | None required |
| **Rate limit** | Rate limited (undocumented; use conservative delays) |
| **Cost** | Free |

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /users/{handle}` | Lookup user profile and userId |
| `GET /metrics/{userId}?type=daily&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD` | Daily post counts |
| `GET /users/{handle}/trackings` | Active tracking windows with live counts |

### Python Example

```python
import requests

BASE_URL = "https://xtracker.polymarket.com"

# Step 1: Get userId
user = requests.get(f"{BASE_URL}/users/elonmusk").json()
user_id = user["id"]

# Step 2: Fetch daily counts
params = {
    "type": "daily",
    "startDate": "2026-01-01",
    "endDate": "2026-02-09",
}
metrics = requests.get(f"{BASE_URL}/metrics/{user_id}", params=params).json()

for day in metrics:
    print(f"{day['date']}: {day['count']} posts")

# Step 3: Get active tracking windows
trackings = requests.get(f"{BASE_URL}/users/elonmusk/trackings").json()
for t in trackings:
    print(f"Window: {t['startDate']} to {t['endDate']}, current count: {t['count']}")
```

### Storage

```
data/sources/xtracker/daily_counts.parquet    # Historical daily counts
data/sources/xtracker/trackings.json          # Snapshot of active tracking windows
```

---

## 2. Kaggle Datasets

| Property | Value |
|----------|-------|
| **Provides** | Historical tweet data with timestamps, text, and engagement metrics |
| **Auth** | Kaggle API key (`~/.kaggle/kaggle.json`) |
| **Rate limit** | Standard Kaggle limits |
| **Cost** | Free |

### Datasets

| Dataset | URL | Coverage |
|---------|-----|----------|
| Elon Musk Tweets 2010-2025 | https://www.kaggle.com/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march | 2010 -- April 2025 |
| Elon Musk Tweets (Daily Updated) | https://www.kaggle.com/datasets/aryansingh0909/elon-musk-tweets-updated-daily | Continuously updated |

### Fields

`createdAt` (timestamp), `id` (tweet ID), `fullText` (content), `likeCount`, `retweetCount`, `replyCount`, `viewCount`

### Python Example

```python
import subprocess
import pandas as pd

# Download datasets (requires ~/.kaggle/kaggle.json)
subprocess.run([
    "kaggle", "datasets", "download",
    "dadalyndell/elon-musk-tweets-2010-to-2025-march",
    "-p", "data/sources/kaggle/",
    "--unzip"
])

subprocess.run([
    "kaggle", "datasets", "download",
    "aryansingh0909/elon-musk-tweets-updated-daily",
    "-p", "data/sources/kaggle/",
    "--unzip"
])

# Load and inspect
df = pd.read_csv("data/sources/kaggle/elon_musk_tweets.csv", parse_dates=["createdAt"])
print(f"Total tweets: {len(df)}")
print(f"Date range: {df['createdAt'].min()} to {df['createdAt'].max()}")

# Filter out replies (XTracker does not count replies)
df_posts = df[df["inReplyToId"].isna()]
print(f"Posts (excluding replies): {len(df_posts)}")

# Daily counts
daily = df_posts.set_index("createdAt").resample("D").size()
print(f"Mean daily posts: {daily.mean():.1f}")
```

### Storage

```
data/sources/kaggle/elon_musk_tweets.csv          # Raw download (archive dataset)
data/sources/kaggle/elon_musk_tweets_daily.csv     # Raw download (daily-updated dataset)
data/sources/kaggle/tweets_cleaned.parquet         # Cleaned, deduplicated, replies filtered
```

---

## 3. GDELT (News and Event Data)

| Property | Value |
|----------|-------|
| **URL** | https://gdeltproject.org |
| **Provides** | Global news events, article counts, sentiment scores |
| **Auth** | None (direct download) or Google Cloud account (BigQuery) |
| **Rate limit** | BigQuery: 1TB free/month; Direct download: no limit |
| **Update frequency** | Every 15 minutes |
| **Cost** | Free |

### Access Methods

1. **Python package** (simplest): `pip install gdelt`
2. **Google BigQuery** (most powerful): Query `gdelt-bq.gdeltv2.gkg` and `gdelt-bq.gdeltv2.events`
3. **Direct download**: CSV files at http://data.gdeltproject.org

### Python Example

```python
import gdelt

gd = gdelt.gdelt(version=2)

# Fetch events mentioning Musk-related entities for a date range
# Note: gdelt package queries by single dates
results = gd.Search(
    date=["2026-02-01", "2026-02-09"],
    table="events",
    coverage=True
)

# Filter for Musk-related events
musk_keywords = ["elon musk", "tesla", "spacex", "doge efficiency", "xai", "neuralink"]
mask = results["SOURCEURL"].str.lower().str.contains("|".join(musk_keywords), na=False)
musk_events = results[mask]

print(f"Musk-related events: {len(musk_events)}")
print(f"Average tone: {musk_events['AvgTone'].mean():.2f}")

# --- Alternative: BigQuery approach (more flexible) ---
from google.cloud import bigquery

client = bigquery.Client()
query = """
SELECT
    DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(SQLDATE AS STRING) || '000000')) as date,
    COUNT(*) as article_count,
    AVG(AvgTone) as avg_tone
FROM `gdelt-bq.gdeltv2.events`
WHERE
    LOWER(Actor1Name) LIKE '%musk%'
    OR LOWER(Actor2Name) LIKE '%musk%'
    OR LOWER(Actor1Name) LIKE '%tesla%'
    OR LOWER(Actor1Name) LIKE '%spacex%'
GROUP BY date
ORDER BY date DESC
LIMIT 100
"""
df = client.query(query).to_dataframe()
```

### Storage

```
data/sources/gdelt/musk_events.parquet          # Filtered events mentioning Musk entities
data/sources/gdelt/daily_news_features.parquet   # Aggregated daily: article count, avg tone
```

---

## 4. SpaceX Launch Library 2

| Property | Value |
|----------|-------|
| **URL** | https://ll.thespacedevs.com/2.0.0 |
| **Provides** | SpaceX launch schedule, dates, status, mission details |
| **Auth** | None required |
| **Rate limit** | 15 requests/hour (free tier) |
| **Cost** | Free |

### Python Example

```python
import requests
import pandas as pd

url = "https://ll.thespacedevs.com/2.0.0/launch/"
params = {
    "lsp__name": "SpaceX",
    "ordering": "-net",  # Most recent first
    "limit": 50,
}

resp = requests.get(url, params=params).json()

launches = []
for launch in resp["results"]:
    launches.append({
        "name": launch["name"],
        "net": launch["net"],           # NET = No Earlier Than (launch date)
        "status": launch["status"]["name"],
        "mission": launch.get("mission", {}).get("name", ""),
    })

df = pd.DataFrame(launches)
df["net"] = pd.to_datetime(df["net"])
print(df[["name", "net", "status"]].to_string())
```

### Alternative: r/SpaceX API

```python
import requests

# Past launches
resp = requests.get("https://api.spacexdata.com/v4/launches/past").json()
# Upcoming launches
resp = requests.get("https://api.spacexdata.com/v4/launches/upcoming").json()
```

GitHub: https://github.com/r-spacex/SpaceX-API

### Storage

```
data/sources/spacex/launches.parquet    # Launch schedule with dates and status
```

---

## 5. SEC EDGAR (Tesla Filings)

| Property | Value |
|----------|-------|
| **URL** | https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=1318605 |
| **Provides** | Tesla SEC filings (earnings dates, material events) |
| **Auth** | None (must include User-Agent header with contact email) |
| **Rate limit** | 10 requests/second |
| **Tesla CIK** | 1318605 |
| **Cost** | Free |

### Python Example

```python
import requests
import pandas as pd

headers = {
    "User-Agent": "ElonTweetMarkets research@example.com",  # Required by SEC
}

# Fetch Tesla filing history
resp = requests.get(
    "https://data.sec.gov/submissions/CIK0001318605.json",
    headers=headers
).json()

recent = resp["filings"]["recent"]
df = pd.DataFrame({
    "form": recent["form"],
    "filingDate": recent["filingDate"],
    "primaryDocument": recent["primaryDocument"],
})

# Filter for earnings-related filings
earnings = df[df["form"].isin(["8-K", "10-Q", "10-K"])]
print(earnings.head(20))
```

### Storage

```
data/sources/sec/tesla_filings.parquet    # Filing dates and types
data/sources/sec/earnings_dates.csv       # Extracted quarterly earnings dates
```

---

## 6. Flight Tracking

| Property | Value |
|----------|-------|
| **Aircraft** | N628TS (Gulfstream G650) |
| **ICAO hex** | a835af |
| **Provides** | Musk's jet location, flight times, destinations |
| **Cost** | $0-10/month |

### Sources

| Source | URL | Auth | Cost |
|--------|-----|------|------|
| ADS-B Exchange | https://globe.adsbexchange.com | RapidAPI key | ~$10/mo |
| OpenSky Network | https://opensky-network.org | Free account | Free |
| ElonJet (Mastodon) | https://mastodon.social/@elonjet | None | Free (24hr delay) |
| plane-notify | https://github.com/Jxck-S/plane-notify | Self-hosted | Free |

### Python Example (OpenSky)

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

# OpenSky REST API -- get flights for N628TS (ICAO: a835af)
# Note: Free tier only returns last 1 hour of state vectors
# For historical data, use the OpenSky historical database (academic access)

ICAO = "a835af"

# Current state (is the plane in the air right now?)
resp = requests.get(
    "https://opensky-network.org/api/states/all",
    params={"icao24": ICAO}
).json()

if resp["states"]:
    state = resp["states"][0]
    print(f"In flight: YES")
    print(f"Latitude: {state[6]}, Longitude: {state[5]}")
    print(f"Altitude: {state[7]}m")
else:
    print("In flight: NO (or not tracked)")

# Historical flights (last 30 days, requires free account)
end = int(datetime.now().timestamp())
begin = int((datetime.now() - timedelta(days=30)).timestamp())

resp = requests.get(
    "https://opensky-network.org/api/flights/aircraft",
    params={"icao24": ICAO, "begin": begin, "end": end},
    auth=("username", "password")  # Free OpenSky account
).json()

flights = pd.DataFrame(resp)
if not flights.empty:
    flights["departure_time"] = pd.to_datetime(flights["firstSeen"], unit="s")
    flights["arrival_time"] = pd.to_datetime(flights["lastSeen"], unit="s")
    print(flights[["departure_time", "arrival_time", "estDepartureAirport", "estArrivalAirport"]])
```

### Python Example (ElonJet via Mastodon API)

```python
import requests

# No auth required for public Mastodon posts
# ElonJet account ID on mastodon.social (look up once, then hardcode)
ACCOUNT_ID = "109348728587973994"  # Verify this

resp = requests.get(
    f"https://mastodon.social/api/v1/accounts/{ACCOUNT_ID}/statuses",
    params={"limit": 20}
).json()

for status in resp:
    print(f"{status['created_at']}: {status['content'][:100]}")
```

### Storage

```
data/sources/flights/flight_history.parquet    # Historical flights (departure, arrival, airports)
data/sources/flights/current_state.json        # Latest known position (updated by cron)
```

---

## 7. Polymarket Data API

| Property | Value |
|----------|-------|
| **Provides** | Market prices, trades, positions, leaderboard |
| **Auth** | None required (public endpoints) |
| **Rate limit** | Reasonable (undocumented) |
| **Cost** | Free |

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET https://gamma-api.polymarket.com/markets` | Browse/search markets |
| `GET https://data-api.polymarket.com/v1/leaderboard` | Top traders by P&L |
| `GET https://data-api.polymarket.com/trades?user={wallet}` | Trade history for a wallet |
| `GET https://data-api.polymarket.com/positions?user={wallet}` | Current positions |

### Python Example

```python
import requests
import pandas as pd

# Step 1: Find Elon tweet markets
resp = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"tag": "elon-musk-tweets", "closed": False}
).json()

for market in resp:
    print(f"ID: {market['id']}")
    print(f"Question: {market['question']}")
    print(f"Liquidity: ${float(market.get('liquidity', 0)):,.0f}")
    print(f"Volume: ${float(market.get('volume', 0)):,.0f}")
    print()

# Step 2: Get token prices for a specific market (bucket prices)
# Each market has multiple tokens (one per bucket)
market_id = resp[0]["id"] if resp else "example-market-id"
market_detail = requests.get(
    f"https://gamma-api.polymarket.com/markets/{market_id}"
).json()

if "tokens" in market_detail:
    for token in market_detail["tokens"]:
        print(f"Bucket: {token['outcome']}, Price: ${float(token.get('price', 0)):.3f}")

# Step 3: Get top traders (leaderboard)
leaderboard = requests.get(
    "https://data-api.polymarket.com/v1/leaderboard",
    params={"limit": 20}
).json()

for trader in leaderboard:
    print(f"Address: {trader['address'][:10]}..., P&L: ${trader.get('pnl', 0):,.0f}")

# Step 4: Track a specific whale's positions
whale_address = "0x..."  # Replace with actual wallet
positions = requests.get(
    "https://data-api.polymarket.com/positions",
    params={"user": whale_address}
).json()
```

### Storage

```
data/sources/polymarket/elon_tweet_markets.parquet    # Market metadata and current prices
data/sources/polymarket/elon_tweet_trades.parquet      # Historical trades
data/sources/polymarket/leaderboard.parquet             # Trader P&L snapshots
data/sources/polymarket/whale_positions.parquet          # Tracked whale positions
```

---

## 8. Twitter/X API (NOT RECOMMENDED)

| Property | Value |
|----------|-------|
| **URL** | https://developer.x.com |
| **Provides** | Direct tweet access (but expensive) |
| **Auth** | OAuth 2.0 / Bearer Token |
| **Cost** | Free: write-only; Basic: $200/mo (15K reads); Pro: $5,000/mo |

**Recommendation**: Do NOT use. Kaggle + XTracker provide all needed data for free. See RESEARCH.md Section 3.3 for full details.

**Backup scrapers** (last resort only):
- twikit: https://github.com/d60/twikit
- twscrape: https://github.com/vladkens/twscrape

---

## Storage Directory Structure

```
data/
  sources/
    xtracker/
      daily_counts.parquet
      trackings.json
    kaggle/
      elon_musk_tweets.csv
      elon_musk_tweets_daily.csv
      tweets_cleaned.parquet
    gdelt/
      musk_events.parquet
      daily_news_features.parquet
    spacex/
      launches.parquet
    sec/
      tesla_filings.parquet
      earnings_dates.csv
    flights/
      flight_history.parquet
      current_state.json
    polymarket/
      elon_tweet_markets.parquet
      elon_tweet_trades.parquet
      leaderboard.parquet
      whale_positions.parquet
  matched/
    xtracker_kaggle_validated.parquet    # XTracker counts cross-validated with Kaggle
  features/
    temporal_features.parquet
    calendar_features.parquet
    news_features.parquet
    flight_features.parquet
    market_features.parquet
    combined_features.parquet
```

---

## Setup Checklist

1. [ ] Create Kaggle account and download API key to `~/.kaggle/kaggle.json`
2. [ ] Install packages: `pip install requests pandas kaggle gdelt google-cloud-bigquery`
3. [ ] Run XTracker fetch to verify API is accessible
4. [ ] Download Kaggle datasets
5. [ ] Cross-validate Kaggle daily counts against XTracker daily counts
6. [ ] Set up OpenSky Network free account for flight tracking
7. [ ] (Optional) Set up ADS-B Exchange via RapidAPI for real-time flight data
8. [ ] (Optional) Set up Google Cloud project for BigQuery GDELT access

---

## Environment Variables

```bash
# Required
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Optional (for enhanced data sources)
GOOGLE_CLOUD_PROJECT=your_project_id     # For BigQuery GDELT queries
RAPIDAPI_KEY=your_key                    # For ADS-B Exchange API
OPENSKY_USERNAME=your_username           # For OpenSky historical data
OPENSKY_PASSWORD=your_password
POLYMARKET_PRIVATE_KEY=your_key          # For placing trades (not data fetching)
```

---

*For detailed research context and edge thesis, see [RESEARCH.md](RESEARCH.md).*
