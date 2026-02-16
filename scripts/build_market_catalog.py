"""
Build a flat parquet market catalog from Polymarket JSON files.

Parses three data sources into one row per bucket per event:
  1. tweet_events_comprehensive.json  (112 negRisk Elon events, Sept 2024 - Feb 2026)
  2. elon_tweet_markets_full.json     (21 early Elon categoricals, Apr-Sept 2024)
  3. elon_tweet_clob_markets.json     (CLOB supplement with winner flags + gap events)

Output: data/processed/market_catalog.parquet  (+ summary JSON)

Usage: python scripts/build_market_catalog.py
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "data" / "sources" / "polymarket"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

COMPREHENSIVE_PATH = SOURCE_DIR / "tweet_events_comprehensive.json"
FULL_PATH = SOURCE_DIR / "elon_tweet_markets_full.json"
CLOB_PATH = SOURCE_DIR / "elon_tweet_clob_markets.json"

# ---------------------------------------------------------------------------
# Exclusion lists
# ---------------------------------------------------------------------------
EXCLUDE_NON_ELON_IDS = {"4681", "6120", "5178", "3250", "3573", "3854", "11984", "13352"}
EXCLUDE_NON_CATEGORICAL_IDS = {"5087", "4987", "4407", "4905", "4742"}
EXCLUDE_IDS = EXCLUDE_NON_ELON_IDS | EXCLUDE_NON_CATEGORICAL_IDS


# ---------------------------------------------------------------------------
# Bucket boundary parsing
# ---------------------------------------------------------------------------
def parse_bucket_bounds(label: str) -> tuple[int, int]:
    """Parse a groupItemTitle string into (lower_bound, upper_bound).

    Examples:
        "<20"         -> (0, 19)
        "20-39"       -> (20, 39)
        "740+"        -> (740, 99999)
        "Less than 50"-> (0, 49)
        "180 or more" -> (180, 99999)
        "100\u201399" -> en-dash variant of 100-139
    """
    s = label.strip()

    # Normalise en-dash / em-dash to ASCII hyphen
    s = s.replace("\u2013", "-").replace("\u2014", "-")

    # "<N" pattern  e.g. "<20", "<150"
    m = re.match(r"^<\s*(\d+)$", s)
    if m:
        return (0, int(m.group(1)) - 1)

    # "Less than N"
    m = re.match(r"^[Ll]ess\s+than\s+(\d+)$", s)
    if m:
        return (0, int(m.group(1)) - 1)

    # "N+" pattern  e.g. "740+", "1400+"
    m = re.match(r"^(\d+)\+$", s)
    if m:
        return (int(m.group(1)), 99999)

    # "N or more"
    m = re.match(r"^(\d+)\s+or\s+more$", s, re.IGNORECASE)
    if m:
        return (int(m.group(1)), 99999)

    # "N-M" range   e.g. "20-39", "260-279"
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # "N to M"
    m = re.match(r"^(\d+)\s+to\s+(\d+)$", s, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # Fallback: try extracting from question text later; return sentinel
    return (-1, -1)


def parse_bucket_from_question(question: str) -> tuple[str, int, int]:
    """Fallback: parse bucket label + bounds from the market question text."""
    q = question.strip()
    # Normalise en-dash / em-dash
    q_norm = q.replace("\u2013", "-").replace("\u2014", "-")

    # "post 0-19 tweets" or "tweet 60-79 times"
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(?:tweets|times|posts)", q_norm, re.IGNORECASE)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (f"{lo}-{hi}", lo, hi)

    # "tweet 275-299 October" - range followed by month/date (no "times" keyword)
    m = re.search(r"(?:tweet|post)\s+(\d+)\s*-\s*(\d+)\s+\w+", q_norm, re.IGNORECASE)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (f"{lo}-{hi}", lo, hi)

    # "less than 60 times" / "less than 100 tweets"
    m = re.search(r"less\s+than\s+(\d+)\s*(?:tweets|times|posts)?", q_norm, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return (f"<{n}", 0, n - 1)

    # "more than N times" e.g. "more than 115 times"
    m = re.search(r"more\s+than\s+(\d+)\s*(?:tweets|times|posts)?", q_norm, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return (f"{n + 1}+", n + 1, 99999)

    # "N or more times/tweets"
    m = re.search(r"(\d+)\s+or\s+more\s*(?:tweets|times|posts)?", q_norm, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return (f"{n}+", n, 99999)

    # "N+ tweets/times"  e.g. "80+ tweets"
    m = re.search(r"(\d+)\+\s*(?:tweets|times|posts)?", q_norm, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return (f"{n}+", n, 99999)

    # "between N and M times"
    m = re.search(r"between\s+(\d[\d,]*)\s+and\s+(\d[\d,]*)\s*(?:tweets|times|posts)?", q_norm, re.IGNORECASE)
    if m:
        lo = int(m.group(1).replace(",", ""))
        hi = int(m.group(2).replace(",", ""))
        return (f"{lo}-{hi}", lo, hi)

    # "have N or more tweets"  (cumulative style from early markets)
    m = re.search(r"have\s+([\d,]+)\s+or\s+more\s+tweets", q_norm, re.IGNORECASE)
    if m:
        n = int(m.group(1).replace(",", ""))
        return (f"{n}+", n, 99999)

    # "have N or fewer tweets"
    m = re.search(r"have\s+([\d,]+)\s+or\s+fewer\s+tweets", q_norm, re.IGNORECASE)
    if m:
        n = int(m.group(1).replace(",", ""))
        return (f"<={n}", 0, n)

    # "have N-M tweets"
    m = re.search(r"have\s+([\d,]+)\s*-\s*([\d,]+)\s+tweets", q_norm, re.IGNORECASE)
    if m:
        lo = int(m.group(1).replace(",", ""))
        hi = int(m.group(2).replace(",", ""))
        return (f"{lo}-{hi}", lo, hi)

    return ("unknown", -1, -1)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def safe_json_loads(val):
    """Parse a JSON string, returning empty list on failure."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def safe_float(val, default=0.0):
    """Convert to float safely."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def parse_iso(val):
    """Parse an ISO datetime string to a timezone-aware datetime (UTC)."""
    if not val:
        return None
    try:
        s = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def compute_market_type(duration_days: int) -> str:
    """Classify market window by duration."""
    if duration_days <= 3:
        return "daily"
    elif duration_days <= 5:
        return "short"
    elif duration_days <= 10:
        return "weekly"
    else:
        return "monthly"


# ---------------------------------------------------------------------------
# Event filtering
# ---------------------------------------------------------------------------
def is_elon_tweet_categorical(event: dict, min_markets: int = 5) -> bool:
    """Determine if an event is an Elon Musk categorical tweet-count event."""
    eid = str(event.get("id", ""))
    if eid in EXCLUDE_IDS:
        return False

    title = event.get("title", "").lower()
    slug = event.get("slug", event.get("ticker", "")).lower()

    # Must reference Elon
    has_elon = "elon" in title or "elon" in slug or "elonmusk" in title or "elonmusk" in slug

    if not has_elon:
        return False

    # Must reference tweets/posts
    has_tweet = any(kw in title for kw in ["tweet", "post", "# of", "# tweet", "number of"])
    if not has_tweet:
        return False

    # Must have enough markets (buckets)
    markets = event.get("markets", [])
    if len(markets) < min_markets:
        return False

    return True


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def build_rows_from_event(event: dict, source_file: str, clob_lookup: dict) -> list[dict]:
    """Build one row per bucket from an event dict."""
    rows = []
    eid = str(event["id"])
    title = event.get("title", "")
    slug = event.get("slug", event.get("ticker", ""))

    # Dates
    start_dt = parse_iso(event.get("startDate"))
    end_dt = parse_iso(event.get("endDate"))

    # Compute duration
    duration_days = None
    if start_dt and end_dt:
        duration_days = max(1, (end_dt - start_dt).days)

    market_type = compute_market_type(duration_days) if duration_days else None

    # Event-level
    is_resolved = bool(event.get("closed", False))
    event_volume = safe_float(event.get("volume", 0))
    markets = event.get("markets", [])
    n_buckets = len(markets)

    for idx, mkt in enumerate(markets):
        # Bucket label
        label = (mkt.get("groupItemTitle") or "").strip()
        lower, upper = (-1, -1)
        if label:
            lower, upper = parse_bucket_bounds(label)

        # If parsing failed, try question text
        if lower == -1 and upper == -1:
            q = mkt.get("question", "")
            fallback_label, lower, upper = parse_bucket_from_question(q)
            if not label:
                label = fallback_label

        # Outcome prices
        prices = safe_json_loads(mkt.get("outcomePrices", "[]"))
        price_yes = safe_float(prices[0]) if len(prices) > 0 else None
        price_no = safe_float(prices[1]) if len(prices) > 1 else None

        # CLOB token IDs
        clob_tokens = safe_json_loads(mkt.get("clobTokenIds", "[]"))
        token_yes = clob_tokens[0] if len(clob_tokens) > 0 else None
        token_no = clob_tokens[1] if len(clob_tokens) > 1 else None

        # Winner detection from prices
        is_winner = False
        if price_yes is not None and price_yes >= 0.99:
            is_winner = True

        # Cross-reference CLOB winner flag if available
        cond_id = mkt.get("conditionId", "")
        if cond_id in clob_lookup:
            clob_entry = clob_lookup[cond_id]
            for tok in clob_entry.get("tokens", []):
                if tok.get("outcome") == "Yes" and tok.get("winner") is True:
                    is_winner = True
                elif tok.get("outcome") == "Yes" and tok.get("winner") is False:
                    # Only override if we also have no price-based signal
                    if price_yes is not None and price_yes < 0.99:
                        is_winner = False

        rows.append({
            "event_id": eid,
            "event_slug": slug,
            "event_title": title,
            "start_date": start_dt,
            "end_date": end_dt,
            "market_type": market_type,
            "duration_days": duration_days,
            "n_buckets": n_buckets,
            "is_resolved": is_resolved,
            "event_volume": event_volume,
            "bucket_idx": idx,
            "bucket_label": label,
            "lower_bound": lower,
            "upper_bound": upper,
            "market_id": str(mkt.get("id", "")),
            "condition_id": cond_id,
            "token_id_yes": token_yes,
            "token_id_no": token_no,
            "price_yes": price_yes,
            "price_no": price_no,
            "is_winner": is_winner,
            "bucket_volume": safe_float(mkt.get("volume", 0)),
            "source_file": source_file,
        })

    return rows


def build_rows_from_clob_group(
    neg_risk_market_id: str,
    clob_entries: list[dict],
    source_file: str,
) -> list[dict]:
    """Build rows from a CLOB-only group (no matching event file entry).

    CLOB entries lack event-level fields, so we synthesise from market-level data.
    """
    rows = []
    n_buckets = len(clob_entries)

    # Try to get a date from the first entry
    sample = clob_entries[0]
    end_dt = parse_iso(sample.get("end_date_iso"))
    # CLOB doesn't have start date; try to infer from question
    start_dt = None
    q_sample = sample.get("question", "")
    # "Sept 6-13?" -> we need start from question
    # We'll try to parse dates from questions later if needed

    # Derive a pseudo event title and slug from the questions
    # Look for common patterns
    titles = [c.get("question", "") for c in clob_entries]
    # Find common date range mentioned
    event_title = _infer_event_title(titles)
    start_dt, end_dt_parsed = _infer_dates_from_questions(titles, end_dt)
    if end_dt_parsed:
        end_dt = end_dt_parsed

    duration_days = None
    if start_dt and end_dt:
        duration_days = max(1, (end_dt - start_dt).days)
    market_type = compute_market_type(duration_days) if duration_days else None

    # Check if resolved (all closed)
    is_resolved = all(c.get("closed", False) for c in clob_entries)

    # Sort entries by bucket bounds for consistent ordering
    parsed_entries = []
    for c in clob_entries:
        q = c.get("question", "")
        label, lower, upper = parse_bucket_from_question(q)
        parsed_entries.append((lower, upper, label, c))
    parsed_entries.sort(key=lambda x: (x[0] if x[0] >= 0 else 999999, x[1]))

    for idx, (lower, upper, label, c) in enumerate(parsed_entries):
        # Tokens
        tokens = c.get("tokens", [])
        token_yes = None
        token_no = None
        price_yes = None
        price_no = None
        is_winner = False
        for tok in tokens:
            if tok.get("outcome") == "Yes":
                token_yes = tok.get("token_id")
                price_yes = safe_float(tok.get("price"))
                if tok.get("winner") is True:
                    is_winner = True
            elif tok.get("outcome") == "No":
                token_no = tok.get("token_id")
                price_no = safe_float(tok.get("price"))

        rows.append({
            "event_id": f"clob_{neg_risk_market_id[:16]}",
            "event_slug": c.get("market_slug", ""),
            "event_title": event_title,
            "start_date": start_dt,
            "end_date": end_dt,
            "market_type": market_type,
            "duration_days": duration_days,
            "n_buckets": n_buckets,
            "is_resolved": is_resolved,
            "event_volume": None,  # Not available from CLOB
            "bucket_idx": idx,
            "bucket_label": label,
            "lower_bound": lower,
            "upper_bound": upper,
            "market_id": f"clob_{c.get('condition_id', '')[:16]}",
            "condition_id": c.get("condition_id", ""),
            "token_id_yes": token_yes,
            "token_id_no": token_no,
            "price_yes": price_yes,
            "price_no": price_no,
            "is_winner": is_winner,
            "bucket_volume": None,  # Not available from CLOB
            "source_file": source_file,
        })

    return rows


def _infer_event_title(questions: list[str]) -> str:
    """Derive an event title from a list of bucket questions."""
    # Find date info from questions
    for q in questions:
        q_norm = q.replace("\u2013", "-").replace("\u2014", "-")
        # "from December 12 to December 19, 2025?"
        m = re.search(r"(?:from|between)\s+(.+?\d{4})", q_norm, re.IGNORECASE)
        if m:
            return f"Elon Musk tweets {m.group(1)}"
        # "Oct 25 - Nov 1?"
        m = re.search(r"times?\s+(.+?)(?:\?|$)", q_norm, re.IGNORECASE)
        if m:
            return f"Elon Musk # tweets {m.group(1).strip().rstrip('?')}"
        # "tweets on September 13, 2025?"
        m = re.search(r"(?:tweets|posts)\s+(?:on|in)\s+(.+?)(?:\?|$)", q_norm, re.IGNORECASE)
        if m:
            return f"Elon Musk # tweets {m.group(1).strip().rstrip('?')}"
    return "Elon Musk # tweets (CLOB)"


def _infer_dates_from_questions(
    questions: list[str], end_dt_fallback
) -> tuple:
    """Try to infer start and end dates from question text."""
    for q in questions:
        q_norm = q.replace("\u2013", "-").replace("\u2014", "-")

        # "from September 18 to September 19, 2025"
        m = re.search(
            r"from\s+(\w+\s+\d+)\s+to\s+(\w+\s+\d+),?\s*(\d{4})",
            q_norm, re.IGNORECASE,
        )
        if m:
            year = m.group(3)
            try:
                start = datetime.strptime(f"{m.group(1)} {year}", "%B %d %Y").replace(tzinfo=timezone.utc)
                end = datetime.strptime(f"{m.group(2)} {year}", "%B %d %Y").replace(tzinfo=timezone.utc)
                return start, end
            except ValueError:
                pass

        # "September 6-13?" or "Sept 6-13?"  with optional year
        m = re.search(
            r"(?:on|in|times?|posts?|tweets?)\s+(\w+)\s+(\d+)\s*-\s*(\d+),?\s*(\d{4})?",
            q_norm, re.IGNORECASE,
        )
        if m:
            month_str = m.group(1)
            d1 = m.group(2)
            d2 = m.group(3)
            year = m.group(4)
            if not year and end_dt_fallback:
                year = str(end_dt_fallback.year)
            if year:
                for fmt in ["%B", "%b"]:
                    try:
                        start = datetime.strptime(f"{month_str} {d1} {year}", f"{fmt} %d %Y").replace(tzinfo=timezone.utc)
                        end = datetime.strptime(f"{month_str} {d2} {year}", f"{fmt} %d %Y").replace(tzinfo=timezone.utc)
                        return start, end
                    except ValueError:
                        continue

        # "Oct 25 - Nov 1?"
        m = re.search(
            r"(\w+)\s+(\d+)\s*-\s*(\w+)\s+(\d+),?\s*(\d{4})?",
            q_norm, re.IGNORECASE,
        )
        if m:
            month1, d1, month2, d2 = m.group(1), m.group(2), m.group(3), m.group(4)
            year = m.group(5)
            if not year and end_dt_fallback:
                year = str(end_dt_fallback.year)
            if year:
                for fmt in ["%B", "%b"]:
                    try:
                        start = datetime.strptime(f"{month1} {d1} {year}", f"{fmt} %d %Y").replace(tzinfo=timezone.utc)
                        end = datetime.strptime(f"{month2} {d2} {year}", f"{fmt} %d %Y").replace(tzinfo=timezone.utc)
                        return start, end
                    except ValueError:
                        continue

        # "on September 13, 2025" (single day)
        m = re.search(
            r"on\s+(\w+\s+\d+),?\s*(\d{4})",
            q_norm, re.IGNORECASE,
        )
        if m:
            year = m.group(2)
            try:
                dt = datetime.strptime(f"{m.group(1)} {year}", "%B %d %Y").replace(tzinfo=timezone.utc)
                return dt, dt
            except ValueError:
                pass

        # "in October 2025" / "in November 2025" (monthly)
        m = re.search(r"in\s+(\w+)\s+(\d{4})", q_norm, re.IGNORECASE)
        if m:
            month_str, year = m.group(1), m.group(2)
            for fmt in ["%B", "%b"]:
                try:
                    start = datetime.strptime(f"{month_str} 1 {year}", f"{fmt} %d %Y").replace(tzinfo=timezone.utc)
                    # End of month: use first of next month
                    if start.month == 12:
                        end = datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc)
                    else:
                        end = datetime(start.year, start.month + 1, 1, tzinfo=timezone.utc)
                    return start, end
                except ValueError:
                    continue

        # "September 1-30?" (monthly-ish)
        m = re.search(
            r"(\w+)\s+(\d+)\s*-\s*(\d+),?\s*(\d{4})?",
            q_norm, re.IGNORECASE,
        )
        if m:
            month_str, d1, d2 = m.group(1), m.group(2), m.group(3)
            year = m.group(4)
            if not year and end_dt_fallback:
                year = str(end_dt_fallback.year)
            if year:
                for fmt in ["%B", "%b"]:
                    try:
                        start = datetime.strptime(f"{month_str} {d1} {year}", f"{fmt} %d %Y").replace(tzinfo=timezone.utc)
                        end = datetime.strptime(f"{month_str} {d2} {year}", f"{fmt} %d %Y").replace(tzinfo=timezone.utc)
                        return start, end
                    except ValueError:
                        continue

    return None, None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load CLOB data for winner cross-referencing
    # ------------------------------------------------------------------
    print("Loading CLOB supplement...")
    clob_lookup = {}  # condition_id -> entry
    clob_groups = defaultdict(list)  # neg_risk_market_id -> [entries]

    with open(CLOB_PATH, encoding="utf-8") as f:
        clob_data = json.load(f)

    for entry in clob_data:
        cid = entry.get("condition_id", "")
        clob_lookup[cid] = entry
        if entry.get("neg_risk"):
            nrmid = entry.get("neg_risk_market_id", "")
            if nrmid:
                clob_groups[nrmid].append(entry)

    print(f"  CLOB: {len(clob_data)} markets, {len(clob_groups)} neg_risk groups")

    # ------------------------------------------------------------------
    # 2. Process comprehensive events (PRIMARY)
    # ------------------------------------------------------------------
    print("\nProcessing comprehensive events...")
    with open(COMPREHENSIVE_PATH, encoding="utf-8") as f:
        comp_events = json.load(f)

    all_rows = []
    seen_event_ids = set()
    comp_condition_ids = set()

    comp_accepted = 0
    for event in comp_events:
        if not is_elon_tweet_categorical(event):
            continue
        eid = str(event["id"])
        seen_event_ids.add(eid)
        comp_accepted += 1

        # Track condition IDs for dedup with CLOB
        for m in event.get("markets", []):
            comp_condition_ids.add(m.get("conditionId", ""))

        rows = build_rows_from_event(event, "tweet_events_comprehensive.json", clob_lookup)
        all_rows.extend(rows)

    print(f"  Accepted: {comp_accepted} events, {len(all_rows)} bucket rows")

    # ------------------------------------------------------------------
    # 3. Process full markets file (SECONDARY - only new events)
    # ------------------------------------------------------------------
    print("\nProcessing full markets file (secondary)...")
    with open(FULL_PATH, encoding="utf-8") as f:
        full_data = json.load(f)

    full_events = full_data.get("events", [])
    full_accepted = 0
    full_rows_start = len(all_rows)

    for event in full_events:
        eid = str(event["id"])
        if eid in seen_event_ids:
            continue  # Already in comprehensive
        if not is_elon_tweet_categorical(event):
            continue
        seen_event_ids.add(eid)
        full_accepted += 1

        # Track condition IDs
        for m in event.get("markets", []):
            comp_condition_ids.add(m.get("conditionId", ""))

        rows = build_rows_from_event(event, "elon_tweet_markets_full.json", clob_lookup)
        all_rows.extend(rows)

    full_rows = len(all_rows) - full_rows_start
    print(f"  Accepted: {full_accepted} new events, {full_rows} bucket rows")

    # ------------------------------------------------------------------
    # 4. Process CLOB-only groups (gap events not in either event file)
    # ------------------------------------------------------------------
    print("\nProcessing CLOB-only groups...")
    clob_only_count = 0
    clob_rows_start = len(all_rows)

    for nrmid, entries in clob_groups.items():
        # Skip if all condition_ids are already covered by event files
        entry_cids = {e["condition_id"] for e in entries}
        if entry_cids.issubset(comp_condition_ids):
            continue

        # Filter to only Elon tweet markets
        elon_entries = []
        for e in entries:
            q = e.get("question", "").lower()
            if ("elon" in q or "elonmusk" in q) and ("tweet" in q or "post" in q or "times" in q):
                elon_entries.append(e)

        if len(elon_entries) < 5:
            continue

        clob_only_count += 1
        rows = build_rows_from_clob_group(nrmid, elon_entries, "elon_tweet_clob_markets.json")
        all_rows.extend(rows)

    clob_rows = len(all_rows) - clob_rows_start
    print(f"  Accepted: {clob_only_count} CLOB-only event groups, {clob_rows} bucket rows")

    # ------------------------------------------------------------------
    # 5. Build DataFrame and clean up
    # ------------------------------------------------------------------
    print(f"\nTotal rows before dedup: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    # Deduplicate by (event_id, bucket_label)
    # For resolved events: prefer comprehensive (most complete historical data)
    # For unresolved events: prefer full (freshest token IDs from API)
    source_priority_resolved = {
        "tweet_events_comprehensive.json": 0,
        "elon_tweet_markets_full.json": 1,
        "elon_tweet_clob_markets.json": 2,
    }
    source_priority_unresolved = {
        "elon_tweet_markets_full.json": 0,
        "tweet_events_comprehensive.json": 1,
        "elon_tweet_clob_markets.json": 2,
    }
    df["_priority"] = df.apply(
        lambda row: source_priority_resolved.get(row["source_file"], 9)
        if row["is_resolved"]
        else source_priority_unresolved.get(row["source_file"], 9),
        axis=1,
    )
    df = df.sort_values("_priority").drop_duplicates(
        subset=["event_id", "bucket_idx"], keep="first"
    )
    df = df.drop(columns=["_priority"])

    # Sort by start_date, then bucket_idx
    df = df.sort_values(["start_date", "event_id", "bucket_idx"]).reset_index(drop=True)

    # Type enforcement
    int_cols = ["duration_days", "n_buckets", "bucket_idx", "lower_bound", "upper_bound"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = ["event_volume", "price_yes", "price_no", "bucket_volume"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    bool_cols = ["is_resolved", "is_winner"]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    print(f"Total rows after dedup: {len(df)}")

    # ------------------------------------------------------------------
    # 6. Validation
    # ------------------------------------------------------------------
    print("\n--- Validation ---")

    # Unique events
    n_events = df["event_id"].nunique()
    print(f"Unique events: {n_events}")

    # Resolved events
    resolved_events = df[df["is_resolved"]]["event_id"].unique()
    print(f"Resolved events: {len(resolved_events)}")

    # Winner coverage for resolved events
    winner_issues = []
    for eid in resolved_events:
        edf = df[df["event_id"] == eid]
        n_winners = edf["is_winner"].sum()
        if n_winners != 1:
            winner_issues.append((eid, n_winners, edf["event_title"].iloc[0][:50]))
    if winner_issues:
        print(f"  WARNING: {len(winner_issues)} resolved events without exactly 1 winner:")
        for eid, nw, title in winner_issues[:10]:
            print(f"    event_id={eid}, winners={nw}, title={title}")
    else:
        print("  All resolved events have exactly 1 winner bucket.")

    # Check for duplicate event+bucket
    dups = df.duplicated(subset=["event_id", "bucket_idx"], keep=False)
    n_dups = dups.sum()
    print(f"Duplicate (event_id, bucket_idx) rows: {n_dups}")

    # Unparsed buckets
    unparsed = df[(df["lower_bound"] == -1) | (df["upper_bound"] == -1)]
    print(f"Unparsed bucket boundaries: {len(unparsed)}")
    if len(unparsed) > 0:
        for _, row in unparsed.head(5).iterrows():
            print(f"    label='{row['bucket_label']}', event={row['event_title'][:50]}")

    # Date range
    valid_starts = df["start_date"].dropna()
    if len(valid_starts) > 0:
        print(f"Date range: {valid_starts.min().strftime('%Y-%m-%d')} to {valid_starts.max().strftime('%Y-%m-%d')}")

    # Market type distribution
    print(f"\nMarket type distribution (events):")
    type_dist = df.groupby("market_type")["event_id"].nunique()
    for mt, cnt in type_dist.items():
        print(f"  {mt}: {cnt} events")

    # Spot-check bucket parsing
    print(f"\nBucket parsing spot-check (first 5 rows):")
    for _, row in df.head(5).iterrows():
        print(f"  '{row['bucket_label']}' -> [{row['lower_bound']}, {row['upper_bound']}]")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    parquet_path = OUTPUT_DIR / "market_catalog.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"\nSaved: {parquet_path} ({len(df)} rows, {parquet_path.stat().st_size / 1024:.1f} KB)")

    # Summary JSON
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_events": int(n_events),
        "total_buckets": len(df),
        "resolved_events": int(len(resolved_events)),
        "unresolved_events": int(n_events - len(resolved_events)),
        "winner_issues": len(winner_issues),
        "unparsed_buckets": int(len(unparsed)),
        "date_range": {
            "start": valid_starts.min().isoformat() if len(valid_starts) > 0 else None,
            "end": valid_starts.max().isoformat() if len(valid_starts) > 0 else None,
        },
        "market_type_distribution": {
            mt: int(cnt) for mt, cnt in type_dist.items()
        },
        "source_file_distribution": {
            src: int(cnt) for src, cnt in df["source_file"].value_counts().items()
        },
        "events_from_comprehensive": comp_accepted,
        "events_from_full": full_accepted,
        "events_from_clob_only": clob_only_count,
    }

    summary_path = OUTPUT_DIR / "market_catalog_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
