"""
Polymarket API client for fetching market data.

Uses two APIs:
- CLOB API (https://clob.polymarket.com): Order books, prices, trades
- Gamma API (https://gamma-api.polymarket.com): Market metadata, categories, events

Adapted from EsportsBetting/BettingMarkets - removed esports-specific methods.
"""
import logging
from datetime import datetime
from typing import Optional
import requests

from .models import Market, Event, Outcome, OrderBook, OrderBookLevel, Trade, PriceHistory

logger = logging.getLogger(__name__)


class PolymarketClient:
    """
    Client for Polymarket CLOB and Gamma APIs.
    Read-only access - no authentication required for market data.
    """

    CLOB_BASE_URL = "https://clob.polymarket.com"
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ElonTweetMarkets/1.0"
        })

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    # ========== Gamma API Methods (Market Metadata) ==========

    def get_markets(self, limit: int = 100, offset: int = 0, active: bool = True,
                    closed: bool = False, order: str = "volume24hr",
                    ascending: bool = False) -> list[Market]:
        params = {
            "limit": min(limit, 100), "offset": offset,
            "active": str(active).lower(), "closed": str(closed).lower(),
            "order": order, "ascending": str(ascending).lower(),
        }
        data = self._get(f"{self.GAMMA_BASE_URL}/markets", params)
        return [self._parse_market(m) for m in data]

    def get_all_markets(self, max_markets: int = 1000, **kwargs) -> list[Market]:
        markets = []
        offset = 0
        limit = 100
        while len(markets) < max_markets:
            batch = self.get_markets(limit=limit, offset=offset, **kwargs)
            if not batch:
                break
            markets.extend(batch)
            offset += limit
            if len(batch) < limit:
                break
        return markets[:max_markets]

    def get_market(self, market_id: str) -> Market:
        data = self._get(f"{self.GAMMA_BASE_URL}/markets/{market_id}")
        return self._parse_market(data)

    def get_events(self, limit: int = 100, offset: int = 0, active: bool = True,
                   closed: bool = False) -> list[Event]:
        params = {
            "limit": min(limit, 100), "offset": offset,
            "active": str(active).lower(), "closed": str(closed).lower(),
        }
        data = self._get(f"{self.GAMMA_BASE_URL}/events", params)
        return [self._parse_event(e) for e in data]

    def search_markets(self, query: str, limit: int = 50) -> list[Market]:
        all_markets = self.get_all_markets(max_markets=500)
        query_lower = query.lower()
        matches = [
            m for m in all_markets
            if query_lower in m.question.lower() or
               query_lower in m.description.lower() or
               any(query_lower in tag.lower() for tag in m.tags)
        ]
        return matches[:limit]

    # ========== CLOB API Methods (Order Book, Prices) ==========

    def get_order_book(self, token_id: str) -> OrderBook:
        data = self._get(f"{self.CLOB_BASE_URL}/book", {"token_id": token_id})
        bids = [OrderBookLevel(price=float(b["price"]), size=float(b["size"])) for b in data.get("bids", [])]
        asks = [OrderBookLevel(price=float(a["price"]), size=float(a["size"])) for a in data.get("asks", [])]
        return OrderBook(
            token_id=token_id,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            timestamp=datetime.now()
        )

    def get_midpoint(self, token_id: str) -> float:
        data = self._get(f"{self.CLOB_BASE_URL}/midpoint", {"token_id": token_id})
        return float(data.get("mid", 0))

    def get_price(self, token_id: str, side: str = "buy") -> float:
        data = self._get(f"{self.CLOB_BASE_URL}/price", {"token_id": token_id, "side": side.upper()})
        return float(data.get("price", 0))

    def get_last_trade_price(self, token_id: str) -> float:
        data = self._get(f"{self.CLOB_BASE_URL}/last-trade-price", {"token_id": token_id})
        return float(data.get("price", 0))

    def get_price_history(self, token_id: str, start_ts: Optional[int] = None,
                          end_ts: Optional[int] = None, fidelity: int = 60) -> list[PriceHistory]:
        params = {"token_id": token_id, "fidelity": fidelity}
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts
        data = self._get(f"{self.CLOB_BASE_URL}/prices-history", params)
        return [PriceHistory(timestamp=datetime.fromtimestamp(p["t"]), price=float(p["p"]))
                for p in data.get("history", [])]

    # ========== Parsing Helpers ==========

    def _parse_market(self, data: dict) -> Market:
        outcomes = []
        token_ids = data.get("clobTokenIds", [])
        prices = data.get("outcomePrices", [])
        outcome_names = data.get("outcomes", ["Yes", "No"])

        if isinstance(token_ids, str):
            import json
            token_ids = json.loads(token_ids)
        if isinstance(prices, str):
            import json
            prices = json.loads(prices)

        for i, (name, token_id) in enumerate(zip(outcome_names, token_ids)):
            price = float(prices[i]) if i < len(prices) else 0.0
            outcomes.append(Outcome(name=str(name), token_id=str(token_id), price=price))

        end_date = None
        if data.get("endDate"):
            try:
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        tags = []
        if data.get("tags"):
            tags = [t.get("label", t.get("slug", "")) for t in data["tags"] if isinstance(t, dict)]

        return Market(
            id=data.get("id", ""), question=data.get("question", ""),
            description=data.get("description", ""), outcomes=outcomes,
            volume=float(data.get("volume", 0) or 0),
            volume_24h=float(data.get("volume24hr", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            end_date=end_date, active=data.get("active", True),
            closed=data.get("closed", False), category=data.get("category", ""),
            tags=tags,
        )

    def _parse_event(self, data: dict) -> Event:
        markets = [self._parse_market(m) for m in data.get("markets", [])]
        tags = []
        if data.get("tags"):
            tags = [t.get("label", t.get("slug", "")) for t in data["tags"] if isinstance(t, dict)]
        start_date = end_date = None
        if data.get("startDate"):
            try:
                start_date = datetime.fromisoformat(data["startDate"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        if data.get("endDate"):
            try:
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        return Event(
            id=data.get("id", ""), title=data.get("title", ""),
            slug=data.get("slug", ""), description=data.get("description", ""),
            markets=markets, category=data.get("category", ""), tags=tags,
            volume=float(data.get("volume", 0) or 0),
            start_date=start_date, end_date=end_date, active=data.get("active", True),
        )
