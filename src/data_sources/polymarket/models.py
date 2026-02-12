"""
Polymarket data models for markets, events, and orderbooks.
Adapted from EsportsBetting/BettingMarkets - removed esports-specific fields.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Outcome:
    """Single outcome in a market (YES/NO or named buckets)."""
    name: str
    token_id: str
    price: float  # 0.00 to 1.00

    @property
    def implied_probability(self) -> float:
        return self.price


@dataclass
class Market:
    """Polymarket prediction market."""
    id: str
    question: str
    description: str
    outcomes: list[Outcome]
    volume: float
    volume_24h: float
    liquidity: float
    end_date: Optional[datetime]
    active: bool
    closed: bool
    category: str
    tags: list[str] = field(default_factory=list)

    # Generic market metadata
    market_type: Optional[str] = None  # e.g., "elon_tweets", "weather", "crypto"

    @property
    def spread(self) -> float:
        if len(self.outcomes) >= 2:
            return abs(self.outcomes[0].price - self.outcomes[1].price)
        return 0.0

    @property
    def n_outcomes(self) -> int:
        return len(self.outcomes)


@dataclass
class Event:
    """Polymarket event containing one or more markets."""
    id: str
    title: str
    slug: str
    description: str
    markets: list[Market]
    category: str
    tags: list[str]
    volume: float
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    active: bool


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def midpoint(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class Trade:
    id: str
    token_id: str
    price: float
    size: float
    side: str
    timestamp: datetime
    market_id: str


@dataclass
class PriceHistory:
    timestamp: datetime
    price: float
    volume: Optional[float] = None
