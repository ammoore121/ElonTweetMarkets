"""
Polymarket Order Placement Client

Copied as-is from EsportsBetting/BettingMarkets (fully game-agnostic).
Supports limit orders, market orders (FOK), and dry-run mode.
"""

import os
import time
from datetime import datetime
from typing import Optional, Literal
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType, MarketOrderArgs
    HAS_CLOB_CLIENT = True
except ImportError:
    HAS_CLOB_CLIENT = False
    ClobClient = None

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137


@dataclass
class MarketState:
    token_id: str
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None
    has_liquidity: bool = False
    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.utcnow()
        if self.best_bid and self.best_ask:
            self.spread = self.best_ask - self.best_bid
            mid = (self.best_ask + self.best_bid) / 2
            self.spread_pct = self.spread / mid if mid > 0 else None
            self.has_liquidity = self.spread < 0.10


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    token_id: str = ""
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    shares: float = 0.0
    dry_run: bool = True
    error: Optional[str] = None
    market_state: Optional[MarketState] = None
    placed_at: datetime = None

    def __post_init__(self):
        if self.placed_at is None:
            self.placed_at = datetime.utcnow()


class PolymarketOrderClient:
    def __init__(self, private_key: str = None):
        if not HAS_CLOB_CLIENT:
            raise ImportError("py-clob-client not installed. Run: pip install py-clob-client")
        self.private_key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not self.private_key:
            raise ValueError("Private key required. Set POLYMARKET_PRIVATE_KEY env var.")
        self.client = ClobClient(host=CLOB_HOST, key=self.private_key, chain_id=POLYGON_CHAIN_ID)
        self._api_creds = None
        self._init_credentials()

    def _init_credentials(self):
        try:
            self._api_creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(self._api_creds)
            print("[OK] Polymarket order client initialized")
        except Exception as e:
            print(f"[WARN] Could not initialize API credentials: {e}")

    def get_market_state(self, token_id: str) -> MarketState:
        try:
            book = self.client.get_order_book(token_id)
            best_bid = float(book.bids[0].price) if book.bids else None
            best_ask = float(book.asks[0].price) if book.asks else None
            bid_size = float(book.bids[0].size) if book.bids else None
            ask_size = float(book.asks[0].size) if book.asks else None
            return MarketState(token_id=token_id, best_bid=best_bid, best_ask=best_ask,
                             bid_size=bid_size, ask_size=ask_size)
        except Exception as e:
            print(f"[ERROR] Failed to get market state: {e}")
            return MarketState(token_id=token_id)

    def place_order(self, token_id: str, side: Literal["BUY", "SELL"], price: float,
                    size: float, dry_run: bool = True) -> OrderResult:
        if not (0.01 <= price <= 0.99):
            return OrderResult(success=False, token_id=token_id, side=side, price=price,
                             size=size, dry_run=dry_run, error=f"Price {price} out of range")
        if size <= 0:
            return OrderResult(success=False, token_id=token_id, side=side, price=price,
                             size=size, dry_run=dry_run, error="Size must be positive")
        market_state = self.get_market_state(token_id)
        shares = size / price
        if dry_run:
            print(f"[DRY RUN] {side} {shares:.2f} shares @ {price*100:.0f}c = ${size:.2f}")
            return OrderResult(success=True, order_id="DRY_RUN_" + str(int(time.time())),
                             token_id=token_id, side=side, price=price, size=size,
                             shares=shares, dry_run=True, market_state=market_state)
        try:
            order_args = OrderArgs(token_id=token_id, price=price, size=shares,
                                  side=side, fee_rate_bps=0)
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTC)
            order_id = response.get("orderID") or response.get("order_id")
            return OrderResult(success=True, order_id=order_id, token_id=token_id,
                             side=side, price=price, size=size, shares=shares,
                             dry_run=False, market_state=market_state)
        except Exception as e:
            return OrderResult(success=False, token_id=token_id, side=side, price=price,
                             size=size, shares=shares, dry_run=False, error=str(e),
                             market_state=market_state)

    def place_market_order(self, token_id: str, side: Literal["BUY", "SELL"],
                           amount: float, dry_run: bool = True) -> OrderResult:
        if amount <= 0:
            return OrderResult(success=False, token_id=token_id, side=side, price=0.0,
                             size=amount, dry_run=dry_run, error="Amount must be positive")
        market_state = self.get_market_state(token_id)
        estimated_price = None
        if side == "BUY" and market_state.best_ask:
            estimated_price = market_state.best_ask
        elif side == "SELL" and market_state.best_bid:
            estimated_price = market_state.best_bid
        estimated_shares = amount / estimated_price if estimated_price else 0
        if dry_run:
            print(f"[DRY RUN] MARKET {side} ${amount:.2f}")
            if estimated_price:
                print(f"  Est. Price: {estimated_price*100:.0f}c, Shares: {estimated_shares:.2f}")
            return OrderResult(success=True, order_id="DRY_RUN_MKT_" + str(int(time.time())),
                             token_id=token_id, side=side, price=estimated_price or 0.0,
                             size=amount, shares=estimated_shares, dry_run=True,
                             market_state=market_state)
        try:
            market_order_args = MarketOrderArgs(token_id=token_id, amount=amount, side=side)
            signed_order = self.client.create_market_order(market_order_args)
            response = self.client.post_order(signed_order, OrderType.FOK)
            order_id = response.get("orderID") or response.get("order_id")
            fill_price = response.get("price") or estimated_price
            fill_shares = response.get("size") or estimated_shares
            return OrderResult(success=True, order_id=order_id, token_id=token_id,
                             side=side, price=fill_price or 0.0, size=amount,
                             shares=fill_shares or 0.0, dry_run=False, market_state=market_state)
        except Exception as e:
            return OrderResult(success=False, token_id=token_id, side=side,
                             price=estimated_price or 0.0, size=amount, shares=0.0,
                             dry_run=False, error=str(e), market_state=market_state)

    def get_open_orders(self) -> list:
        try:
            return self.client.get_orders()
        except Exception as e:
            print(f"[ERROR] Failed to get orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.cancel(order_id)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to cancel order: {e}")
            return False
