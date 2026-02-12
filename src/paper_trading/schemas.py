"""
Paper Trading Schemas for Elon Tweet Count Prediction Markets.

Relational dataclasses for the live betting pipeline:
    MarketOdds -> Signal -> Betslip -> Fill -> Settlement

Key differences from backtesting schemas (src/backtesting/schemas.py):
- MarketOdds: Append-only market snapshots with change detection
- Signal: Full distributional prediction with strategy attribution
- Betslip: Tracks a bet on ONE bucket with fill aggregation
- Fill: Partial execution records
- Settlement: Resolution with running cumulative stats

Designed for CATEGORICAL markets (10-30 buckets of tweet count ranges),
not binary markets. Each bucket has its own YES price.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Literal


def make_id() -> str:
    """Generate a short unique ID."""
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# MarketOdds - Point-in-time market snapshot
# ---------------------------------------------------------------------------

@dataclass
class MarketOdds:
    """Point-in-time market snapshot for an Elon tweet count event.

    Created only when prices change from the last capture (append-only).
    Stores YES prices for ALL buckets in the event.

    Example bucket_prices:
        {"0-19": 0.02, "20-39": 0.05, "40-59": 0.08, ..., "740+": 0.01}
    """

    odds_id: str = field(default_factory=make_id)
    event_slug: str = ""
    event_id: str = ""
    market_type: str = ""               # weekly, daily, monthly, short
    bucket_prices: dict[str, float] = field(default_factory=dict)  # bucket_label -> YES price
    n_buckets: int = 0
    implied_ev: float = 0.0             # Probability-weighted expected count
    total_liquidity: Optional[float] = None  # Estimated total liquidity ($)
    captured_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):
        if self.n_buckets == 0 and self.bucket_prices:
            self.n_buckets = len(self.bucket_prices)

    def to_dict(self) -> dict:
        """Convert to dictionary for parquet/JSON storage."""
        return {
            "odds_id": self.odds_id,
            "event_slug": self.event_slug,
            "event_id": self.event_id,
            "market_type": self.market_type,
            "bucket_prices": json.dumps(self.bucket_prices),
            "n_buckets": self.n_buckets,
            "implied_ev": round(self.implied_ev, 2),
            "total_liquidity": self.total_liquidity,
            "captured_at": self.captured_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MarketOdds:
        """Create from dictionary (parquet row or JSON)."""
        bp = d.get("bucket_prices", {})
        if isinstance(bp, str):
            bp = json.loads(bp)
        ca = d.get("captured_at")
        if ca and isinstance(ca, str):
            ca = datetime.fromisoformat(ca)
        elif ca is None:
            ca = datetime.now(timezone.utc)
        return cls(
            odds_id=d.get("odds_id", make_id()),
            event_slug=d.get("event_slug", ""),
            event_id=d.get("event_id", ""),
            market_type=d.get("market_type", ""),
            bucket_prices=bp,
            n_buckets=d.get("n_buckets", len(bp)),
            implied_ev=d.get("implied_ev", 0.0),
            total_liquidity=d.get("total_liquidity"),
            captured_at=ca,
        )

    def prices_match(self, other: MarketOdds, tolerance: float = 0.005) -> bool:
        """Check if prices are essentially unchanged (for deduplication).

        Returns True if all bucket prices differ by less than tolerance.
        Different bucket sets are always considered changed.
        """
        if set(self.bucket_prices.keys()) != set(other.bucket_prices.keys()):
            return False
        for label, price in self.bucket_prices.items():
            other_price = other.bucket_prices.get(label, 0.0)
            if abs(price - other_price) >= tolerance:
                return False
        return True


# ---------------------------------------------------------------------------
# Signal - Model prediction + betting decision
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Model prediction and betting decision for a tweet count event.

    Contains the FULL distributional prediction (probabilities for every bucket)
    plus the identified best bet and Kelly sizing.

    References a specific MarketOdds snapshot via odds_id.
    """

    signal_id: str = field(default_factory=make_id)
    odds_id: str = ""                       # FK -> MarketOdds

    # Model Identity
    model_id: str = ""
    strategy_id: str = ""                   # FK -> strategy_registry

    # Predictions (full distribution)
    predicted_probs: dict[str, float] = field(default_factory=dict)  # bucket_label -> prob
    predicted_ev: float = 0.0               # Model's expected tweet count

    # Best bet identified
    best_bucket: str = ""                   # Bucket with highest edge
    best_bucket_edge: float = 0.0           # model_prob - market_price for that bucket
    best_bucket_model_prob: float = 0.0
    best_bucket_market_price: float = 0.0

    # Decision
    meets_criteria: bool = False            # Passed strategy filters?
    n_buckets_with_edge: int = 0            # How many buckets have edge > min_edge

    # Kelly sizing for best bucket
    kelly_fraction: float = 0.0
    recommended_wager: float = 0.0

    # Strategy filter details
    strategy_ids: str = ""                  # Comma-separated matching strategies
    n_strategies: int = 0

    # Features snapshot (for audit)
    feature_summary: dict = field(default_factory=dict)

    # Metadata
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for parquet/JSON storage."""
        return {
            "signal_id": self.signal_id,
            "odds_id": self.odds_id,
            "model_id": self.model_id,
            "strategy_id": self.strategy_id,
            "predicted_probs": json.dumps(
                {k: round(v, 6) for k, v in self.predicted_probs.items()}
            ),
            "predicted_ev": round(self.predicted_ev, 2),
            "best_bucket": self.best_bucket,
            "best_bucket_edge": round(self.best_bucket_edge, 6),
            "best_bucket_model_prob": round(self.best_bucket_model_prob, 6),
            "best_bucket_market_price": round(self.best_bucket_market_price, 6),
            "meets_criteria": self.meets_criteria,
            "n_buckets_with_edge": self.n_buckets_with_edge,
            "kelly_fraction": round(self.kelly_fraction, 6),
            "recommended_wager": round(self.recommended_wager, 2),
            "strategy_ids": self.strategy_ids,
            "n_strategies": self.n_strategies,
            "feature_summary": json.dumps(self.feature_summary),
            "generated_at": self.generated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Signal:
        """Create from dictionary (ignores unknown fields)."""
        pp = d.get("predicted_probs", {})
        if isinstance(pp, str):
            pp = json.loads(pp)
        fs = d.get("feature_summary", {})
        if isinstance(fs, str):
            fs = json.loads(fs)
        ga = d.get("generated_at")
        if ga and isinstance(ga, str):
            ga = datetime.fromisoformat(ga)
        elif ga is None:
            ga = datetime.now(timezone.utc)
        return cls(
            signal_id=d.get("signal_id", make_id()),
            odds_id=d.get("odds_id", ""),
            model_id=d.get("model_id", ""),
            strategy_id=d.get("strategy_id", ""),
            predicted_probs=pp,
            predicted_ev=d.get("predicted_ev", 0.0),
            best_bucket=d.get("best_bucket", ""),
            best_bucket_edge=d.get("best_bucket_edge", 0.0),
            best_bucket_model_prob=d.get("best_bucket_model_prob", 0.0),
            best_bucket_market_price=d.get("best_bucket_market_price", 0.0),
            meets_criteria=bool(d.get("meets_criteria", False)),
            n_buckets_with_edge=int(d.get("n_buckets_with_edge", 0)),
            kelly_fraction=d.get("kelly_fraction", 0.0),
            recommended_wager=d.get("recommended_wager", 0.0),
            strategy_ids=d.get("strategy_ids", ""),
            n_strategies=int(d.get("n_strategies", 0)),
            feature_summary=fs,
            generated_at=ga,
        )


# ---------------------------------------------------------------------------
# Betslip - A placed bet on one bucket
# ---------------------------------------------------------------------------

@dataclass
class Betslip:
    """A placed bet on ONE bucket of one tweet count event.

    In categorical markets, each bet is on a single bucket (e.g., "60-79").
    The bet_side is always "YES" because we buy the YES token for that bucket.

    References Signal via signal_id. Key fields are denormalized from
    Signal/MarketOdds for practical querying without joins.
    """

    betslip_id: str = field(default_factory=make_id)
    signal_id: str = ""                     # FK -> Signal

    # Event context (denormalized for querying)
    event_slug: str = ""
    event_id: str = ""
    market_type: str = ""

    # Bet details
    bucket_label: str = ""                  # Which bucket we are betting on
    bet_side: Literal["YES"] = "YES"        # Always YES for bucket markets

    # Pricing
    price_paid: float = 0.0                 # Entry price (YES price for this bucket)
    model_prob: float = 0.0                 # Model's probability at bet time
    edge_at_bet: float = 0.0               # model_prob - price_paid

    # Sizing
    wager: float = 0.0                      # Amount risked ($)
    shares: float = 0.0                     # wager / price_paid
    to_win: float = 0.0                     # shares - wager (profit if correct)

    # Fill tracking (updated as fills come in)
    fills_count: int = 0
    total_wager: float = 0.0               # Sum of fill amounts
    avg_price: float = 0.0                  # Weighted average fill price
    total_shares: float = 0.0              # Sum of fill shares

    # Metadata
    placed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    placed_by: Literal["paper", "live"] = "paper"
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for parquet/JSON storage."""
        return {
            "betslip_id": self.betslip_id,
            "signal_id": self.signal_id,
            "event_slug": self.event_slug,
            "event_id": self.event_id,
            "market_type": self.market_type,
            "bucket_label": self.bucket_label,
            "bet_side": self.bet_side,
            "price_paid": round(self.price_paid, 6),
            "model_prob": round(self.model_prob, 6),
            "edge_at_bet": round(self.edge_at_bet, 6),
            "wager": round(self.wager, 2),
            "shares": round(self.shares, 6),
            "to_win": round(self.to_win, 2),
            "fills_count": self.fills_count,
            "total_wager": round(self.total_wager, 2),
            "avg_price": round(self.avg_price, 6),
            "total_shares": round(self.total_shares, 6),
            "placed_at": self.placed_at.isoformat(),
            "placed_by": self.placed_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Betslip:
        """Create from dictionary (ignores unknown fields)."""
        pa = d.get("placed_at")
        if pa and isinstance(pa, str):
            pa = datetime.fromisoformat(pa)
        elif pa is None:
            pa = datetime.now(timezone.utc)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k in valid_fields:
                filtered[k] = v
        filtered["placed_at"] = pa
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Fill - Partial execution of a betslip
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """A partial execution of a betslip at a specific price.

    Multiple fills can belong to the same betslip (partial fills at
    different prices). Betslip fill aggregates are recalculated after
    each fill is recorded.
    """

    fill_id: str = field(default_factory=make_id)
    betslip_id: str = ""                    # FK -> Betslip
    price: float = 0.0                      # Execution price
    amount: float = 0.0                     # Dollar amount at this price
    shares: float = 0.0                     # amount / price
    filled_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for parquet/JSON storage."""
        return {
            "fill_id": self.fill_id,
            "betslip_id": self.betslip_id,
            "price": round(self.price, 6),
            "amount": round(self.amount, 2),
            "shares": round(self.shares, 6),
            "filled_at": self.filled_at.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Fill:
        """Create from dictionary."""
        fa = d.get("filled_at")
        if fa and isinstance(fa, str):
            fa = datetime.fromisoformat(fa)
        elif fa is None:
            fa = datetime.now(timezone.utc)
        return cls(
            fill_id=d.get("fill_id", make_id()),
            betslip_id=d.get("betslip_id", ""),
            price=d.get("price", 0.0),
            amount=d.get("amount", 0.0),
            shares=d.get("shares", 0.0),
            filled_at=fa,
            notes=d.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Settlement - Resolution of a bet after event ends
# ---------------------------------------------------------------------------

@dataclass
class Settlement:
    """Resolution of a bet after event ends and XTracker publishes the count.

    Created when we know the winning bucket. Links back to the Betslip
    and computes P&L. Maintains running cumulative stats for the audit trail.
    """

    settlement_id: str = field(default_factory=make_id)
    betslip_id: str = ""                    # FK -> Betslip

    # Event outcome
    event_slug: str = ""
    winning_bucket: str = ""                # Which bucket won
    xtracker_count: Optional[int] = None    # Actual tweet count (if known)

    # P&L
    bucket_bet: str = ""                    # Which bucket we bet on
    won: bool = False                       # bucket_bet == winning_bucket
    wager: float = 0.0
    payout: float = 0.0                     # shares if won, 0 if lost
    pnl: float = 0.0                        # payout - wager

    # Running totals (updated with each settlement)
    cumul_wager: float = 0.0
    cumul_pnl: float = 0.0
    cumul_roi_pct: float = 0.0
    total_bets: int = 0
    total_wins: int = 0
    win_rate_pct: float = 0.0

    # Metadata
    settled_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for parquet/JSON storage."""
        return {
            "settlement_id": self.settlement_id,
            "betslip_id": self.betslip_id,
            "event_slug": self.event_slug,
            "winning_bucket": self.winning_bucket,
            "xtracker_count": self.xtracker_count,
            "bucket_bet": self.bucket_bet,
            "won": self.won,
            "wager": round(self.wager, 2),
            "payout": round(self.payout, 2),
            "pnl": round(self.pnl, 2),
            "cumul_wager": round(self.cumul_wager, 2),
            "cumul_pnl": round(self.cumul_pnl, 2),
            "cumul_roi_pct": round(self.cumul_roi_pct, 2),
            "total_bets": self.total_bets,
            "total_wins": self.total_wins,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "settled_at": self.settled_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Settlement:
        """Create from dictionary."""
        sa = d.get("settled_at")
        if sa and isinstance(sa, str):
            sa = datetime.fromisoformat(sa)
        elif sa is None:
            sa = datetime.now(timezone.utc)
        return cls(
            settlement_id=d.get("settlement_id", make_id()),
            betslip_id=d.get("betslip_id", ""),
            event_slug=d.get("event_slug", ""),
            winning_bucket=d.get("winning_bucket", ""),
            xtracker_count=d.get("xtracker_count"),
            bucket_bet=d.get("bucket_bet", ""),
            won=bool(d.get("won", False)),
            wager=d.get("wager", 0.0),
            payout=d.get("payout", 0.0),
            pnl=d.get("pnl", 0.0),
            cumul_wager=d.get("cumul_wager", 0.0),
            cumul_pnl=d.get("cumul_pnl", 0.0),
            cumul_roi_pct=d.get("cumul_roi_pct", 0.0),
            total_bets=int(d.get("total_bets", 0)),
            total_wins=int(d.get("total_wins", 0)),
            win_rate_pct=d.get("win_rate_pct", 0.0),
            settled_at=sa,
        )
