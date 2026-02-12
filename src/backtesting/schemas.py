"""
Data schemas for the backtesting engine.

Follows the EsportsBetting pattern of typed dataclasses with to_dict()/from_dict()
methods for serialization.  The pipeline stages are:

    MarketSnapshot  (point-in-time prices)
        -> PredictionSignal  (model output)
            -> Trade  (Kelly-sized bet on one bucket)
                -> Settlement  (resolved after event ends)
                    -> BacktestResult  (aggregate across events)

Each stage carries foreign-key references (event_slug, trade_id) so that the
full audit trail can be reconstructed from persisted JSON / parquet.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# MarketSnapshot
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """Point-in-time market prices for an event's buckets."""

    event_slug: str
    snapshot_time: Optional[datetime]
    bucket_prices: dict[str, float]   # bucket_label -> price (0-1)
    market_type: str = ""             # weekly, daily, monthly, short
    event_id: str = ""

    def to_dict(self) -> dict:
        return {
            "event_slug": self.event_slug,
            "event_id": self.event_id,
            "snapshot_time": (
                self.snapshot_time.isoformat() if self.snapshot_time else None
            ),
            "bucket_prices": dict(self.bucket_prices),
            "market_type": self.market_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MarketSnapshot:
        st = d.get("snapshot_time")
        if st and isinstance(st, str):
            st = datetime.fromisoformat(st)
        return cls(
            event_slug=d["event_slug"],
            event_id=d.get("event_id", ""),
            snapshot_time=st,
            bucket_prices=d.get("bucket_prices", {}),
            market_type=d.get("market_type", ""),
        )


# ---------------------------------------------------------------------------
# PredictionSignal
# ---------------------------------------------------------------------------

@dataclass
class PredictionSignal:
    """Model's probability predictions for an event."""

    event_slug: str
    model_name: str
    model_version: str
    predicted_probs: dict[str, float]  # bucket_label -> prob (sums to 1)
    predicted_winner: str = ""
    predicted_winner_prob: float = 0.0
    features_used: dict = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "event_slug": self.event_slug,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predicted_probs": {
                k: round(v, 6) for k, v in self.predicted_probs.items()
            },
            "predicted_winner": self.predicted_winner,
            "predicted_winner_prob": round(self.predicted_winner_prob, 6),
            "features_used": self.features_used,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PredictionSignal:
        ca = d.get("created_at")
        if ca and isinstance(ca, str):
            ca = datetime.fromisoformat(ca)
        else:
            ca = datetime.now(timezone.utc)
        return cls(
            event_slug=d["event_slug"],
            model_name=d["model_name"],
            model_version=d.get("model_version", ""),
            predicted_probs=d.get("predicted_probs", {}),
            predicted_winner=d.get("predicted_winner", ""),
            predicted_winner_prob=d.get("predicted_winner_prob", 0.0),
            features_used=d.get("features_used", {}),
            created_at=ca,
        )


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single bet on one bucket of one event."""

    trade_id: str
    event_slug: str
    bucket_label: str
    entry_price: float      # market price at entry
    model_prob: float       # model's predicted probability
    edge: float             # model_prob - entry_price
    kelly_fraction: float
    wager: float
    shares: float           # wager / entry_price
    entry_time: Optional[datetime] = None
    model_name: str = ""

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "event_slug": self.event_slug,
            "bucket_label": self.bucket_label,
            "entry_price": round(self.entry_price, 6),
            "model_prob": round(self.model_prob, 6),
            "edge": round(self.edge, 6),
            "kelly_fraction": round(self.kelly_fraction, 6),
            "wager": round(self.wager, 2),
            "shares": round(self.shares, 6),
            "entry_time": (
                self.entry_time.isoformat() if self.entry_time else None
            ),
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trade:
        et = d.get("entry_time")
        if et and isinstance(et, str):
            et = datetime.fromisoformat(et)
        return cls(
            trade_id=d["trade_id"],
            event_slug=d["event_slug"],
            bucket_label=d["bucket_label"],
            entry_price=d["entry_price"],
            model_prob=d["model_prob"],
            edge=d["edge"],
            kelly_fraction=d.get("kelly_fraction", 0.0),
            wager=d["wager"],
            shares=d.get("shares", 0.0),
            entry_time=et,
            model_name=d.get("model_name", ""),
        )


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

@dataclass
class Settlement:
    """Settlement of a trade after event resolution."""

    trade_id: str
    event_slug: str
    bucket_label: str
    winning_bucket: str
    won: bool
    wager: float
    payout: float   # shares if won, 0 if lost
    pnl: float      # payout - wager
    settled_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "event_slug": self.event_slug,
            "bucket_label": self.bucket_label,
            "winning_bucket": self.winning_bucket,
            "won": self.won,
            "wager": round(self.wager, 2),
            "payout": round(self.payout, 2),
            "pnl": round(self.pnl, 2),
            "settled_at": (
                self.settled_at.isoformat() if self.settled_at else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Settlement:
        sa = d.get("settled_at")
        if sa and isinstance(sa, str):
            sa = datetime.fromisoformat(sa)
        return cls(
            trade_id=d["trade_id"],
            event_slug=d["event_slug"],
            bucket_label=d["bucket_label"],
            winning_bucket=d["winning_bucket"],
            won=d["won"],
            wager=d["wager"],
            payout=d["payout"],
            pnl=d["pnl"],
            settled_at=sa,
        )


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Aggregate results of a backtest run."""

    model_name: str
    model_version: str
    config: dict
    n_events: int
    n_traded: int
    n_bets: int
    n_wins: int
    total_wagered: float
    total_pnl: float
    roi: float
    brier_score: float
    log_loss: Optional[float]
    accuracy: float
    accuracy_str: str
    per_event: list[dict]
    trades: list[Trade] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "config": self.config,
            "n_events": self.n_events,
            "n_traded": self.n_traded,
            "n_bets": self.n_bets,
            "n_wins": self.n_wins,
            "total_wagered": round(self.total_wagered, 2),
            "total_pnl": round(self.total_pnl, 2),
            "roi": round(self.roi, 2),
            "brier_score": round(self.brier_score, 6) if self.brier_score is not None else None,
            "log_loss": round(self.log_loss, 6) if self.log_loss is not None else None,
            "accuracy": round(self.accuracy, 4),
            "accuracy_str": self.accuracy_str,
            "per_event": self.per_event,
            "trades": [t.to_dict() for t in self.trades],
            "settlements": [s.to_dict() for s in self.settlements],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> BacktestResult:
        ca = d.get("created_at")
        if ca and isinstance(ca, str):
            ca = datetime.fromisoformat(ca)
        else:
            ca = datetime.now(timezone.utc)
        return cls(
            model_name=d["model_name"],
            model_version=d.get("model_version", ""),
            config=d.get("config", {}),
            n_events=d["n_events"],
            n_traded=d["n_traded"],
            n_bets=d["n_bets"],
            n_wins=d.get("n_wins", 0),
            total_wagered=d["total_wagered"],
            total_pnl=d["total_pnl"],
            roi=d["roi"],
            brier_score=d["brier_score"],
            log_loss=d.get("log_loss"),
            accuracy=d["accuracy"],
            accuracy_str=d["accuracy_str"],
            per_event=d.get("per_event", []),
            trades=[Trade.from_dict(t) for t in d.get("trades", [])],
            settlements=[
                Settlement.from_dict(s) for s in d.get("settlements", [])
            ],
            created_at=ca,
        )

    def summary(self) -> str:
        """Return a compact one-line summary string."""
        return (
            "BacktestResult(model={}, events={}, bets={}, "
            "Brier={:.4f}, ROI={:+.1f}%, accuracy={})".format(
                self.model_name,
                self.n_events,
                self.n_bets,
                self.brier_score if self.brier_score is not None else 0.0,
                self.roi,
                self.accuracy_str,
            )
        )


def make_trade_id() -> str:
    """Generate a unique trade ID."""
    return str(uuid.uuid4())[:12]
