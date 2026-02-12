"""
Paper Trading for Elon Tweet Count Prediction Markets.

Pipeline: MarketOdds -> Signal -> Betslip -> Fill -> Settlement

Exports:
    MarketOdds          - Point-in-time bucket prices for an event
    Signal              - Model prediction + betting decision
    Betslip             - A placed bet on one bucket
    Fill                - Partial execution of a betslip
    Settlement          - Resolution of a bet after event ends
    PerformanceTracker  - Persistent storage and relational queries
    ValidationError     - Raised when validation fails
"""

from src.paper_trading.schemas import (
    MarketOdds,
    Signal,
    Betslip,
    Fill,
    Settlement,
    make_id,
)
from src.paper_trading.tracker import PerformanceTracker
from src.paper_trading.validators import ValidationError

__all__ = [
    "MarketOdds",
    "Signal",
    "Betslip",
    "Fill",
    "Settlement",
    "make_id",
    "PerformanceTracker",
    "ValidationError",
]
