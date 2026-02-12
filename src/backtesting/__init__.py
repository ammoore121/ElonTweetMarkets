"""
Backtesting engine for Elon Musk tweet count prediction markets.

Exports:
    BacktestEngine  - Main engine class (run, print_report, save_result)
    BacktestResult  - Aggregate result dataclass
    MarketSnapshot  - Point-in-time market prices
    PredictionSignal - Model output probabilities
    Trade           - A single bucket bet
    Settlement      - Resolved trade outcome
"""

from src.backtesting.engine import BacktestEngine
from src.backtesting.schemas import (
    BacktestResult,
    MarketSnapshot,
    PredictionSignal,
    Settlement,
    Trade,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "MarketSnapshot",
    "PredictionSignal",
    "Settlement",
    "Trade",
]
