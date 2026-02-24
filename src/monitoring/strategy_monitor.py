"""Post-settlement strategy health monitor.

Analyzes settled bets per strategy and raises alerts when performance
degrades beyond thresholds. Designed to catch the class of failures
discovered in Feb 2026: systematic losses from regime shifts, stale
features, and OOD model predictions.

Thresholds:
    WARNING:  5 consecutive losses OR trailing 10-bet ROI < -30%
    CRITICAL: 8 consecutive losses OR trailing 10-bet ROI < -60%
              OR total strategy ROI < -50%

Usage:
    from src.monitoring.strategy_monitor import StrategyMonitor
    from src.paper_trading.tracker import PerformanceTracker

    tracker = PerformanceTracker(...)
    monitor = StrategyMonitor()
    alerts = monitor.check_strategy_health(tracker)
    for alert in alerts:
        print(alert)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
CONSEC_LOSS_WARNING = 5
CONSEC_LOSS_CRITICAL = 8
TRAILING_ROI_WARNING = -30.0      # percent
TRAILING_ROI_CRITICAL = -60.0     # percent
TRAILING_WINDOW = 10              # number of bets
TOTAL_ROI_CRITICAL = -50.0        # percent
MIN_BETS_FOR_ROI_CHECK = 5        # need at least this many bets


class StrategyMonitor:
    """Monitors strategy health after settlements and flags degradation."""

    def check_strategy_health(self, tracker) -> list[dict]:
        """Run all health checks on settled strategies.

        Args:
            tracker: PerformanceTracker instance with settlement data.

        Returns:
            List of alert dicts, each with keys:
                strategy_id, alert_type, severity, message, recommendation
        """
        settlements_df = tracker.get_all_settlements()
        if settlements_df.empty:
            return []

        # We need strategy_id on settlements. Settlements link to betslips
        # via betslip_id, and betslips link to signals via signal_id.
        # Build the mapping: betslip_id -> strategy_id
        betslips_df = tracker.get_all_betslips()
        signals_df = tracker.get_all_signals()

        if betslips_df.empty or signals_df.empty:
            return []

        # betslip -> signal_id -> strategy_id
        betslip_to_signal = {}
        if "signal_id" in betslips_df.columns:
            betslip_to_signal = dict(
                zip(betslips_df["betslip_id"], betslips_df["signal_id"])
            )

        signal_to_strategy = {}
        if "strategy_id" in signals_df.columns:
            signal_to_strategy = dict(
                zip(signals_df["signal_id"], signals_df["strategy_id"])
            )

        # Attach strategy_id to settlements
        strategy_ids = []
        for _, row in settlements_df.iterrows():
            bid = row.get("betslip_id", "")
            sid = betslip_to_signal.get(bid, "")
            strat = signal_to_strategy.get(sid, "unknown")
            strategy_ids.append(strat)
        settlements_df = settlements_df.copy()
        settlements_df["strategy_id"] = strategy_ids

        # Run checks per strategy
        alerts = []
        for strategy_id, group in settlements_df.groupby("strategy_id"):
            if strategy_id == "unknown":
                continue
            # Sort by settlement time
            if "settled_at" in group.columns:
                group = group.sort_values("settled_at")

            alert = self._check_consecutive_losses(strategy_id, group)
            if alert:
                alerts.append(alert)

            alert = self._check_rolling_roi(strategy_id, group)
            if alert:
                alerts.append(alert)

            alert = self._check_total_roi(strategy_id, group)
            if alert:
                alerts.append(alert)

        return alerts

    def _check_consecutive_losses(
        self, strategy_id: str, settlements: pd.DataFrame
    ) -> Optional[dict]:
        """Alert if strategy has 5+ consecutive losses (most recent)."""
        if "won" not in settlements.columns:
            return None

        won_values = settlements["won"].tolist()
        # Count consecutive losses from the end
        consec = 0
        for w in reversed(won_values):
            if not w:
                consec += 1
            else:
                break

        if consec >= CONSEC_LOSS_CRITICAL:
            return {
                "strategy_id": strategy_id,
                "alert_type": "consecutive_losses",
                "severity": "critical",
                "message": "{} consecutive losses (threshold: {})".format(
                    consec, CONSEC_LOSS_CRITICAL
                ),
                "recommendation": "Suspend strategy immediately and investigate.",
            }
        elif consec >= CONSEC_LOSS_WARNING:
            return {
                "strategy_id": strategy_id,
                "alert_type": "consecutive_losses",
                "severity": "warning",
                "message": "{} consecutive losses (threshold: {})".format(
                    consec, CONSEC_LOSS_WARNING
                ),
                "recommendation": "Review strategy parameters and recent feature quality.",
            }
        return None

    def _check_rolling_roi(
        self, strategy_id: str, settlements: pd.DataFrame
    ) -> Optional[dict]:
        """Alert if trailing N-bet ROI is below threshold."""
        if len(settlements) < MIN_BETS_FOR_ROI_CHECK:
            return None
        if "pnl" not in settlements.columns or "wager" not in settlements.columns:
            return None

        recent = settlements.tail(TRAILING_WINDOW)
        total_wager = recent["wager"].sum()
        total_pnl = recent["pnl"].sum()

        if total_wager <= 0:
            return None

        roi_pct = (total_pnl / total_wager) * 100.0

        if roi_pct <= TRAILING_ROI_CRITICAL:
            return {
                "strategy_id": strategy_id,
                "alert_type": "rolling_roi",
                "severity": "critical",
                "message": "Trailing {}-bet ROI: {:.1f}% (threshold: {}%)".format(
                    min(len(recent), TRAILING_WINDOW), roi_pct, TRAILING_ROI_CRITICAL
                ),
                "recommendation": "Suspend strategy. Check for regime shift or data issues.",
            }
        elif roi_pct <= TRAILING_ROI_WARNING:
            return {
                "strategy_id": strategy_id,
                "alert_type": "rolling_roi",
                "severity": "warning",
                "message": "Trailing {}-bet ROI: {:.1f}% (threshold: {}%)".format(
                    min(len(recent), TRAILING_WINDOW), roi_pct, TRAILING_ROI_WARNING
                ),
                "recommendation": "Monitor closely. Consider reducing position sizing.",
            }
        return None

    def _check_total_roi(
        self, strategy_id: str, settlements: pd.DataFrame
    ) -> Optional[dict]:
        """Alert if total lifetime ROI is critically negative."""
        if len(settlements) < MIN_BETS_FOR_ROI_CHECK:
            return None
        if "pnl" not in settlements.columns or "wager" not in settlements.columns:
            return None

        total_wager = settlements["wager"].sum()
        total_pnl = settlements["pnl"].sum()

        if total_wager <= 0:
            return None

        roi_pct = (total_pnl / total_wager) * 100.0

        if roi_pct <= TOTAL_ROI_CRITICAL:
            return {
                "strategy_id": strategy_id,
                "alert_type": "total_roi",
                "severity": "critical",
                "message": "Total ROI: {:.1f}% over {} bets (threshold: {}%)".format(
                    roi_pct, len(settlements), TOTAL_ROI_CRITICAL
                ),
                "recommendation": "Suspend strategy. Fundamental edge may not exist in live trading.",
            }
        return None
