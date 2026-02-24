"""
Paper Trading Performance Tracker for Elon Tweet Count Prediction Markets.

Manages persistent storage and relational queries for the paper trading pipeline:
    MarketOdds -> Signal -> Betslip -> Fill -> Settlement

Storage: parquet files in data/paper_trading/
    - odds.parquet:       Append-only market snapshots
    - signals.parquet:    Append-only model predictions
    - betslips.parquet:   Placed bets (updated with fill aggregates)
    - fills.parquet:      Partial executions
    - settlements.parquet: Resolved bet outcomes

Adapted from EsportsBetting/BettingMarkets pattern for categorical markets.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .schemas import MarketOdds, Signal, Betslip, Fill, Settlement, make_id
from .validators import (
    validate_odds,
    validate_signal,
    validate_betslip,
    validate_settlement,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Manages persistent storage and relational queries for paper trading.

    All data is stored as parquet files for efficient columnar access.
    Append-only pattern: records are only added, never deleted.

    Usage:
        tracker = PerformanceTracker()
        odds_id, recorded = tracker.record_odds(odds)
        signal_id = tracker.record_signal(signal)
        betslip_id = tracker.create_betslip_from_signal(signal_id)
        fill_id = tracker.add_fill(betslip_id, price=0.35, amount=50.0)
        settlement_id = tracker.settle_bet(betslip_id, "60-79", xtracker_count=72)
        tracker.print_performance()
    """

    def __init__(self, data_dir: Optional[str] = None, odds_dir: Optional[str] = None, signals_dir: Optional[str] = None):
        if data_dir is None:
            self.data_dir = PROJECT_DIR / "data" / "paper_trading"
        else:
            self.data_dir = Path(data_dir)

        self.odds_dir = Path(odds_dir) if odds_dir else self.data_dir
        self.signals_dir = Path(signals_dir) if signals_dir else self.data_dir

        self.data_dir.mkdir(parents=True, exist_ok=True)
        if odds_dir:
            self.odds_dir.mkdir(parents=True, exist_ok=True)
        if signals_dir:
            self.signals_dir.mkdir(parents=True, exist_ok=True)

        # Parquet file paths
        self.odds_path = self.odds_dir / "odds.parquet"
        self.signals_path = self.signals_dir / "signals.parquet"
        self.betslips_path = self.data_dir / "betslips.parquet"
        self.fills_path = self.data_dir / "fills.parquet"
        self.settlements_path = self.data_dir / "settlements.parquet"

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read parquet file, returning empty DataFrame if not found."""
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def _append_record(self, path: Path, record: dict) -> None:
        """Append a single record to a parquet file."""
        if path.exists():
            df = pd.read_parquet(path)
            records = df.to_dict("records")
        else:
            records = []
        records.append(record)
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)

    def _update_record(self, path: Path, id_col: str, id_val: str,
                       updates: dict) -> bool:
        """Update a single record in a parquet file by ID column.

        Returns True if record was found and updated.
        """
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        idx = df[df[id_col] == id_val].index
        if len(idx) == 0:
            return False
        for col, val in updates.items():
            df.loc[idx[0], col] = val
        df.to_parquet(path, index=False)
        return True

    # =====================================================================
    # ODDS OPERATIONS
    # =====================================================================

    def record_odds(self, odds: MarketOdds, check_changed: bool = True) -> tuple[str, bool]:
        """Record a market snapshot. Returns (odds_id, was_recorded).

        If check_changed=True, skips recording if prices haven't changed
        from the last snapshot for this event_slug. This deduplicates
        repeated polls that show identical prices.
        """
        # Validate
        errors = validate_odds(odds)
        if errors:
            raise ValidationError("Odds validation failed: {}".format(errors))

        # Check for unchanged prices
        if check_changed:
            latest = self.get_latest_odds(odds.event_slug)
            if latest and odds.prices_match(latest):
                return latest.odds_id, False

        self._append_record(self.odds_path, odds.to_dict())
        return odds.odds_id, True

    def get_odds(self, odds_id: str) -> Optional[MarketOdds]:
        """Get a specific odds snapshot by ID."""
        df = self._read_parquet(self.odds_path)
        if df.empty:
            return None
        matches = df[df["odds_id"] == odds_id]
        if matches.empty:
            return None
        return MarketOdds.from_dict(matches.iloc[0].to_dict())

    def get_latest_odds(self, event_slug: str) -> Optional[MarketOdds]:
        """Get the most recent odds snapshot for an event."""
        df = self._read_parquet(self.odds_path)
        if df.empty:
            return None
        matches = df[df["event_slug"] == event_slug]
        if matches.empty:
            return None
        matches = matches.sort_values("captured_at", ascending=False)
        return MarketOdds.from_dict(matches.iloc[0].to_dict())

    def get_odds_history(self, event_slug: str) -> list[MarketOdds]:
        """Get all odds snapshots for an event, chronologically."""
        df = self._read_parquet(self.odds_path)
        if df.empty:
            return []
        matches = df[df["event_slug"] == event_slug].sort_values("captured_at")
        return [MarketOdds.from_dict(row.to_dict()) for _, row in matches.iterrows()]

    # =====================================================================
    # SIGNAL OPERATIONS
    # =====================================================================

    def record_signal(self, signal: Signal) -> str:
        """Record a model prediction. Returns signal_id."""
        errors = validate_signal(signal)
        if errors:
            raise ValidationError("Signal validation failed: {}".format(errors))

        self._append_record(self.signals_path, signal.to_dict())
        return signal.signal_id

    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """Get a specific signal by ID."""
        df = self._read_parquet(self.signals_path)
        if df.empty:
            return None
        matches = df[df["signal_id"] == signal_id]
        if matches.empty:
            return None
        return Signal.from_dict(matches.iloc[0].to_dict())

    def get_signal_with_odds(self, signal_id: str) -> Optional[tuple[Signal, MarketOdds]]:
        """Get a signal and its referenced odds snapshot."""
        signal = self.get_signal(signal_id)
        if signal is None:
            return None
        odds = self.get_odds(signal.odds_id)
        if odds is None:
            return None
        return signal, odds

    def get_signals(
        self,
        meets_criteria: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[Signal]:
        """Get signals with optional filters."""
        df = self._read_parquet(self.signals_path)
        if df.empty:
            return []

        if meets_criteria is not None:
            df = df[df["meets_criteria"] == meets_criteria]

        if start_date is not None:
            df["generated_at"] = pd.to_datetime(df["generated_at"])
            start_ts = pd.Timestamp(start_date)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            df = df[df["generated_at"] >= start_ts]

        if end_date is not None:
            df["generated_at"] = pd.to_datetime(df["generated_at"])
            end_ts = pd.Timestamp(end_date)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            df = df[df["generated_at"] <= end_ts]

        return [Signal.from_dict(row.to_dict()) for _, row in df.iterrows()]

    def get_unbetted_signals(self, max_age_hours: int = 48) -> list[Signal]:
        """Get actionable signals that haven't been converted to betslips yet.

        Only returns signals that:
        - Meet strategy criteria (meets_criteria=True)
        - Were generated within the last max_age_hours
        - Have no corresponding betslip
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        signals = self.get_signals(meets_criteria=True, start_date=cutoff)

        # Get signal IDs that already have betslips
        betslip_signal_ids = set()
        df = self._read_parquet(self.betslips_path)
        if not df.empty:
            betslip_signal_ids = set(df["signal_id"].tolist())

        return [s for s in signals if s.signal_id not in betslip_signal_ids]

    # =====================================================================
    # BETSLIP OPERATIONS
    # =====================================================================

    def create_betslip_from_signal(
        self,
        signal_id: str,
        placed_by: str = "paper",
    ) -> str:
        """Create a betslip from an existing signal. Returns betslip_id.

        Pulls signal + odds data, constructs the betslip with proper
        pricing and sizing for the best bucket identified by the signal.
        """
        result = self.get_signal_with_odds(signal_id)
        if result is None:
            raise ValueError(
                "Signal {} or its odds not found".format(signal_id)
            )
        signal, odds = result

        # Compute bet sizing
        price = signal.best_bucket_market_price
        wager = signal.recommended_wager
        shares = wager / price if price > 0 else 0.0
        to_win = shares - wager if shares > 0 else 0.0

        betslip = Betslip(
            signal_id=signal_id,
            event_slug=odds.event_slug,
            event_id=odds.event_id,
            market_type=odds.market_type,
            bucket_label=signal.best_bucket,
            bet_side="YES",
            price_paid=price,
            model_prob=signal.best_bucket_model_prob,
            edge_at_bet=signal.best_bucket_edge,
            wager=wager,
            shares=shares,
            to_win=to_win,
            placed_by=placed_by,
        )

        # Validate
        errors = validate_betslip(betslip, signal)
        if errors:
            raise ValidationError("Betslip validation failed: {}".format(errors))

        self._append_record(self.betslips_path, betslip.to_dict())
        return betslip.betslip_id

    def get_betslip(self, betslip_id: str) -> Optional[Betslip]:
        """Get a specific betslip by ID."""
        df = self._read_parquet(self.betslips_path)
        if df.empty:
            return None
        matches = df[df["betslip_id"] == betslip_id]
        if matches.empty:
            return None
        return Betslip.from_dict(matches.iloc[0].to_dict())

    def get_open_betslips(self) -> list[Betslip]:
        """Get all betslips that have not been settled yet."""
        betslips_df = self._read_parquet(self.betslips_path)
        if betslips_df.empty:
            return []

        # Get settled betslip IDs
        settled_ids = set()
        settlements_df = self._read_parquet(self.settlements_path)
        if not settlements_df.empty:
            settled_ids = set(settlements_df["betslip_id"].tolist())

        unsettled = betslips_df[~betslips_df["betslip_id"].isin(settled_ids)]
        return [Betslip.from_dict(row.to_dict()) for _, row in unsettled.iterrows()]

    def has_open_position(
        self, event_slug: str, strategy_id: Optional[str] = None
    ) -> bool:
        """Check if a strategy already has an open position on an event.

        Used for deduplication: a strategy may only place 1 bet per event,
        regardless of which bucket it targets. Different strategies are
        independent and may each have their own position on the same event.

        If strategy_id is None, checks for any open position on the event
        across all strategies (used for event-level checks).

        The uniqueness key is (event_slug, strategy_id): one bet per event
        per strategy, enforced by looking up the strategy_id stored on the
        signal that originated each open betslip.
        """
        open_bets = self.get_open_betslips()
        for bet in open_bets:
            if bet.event_slug != event_slug:
                continue
            if strategy_id is None:
                # No strategy filter: any open position on this event matches
                return True
            # Resolve strategy_id through the signal that created this betslip
            signal = self.get_signal(bet.signal_id)
            if signal is not None and signal.strategy_id == strategy_id:
                return True
        return False

    def get_betslips_for_event(self, event_slug: str) -> list[Betslip]:
        """Get all betslips for a specific event."""
        df = self._read_parquet(self.betslips_path)
        if df.empty:
            return []
        matches = df[df["event_slug"] == event_slug]
        return [Betslip.from_dict(row.to_dict()) for _, row in matches.iterrows()]

    # =====================================================================
    # FILL OPERATIONS
    # =====================================================================

    def add_fill(
        self,
        betslip_id: str,
        price: float,
        amount: float,
        filled_at: Optional[datetime] = None,
        notes: str = "",
    ) -> str:
        """Record a partial fill for a betslip. Returns fill_id.

        After recording the fill, recalculates the betslip's fill
        aggregate fields (fills_count, total_wager, avg_price, total_shares).
        """
        betslip = self.get_betslip(betslip_id)
        if betslip is None:
            raise ValueError("Betslip {} not found".format(betslip_id))

        if price <= 0:
            raise ValueError("Fill price must be positive, got {}".format(price))
        if amount <= 0:
            raise ValueError("Fill amount must be positive, got {}".format(amount))

        shares = amount / price

        fill = Fill(
            betslip_id=betslip_id,
            price=price,
            amount=amount,
            shares=shares,
            filled_at=filled_at or datetime.now(timezone.utc),
            notes=notes,
        )

        self._append_record(self.fills_path, fill.to_dict())

        # Recalculate betslip aggregates
        self._recalculate_betslip_aggregates(betslip_id)

        return fill.fill_id

    def get_fills_for_betslip(self, betslip_id: str) -> list[Fill]:
        """Get all fills for a specific betslip."""
        df = self._read_parquet(self.fills_path)
        if df.empty:
            return []
        matches = df[df["betslip_id"] == betslip_id]
        return [Fill.from_dict(row.to_dict()) for _, row in matches.iterrows()]

    def _recalculate_betslip_aggregates(self, betslip_id: str) -> None:
        """Recalculate fill aggregates on a betslip after a new fill."""
        fills = self.get_fills_for_betslip(betslip_id)
        if not fills:
            return

        fills_count = len(fills)
        total_wager = sum(f.amount for f in fills)
        total_shares = sum(f.shares for f in fills)
        avg_price = (
            sum(f.price * f.amount for f in fills) / total_wager
            if total_wager > 0
            else 0.0
        )

        updates = {
            "fills_count": fills_count,
            "total_wager": round(total_wager, 2),
            "avg_price": round(avg_price, 6),
            "total_shares": round(total_shares, 6),
            # Also update the primary wager/shares/price fields
            "wager": round(total_wager, 2),
            "shares": round(total_shares, 6),
            "price_paid": round(avg_price, 6),
            "to_win": round(total_shares - total_wager, 2),
        }

        self._update_record(self.betslips_path, "betslip_id", betslip_id, updates)

    # =====================================================================
    # SETTLEMENT OPERATIONS
    # =====================================================================

    def settle_bet(
        self,
        betslip_id: str,
        winning_bucket: str,
        xtracker_count: Optional[int] = None,
    ) -> str:
        """Settle a bet after event resolution. Returns settlement_id.

        Computes P&L based on whether the betslip's bucket matches the
        winning bucket. Updates running cumulative stats.
        """
        betslip = self.get_betslip(betslip_id)
        if betslip is None:
            raise ValueError("Betslip {} not found".format(betslip_id))

        # Check not already settled
        settlements_df = self._read_parquet(self.settlements_path)
        if not settlements_df.empty:
            already = settlements_df[settlements_df["betslip_id"] == betslip_id]
            if not already.empty:
                raise ValueError(
                    "Betslip {} already settled".format(betslip_id)
                )

        # Compute P&L
        won = betslip.bucket_label == winning_bucket
        payout = betslip.shares if won else 0.0
        pnl = payout - betslip.wager

        # Get running cumulative stats
        cumul_wager, cumul_pnl, total_bets, total_wins = self._get_cumulative_stats()
        cumul_wager += betslip.wager
        cumul_pnl += pnl
        total_bets += 1
        if won:
            total_wins += 1
        cumul_roi_pct = (
            (cumul_pnl / cumul_wager * 100) if cumul_wager > 0 else 0.0
        )
        win_rate_pct = (
            (total_wins / total_bets * 100) if total_bets > 0 else 0.0
        )

        settlement = Settlement(
            betslip_id=betslip_id,
            event_slug=betslip.event_slug,
            winning_bucket=winning_bucket,
            xtracker_count=xtracker_count,
            bucket_bet=betslip.bucket_label,
            won=won,
            wager=betslip.wager,
            payout=payout,
            pnl=pnl,
            cumul_wager=cumul_wager,
            cumul_pnl=cumul_pnl,
            cumul_roi_pct=cumul_roi_pct,
            total_bets=total_bets,
            total_wins=total_wins,
            win_rate_pct=win_rate_pct,
        )

        # Validate
        errors = validate_settlement(settlement, betslip)
        if errors:
            raise ValidationError("Settlement validation failed: {}".format(errors))

        self._append_record(self.settlements_path, settlement.to_dict())
        return settlement.settlement_id

    def _get_cumulative_stats(self) -> tuple[float, float, int, int]:
        """Get running cumulative stats from the last settlement.

        Returns (cumul_wager, cumul_pnl, total_bets, total_wins).
        """
        df = self._read_parquet(self.settlements_path)
        if df.empty:
            return 0.0, 0.0, 0, 0
        last = df.iloc[-1]
        return (
            float(last.get("cumul_wager", 0)),
            float(last.get("cumul_pnl", 0)),
            int(last.get("total_bets", 0)),
            int(last.get("total_wins", 0)),
        )

    def get_settlements(self) -> list[Settlement]:
        """Get all settlements, chronologically."""
        df = self._read_parquet(self.settlements_path)
        if df.empty:
            return []
        return [Settlement.from_dict(row.to_dict()) for _, row in df.iterrows()]

    def get_settlement_for_betslip(self, betslip_id: str) -> Optional[Settlement]:
        """Get the settlement for a specific betslip."""
        df = self._read_parquet(self.settlements_path)
        if df.empty:
            return None
        matches = df[df["betslip_id"] == betslip_id]
        if matches.empty:
            return None
        return Settlement.from_dict(matches.iloc[0].to_dict())

    # =====================================================================
    # SETTLE ALL BETS FOR AN EVENT
    # =====================================================================

    def settle_event(
        self,
        event_slug: str,
        winning_bucket: str,
        xtracker_count: Optional[int] = None,
    ) -> list[str]:
        """Settle all open bets for an event. Returns list of settlement_ids.

        Convenience method that finds all open betslips for the event
        and settles them with the same winning bucket.
        """
        open_bets = self.get_open_betslips()
        event_bets = [b for b in open_bets if b.event_slug == event_slug]

        settlement_ids = []
        for betslip in event_bets:
            sid = self.settle_bet(
                betslip.betslip_id,
                winning_bucket,
                xtracker_count=xtracker_count,
            )
            settlement_ids.append(sid)

        return settlement_ids

    # =====================================================================
    # PERFORMANCE SUMMARY
    # =====================================================================

    def get_performance(self) -> dict:
        """Get performance summary statistics.

        Returns dict with keys: total_signals, actionable_signals,
        total_bets, open_bets, settled_bets, total_wins, total_losses,
        win_rate_pct, cumul_wager, cumul_pnl, cumul_roi_pct,
        avg_edge, avg_pnl_per_bet.
        """
        cumul_wager, cumul_pnl, total_bets, total_wins = self._get_cumulative_stats()

        # Signal counts
        total_signals = 0
        actionable_signals = 0
        signals_df = self._read_parquet(self.signals_path)
        if not signals_df.empty:
            total_signals = len(signals_df)
            actionable_signals = len(
                signals_df[signals_df["meets_criteria"] == True]  # noqa: E712
            )

        open_bets = len(self.get_open_betslips())

        # Average edge on betslips
        avg_edge = 0.0
        betslips_df = self._read_parquet(self.betslips_path)
        if not betslips_df.empty and "edge_at_bet" in betslips_df.columns:
            avg_edge = betslips_df["edge_at_bet"].mean()

        avg_pnl = cumul_pnl / total_bets if total_bets > 0 else 0.0

        return {
            "total_signals": total_signals,
            "actionable_signals": actionable_signals,
            "total_bets": total_bets,
            "open_bets": open_bets,
            "settled_bets": total_bets,
            "total_wins": total_wins,
            "total_losses": total_bets - total_wins,
            "win_rate_pct": (
                total_wins / total_bets * 100 if total_bets > 0 else 0.0
            ),
            "cumul_wager": round(cumul_wager, 2),
            "cumul_pnl": round(cumul_pnl, 2),
            "cumul_roi_pct": (
                round(cumul_pnl / cumul_wager * 100, 2)
                if cumul_wager > 0
                else 0.0
            ),
            "avg_edge": round(avg_edge, 4),
            "avg_pnl_per_bet": round(avg_pnl, 2),
        }

    def print_performance(self) -> None:
        """Print formatted performance summary."""
        perf = self.get_performance()

        sep = "=" * 55
        print()
        print(sep)
        print("  PAPER TRADING PERFORMANCE")
        print(sep)
        print()
        print("  Signals Generated:      {:>6d}".format(perf["total_signals"]))
        print("  Actionable Signals:     {:>6d}".format(perf["actionable_signals"]))
        print()
        print("  Bets Placed:            {:>6d}".format(
            perf["total_bets"] + perf["open_bets"]
        ))
        print("  Open Bets:              {:>6d}".format(perf["open_bets"]))
        print("  Settled Bets:           {:>6d}".format(perf["settled_bets"]))
        print()
        print("  Wins:                   {:>6d}".format(perf["total_wins"]))
        print("  Losses:                 {:>6d}".format(perf["total_losses"]))
        print("  Win Rate:               {:>5.1f}%".format(perf["win_rate_pct"]))
        print()
        print("  Total Wagered:        ${:>9.2f}".format(perf["cumul_wager"]))
        print("  Cumulative P&L:       ${:>+9.2f}".format(perf["cumul_pnl"]))
        print("  ROI:                  {:>+8.1f}%".format(perf["cumul_roi_pct"]))
        print()
        print("  Avg Edge at Bet:        {:>5.2f}%".format(
            perf["avg_edge"] * 100
        ))
        print("  Avg P&L per Bet:      ${:>+9.2f}".format(perf["avg_pnl_per_bet"]))
        print(sep)
        print()

    # =====================================================================
    # DATA ACCESS (for analysis)
    # =====================================================================

    def get_all_odds(self) -> pd.DataFrame:
        """Get all odds records as a DataFrame."""
        return self._read_parquet(self.odds_path)

    def get_all_signals(self) -> pd.DataFrame:
        """Get all signal records as a DataFrame."""
        return self._read_parquet(self.signals_path)

    def get_all_betslips(self) -> pd.DataFrame:
        """Get all betslip records as a DataFrame."""
        return self._read_parquet(self.betslips_path)

    def get_all_fills(self) -> pd.DataFrame:
        """Get all fill records as a DataFrame."""
        return self._read_parquet(self.fills_path)

    def get_all_settlements(self) -> pd.DataFrame:
        """Get all settlement records as a DataFrame."""
        return self._read_parquet(self.settlements_path)

    def clear_all(self) -> None:
        """Delete all parquet files. USE WITH CAUTION."""
        for path in [
            self.odds_path,
            self.signals_path,
            self.betslips_path,
            self.fills_path,
            self.settlements_path,
        ]:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    tracker = PerformanceTracker()
    tracker.print_performance()
