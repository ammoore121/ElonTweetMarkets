"""
Backtesting engine for categorical tweet count prediction markets.

Follows EsportsBetting pattern: separate data loading, prediction,
trade execution, settlement, and reporting.

Usage:
    from src.backtesting.engine import BacktestEngine

    engine = BacktestEngine(config={"bankroll": 1000})
    result = engine.run(model, events)
    engine.print_report(result)
    engine.save_result(result)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.backtesting.schemas import (
    BacktestResult,
    MarketSnapshot,
    PredictionSignal,
    Settlement,
    Trade,
    make_trade_id,
)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
BACKTEST_DIR = PROJECT_DIR / "data" / "backtest"
INDEX_PATH = BACKTEST_DIR / "backtest_index.json"
EVENTS_DIR = BACKTEST_DIR / "events"
RESULTS_DIR = PROJECT_DIR / "data" / "results"


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Backtesting engine for categorical tweet count prediction markets.

    Follows EsportsBetting pattern: separate data loading, prediction,
    trade execution, settlement, and reporting.
    """

    DEFAULT_CONFIG = {
        "bankroll": 1000.0,
        "kelly_fraction": 0.25,
        "min_edge": 0.05,
        "max_bet_pct": 0.05,
        "entry_hours_before_close": 24,
        "entry_window_hours": 6,
        "dry_run": False,
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, model, events: list[dict]) -> BacktestResult:
        """Run backtest across events.

        Args:
            model: Any model with a .predict(features, buckets, context) method
                   and a .name attribute.
            events: List of event dicts from the backtest index, each having
                    at least "event_slug" and optionally "ground_truth_tier".

        Returns:
            BacktestResult with all trades, settlements, and per-event detail.
        """
        event_results = []
        all_trades: list[Trade] = []
        all_settlements: list[Settlement] = []

        for evt in events:
            slug = evt["event_slug"]
            metadata, features, prices_df = self._load_event_data(slug)
            if metadata is None:
                continue

            result = self._backtest_event(
                model=model,
                metadata=metadata,
                features=features or {},
                prices_df=prices_df,
            )
            event_results.append(result)

            # Collect typed Trade / Settlement objects from the per-event dict
            for t in result.get("_trades_typed", []):
                all_trades.append(t)
            for s in result.get("_settlements_typed", []):
                all_settlements.append(s)

        # Aggregate
        agg = self._compute_aggregate_stats(event_results)

        config_info = {
            "model": model.name,
            "bankroll": self.config["bankroll"],
            "kelly_fraction": self.config["kelly_fraction"],
            "min_edge": self.config["min_edge"],
            "entry_hours_before_close": self.config["entry_hours_before_close"],
        }

        # Count wins across all settlements
        n_wins = sum(1 for s in all_settlements if s.won)

        return BacktestResult(
            model_name=model.name,
            model_version="1.0",
            config=config_info,
            n_events=agg["n_events"],
            n_traded=agg["n_events_with_bets"],
            n_bets=agg["n_total_bets"],
            n_wins=n_wins,
            total_wagered=agg["total_wagered"],
            total_pnl=agg["total_pnl"],
            roi=agg["roi_pct"],
            brier_score=agg["mean_brier"],
            log_loss=agg["mean_log_loss"],
            accuracy=agg["accuracy"],
            accuracy_str=agg["accuracy_str"],
            per_event=self._strip_typed_objects(event_results),
            trades=all_trades,
            settlements=all_settlements,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_event_data(self, event_slug: str):
        """Load metadata, features, and prices for a single event.

        Returns (metadata, features, prices_df_or_none).
        """
        event_dir = EVENTS_DIR / event_slug

        # Metadata
        meta_path = event_dir / "metadata.json"
        if not meta_path.exists():
            return None, None, None
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Features
        feat_path = event_dir / "features.json"
        features = None
        if feat_path.exists():
            with open(feat_path, "r", encoding="utf-8") as f:
                features = json.load(f)

        # Prices (optional -- loaded as DataFrame from parquet if available)
        prices = None
        prices_path = event_dir / "prices.parquet"
        if prices_path.exists():
            try:
                import pandas as pd

                prices = pd.read_parquet(prices_path)
            except Exception:
                pass

        return metadata, features, prices

    # ------------------------------------------------------------------
    # Entry price extraction
    # ------------------------------------------------------------------

    def _get_entry_prices(self, metadata: dict, prices_df) -> MarketSnapshot:
        """Get market prices at entry time.

        If price history is available, finds the snapshot closest to
        (end_date - entry_hours) hours.  Otherwise falls back to the
        final market prices in metadata (price_yes).

        Returns a MarketSnapshot with bucket_prices.
        """
        buckets = metadata.get("buckets", [])
        entry_hours = self.config["entry_hours_before_close"]
        window_hours = self.config["entry_window_hours"]
        snapshot_time = None

        bucket_prices: dict[str, float] = {}

        # Try price history first
        if prices_df is not None and not prices_df.empty:
            try:
                import pandas as pd

                end_str = metadata.get("end_date")
                if end_str:
                    end_dt = pd.Timestamp(end_str, tz="UTC")
                    target_time = end_dt - pd.Timedelta(hours=entry_hours)

                    # Window: +/- entry_window_hours around target
                    window_start = target_time - pd.Timedelta(hours=window_hours)
                    window_end = target_time + pd.Timedelta(hours=window_hours)

                    ts_col = "timestamp"
                    if ts_col in prices_df.columns:
                        # Ensure timestamp column is datetime
                        if not pd.api.types.is_datetime64_any_dtype(
                            prices_df[ts_col]
                        ):
                            prices_df = prices_df.copy()
                            prices_df[ts_col] = pd.to_datetime(
                                prices_df[ts_col], utc=True
                            )

                        windowed = prices_df[
                            (prices_df[ts_col] >= window_start)
                            & (prices_df[ts_col] <= window_end)
                        ]

                        if not windowed.empty:
                            windowed = windowed.copy()
                            windowed["_diff"] = (
                                windowed[ts_col] - target_time
                            ).abs()
                            closest = windowed.sort_values(
                                "_diff"
                            ).drop_duplicates(
                                subset=["bucket_label"], keep="first"
                            )

                            for _, row in closest.iterrows():
                                bucket_prices[str(row["bucket_label"])] = float(
                                    row["price"]
                                )

                            if bucket_prices:
                                snapshot_time = target_time.to_pydatetime()
                                return MarketSnapshot(
                                    event_slug=metadata.get("event_slug", ""),
                                    snapshot_time=snapshot_time,
                                    bucket_prices=bucket_prices,
                                    market_type=metadata.get(
                                        "market_type", ""
                                    ),
                                )
            except Exception:
                pass

        # Fallback: use final market prices from metadata
        for bucket in buckets:
            label = bucket["bucket_label"]
            price = bucket.get("price_yes")
            if price is not None:
                bucket_prices[label] = float(price)
            else:
                bucket_prices[label] = 0.0

        return MarketSnapshot(
            event_slug=metadata.get("event_slug", ""),
            snapshot_time=snapshot_time,
            bucket_prices=bucket_prices,
            market_type=metadata.get("market_type", ""),
        )

    # ------------------------------------------------------------------
    # Trade generation (Kelly criterion)
    # ------------------------------------------------------------------

    def _generate_trades(
        self,
        signal: PredictionSignal,
        snapshot: MarketSnapshot,
        buckets: list[dict],
        winning_label: str,
    ) -> tuple[list[Trade], list[Settlement]]:
        """Kelly-criterion trade generation and immediate settlement.

        For each bucket with positive edge >= min_edge, generates a Trade
        and its corresponding Settlement.

        Returns (trades, settlements) tuple.
        """
        bankroll = self.config["bankroll"]
        kelly_frac = self.config["kelly_fraction"]
        min_edge = self.config["min_edge"]
        max_bet_pct = self.config["max_bet_pct"]

        trades: list[Trade] = []
        settlements: list[Settlement] = []

        if self.config.get("dry_run", False):
            return trades, settlements

        for bucket in buckets:
            label = bucket["bucket_label"]
            model_prob = signal.predicted_probs.get(label, 0.0)
            market_price = snapshot.bucket_prices.get(label, 0.0)

            wager, edge, kelly_f = self._kelly_wager(
                model_prob, market_price, bankroll, kelly_frac, min_edge,
                max_bet_pct,
            )

            if wager <= 0:
                continue

            won = label == winning_label

            # Compute payout and P&L (keep unrounded for accurate summing)
            if won:
                if market_price > 0:
                    payout = wager / market_price
                    pnl = payout - wager
                else:
                    payout = 0.0
                    pnl = 0.0
            else:
                payout = 0.0
                pnl = -wager

            tid = make_trade_id()

            trade = Trade(
                trade_id=tid,
                event_slug=signal.event_slug,
                bucket_label=label,
                entry_price=market_price,
                model_prob=model_prob,
                edge=edge,
                kelly_fraction=kelly_f,
                wager=wager,
                shares=wager / market_price if market_price > 0 else 0.0,
                entry_time=snapshot.snapshot_time,
                model_name=signal.model_name,
            )

            settlement = Settlement(
                trade_id=tid,
                event_slug=signal.event_slug,
                bucket_label=label,
                winning_bucket=winning_label,
                won=won,
                wager=wager,
                payout=payout,
                pnl=pnl,
                settled_at=datetime.now(timezone.utc),
            )

            trades.append(trade)
            settlements.append(settlement)

        return trades, settlements

    @staticmethod
    def _kelly_wager(
        model_prob: float,
        market_prob: float,
        bankroll: float,
        kelly_fraction: float,
        min_edge: float,
        max_bet_pct: float,
    ) -> tuple[float, float, float]:
        """Compute Kelly-criterion wager for a single bucket.

        Returns (wager, edge, kelly_f) tuple.  wager=0 if no bet.
        """
        if market_prob <= 0 or market_prob >= 1:
            return 0.0, 0.0, 0.0

        edge = model_prob - market_prob
        if edge < min_edge:
            return 0.0, edge, 0.0

        kelly_f = edge / (1.0 - market_prob)

        # Apply fractional Kelly and cap at max_bet_pct of bankroll per bucket
        wager = bankroll * kelly_f * kelly_fraction
        wager = max(0.0, min(wager, bankroll * max_bet_pct))

        return wager, edge, kelly_f

    # ------------------------------------------------------------------
    # Scoring metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _brier_score(predicted_probs: dict[str, float], winning_label: str) -> float:
        """Compute Brier score for a single event.

        Brier = sum over all buckets of (predicted_prob - actual)^2
        where actual = 1 for winning bucket, 0 for others.

        Lower is better. Range [0, 2]. Random guessing with K buckets: ~(K-1)/K.
        """
        score = 0.0
        for label, prob in predicted_probs.items():
            actual = 1.0 if label == winning_label else 0.0
            score += (prob - actual) ** 2
        return score

    @staticmethod
    def _log_loss_score(
        predicted_probs: dict[str, float], winning_label: str
    ) -> Optional[float]:
        """Compute log loss for a single event.

        log_loss = -log(P(winning_bucket))
        Lower is better. Returns None if winning bucket not in predictions.
        """
        prob = predicted_probs.get(winning_label)
        if prob is None or prob <= 0:
            return None
        return -math.log(prob)

    # ------------------------------------------------------------------
    # Single-event backtest
    # ------------------------------------------------------------------

    def _backtest_event(
        self,
        model,
        metadata: dict,
        features: dict,
        prices_df,
    ) -> dict:
        """Run backtest on a single event.

        Returns a result dict with predictions, trades, and scoring.
        This dict format is identical to the original run_backtest.py output
        to preserve backward compatibility.
        """
        event_slug = metadata["event_slug"]
        buckets = metadata.get("buckets", [])
        winning_label = metadata.get("winning_bucket")
        duration_days = metadata.get("duration_days", 7)

        if not buckets:
            return {
                "event_slug": event_slug,
                "skipped": True,
                "reason": "no_buckets",
            }

        # Get entry prices
        snapshot = self._get_entry_prices(metadata, prices_df)

        # Build context for model
        context = {
            "duration_days": duration_days,
            "entry_prices": snapshot.bucket_prices,
        }

        # Get model predictions
        predicted_probs = model.predict(features, buckets, context=context)

        if not predicted_probs:
            return {
                "event_slug": event_slug,
                "skipped": True,
                "reason": "no_predictions",
            }

        # Build PredictionSignal
        signal = PredictionSignal(
            event_slug=event_slug,
            model_name=model.name,
            model_version="1.0",
            predicted_probs=predicted_probs,
            predicted_winner=max(predicted_probs, key=predicted_probs.get),
            predicted_winner_prob=max(predicted_probs.values()),
        )

        # Compute scores
        brier = self._brier_score(predicted_probs, winning_label)
        logloss = self._log_loss_score(predicted_probs, winning_label)

        # Generate trades and settlements
        trades, settlements = self._generate_trades(
            signal, snapshot, buckets, winning_label
        )

        # Aggregate trade stats
        total_wagered = sum(t.wager for t in trades)
        total_pnl = sum(s.pnl for s in settlements)
        n_bets = len(trades)

        # Build legacy-compatible trades list (for per-event JSON output)
        trades_legacy = []
        for trade, settlement in zip(trades, settlements):
            trades_legacy.append({
                "bucket_label": trade.bucket_label,
                "model_prob": round(trade.model_prob, 6),
                "market_price": round(trade.entry_price, 6),
                "edge": round(trade.edge, 6),
                "wager": round(trade.wager, 2),
                "won": settlement.won,
                "pnl": round(settlement.pnl, 2),
            })

        result = {
            "event_slug": event_slug,
            "event_title": metadata.get("event_title", ""),
            "start_date": metadata.get("start_date"),
            "end_date": metadata.get("end_date"),
            "duration_days": duration_days,
            "ground_truth_tier": metadata.get("ground_truth_tier"),
            "winning_bucket": winning_label,
            "xtracker_count": metadata.get("xtracker_count"),
            "n_buckets": len(buckets),
            "skipped": False,
            "model_name": model.name,
            "predicted_probs": {
                k: round(v, 6) for k, v in predicted_probs.items()
            },
            "predicted_winner": signal.predicted_winner,
            "predicted_winner_prob": round(signal.predicted_winner_prob, 6),
            "winning_bucket_prob": round(
                predicted_probs.get(winning_label, 0.0), 6
            ),
            "brier_score": round(brier, 6),
            "log_loss": round(logloss, 6) if logloss is not None else None,
            "n_bets": n_bets,
            "total_wagered": round(total_wagered, 2),
            "total_pnl": round(total_pnl, 2),
            "trades": trades_legacy,
            "used_temporal": (
                features.get("temporal", {}).get("rolling_avg_7d") is not None
                if features
                else False
            ),
            # Internal: typed objects for aggregation (stripped before JSON save)
            "_trades_typed": trades,
            "_settlements_typed": settlements,
        }
        return result

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_aggregate_stats(results: list[dict]) -> dict:
        """Compute aggregate statistics across all backtested events."""
        active = [r for r in results if not r.get("skipped", True)]

        if not active:
            return {
                "n_events": 0,
                "n_events_with_bets": 0,
                "n_total_bets": 0,
                "mean_brier": None,
                "mean_log_loss": None,
                "total_pnl": 0.0,
                "total_wagered": 0.0,
                "roi_pct": 0.0,
                "accuracy": 0.0,
                "accuracy_str": "0/0",
            }

        brier_scores = [
            r["brier_score"] for r in active if r["brier_score"] is not None
        ]
        log_losses = [
            r["log_loss"] for r in active if r["log_loss"] is not None
        ]
        correct = sum(
            1
            for r in active
            if r.get("predicted_winner") == r.get("winning_bucket")
        )

        total_pnl = sum(r.get("total_pnl", 0.0) for r in active)
        total_wagered = sum(r.get("total_wagered", 0.0) for r in active)
        n_with_bets = sum(1 for r in active if r.get("n_bets", 0) > 0)

        return {
            "n_events": len(active),
            "n_events_with_bets": n_with_bets,
            "n_total_bets": sum(r.get("n_bets", 0) for r in active),
            "mean_brier": (
                round(sum(brier_scores) / len(brier_scores), 6)
                if brier_scores
                else None
            ),
            "mean_log_loss": (
                round(sum(log_losses) / len(log_losses), 6)
                if log_losses
                else None
            ),
            "total_pnl": round(total_pnl, 2),
            "total_wagered": round(total_wagered, 2),
            "roi_pct": (
                round(100.0 * total_pnl / total_wagered, 2)
                if total_wagered > 0
                else 0.0
            ),
            "accuracy": round(correct / len(active), 4) if active else 0.0,
            "accuracy_str": "{}/{}".format(correct, len(active)),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_typed_objects(event_results: list[dict]) -> list[dict]:
        """Remove internal typed objects from event results before saving."""
        cleaned = []
        for r in event_results:
            c = {k: v for k, v in r.items() if not k.startswith("_")}
            cleaned.append(c)
        return cleaned

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, result: BacktestResult) -> None:
        """Print formatted backtest report.

        Output format is identical to the original run_backtest.py for
        backward compatibility.
        """
        config_info = (
            "model={}, bankroll=${:,.0f}, kelly={}, "
            "min_edge={:.0%}, entry=T-{}h".format(
                result.model_name,
                result.config.get("bankroll", 0),
                result.config.get("kelly_fraction", 0),
                result.config.get("min_edge", 0),
                result.config.get("entry_hours_before_close", 0),
            )
        )

        sep = "=" * 70
        print(sep)
        print("BACKTEST REPORT")
        print(sep)
        print()
        print("Config: {}".format(config_info))
        print()
        print(
            "Events: {} processed, {} with bets".format(
                result.n_events, result.n_traded
            )
        )
        print("Total bets: {}".format(result.n_bets))
        print()
        print("--- P&L ---")
        print("  Total wagered:  ${:,.2f}".format(result.total_wagered))
        print("  Total P&L:      ${:+,.2f}".format(result.total_pnl))
        print("  ROI:            {:+.1f}%".format(result.roi))
        print()
        print("--- Scoring ---")
        if result.brier_score is not None:
            print("  Avg Brier score: {:.4f}".format(result.brier_score))
        if result.log_loss is not None:
            print("  Avg log loss:    {:.4f}".format(result.log_loss))
        print(
            "  Accuracy (top bucket correct): {}".format(result.accuracy_str)
        )
        print()

        # Per-event table
        print("--- Per-Event Results ---")
        header = "{:<55s} {:>6s} {:>5s} {:>10s} {:>10s} {:>15s}".format(
            "Event", "Type", "Bets", "Wagered", "P&L", "Winner"
        )
        print(header)
        print("-" * len(header))
        for r in result.per_event:
            slug = r.get("event_slug", "")[:54]
            mtype = r.get("ground_truth_tier", "?")[:6]
            n_bets = r.get("n_bets", 0)
            wagered = r.get("total_wagered", 0)
            pnl = r.get("total_pnl", 0)
            winner = r.get("winning_bucket", "?")[:15]
            print(
                "{:<55s} {:>6s} {:>5d} {:>10s} {:>10s} {:>15s}".format(
                    slug,
                    mtype,
                    n_bets,
                    "${:,.2f}".format(wagered),
                    "${:+,.2f}".format(pnl),
                    winner,
                )
            )
        print(sep)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_result(self, result: BacktestResult, path: Optional[Path] = None) -> Path:
        """Save results to JSON.

        The JSON structure matches the original run_backtest.py output:
        { "config": str, "aggregate": dict, "events": list[dict] }

        Returns the path the file was saved to.
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        if path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / "backtest_{}_{}.json".format(
                result.model_name, timestamp
            )

        # Build legacy config string for backward compatibility
        config_str = (
            "model={}, bankroll=${:,.0f}, kelly={}, "
            "min_edge={:.0%}, entry=T-{}h".format(
                result.model_name,
                result.config.get("bankroll", 0),
                result.config.get("kelly_fraction", 0),
                result.config.get("min_edge", 0),
                result.config.get("entry_hours_before_close", 0),
            )
        )

        # Build legacy aggregate dict for backward compatibility
        agg = {
            "n_events": result.n_events,
            "n_events_with_bets": result.n_traded,
            "n_total_bets": result.n_bets,
            "mean_brier": (
                round(result.brier_score, 6)
                if result.brier_score is not None
                else None
            ),
            "mean_log_loss": (
                round(result.log_loss, 6)
                if result.log_loss is not None
                else None
            ),
            "total_pnl": round(result.total_pnl, 2),
            "total_wagered": round(result.total_wagered, 2),
            "roi_pct": round(result.roi, 2),
            "accuracy": round(result.accuracy, 4),
            "accuracy_str": result.accuracy_str,
        }

        output = {
            "config": config_str,
            "aggregate": agg,
            "events": result.per_event,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        return path
