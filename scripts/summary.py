"""Paper trading summary dashboard.

Prints overall stats, recent settlements, open positions, and daily P&L.
Designed to be invoked by the /summary slash command.

Usage:
    python scripts/summary.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from src.paper_trading.tracker import PerformanceTracker


def main():
    tracker = PerformanceTracker(
        odds_dir=str(PROJECT_DIR / "data" / "odds"),
        signals_dir=str(PROJECT_DIR / "data" / "signals"),
    )

    perf = tracker.get_performance()
    betslips = tracker.get_all_betslips()
    settlements = tracker.get_all_settlements()

    # get_open_betslips() returns list[Betslip], convert to DataFrame
    open_list = tracker.get_open_betslips()
    if open_list:
        open_bets = pd.DataFrame([b.to_dict() for b in open_list])
    else:
        open_bets = pd.DataFrame()

    sep = "=" * 65
    thin = "-" * 65

    # ------------------------------------------------------------------
    # 1. OVERALL STATS
    # ------------------------------------------------------------------
    print()
    print(sep)
    print("  PAPER TRADING DASHBOARD  ({})".format(
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    print(sep)
    print()
    print("  Signals Generated:      {:>6d}".format(perf["total_signals"]))
    print("  Actionable Signals:     {:>6d}".format(perf["actionable_signals"]))
    print()
    print("  Bets Placed:            {:>6d}".format(
        perf["total_bets"] + perf["open_bets"]))
    print("  Settled:                {:>6d}".format(perf["settled_bets"]))
    print("  Open:                   {:>6d}".format(perf["open_bets"]))
    print()
    print("  Wins:                   {:>6d}".format(perf["total_wins"]))
    print("  Losses:                 {:>6d}".format(perf["total_losses"]))
    print("  Win Rate:               {:>5.1f}%".format(perf["win_rate_pct"]))
    print()
    print("  Total Wagered:        ${:>9.2f}".format(perf["cumul_wager"]))
    print("  Cumulative P&L:       ${:>+9.2f}".format(perf["cumul_pnl"]))
    print("  ROI:                  {:>+8.1f}%".format(perf["cumul_roi_pct"]))
    print("  Avg Edge at Bet:        {:>5.2f}%".format(perf["avg_edge"] * 100))
    print("  Avg P&L per Bet:      ${:>+9.2f}".format(perf["avg_pnl_per_bet"]))

    # ------------------------------------------------------------------
    # 2. DAILY P&L
    # ------------------------------------------------------------------
    if not settlements.empty and "settled_at" in settlements.columns:
        sett = settlements.copy()
        sett["date"] = pd.to_datetime(sett["settled_at"]).dt.date

        print()
        print(thin)
        print("  DAILY P&L")
        print(thin)

        daily = sett.groupby("date").agg(
            wins=("won", "sum"),
            total=("won", "count"),
            pnl=("pnl", "sum"),
        ).reset_index()
        daily["losses"] = daily["total"] - daily["wins"]
        daily["wins"] = daily["wins"].astype(int)

        cumul = 0.0
        for _, row in daily.iterrows():
            cumul += row["pnl"]
            print("  {} | {:>2d}-{:<2d} | day ${:>+8.2f} | cumul ${:>+8.2f}".format(
                row["date"], row["wins"], row["losses"], row["pnl"], cumul))

    # ------------------------------------------------------------------
    # 3. RECENT SETTLEMENTS (last 15)
    # ------------------------------------------------------------------
    if not settlements.empty:
        print()
        print(thin)
        print("  RECENT SETTLEMENTS (last 15)")
        print(thin)

        recent = settlements.sort_values("settled_at", ascending=False).head(15)

        # Join with betslips for wager/bucket info
        if not betslips.empty:
            bet_cols = [c for c in ["betslip_id", "bucket_label", "wager",
                                     "price_paid", "edge_at_bet", "placed_by"]
                        if c in betslips.columns]
            merged = recent.merge(betslips[bet_cols], on="betslip_id", how="left")
        else:
            merged = recent

        for _, row in merged.iterrows():
            won = "WIN " if row.get("won", False) else "LOSS"
            bucket = row.get("bucket_label", row.get("winning_bucket", "?"))
            wager = row.get("wager", 0) or 0
            strategy = str(row.get("placed_by", "paper"))
            slug = str(row.get("event_slug", "?"))
            if len(slug) > 40:
                slug = slug[:37] + "..."
            print("  {} | {:<10s} | ${:>+7.2f} | ${:>6.2f} on {:<10s} | {}".format(
                won, strategy[:10], row["pnl"], wager, bucket, slug))

    # ------------------------------------------------------------------
    # 4. OPEN POSITIONS
    # ------------------------------------------------------------------
    if not open_bets.empty:
        print()
        print(thin)
        print("  OPEN POSITIONS ({})".format(len(open_bets)))
        print(thin)

        for slug in open_bets["event_slug"].unique():
            event_bets = open_bets[open_bets["event_slug"] == slug]
            total_wager = event_bets["wager"].sum()
            print()
            print("  {} (${:.2f} wagered)".format(slug, total_wager))
            for _, b in event_bets.iterrows():
                strategy = str(b.get("placed_by", "paper"))
                print("    {:<10s} @ ${:.3f} | edge={:.1f}% | ${:.2f} | {}".format(
                    b["bucket_label"], b["price_paid"],
                    b["edge_at_bet"] * 100, b["wager"], strategy))

    # ------------------------------------------------------------------
    # 5. STRATEGY BREAKDOWN (via signal linkage)
    # ------------------------------------------------------------------
    signals = tracker.get_all_signals()
    if not settlements.empty and not betslips.empty and not signals.empty:
        print()
        print(thin)
        print("  STRATEGY BREAKDOWN")
        print(thin)

        settled_ids = set(settlements["betslip_id"])
        settled_bets = betslips[betslips["betslip_id"].isin(settled_ids)].copy()

        # Link betslip -> signal -> strategy_id
        if "signal_id" in settled_bets.columns and "strategy_id" in signals.columns:
            sig_map = signals[["signal_id", "strategy_id"]].drop_duplicates("signal_id")
            settled_bets = settled_bets.merge(sig_map, on="signal_id", how="left")
        else:
            settled_bets["strategy_id"] = settled_bets.get("placed_by", "paper")

        settled_bets = settled_bets.merge(
            settlements[["betslip_id", "won", "pnl"]],
            on="betslip_id", how="left",
        )

        # Fill missing strategy_id
        settled_bets["strategy_id"] = settled_bets["strategy_id"].fillna("unknown")

        by_strategy = settled_bets.groupby("strategy_id").agg(
            n_bets=("betslip_id", "count"),
            wins=("won", "sum"),
            total_wager=("wager", "sum"),
            total_pnl=("pnl", "sum"),
        ).reset_index()
        by_strategy["roi"] = (
            by_strategy["total_pnl"] / by_strategy["total_wager"].replace(0, 1) * 100
        ).round(1)
        by_strategy["wins"] = by_strategy["wins"].astype(int)
        by_strategy = by_strategy.sort_values("total_pnl", ascending=False)

        for _, row in by_strategy.iterrows():
            print("  {:<22s} | {:>2d} bets | {:>2d}W | ${:>+8.2f} | {:>+6.1f}% ROI".format(
                str(row["strategy_id"])[:22], row["n_bets"], row["wins"],
                row["total_pnl"], row["roi"]))

    print()
    print(sep)
    print()


if __name__ == "__main__":
    main()
