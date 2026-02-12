"""
Smoke test for paper trading schemas and PerformanceTracker.

Tests the full pipeline:
    MarketOdds -> Signal -> Betslip -> Fill -> Settlement

Verifies:
1. Schema creation and field computation
2. Round-trip serialization (to_dict / from_dict)
3. PerformanceTracker persistence (parquet round-trip)
4. Full pipeline flow with realistic data
5. Deduplication (prices_match)
6. Validation rules
7. Performance summary output
"""

import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.paper_trading.schemas import MarketOdds, Signal, Betslip, Fill, Settlement, make_id
from src.paper_trading.tracker import PerformanceTracker
from src.paper_trading.validators import (
    validate_odds,
    validate_signal,
    validate_betslip,
    validate_settlement,
    ValidationError,
)


def test_schema_round_trip():
    """Test that all schemas survive to_dict -> from_dict round-trip."""
    print("--- Test 1: Schema Round-Trip Serialization ---")

    # MarketOdds
    odds = MarketOdds(
        event_slug="elon-musk-of-tweets-feb-3-9",
        event_id="evt-001",
        market_type="weekly",
        bucket_prices={
            "0-19": 0.02, "20-39": 0.05, "40-59": 0.08,
            "60-79": 0.12, "80-99": 0.15, "100-119": 0.13,
            "120-139": 0.10, "140-159": 0.08, "160-179": 0.07,
            "180-199": 0.05, "200-219": 0.04, "220-239": 0.03,
            "240-259": 0.02, "260-279": 0.02, "280+": 0.04,
        },
        implied_ev=125.5,
        total_liquidity=50000.0,
    )
    odds_dict = odds.to_dict()
    odds2 = MarketOdds.from_dict(odds_dict)
    assert odds.odds_id == odds2.odds_id
    assert odds.event_slug == odds2.event_slug
    assert odds.n_buckets == 15
    assert odds2.n_buckets == 15
    assert odds.bucket_prices == odds2.bucket_prices
    print("  MarketOdds round-trip: PASS (n_buckets={})".format(odds.n_buckets))

    # Signal
    signal = Signal(
        odds_id=odds.odds_id,
        model_id="poisson_v1",
        strategy_id="high_edge",
        predicted_probs={
            "0-19": 0.01, "20-39": 0.03, "40-59": 0.06,
            "60-79": 0.10, "80-99": 0.14, "100-119": 0.16,
            "120-139": 0.14, "140-159": 0.11, "160-179": 0.08,
            "180-199": 0.06, "200-219": 0.04, "220-239": 0.03,
            "240-259": 0.02, "260-279": 0.01, "280+": 0.01,
        },
        predicted_ev=118.0,
        best_bucket="100-119",
        best_bucket_edge=0.03,  # 0.16 - 0.13
        best_bucket_model_prob=0.16,
        best_bucket_market_price=0.13,
        meets_criteria=True,
        n_buckets_with_edge=3,
        kelly_fraction=0.0345,
        recommended_wager=8.62,
        strategy_ids="high_edge",
        n_strategies=1,
        feature_summary={"rolling_avg_7d": 95.2, "day_of_week": 1},
    )
    signal_dict = signal.to_dict()
    signal2 = Signal.from_dict(signal_dict)
    assert signal.signal_id == signal2.signal_id
    assert signal.predicted_probs == signal2.predicted_probs
    assert signal.feature_summary == signal2.feature_summary
    assert signal.meets_criteria is True
    print("  Signal round-trip: PASS (best_bucket={}, edge={:.4f})".format(
        signal.best_bucket, signal.best_bucket_edge
    ))

    # Betslip
    betslip = Betslip(
        signal_id=signal.signal_id,
        event_slug="elon-musk-of-tweets-feb-3-9",
        event_id="evt-001",
        market_type="weekly",
        bucket_label="100-119",
        price_paid=0.13,
        model_prob=0.16,
        edge_at_bet=0.03,
        wager=8.62,
        shares=66.307692,
        to_win=57.69,
        placed_by="paper",
    )
    betslip_dict = betslip.to_dict()
    betslip2 = Betslip.from_dict(betslip_dict)
    assert betslip.betslip_id == betslip2.betslip_id
    assert betslip.bucket_label == betslip2.bucket_label
    assert betslip.placed_by == "paper"
    print("  Betslip round-trip: PASS (bucket={}, wager=${:.2f})".format(
        betslip.bucket_label, betslip.wager
    ))

    # Fill
    fill = Fill(
        betslip_id=betslip.betslip_id,
        price=0.13,
        amount=8.62,
        shares=66.307692,
    )
    fill_dict = fill.to_dict()
    fill2 = Fill.from_dict(fill_dict)
    assert fill.fill_id == fill2.fill_id
    assert abs(fill.shares - fill2.shares) < 0.01
    print("  Fill round-trip: PASS (price={}, amount=${:.2f})".format(
        fill.price, fill.amount
    ))

    # Settlement (winning case)
    settlement = Settlement(
        betslip_id=betslip.betslip_id,
        event_slug="elon-musk-of-tweets-feb-3-9",
        winning_bucket="100-119",
        xtracker_count=107,
        bucket_bet="100-119",
        won=True,
        wager=8.62,
        payout=66.31,
        pnl=57.69,
        cumul_wager=8.62,
        cumul_pnl=57.69,
        cumul_roi_pct=669.37,
        total_bets=1,
        total_wins=1,
        win_rate_pct=100.0,
    )
    settlement_dict = settlement.to_dict()
    settlement2 = Settlement.from_dict(settlement_dict)
    assert settlement.settlement_id == settlement2.settlement_id
    assert settlement.won is True
    assert settlement.xtracker_count == 107
    print("  Settlement round-trip: PASS (won={}, pnl=${:.2f})".format(
        settlement.won, settlement.pnl
    ))

    print("  ALL ROUND-TRIP TESTS PASSED\n")


def test_prices_match():
    """Test MarketOdds.prices_match deduplication."""
    print("--- Test 2: Prices Match (Deduplication) ---")

    odds1 = MarketOdds(
        event_slug="test-event",
        bucket_prices={"0-19": 0.10, "20-39": 0.30, "40-59": 0.35, "60+": 0.25},
    )

    # Same prices
    odds2 = MarketOdds(
        event_slug="test-event",
        bucket_prices={"0-19": 0.10, "20-39": 0.30, "40-59": 0.35, "60+": 0.25},
    )
    assert odds1.prices_match(odds2), "Identical prices should match"
    print("  Identical prices match: PASS")

    # Within tolerance
    odds3 = MarketOdds(
        event_slug="test-event",
        bucket_prices={"0-19": 0.1004, "20-39": 0.3003, "40-59": 0.3499, "60+": 0.2502},
    )
    assert odds1.prices_match(odds3), "Prices within tolerance should match"
    print("  Within-tolerance match: PASS")

    # Exceeded tolerance
    odds4 = MarketOdds(
        event_slug="test-event",
        bucket_prices={"0-19": 0.10, "20-39": 0.35, "40-59": 0.30, "60+": 0.25},
    )
    assert not odds1.prices_match(odds4), "Changed prices should not match"
    print("  Changed prices don't match: PASS")

    # Different buckets
    odds5 = MarketOdds(
        event_slug="test-event",
        bucket_prices={"0-19": 0.10, "20-39": 0.30, "40-59": 0.35, "60-79": 0.15, "80+": 0.10},
    )
    assert not odds1.prices_match(odds5), "Different bucket sets should not match"
    print("  Different buckets don't match: PASS")

    print("  ALL DEDUPLICATION TESTS PASSED\n")


def test_validators():
    """Test validation rules."""
    print("--- Test 3: Validators ---")

    # Valid odds
    odds = MarketOdds(
        event_slug="test",
        bucket_prices={"A": 0.4, "B": 0.6},
    )
    errors = validate_odds(odds)
    assert errors == [], "Valid odds should have no errors, got: {}".format(errors)
    print("  Valid odds: PASS")

    # Invalid odds: empty slug
    bad_odds = MarketOdds(bucket_prices={"A": 0.5, "B": 0.5})
    errors = validate_odds(bad_odds)
    assert any("event_slug" in e for e in errors), "Should catch missing event_slug"
    print("  Missing event_slug: PASS")

    # Invalid odds: prices out of range
    bad_odds2 = MarketOdds(
        event_slug="test",
        bucket_prices={"A": 1.5, "B": -0.1},
    )
    errors = validate_odds(bad_odds2)
    assert len(errors) >= 2, "Should catch invalid prices"
    print("  Invalid bucket prices: PASS")

    # Valid signal
    signal = Signal(
        odds_id="abc",
        model_id="test_model",
        predicted_probs={"A": 0.4, "B": 0.6},
        best_bucket="B",
        kelly_fraction=0.05,
    )
    errors = validate_signal(signal)
    assert errors == [], "Valid signal should have no errors, got: {}".format(errors)
    print("  Valid signal: PASS")

    # Invalid signal: probs don't sum to 1
    bad_signal = Signal(
        odds_id="abc",
        model_id="test_model",
        predicted_probs={"A": 0.3, "B": 0.3},
        kelly_fraction=0.05,
    )
    errors = validate_signal(bad_signal)
    assert any("sum" in e for e in errors), "Should catch probs not summing to 1"
    print("  Probs sum check: PASS")

    # Valid betslip
    betslip = Betslip(
        signal_id="sig1",
        event_slug="test",
        bucket_label="B",
        price_paid=0.30,
        model_prob=0.40,
        edge_at_bet=0.10,
        wager=10.0,
        shares=33.333333,
    )
    errors = validate_betslip(betslip)
    assert errors == [], "Valid betslip should have no errors, got: {}".format(errors)
    print("  Valid betslip: PASS")

    # Valid settlement
    settlement = Settlement(
        betslip_id="bet1",
        event_slug="test",
        winning_bucket="B",
        bucket_bet="B",
        won=True,
        wager=10.0,
        payout=33.33,
        pnl=23.33,
    )
    errors = validate_settlement(settlement)
    assert errors == [], "Valid settlement should have no errors, got: {}".format(errors)
    print("  Valid settlement: PASS")

    # Settlement with wrong won flag
    bad_settlement = Settlement(
        betslip_id="bet1",
        winning_bucket="A",
        bucket_bet="B",
        won=True,  # Should be False
        wager=10.0,
        payout=0.0,
        pnl=-10.0,
    )
    errors = validate_settlement(bad_settlement)
    assert any("won" in e for e in errors), "Should catch inconsistent won flag"
    print("  Won flag consistency: PASS")

    print("  ALL VALIDATOR TESTS PASSED\n")


def test_full_pipeline():
    """Test the full MarketOdds -> Signal -> Betslip -> Fill -> Settlement pipeline."""
    print("--- Test 4: Full Pipeline with PerformanceTracker ---")

    # Use a temp directory for parquet files
    tmp_dir = tempfile.mkdtemp(prefix="elon_paper_trading_test_")
    print("  Temp dir: {}".format(tmp_dir))

    try:
        tracker = PerformanceTracker(data_dir=tmp_dir)

        # ===== Step 1: Record MarketOdds =====
        bucket_prices = {
            "0-39": 0.03,
            "40-79": 0.08,
            "80-119": 0.15,
            "120-159": 0.18,
            "160-199": 0.16,
            "200-239": 0.12,
            "240-279": 0.08,
            "280-319": 0.06,
            "320-359": 0.04,
            "360-399": 0.03,
            "400+": 0.07,
        }

        odds = MarketOdds(
            event_slug="elon-musk-of-tweets-feb-10-16",
            event_id="evt-weekly-feb10",
            market_type="weekly",
            bucket_prices=bucket_prices,
            implied_ev=185.0,
            total_liquidity=48000.0,
        )

        odds_id, recorded = tracker.record_odds(odds)
        assert recorded is True, "First odds should be recorded"
        print("  Step 1: Recorded odds (id={}, {} buckets)".format(odds_id, odds.n_buckets))

        # Verify dedup: same prices should not be recorded again
        odds_dup = MarketOdds(
            event_slug="elon-musk-of-tweets-feb-10-16",
            event_id="evt-weekly-feb10",
            market_type="weekly",
            bucket_prices=dict(bucket_prices),
            implied_ev=185.0,
        )
        dup_id, dup_recorded = tracker.record_odds(odds_dup)
        assert dup_recorded is False, "Duplicate odds should not be recorded"
        assert dup_id == odds_id, "Should return existing odds_id"
        print("  Step 1b: Dedup verified (not re-recorded)")

        # Verify retrieval
        retrieved_odds = tracker.get_odds(odds_id)
        assert retrieved_odds is not None
        assert retrieved_odds.n_buckets == 11
        assert retrieved_odds.event_slug == "elon-musk-of-tweets-feb-10-16"
        print("  Step 1c: Odds retrieval verified")

        # ===== Step 2: Record Signal =====
        predicted_probs = {
            "0-39": 0.01,
            "40-79": 0.04,
            "80-119": 0.10,
            "120-159": 0.16,
            "160-199": 0.20,
            "200-239": 0.18,
            "240-279": 0.12,
            "280-319": 0.08,
            "320-359": 0.05,
            "360-399": 0.03,
            "400+": 0.03,
        }

        # Identify best bucket (highest edge)
        best_bucket = None
        best_edge = -999
        for label in predicted_probs:
            edge = predicted_probs[label] - bucket_prices[label]
            if edge > best_edge:
                best_edge = edge
                best_bucket = label

        # Kelly sizing for best bucket
        model_p = predicted_probs[best_bucket]
        market_p = bucket_prices[best_bucket]
        kelly_f = best_edge / (1.0 - market_p)
        bankroll = 1000.0
        wager = bankroll * kelly_f * 0.25  # Quarter Kelly
        wager = min(wager, bankroll * 0.05)  # Max 5% of bankroll

        signal = Signal(
            odds_id=odds_id,
            model_id="poisson_v2",
            strategy_id="edge_hunter",
            predicted_probs=predicted_probs,
            predicted_ev=195.0,
            best_bucket=best_bucket,
            best_bucket_edge=best_edge,
            best_bucket_model_prob=model_p,
            best_bucket_market_price=market_p,
            meets_criteria=True,
            n_buckets_with_edge=5,
            kelly_fraction=kelly_f * 0.25,
            recommended_wager=round(wager, 2),
            strategy_ids="edge_hunter",
            n_strategies=1,
            feature_summary={"rolling_avg_7d": 180.5, "regime": "high"},
        )

        signal_id = tracker.record_signal(signal)
        print("  Step 2: Recorded signal (id={}, best={}, edge={:.4f}, wager=${:.2f})".format(
            signal_id, best_bucket, best_edge, wager
        ))

        # Verify retrieval
        retrieved_signal = tracker.get_signal(signal_id)
        assert retrieved_signal is not None
        assert retrieved_signal.predicted_probs == predicted_probs
        assert retrieved_signal.feature_summary["regime"] == "high"
        print("  Step 2b: Signal retrieval verified")

        # Verify unbetted signals
        unbetted = tracker.get_unbetted_signals()
        assert len(unbetted) == 1
        print("  Step 2c: Unbetted signals: {} (expected 1)".format(len(unbetted)))

        # ===== Step 3: Create Betslip =====
        betslip_id = tracker.create_betslip_from_signal(signal_id, placed_by="paper")
        print("  Step 3: Created betslip (id={})".format(betslip_id))

        # Verify
        betslip = tracker.get_betslip(betslip_id)
        assert betslip is not None
        assert betslip.bucket_label == best_bucket
        assert betslip.placed_by == "paper"
        assert betslip.wager > 0
        print("  Step 3b: Betslip verified (bucket={}, wager=${:.2f}, shares={:.2f})".format(
            betslip.bucket_label, betslip.wager, betslip.shares
        ))

        # Verify dedup
        assert tracker.has_open_position("elon-musk-of-tweets-feb-10-16")
        assert tracker.has_open_position("elon-musk-of-tweets-feb-10-16", best_bucket)
        assert not tracker.has_open_position("elon-musk-of-tweets-feb-10-16", "0-39")
        print("  Step 3c: Position dedup verified")

        # Unbetted signals should now be 0
        unbetted = tracker.get_unbetted_signals()
        assert len(unbetted) == 0
        print("  Step 3d: Unbetted signals: {} (expected 0)".format(len(unbetted)))

        # ===== Step 4: Add Fill =====
        fill_id = tracker.add_fill(
            betslip_id,
            price=market_p,
            amount=betslip.wager,
            notes="paper fill at market price",
        )
        print("  Step 4: Added fill (id={})".format(fill_id))

        # Verify fill aggregation
        updated_betslip = tracker.get_betslip(betslip_id)
        assert updated_betslip.fills_count == 1
        assert abs(updated_betslip.total_wager - betslip.wager) < 0.01
        print("  Step 4b: Fill aggregation verified (fills_count={})".format(
            updated_betslip.fills_count
        ))

        # ===== Step 5: Verify Open Positions =====
        open_bets = tracker.get_open_betslips()
        assert len(open_bets) == 1
        print("  Step 5: Open betslips: {} (expected 1)".format(len(open_bets)))

        # ===== Step 6: Settle the Bet =====
        # Scenario: actual count was 210 -> winning bucket is "200-239"
        actual_count = 210
        winning_bucket = "200-239"

        settlement_id = tracker.settle_bet(
            betslip_id,
            winning_bucket=winning_bucket,
            xtracker_count=actual_count,
        )
        print("  Step 6: Settled bet (id={}, winning_bucket={})".format(
            settlement_id, winning_bucket
        ))

        # Verify settlement
        settlement = tracker.get_settlement_for_betslip(betslip_id)
        assert settlement is not None
        bet_won = (best_bucket == winning_bucket)
        assert settlement.won == bet_won
        assert settlement.xtracker_count == actual_count
        print("  Step 6b: Settlement verified (won={}, pnl=${:.2f})".format(
            settlement.won, settlement.pnl
        ))

        # Verify open bets is now 0
        open_bets = tracker.get_open_betslips()
        assert len(open_bets) == 0
        print("  Step 6c: Open betslips: {} (expected 0)".format(len(open_bets)))

        # ===== Step 7: Second Bet (different event, winning scenario) =====
        print("\n  --- Adding second bet (winning scenario) ---")

        odds2 = MarketOdds(
            event_slug="elon-musk-of-tweets-feb-17-23",
            event_id="evt-weekly-feb17",
            market_type="weekly",
            bucket_prices={
                "0-39": 0.02, "40-79": 0.06, "80-119": 0.12,
                "120-159": 0.15, "160-199": 0.20, "200-239": 0.15,
                "240-279": 0.10, "280-319": 0.07, "320+": 0.13,
            },
        )
        odds2_id, _ = tracker.record_odds(odds2)

        signal2 = Signal(
            odds_id=odds2_id,
            model_id="poisson_v2",
            strategy_id="edge_hunter",
            predicted_probs={
                "0-39": 0.01, "40-79": 0.03, "80-119": 0.08,
                "120-159": 0.12, "160-199": 0.18, "200-239": 0.20,
                "240-279": 0.15, "280-319": 0.10, "320+": 0.13,
            },
            predicted_ev=230.0,
            best_bucket="200-239",
            best_bucket_edge=0.05,
            best_bucket_model_prob=0.20,
            best_bucket_market_price=0.15,
            meets_criteria=True,
            n_buckets_with_edge=4,
            kelly_fraction=0.0147,
            recommended_wager=14.71,
            strategy_ids="edge_hunter",
            n_strategies=1,
            feature_summary={"rolling_avg_7d": 210.0, "regime": "high"},
        )
        signal2_id = tracker.record_signal(signal2)
        betslip2_id = tracker.create_betslip_from_signal(signal2_id)
        tracker.add_fill(betslip2_id, price=0.15, amount=14.71)

        # This time we WIN
        tracker.settle_bet(betslip2_id, winning_bucket="200-239", xtracker_count=225)

        settlement2 = tracker.get_settlement_for_betslip(betslip2_id)
        assert settlement2.won is True
        print("  Bet 2: won={}, pnl=${:.2f}".format(settlement2.won, settlement2.pnl))

        # ===== Step 8: Performance Summary =====
        print()
        tracker.print_performance()

        # Verify performance dict
        perf = tracker.get_performance()
        assert perf["total_signals"] == 2
        assert perf["actionable_signals"] == 2
        assert perf["total_bets"] == 2
        assert perf["open_bets"] == 0
        print("  Performance dict verified: {} bets, {} wins, ROI={:.1f}%".format(
            perf["total_bets"], perf["total_wins"], perf["cumul_roi_pct"]
        ))

        # ===== Step 9: Verify parquet files exist =====
        for name in ["odds", "signals", "betslips", "fills", "settlements"]:
            path = Path(tmp_dir) / "{}.parquet".format(name)
            assert path.exists(), "{} should exist".format(path)
        print("  All 5 parquet files verified")

        # ===== Step 10: DataFrame access =====
        odds_df = tracker.get_all_odds()
        signals_df = tracker.get_all_signals()
        betslips_df = tracker.get_all_betslips()
        fills_df = tracker.get_all_fills()
        settlements_df = tracker.get_all_settlements()
        print("  DataFrames: odds={}, signals={}, betslips={}, fills={}, settlements={}".format(
            len(odds_df), len(signals_df), len(betslips_df), len(fills_df), len(settlements_df)
        ))
        assert len(odds_df) == 2  # Two events
        assert len(signals_df) == 2
        assert len(betslips_df) == 2
        assert len(fills_df) == 2
        assert len(settlements_df) == 2

        print("\n  ALL PIPELINE TESTS PASSED\n")

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  Cleaned up temp dir")


def test_settle_event():
    """Test settling all bets for an event at once."""
    print("--- Test 5: Settle Event (Bulk Settlement) ---")

    tmp_dir = tempfile.mkdtemp(prefix="elon_settle_event_test_")
    try:
        tracker = PerformanceTracker(data_dir=tmp_dir)

        # Create event with odds
        odds = MarketOdds(
            event_slug="elon-bulk-test",
            bucket_prices={"A": 0.20, "B": 0.30, "C": 0.30, "D": 0.20},
        )
        odds_id, _ = tracker.record_odds(odds)

        # Create two signals betting on different buckets
        for bucket, prob in [("B", 0.40), ("C", 0.38)]:
            signal = Signal(
                odds_id=odds_id,
                model_id="test",
                predicted_probs={"A": 0.10, "B": 0.30, "C": 0.30, "D": 0.30},
                best_bucket=bucket,
                best_bucket_edge=prob - odds.bucket_prices[bucket],
                best_bucket_model_prob=prob,
                best_bucket_market_price=odds.bucket_prices[bucket],
                meets_criteria=True,
                kelly_fraction=0.05,
                recommended_wager=10.0,
            )
            sid = tracker.record_signal(signal)
            tracker.create_betslip_from_signal(sid)

        # Should have 2 open bets
        assert len(tracker.get_open_betslips()) == 2
        print("  2 open bets created")

        # Settle all at once
        settlement_ids = tracker.settle_event("elon-bulk-test", "B", xtracker_count=45)
        assert len(settlement_ids) == 2
        print("  Settled {} bets in one call".format(len(settlement_ids)))

        # Verify: one won, one lost
        settlements = tracker.get_settlements()
        wins = sum(1 for s in settlements if s.won)
        losses = sum(1 for s in settlements if not s.won)
        assert wins == 1 and losses == 1
        print("  Results: {} win, {} loss".format(wins, losses))

        # No more open bets
        assert len(tracker.get_open_betslips()) == 0
        print("  0 open bets remaining")

        print("  SETTLE EVENT TEST PASSED\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_already_settled_guard():
    """Test that settling the same betslip twice raises an error."""
    print("--- Test 6: Double Settlement Guard ---")

    tmp_dir = tempfile.mkdtemp(prefix="elon_double_settle_test_")
    try:
        tracker = PerformanceTracker(data_dir=tmp_dir)

        odds = MarketOdds(
            event_slug="guard-test",
            bucket_prices={"X": 0.50, "Y": 0.50},
        )
        odds_id, _ = tracker.record_odds(odds)

        signal = Signal(
            odds_id=odds_id,
            model_id="test",
            predicted_probs={"X": 0.60, "Y": 0.40},
            best_bucket="X",
            best_bucket_edge=0.10,
            best_bucket_model_prob=0.60,
            best_bucket_market_price=0.50,
            meets_criteria=True,
            kelly_fraction=0.05,
            recommended_wager=10.0,
        )
        sid = tracker.record_signal(signal)
        bid = tracker.create_betslip_from_signal(sid)

        # First settle should work
        tracker.settle_bet(bid, "X")

        # Second settle should raise
        try:
            tracker.settle_bet(bid, "X")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already settled" in str(e).lower()
            print("  Double settlement blocked: PASS")

        print("  DOUBLE SETTLEMENT GUARD PASSED\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  PAPER TRADING SMOKE TEST")
    print("=" * 60)
    print()

    test_schema_round_trip()
    test_prices_match()
    test_validators()
    test_full_pipeline()
    test_settle_event()
    test_already_settled_guard()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    print()
